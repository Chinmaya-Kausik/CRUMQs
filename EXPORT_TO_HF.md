# Exporting CRUMQs-selected questions to HuggingFace

Merge outputs from the RunPod run (TREC RAG) with local outputs (NeuCLIR + any
older TREC RAG), apply the CRUMQs selection filter, and push the result to an
HF dataset.

## What "selected" means

`postprocess_metadata.py` (and `export_selected.py`) apply this filter per
question:

```
context_necessity       >= 3
context_sufficiency     >= 2
context_sufficiency_internal <= 1
answer_correctness      >= 2
```

Only ~4% of generated candidates pass. Those are the questions CRUMQs would
choose. Everything else is intermediate scoring data.

## Prerequisites

- `src/crumqs_generation/postprocess_metadata.py` and `export_selected.py` on
  the branch (both committed on `runpod-fixes`).
- Repo venv active with `PYTHONPATH=.`.
- SSH access to the RunPod pod (ask the user for the connection string).
- A target HF dataset repo and a logged-in `huggingface-cli` / HF token.

## Steps for the local Claude Code agent

> **Placeholders to fill in from the user:**
> - `<POD>`: ssh target for the RunPod pod (e.g. `runpod-3090` or
>   `root@<ip> -p <port>`)
> - `<HF_REPO>`: HF dataset repo (e.g. `myuser/crumqs-selected`)
> - `<HF_SPLIT>`: split name to write to (typically `train`)

### 1. Pull the export script

```bash
git checkout runpod-fixes
git pull
```

This makes `export_selected.py` available locally.

### 2. Grab the pod's pre-exported JSONL

The pod already ran the export for its completed TREC RAG runs. Copy it down:

```bash
scp <POD>:/workspace/CRUMQs/generated_data/selected_trecrag_pod.jsonl ./selected_trecrag_pod.jsonl
```

(If the pod has completed more requests since last export, ask the user
whether to refresh by running the export again on the pod first.)

### 3. Run the export locally for NeuCLIR and any local TREC RAG

```bash
PYTHONPATH=. python3 export_selected.py mh_neuclir selected_neuclir_laptop.jsonl --backfill --source-label neuclir
PYTHONPATH=. python3 export_selected.py mh_trecrag selected_trecrag_laptop.jsonl --backfill --source-label trecrag
```

`--backfill` runs `postprocess_metadata.backfill_all` first so each record has
populated `sources` / `doc_ids` before filtering.

If `mh_trecrag` doesn't exist locally (no pre-pod runs), skip that second
command and just use the pod's JSONL.

### 4. Merge with dedup (pod wins on TREC RAG conflicts)

The pod re-ran some TREC RAG IDs with fixes, so pod records take priority for
any overlapping `rr_id`. NeuCLIR and TREC RAG don't share an `rr_id`
namespace, so those concatenate cleanly.

```bash
python3 <<'PY'
import json, os

out_path = 'selected_merged.jsonl'
pod_trec_ids = set()
kept_laptop = skipped_laptop = neu = 0

with open(out_path, 'w') as out:
    # Pod TREC RAG first — its rr_ids take priority
    with open('selected_trecrag_pod.jsonl') as f:
        for line in f:
            rec = json.loads(line)
            pod_trec_ids.add(rec['rr_id'])
            out.write(line)

    # Laptop TREC RAG — skip anything the pod already has
    if os.path.exists('selected_trecrag_laptop.jsonl'):
        with open('selected_trecrag_laptop.jsonl') as f:
            for line in f:
                rec = json.loads(line)
                if rec['rr_id'] in pod_trec_ids:
                    skipped_laptop += 1
                else:
                    out.write(line); kept_laptop += 1

    # NeuCLIR (separate namespace)
    if os.path.exists('selected_neuclir_laptop.jsonl'):
        with open('selected_neuclir_laptop.jsonl') as f:
            for line in f:
                out.write(line); neu += 1

print(f'merged -> {out_path}')
print(f'  pod trecrag rr_ids: {len(pod_trec_ids)}')
print(f'  laptop trecrag kept: {kept_laptop}  skipped (dup with pod): {skipped_laptop}')
print(f'  neuclir records: {neu}')
print(f'  total lines: {sum(1 for _ in open(out_path))}')
print(f'  size: {os.path.getsize(out_path)/1e6:.2f} MB')
PY
```

**Pause after this and show the user the totals before pushing to HF.**

### 5. Push to HF

Confirm with the user: the exact `<HF_REPO>`, the target `<HF_SPLIT>`, and
whether to concatenate with existing split data or overwrite. If the HF
dataset repo already exists and has a split you're updating, load it first,
concatenate, dedup by (rr_id, question), and push:

```python
from datasets import Dataset, load_dataset, concatenate_datasets
import json

new = Dataset.from_json('selected_merged.jsonl')

# Concatenate with existing split if present
try:
    existing = load_dataset('<HF_REPO>', split='<HF_SPLIT>')
    # dedup by (source_dataset, rr_id, question) — keep first occurrence
    seen = set()
    def keep(r):
        k = (r['source_dataset'], r['rr_id'], r['question'])
        if k in seen: return False
        seen.add(k); return True
    combined = concatenate_datasets([new, existing]).filter(keep)
except Exception:
    combined = new

combined.push_to_hub('<HF_REPO>', split='<HF_SPLIT>')
```

Don't create a new HF repo unless the user explicitly confirms.

## Refreshing when the pod finishes more runs

The pod's run has ~100 TREC RAG requests total. Each completion adds another
~20 selected questions to the pool. To refresh:

1. Re-run the export on the pod (ssh in):
   `PYTHONPATH=. python3 export_selected.py mh_trecrag generated_data/selected_trecrag_pod.jsonl --backfill --source-label trecrag`
2. Repeat steps 2–5 above locally.

The HF push in step 5 will dedup so re-pushes are idempotent.
