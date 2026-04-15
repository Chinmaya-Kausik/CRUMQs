# CRUMQs RunPod Setup

## Context

This repo generates unanswerable multi-hop questions using the CRUMQs pipeline.
We've made several fixes to the original code that need to be on whatever branch you clone.

### Changes from original (branch: `runpod-fixes`)
1. **Crawler**: Replaced broken Google Search HTML scraper with DDGS + gnews multi-source search. 3s timeouts via trafilatura, early exit at 10 articles per topic.
2. **Science pipeline**: Added preprint PDF fallback via arXiv + OpenAlex API. Capped xrxiv keyword search results to avoid downloading 40K+ papers per topic.
3. **Metadata bug fix**: `create_dataset.py` iterated `node_dict` keys instead of `.values()`, producing empty `sources`/`doc_ids`.
4. **Parallel run scripts**: `run_trecrag_parallel.sh` runs N workers in parallel, skipping completed requests.
5. **Post-processing**: `postprocess_metadata.py` backfills source URLs into question data.

### What's done
- **NeuCLIR**: All 19 requests COMPLETE (including 4 science re-runs with xrxiv)
- **TREC RAG**: 4/104 complete, 100 remaining

## Pod type: CPU-only is fine (recommended)

This pipeline is **API-bound**, not compute-bound. Per TREC RAG request: ~20 min of network-bound
crawling + ~100 min of OpenRouter LLM scoring calls. A GPU gives ~0% speedup because:
- LLM scoring → OpenRouter (Llama 3.3 70B) — external API, GPU irrelevant
- Embeddings / reranker → OpenAI API by default — GPU irrelevant
- Crawling, PDF parsing → network / single-threaded CPU — GPU irrelevant

The real throughput bottleneck is OpenRouter rate limits on the 70B model. Bump `NUM_WORKERS` in
`run_trecrag_parallel.sh` (3-4) to parallelize around rate limits; a GPU does not help here.

**Recommendation:** rent a CPU-only pod. Save the money.

## Persistence / moving between pods

On RunPod, `/workspace` is typically a network volume (check: `df -h /workspace` should show an
MFS mount like `mfs#*.runpod.net`). Everything under `/workspace/CRUMQs/` persists across pods
when you detach the volume and attach it to a new pod:

- ✅ `unans_env/` (venv) — persists. **Must be mounted at the same path** in the new pod, since
  venv shebangs are absolute (`#!/workspace/CRUMQs/unans_env/bin/python3.11`).
- ✅ `_database/`, `env.sh`, `generated_data/` — persist.
- ❌ `/root/.cache/pip` — does NOT persist (lives outside `/workspace`). Re-downloadable; doesn't
  matter unless you're reinstalling.

### Quickstart when reattaching an existing volume to a new pod

If you already completed the full setup on a previous pod and the volume is intact, skip the
full Setup Steps below and just do this:

```bash
cd /workspace/CRUMQs

# Sanity-check the environment survived the move
test -x unans_env/bin/python3.11 && echo "venv OK"  || echo "venv broken — see note below"
test -f _database/test_requests_trecrag2025.jsonl && echo "data OK" || echo "data missing"
test -f env.sh && echo "env.sh OK" || echo "env.sh missing — re-create from RUNPOD_SETUP step 4"

# Activate + load keys
source unans_env/bin/activate
source env.sh

# Verify the interpreter actually works (shebangs can break if volume mount path differs)
python3 -c "import llama_index, langchain, trafilatura, ddgs, gnews, paperscraper; print('imports OK')"

# Check Python 3.11 exists on the new pod (needed only if you ever reinstall)
which python3.11 || echo "WARNING: python3.11 missing on this pod — install if you plan to rebuild the venv"

# Launch
bash run_trecrag_parallel.sh > generated_data/run_trecrag_parallel.log 2>&1 &
tail -f generated_data/run_trecrag_parallel.log
```

**If the venv check fails**, the new pod mounted `/workspace` at a different path (or is missing
glibc compatible with the wheels). Easiest fix is to rebuild: `rm -rf unans_env` and redo
Setup Step 2. All data in `_database/` is untouched so this is the only thing you lose.

**If `python3 -c "import ..."` fails on any package**, something in the venv got corrupted. Usually
safer to rebuild the venv than chase the specific failure.

## Setup Steps

### 1. Clone and checkout

```bash
git clone https://github.com/pybeebee/CRUMQs.git
cd CRUMQs
git checkout runpod-fixes  # or whatever branch has our changes
```

### 2. Create Python environment

Python 3.11 is required. If the pod image doesn't ship it, install it first:
`apt-get update && apt-get install -y python3.11 python3.11-venv python3.11-dev`.

```bash
python3.11 -m venv unans_env
source ./unans_env/bin/activate
pip install --upgrade pip

# 2a) Base requirements (the repo has NO setup.py/pyproject.toml, so `pip install -e .` does
#     NOT work — use requirements.txt). This step takes ~10-15 min (downloads torch + cu12 wheels).
pip install -r requirements.txt

# 2b) Extra llama-index + crawler packages (from setup.sh, minus the broken
#     `llama-index-retrievers` which doesn't exist on PyPI).
pip install \
  "llama-index[cohere,huggingface,bm25,raptor]" \
  llama-index-llms-cohere llama-index-embeddings-cohere \
  llama-index-postprocessor-cohere-rerank \
  llama-index-packs-raptor \
  llama-index-retrievers-bm25 \
  llama-index-embeddings-huggingface \
  llama-index-llms-huggingface \
  llama-index-llms-openai-like==0.3.0 \
  termcolor ipdb paperscraper PyMuPDF \
  langchain-together langchain-anthropic langchain-huggingface \
  json-repair google-generativeai google-genai \
  gnews googlenewsdecoder ddgs

# 2c) Pin versions (resolver will warn about conflicts — those warnings are benign for this pipeline).
pip install \
  langchain-core==0.3.12 \
  langchain-openai==0.2.2 \
  openai==1.51.2 \
  pydantic==2.9.2 \
  numpy==1.26.4 \
  llama-index==0.10.22

# 2d) CPU-only pod? Strip the CUDA wheels (~3GB saved, no functional change since nothing uses GPU).
# Skip this block on GPU pods.
pip uninstall -y $(pip list | grep -E '^nvidia-|^triton' | awk '{print $1}')
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu

# 2e) Verify imports (catches corrupted wheels early before you waste hours on a crawl).
python3 -c "import llama_index, langchain, trafilatura, ddgs, gnews, paperscraper, ragas; print('imports OK')"
```

If step 2e reports a ragas `VertexAI` metaclass error on import, comment out the VertexAI import
line in `unans_env/lib/python3.11/site-packages/ragas/llms/base.py`. Recent ragas versions no
longer need this patch.

### 3. Download databases

All database files (NeuCLIR, TREC RAG, xrxiv dumps) are bundled in a GitHub release:

```bash
# Download and extract database tarball (148MB compressed, 477MB extracted)
gh release download v0.1-data --repo Chinmaya-Kausik/CRUMQs --pattern '*.tar.gz'
tar xzf crumqs_database.tar.gz
rm crumqs_database.tar.gz
```

This gives you:
- `_database/neuclir_test/` — NeuCLIR test documents
- `_database/trecrag2025/` — TREC RAG 2025 documents (105K articles)
- `_database/test_requests_neuclir.jsonl` — NeuCLIR request definitions
- `_database/test_requests_trecrag2025.jsonl` — TREC RAG request definitions
- `_database/biorxiv.jsonl` — biorxiv dump (276MB)
- `_database/medrxiv.jsonl` — medrxiv dump (90MB)
- `_database/chemrxiv.jsonl` — chemrxiv dump (12MB)

### 4. Set environment variables

Put these in `env.sh` in the repo root (it's gitignored, so it won't leak secrets, and it
persists with the volume across pods):

```bash
cat > env.sh <<'EOF'
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=true
export LITELLM_API_KEY="<your-openrouter-key>"
export LITELLM_IP="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="<your-openai-key>"
EOF
chmod 600 env.sh  # some filesystems ignore chmod; gitignore is the real protection
source env.sh
```

After this, any future session just needs `source env.sh` to re-load keys.

### 5. Transfer completed results (optional)

If you want to skip re-running NeuCLIR and already-done TREC RAG requests,
copy `generated_data/` from the local machine. The parallel script auto-skips
requests that have 5+ .pkl files in `_data_multihop/`.

### 6. Verify the pipeline with tests

Before running the full pipeline, sanity-check the crawler and science pipeline:

```bash
PYTHONPATH=. python3 test_crawlers.py
```

This runs 5 isolated tests (URL search, news crawler, preprint fallback, science crawl
non-bio, science crawl bio). Takes ~5-10 minutes. All should PASS.

For a quick science-only check:

```bash
PYTHONPATH=. python3 test_science_crawl.py
```

### 7. Run TREC RAG

```bash
# Run in background with 2 parallel workers (configurable in the script)
bash run_trecrag_parallel.sh > generated_data/run_trecrag_parallel.log 2>&1 &

# Monitor:
tail -f generated_data/run_trecrag_parallel.log
```

### 8. Post-process when done

```bash
python src/crumqs_generation/postprocess_metadata.py
```

## Key files

- `run_trecrag_parallel.sh` — parallel TREC RAG runner (set NUM_WORKERS)
- `src/crumqs_generation/crawler.py` — web article crawler (DDGS + gnews)
- `src/crumqs_generation/crawl_science.py` — science article crawler (PubMed + arXiv + xrxiv + preprint fallback)
- `src/crumqs_generation/create_dataset.py` — question generation + scoring
- `src/crumqs_generation/postprocess_metadata.py` — backfill source metadata
- `src/crumqs_generation/run.py` — main pipeline entry point

## Expected timing

- ~2 hours per TREC RAG request (20 min crawl + 100 min LLM scoring)
- 100 remaining requests / 2 workers = ~100 hours (~4 days)
- LLM calls go to OpenRouter (Llama 3.3 70B) — rate limits may slow things down

## Troubleshooting

- **`pip install -e .` fails with "not a Python project"**: The repo has no `setup.py` /
  `pyproject.toml`. Use `pip install -r requirements.txt` instead (see Step 2).
- **`llama-index-retrievers` not found**: Doesn't exist on PyPI. Remove it from the install
  list. `llama-index-retrievers-bm25` is the real package name.
- **venv shebang errors like `bad interpreter: /workspace/.../python3.11`** after moving to a
  new pod: the volume is mounted at a different path. Either re-mount at `/workspace` or
  rebuild the venv (step 2).
- **"No module named 'gnews'"**: `pip install gnews googlenewsdecoder ddgs`.
- **xrxiv "No dump found"**: Download dumps per step 3. The import-time warning from
  paperscraper is harmless.
- **429 rate limit errors**: Reduce NUM_WORKERS to 1 (or increase cautiously if OpenRouter
  allows — it's the main throughput knob).
- **PDF download stuck on thousands of papers**: The xrxiv index cap fix should prevent this.
  If it recurs, delete the bloated `_ood_articles/metadata/<source>/index.jsonl` and re-run.
- **JSON decode errors**: Usually a transient OpenRouter issue. The script continues to the
  next request.
- **Elsevier / Wiley 400/403 errors during test_crawlers.py test 5**: Expected — those are
  paywalled sources and we don't have institutional API keys. The test should still PASS as
  long as biorxiv/medrxiv/arXiv/PubMed provide enough articles (they do).
