"""Export CRUMQs-selected questions to a flat JSONL.

Applies the `postprocess_metadata.py` selection filter (context_necessity>=3,
context_sufficiency>=2, context_sufficiency_internal<=1, answer_correctness>=2)
to every completed request and emits one JSON object per selected question.

Optionally backfills source/doc_id metadata first (same backfill_all used by
postprocess_smoke_test).

Usage:
    python3 export_selected.py <dataset_subdir> <output.jsonl> [--backfill] [--source-label LABEL]

Example:
    python3 export_selected.py mh_trecrag selected_trecrag.jsonl --backfill --source-label trecrag
    python3 export_selected.py mh_neuclir selected_neuclir.jsonl --backfill --source-label neuclir
"""
import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.crumqs_generation.postprocess_metadata import backfill_all


def get_score(s):
    if isinstance(s, tuple):
        return s[1]
    if isinstance(s, list) and len(s) > 0:
        if isinstance(s[-1], (int, float)):
            return s[-1]
        if isinstance(s[0], tuple):
            return s[0][1]
        if isinstance(s[0], str) and len(s) > 1 and isinstance(s[1], int):
            return s[1]
    return None


def passes_filter(cn, cs, csi, ac):
    return (cn is not None and cs is not None and csi is not None and ac is not None
            and cn >= 3 and cs >= 2 and csi <= 1 and ac >= 2)


def export(dataset_dir, out_path, source_label):
    n_total = 0
    n_selected = 0
    per_rr = {}

    with open(out_path, 'w') as out:
        for rr_id in sorted(os.listdir(dataset_dir), key=lambda x: (x.isdigit() and int(x)) or x):
            rr_path = os.path.join(dataset_dir, rr_id)
            if not os.path.isdir(rr_path):
                continue
            pkl_path = os.path.join(
                rr_path, '_data_multihop',
                f'{rr_id}__multihop_backup_shuffled__mode_documents.pkl'
            )
            if not os.path.exists(pkl_path):
                continue
            with open(pkl_path, 'rb') as f:
                d = pickle.load(f)

            rr_total = len(d.get('qa_pairs', []))
            rr_selected = 0
            for i in range(rr_total):
                n_total += 1
                qa = d['qa_pairs'][i]
                scores = d['per_qa_scores'][i] if 'per_qa_scores' in d else {}

                if isinstance(qa, tuple):
                    q, a = qa[0], (qa[1] if len(qa) > 1 else '')
                elif isinstance(qa, dict):
                    q, a = qa.get('question', ''), qa.get('answer', '')
                else:
                    continue

                cn = get_score(scores.get('context_necessity'))
                cs = get_score(scores.get('context_sufficiency'))
                csi = get_score(scores.get('context_sufficiency_internal'))
                ac = get_score(scores.get('answer_correctness'))
                au = get_score(scores.get('answer_uniqueness'))

                if not passes_filter(cn, cs, csi, ac):
                    continue

                record = {
                    'source_dataset': source_label,
                    'rr_id': rr_id,
                    'question': q,
                    'answer': a,
                    'context': d['contexts'][i] if i < len(d.get('contexts', [])) else None,
                    'internal_context': d['internal_contexts'][i] if i < len(d.get('internal_contexts', [])) else None,
                    'metadata': d['metadata'][i] if i < len(d.get('metadata', [])) else None,
                    'cot_annotation': d['cot_annotations'][i] if i < len(d.get('cot_annotations', [])) else None,
                    'scores': {
                        'context_necessity': cn,
                        'context_sufficiency': cs,
                        'context_sufficiency_internal': csi,
                        'answer_correctness': ac,
                        'answer_uniqueness': au,
                    },
                }
                out.write(json.dumps(record, default=str) + '\n')
                n_selected += 1
                rr_selected += 1

            per_rr[rr_id] = (rr_selected, rr_total)

    return n_selected, n_total, per_rr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset_subdir', help='e.g. mh_trecrag or mh_neuclir')
    p.add_argument('out_path', help='output .jsonl path')
    p.add_argument('--backfill', action='store_true',
                   help='run postprocess_metadata.backfill_all first')
    p.add_argument('--source-label', default=None,
                   help='tag each record with this source_dataset value (defaults to dataset_subdir)')
    p.add_argument('--generated-dir', default='generated_data',
                   help='parent generated_data dir (default: generated_data)')
    args = p.parse_args()

    dataset_dir = os.path.join(args.generated_dir, args.dataset_subdir)
    if not os.path.isdir(dataset_dir):
        print(f'ERROR: {dataset_dir} not found', file=sys.stderr)
        sys.exit(1)

    source_label = args.source_label or args.dataset_subdir

    if args.backfill:
        print(f'Running backfill_all({args.generated_dir}, {args.dataset_subdir}) ...')
        backfill_all(args.generated_dir, args.dataset_subdir)

    print(f'\nExporting from {dataset_dir} -> {args.out_path} (source_label={source_label})')
    n_selected, n_total, per_rr = export(dataset_dir, args.out_path, source_label)

    print(f'\n=== Summary ===')
    print(f'  {len(per_rr)} requests processed')
    print(f'  {n_selected} / {n_total} questions selected ({100*n_selected/max(n_total,1):.1f}%)')
    print(f'  wrote {args.out_path} ({os.path.getsize(args.out_path)/1e6:.2f} MB)')
    print(f'\nPer-request retention:')
    for rr_id in sorted(per_rr, key=lambda x: (x.isdigit() and int(x)) or x):
        s, t = per_rr[rr_id]
        print(f'  rr_id={rr_id}: {s}/{t}')


if __name__ == '__main__':
    main()
