"""Post-process CRUMQs question data to backfill source/doc_id metadata.

The original code had a bug where `for node in node_dict` iterated over dict keys
(strings) instead of values (Document objects), producing empty sources/doc_ids.

This script matches context article text back to the crawled articles to recover
the correct metadata.
"""

import json
import pickle
import os
import re
from collections import defaultdict


def load_articles(articles_path):
    """Load crawled articles from full_articles.json."""
    with open(articles_path) as f:
        return json.load(f)


def extract_context_chunks(context_str):
    """Split a context string like 'Article 1: ... Article 2: ...' into chunks."""
    # Split on "Article N:" pattern
    parts = re.split(r'Article \d+:\s*', context_str)
    return [p.strip() for p in parts if p.strip()]


def match_chunk_to_article(chunk, articles, min_overlap=50):
    """Find the best matching article for a context chunk.
    Tries multiple snippets from the chunk for robust matching.
    """
    # Try several positions in the chunk to handle tables/numbers at the start
    for start in [0, 50, 100, 200]:
        snippet = chunk[start:start + min_overlap].strip()
        if not snippet or len(snippet) < 20:
            continue
        for article in articles:
            text = article.get('text', '')
            if snippet in text:
                return {
                    'source': article.get('source', ''),
                    'doc_id': article.get('doc_id', ''),
                    'title': article.get('title', ''),
                    'topic': article.get('topic', ''),
                }
    return None


def backfill_metadata_for_request(rr_id, data_dir, articles_dir):
    """Backfill sources/doc_ids for a single request's question data."""
    # Load articles
    articles_path = os.path.join(articles_dir, 'full_articles.json')
    if not os.path.exists(articles_path):
        print(f"  No articles found for {rr_id}, skipping")
        return None

    articles = load_articles(articles_path)

    # Load question data
    pkl_path = os.path.join(
        data_dir, f'{rr_id}__multihop_backup_shuffled__mode_documents.pkl'
    )
    if not os.path.exists(pkl_path):
        print(f"  No question data found for {rr_id}, skipping")
        return None

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    updated = 0
    for i in range(len(data['metadata'])):
        meta = data['metadata'][i]

        # Skip if all sources are already populated (no empty strings)
        if meta.get('sources') and all(s for s in meta['sources']):
            continue

        # Match context chunks to articles
        context = data['contexts'][i]
        chunks = extract_context_chunks(context)

        sources = []
        doc_ids = []
        for chunk in chunks:
            match = match_chunk_to_article(chunk, articles)
            if match:
                sources.append(match['source'])
                doc_ids.append(match['doc_id'])
            else:
                sources.append('')
                doc_ids.append('')

        meta['sources'] = sources
        meta['doc_ids'] = doc_ids
        updated += 1

    print(f"  {rr_id}: backfilled {updated}/{len(data['metadata'])} questions")
    return data


def backfill_all(generated_data_dir, save_dir_name='mh_neuclir'):
    """Backfill metadata for all completed requests."""
    base_dir = os.path.join(generated_data_dir, save_dir_name)
    for rr_id in sorted(os.listdir(base_dir)):
        rr_path = os.path.join(base_dir, rr_id)
        if not os.path.isdir(rr_path):
            continue

        data_dir = os.path.join(rr_path, '_data_multihop')
        articles_dir = os.path.join(rr_path, '_ood_articles')
        pkl_path = os.path.join(
            data_dir, f'{rr_id}__multihop_backup_shuffled__mode_documents.pkl'
        )
        if not os.path.exists(pkl_path):
            continue

        data = backfill_metadata_for_request(rr_id, data_dir, articles_dir)
        if data:
            with open(pkl_path, 'wb') as f:
                pickle.dump(data, f)


def postprocess_smoke_test(smoke_test_path, generated_data_dir, save_dir_name='mh_neuclir'):
    """Re-export smoke test with backfilled metadata."""
    import random
    random.seed(42)

    # First backfill the pkl files
    backfill_all(generated_data_dir, save_dir_name)

    # Then re-export
    all_filtered = []
    base_dir = os.path.join(generated_data_dir, save_dir_name)

    for rr_id in sorted(os.listdir(base_dir)):
        rr_path = os.path.join(base_dir, rr_id)
        pkl_path = os.path.join(
            rr_path, '_data_multihop',
            f'{rr_id}__multihop_backup_shuffled__mode_documents.pkl'
        )
        if not os.path.exists(pkl_path):
            continue

        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)

        for i in range(len(d['qa_pairs'])):
            qa = d['qa_pairs'][i]
            scores = d['per_qa_scores'][i]

            if isinstance(qa, tuple):
                q, a = qa[0], qa[1] if len(qa) > 1 else ''
            elif isinstance(qa, dict):
                q, a = qa.get('question', ''), qa.get('answer', '')
            else:
                continue

            def get_score(s):
                if isinstance(s, tuple): return s[1]
                if isinstance(s, list) and len(s) > 0:
                    if isinstance(s[-1], (int, float)): return s[-1]
                    if isinstance(s[0], tuple): return s[0][1]
                    if isinstance(s[0], str) and len(s) > 1 and isinstance(s[1], int): return s[1]
                return None

            cn = get_score(scores.get('context_necessity'))
            cs = get_score(scores.get('context_sufficiency'))
            csi = get_score(scores.get('context_sufficiency_internal'))
            ac = get_score(scores.get('answer_correctness'))
            au = get_score(scores.get('answer_uniqueness'))

            if None in [cn, cs, csi, ac]:
                continue

            if cn >= 3 and cs >= 2 and csi <= 1 and ac >= 2:
                all_filtered.append({
                    'rr_id': rr_id,
                    'question': q,
                    'answer': a,
                    'context': d['contexts'][i],
                    'internal_context': d['internal_contexts'][i],
                    'metadata': d['metadata'][i],
                    'cot_annotation': d['cot_annotations'][i],
                    'scores': {
                        'context_necessity': cn,
                        'context_sufficiency': cs,
                        'context_sufficiency_internal': csi,
                        'answer_correctness': ac,
                        'answer_uniqueness': au,
                    },
                })

    print(f'\nTotal filtered questions: {len(all_filtered)}')
    sample = all_filtered if len(all_filtered) <= 100 else random.sample(all_filtered, 100)

    with open(smoke_test_path, 'w') as f:
        json.dump(sample, f, indent=2, default=str)
    print(f'Saved {len(sample)} questions to {smoke_test_path}')

    return all_filtered


if __name__ == '__main__':
    postprocess_smoke_test(
        smoke_test_path='./generated_data/smoke_test_100.json',
        generated_data_dir='./generated_data',
        save_dir_name='mh_neuclir',
    )
