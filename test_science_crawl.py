"""Isolated test for the science crawl pipeline.

Checks that xrxiv abstracts are actually being collected when dumps are available.
"""
import os
import sys
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.crumqs_generation.crawl_science import crawl_scientific_articles


def test_xrxiv_abstracts():
    """Test that biorxiv/medrxiv/chemrxiv abstracts are collected for a science topic."""
    topic = ["COVID-19 mRNA vaccine immune response"]

    tmp_save_dir = tempfile.mkdtemp() + "/_ood_articles"
    os.makedirs(tmp_save_dir, exist_ok=True)
    # Create dummy logging file
    with open(tmp_save_dir.replace("_ood_articles", "") + "logging.txt", "w") as f:
        pass

    try:
        articles = crawl_scientific_articles(
            topic=topic,
            save_dir=tmp_save_dir,
            articles_per_source=10,  # What the pipeline uses
            start_date="2024-01-01",
        )

        print(f"\n{'='*60}")
        print(f"TEST RESULTS for topic: {topic[0]}")
        print(f"{'='*60}")
        print(f"Total science articles: {len(articles)}")

        # Break down by source
        from collections import Counter
        sources = Counter(a['source'] for a in articles)
        print(f"By source: {dict(sources)}")

        # Check doc_id patterns
        doi_patterns = Counter()
        for a in articles:
            doc_id = a.get('doc_id', '')
            if doc_id.startswith('10.1101'):
                doi_patterns['biorxiv/medrxiv (10.1101)'] += 1
            elif 'arXiv' in doc_id or 'arxiv' in doc_id:
                doi_patterns['arxiv'] += 1
            elif doc_id.startswith('10.'):
                doi_patterns['other DOI'] += 1
            elif doc_id.startswith('http'):
                doi_patterns['URL'] += 1
            else:
                doi_patterns['other'] += 1
        print(f"By doc_id pattern: {dict(doi_patterns)}")

        # Check index files
        print(f"\nIndex file sizes:")
        for src in ['pubmed', 'arxiv', 'biorxiv', 'medrxiv', 'chemrxiv']:
            idx = f"{tmp_save_dir}/metadata/{src}/index.jsonl"
            if os.path.exists(idx):
                with open(idx) as f:
                    lines = len([l for l in f if l.strip()])
                print(f"  {src}: {lines} papers")
            else:
                print(f"  {src}: not created")

        # Success criteria: at least 5 biorxiv or medrxiv abstracts
        xrxiv_count = doi_patterns.get('biorxiv/medrxiv (10.1101)', 0)
        if xrxiv_count >= 5:
            print(f"\n✓ PASS: Got {xrxiv_count} xrxiv abstracts (>= 5)")
            return True
        else:
            print(f"\n✗ FAIL: Only {xrxiv_count} xrxiv abstracts (expected >= 5)")
            return False

    finally:
        # Cleanup
        parent = tmp_save_dir.replace("_ood_articles", "")
        if os.path.exists(parent):
            shutil.rmtree(parent)


if __name__ == '__main__':
    os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
    passed = test_xrxiv_abstracts()
    sys.exit(0 if passed else 1)
