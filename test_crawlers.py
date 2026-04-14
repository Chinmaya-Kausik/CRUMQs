"""Isolated tests for the CRUMQs crawler components.

Tests:
1. News crawler (DDGS + gnews + trafilatura extraction)
2. Science crawler (PubMed + arXiv + xrxiv)
3. Preprint PDF fallback
4. End-to-end: both news and science for a single topic

Run: PYTHONPATH=. python3 test_crawlers.py
"""
import os
import sys
import time
import tempfile
import shutil
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _setup_save_dir():
    tmp = tempfile.mkdtemp()
    save_dir = os.path.join(tmp, "_ood_articles")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(tmp, "logging.txt"), "w"):
        pass
    return save_dir, tmp


def _doi_pattern(doc_id):
    if doc_id.startswith("10.1101"):
        return "biorxiv/medrxiv"
    if "arXiv" in doc_id or doc_id.startswith("arxiv"):
        return "arxiv"
    if doc_id.startswith("10."):
        return "other DOI"
    if doc_id.startswith("http"):
        return "URL"
    return "other"


def test_news_crawler():
    """Test news article crawling via DDGS + gnews + trafilatura."""
    print(f"\n{'='*60}")
    print("TEST 1: News crawler (DDGS + gnews + trafilatura)")
    print(f"{'='*60}")

    from src.crumqs_generation.crawler import crawl_gnews_articles

    save_dir, tmp = _setup_save_dir()
    try:
        start = time.time()
        articles = crawl_gnews_articles(
            keywords=["COVID-19 vaccine development"],
            save_dir=save_dir,
            articles_per_feed=10,
        )
        elapsed = time.time() - start

        print(f"Time: {elapsed:.1f}s")
        print(f"Articles extracted: {len(articles)}")
        if articles:
            print(f"Sample titles:")
            for a in articles[:3]:
                print(f"  [{len(a['text'])} chars] {a.get('title', 'no title')[:80]}")

        passed = len(articles) >= 3
        print(f"\n{'PASS' if passed else 'FAIL'}: Got {len(articles)} articles (expected >= 3)")
        return passed
    finally:
        shutil.rmtree(tmp)


def test_url_search():
    """Test that combined URL search (DDGS + gnews) returns enough URLs."""
    print(f"\n{'='*60}")
    print("TEST 2: URL search (DDGS + gnews combined)")
    print(f"{'='*60}")

    from src.crumqs_generation.crawler import get_google_news_article

    start = time.time()
    urls = get_google_news_article("Japan COVID-19 suicide rates", 10)
    elapsed = time.time() - start

    print(f"Time: {elapsed:.1f}s")
    print(f"URLs returned: {len(urls)}")
    for u in urls[:5]:
        print(f"  {u}")

    passed = len(urls) >= 5
    print(f"\n{'PASS' if passed else 'FAIL'}: Got {len(urls)} URLs (expected >= 5)")
    return passed


def test_preprint_fallback():
    """Test arXiv + OpenAlex preprint PDF fallback."""
    print(f"\n{'='*60}")
    print("TEST 3: Preprint PDF fallback (arXiv + OpenAlex)")
    print(f"{'='*60}")

    from src.crumqs_generation.crawl_science import _try_preprint_pdf

    tmp_dir = tempfile.mkdtemp()
    try:
        # Test 1: Known arXiv paper
        r1 = _try_preprint_pdf("Attention Is All You Need", "Ashish Vaswani", tmp_dir)
        size1 = os.path.getsize(r1) if r1 else 0
        print(f"arXiv paper: {'OK' if r1 else 'FAIL'} ({size1} bytes)")

        # Test 2: Nature paper with OpenAlex coverage
        r2 = _try_preprint_pdf(
            "Increase in suicide following an initial decline during the COVID-19 pandemic in Japan",
            "Takanao Tanaka", tmp_dir,
        )
        size2 = os.path.getsize(r2) if r2 else 0
        print(f"Nature/OpenAlex: {'OK' if r2 else 'FAIL'} ({size2} bytes)")

        passed = bool(r1) and size1 > 10000
        print(f"\n{'PASS' if passed else 'FAIL'}: arXiv fallback working")
        return passed
    finally:
        shutil.rmtree(tmp_dir)


def test_science_crawl_non_bio():
    """Test science crawl with a non-biomedical topic (Mars/space)."""
    print(f"\n{'='*60}")
    print("TEST 4: Science crawl (non-biomedical topic)")
    print(f"{'='*60}")

    from src.crumqs_generation.crawl_science import crawl_scientific_articles

    save_dir, tmp = _setup_save_dir()
    try:
        articles = crawl_scientific_articles(
            topic=["Mars oxygen production electrolysis"],
            save_dir=save_dir,
            articles_per_source=10,
            start_date="2024-01-01",
        )
        print(f"Science articles: {len(articles)}")
        patterns = Counter(_doi_pattern(a.get("doc_id", "")) for a in articles)
        print(f"By type: {dict(patterns)}")

        # Non-bio topic may have few xrxiv hits — accept any science articles
        passed = len(articles) >= 5
        print(f"\n{'PASS' if passed else 'FAIL'}: Got {len(articles)} science articles")
        return passed
    finally:
        shutil.rmtree(tmp)


def test_science_crawl_bio():
    """Test science crawl with biomedical topic — should hit xrxiv heavily."""
    print(f"\n{'='*60}")
    print("TEST 5: Science crawl (biomedical topic, expects xrxiv hits)")
    print(f"{'='*60}")

    from src.crumqs_generation.crawl_science import crawl_scientific_articles

    save_dir, tmp = _setup_save_dir()
    try:
        articles = crawl_scientific_articles(
            topic=["COVID-19 mRNA vaccine immune response"],
            save_dir=save_dir,
            articles_per_source=10,
            start_date="2024-01-01",
        )
        patterns = Counter(_doi_pattern(a.get("doc_id", "")) for a in articles)
        print(f"Science articles: {len(articles)}")
        print(f"By type: {dict(patterns)}")

        xrxiv = patterns.get("biorxiv/medrxiv", 0)
        passed = xrxiv >= 10
        print(f"\n{'PASS' if passed else 'FAIL'}: Got {xrxiv} xrxiv abstracts (expected >= 10)")
        return passed
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    tests = [
        ("URL search", test_url_search),
        ("News crawler", test_news_crawler),
        ("Preprint fallback", test_preprint_fallback),
        ("Science (non-bio)", test_science_crawl_non_bio),
        ("Science (bio)", test_science_crawl_bio),
    ]

    results = []
    for name, fn in tests:
        try:
            passed = fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n{name} RAISED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print(f"\n\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(p for _, p in results)
    sys.exit(0 if all_passed else 1)
