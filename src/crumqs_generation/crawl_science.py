import os, sys
import contextlib
from sympy import N
from termcolor import colored
import fitz 
import pandas as pd

from src.crumqs_generation.utils_deduplication import log

@contextlib.contextmanager
def suppress_output(enabled=True):
    if not enabled:
        yield   # Do nothing
    else:
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = devnull
                sys.stderr = devnull
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

with suppress_output():
    from paperscraper.get_dumps import biorxiv, medrxiv, chemrxiv
    from paperscraper.pubmed import get_and_dump_pubmed_papers
    from paperscraper.arxiv import get_and_dump_arxiv_papers
    from paperscraper.xrxiv.xrxiv_query import XRXivQuery
    from paperscraper.scholar import get_and_dump_scholar_papers
    from paperscraper.pdf import save_pdf_from_dump

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def _try_preprint_pdf(title, authors_str, pdf_dir):
    """Try to find and download an open-access PDF for a paper.

    Strategy:
    1. arXiv search by title + first author (most reliable for preprints)
    2. OpenAlex API to find OA PDF across all repositories (biorxiv, medrxiv, chemrxiv, PMC, etc.)

    Returns the path to the downloaded PDF, or None if not found.
    """
    import arxiv
    import re
    import requests as req

    first_author = ""
    if authors_str:
        first_author = authors_str.split(",")[0].strip()
        first_author = first_author.split()[-1] if first_author else ""

    clean_title = re.sub(r'[^\w\s]', '', title).strip()
    if len(clean_title) < 10:
        return None

    # 1. Try arXiv (open access, reliable PDFs)
    query = f'ti:"{clean_title[:200]}"'
    if first_author:
        query += f' AND au:{first_author}'
    try:
        results = list(arxiv.Client().results(
            arxiv.Search(query=query, max_results=1)
        ))
        if results:
            paper = results[0]
            if clean_title[:50].lower() in paper.title.lower() or paper.title.lower()[:50] in clean_title.lower():
                pdf_filename = paper.get_short_id() + ".pdf"
                pdf_filepath = os.path.join(pdf_dir, pdf_filename)
                if not os.path.exists(pdf_filepath):
                    paper.download_pdf(dirpath=pdf_dir, filename=pdf_filename)
                if os.path.exists(pdf_filepath):
                    return pdf_filepath
    except Exception:
        pass

    # 2. Try OpenAlex (indexes biorxiv, medrxiv, chemrxiv, PMC, institutional repos)
    try:
        resp = req.get(
            'https://api.openalex.org/works',
            params={'search': title[:200], 'per_page': 1},
            timeout=10,
        )
        if resp.status_code == 200:
            works = resp.json().get('results', [])
            if works:
                work = works[0]
                # Verify title match
                work_title = re.sub(r'[^\w\s]', '', work.get('title', '')).strip()
                if clean_title[:50].lower() not in work_title.lower() and work_title[:50].lower() not in clean_title.lower():
                    return None

                # Collect all candidate PDF URLs
                pdf_urls = []
                oa_url = work.get('open_access', {}).get('oa_url', '')
                if oa_url and oa_url.endswith('.pdf'):
                    pdf_urls.append(oa_url)
                for loc in work.get('locations', []):
                    pdf_url = loc.get('pdf_url')
                    if pdf_url:
                        pdf_urls.append(pdf_url)
                    # biorxiv/medrxiv landing pages can be converted to PDF URLs
                    landing = loc.get('landing_page_url', '')
                    if '10.1101' in landing and ('biorxiv' in landing or 'medrxiv' in landing):
                        pdf_urls.append(landing + '.full.pdf')

                # Try downloading the first available PDF
                for pdf_url in pdf_urls:
                    try:
                        pdf_resp = req.get(pdf_url, timeout=10)
                        if pdf_resp.status_code == 200 and len(pdf_resp.content) > 5000:
                            safe_name = re.sub(r'[^\w]', '_', clean_title[:60])
                            pdf_filepath = os.path.join(pdf_dir, f"oa_{safe_name}.pdf")
                            with open(pdf_filepath, "wb") as f:
                                f.write(pdf_resp.content)
                            return pdf_filepath
                    except Exception:
                        continue
    except Exception:
        pass

    return None

def index_scientific_articles(
        start_date="2024-01-01",
        save_dir="./_database"
    ):
    """Check for pre-downloaded xrxiv dump indexes.

    These dumps are large (100K+ papers) and take 30+ minutes to download.
    Run this separately before the pipeline if you want xrxiv coverage:

        python -c "from paperscraper.get_dumps import biorxiv, medrxiv, chemrxiv; \\
            biorxiv(start_date='2024-01-01', save_path='./_database/biorxiv.jsonl'); \\
            medrxiv(start_date='2024-01-01', save_path='./_database/medrxiv.jsonl'); \\
            chemrxiv(start_date='2024-01-01', save_path='./_database/chemrxiv.jsonl')"
    """
    for name, path in [
        ("MedrXiv", f"{save_dir}/medrxiv.jsonl"),
        ("BiorXiv", f"{save_dir}/biorxiv.jsonl"),
        ("ChemrXiv", f"{save_dir}/chemrxiv.jsonl"),
    ]:
        if os.path.exists(path):
            print(colored(f"{name} dump found at {path}", "green"))
        else:
            print(colored(f"{name} dump not found at {path} — skipping. Pre-download for xrxiv coverage.", "yellow"))


def crawl_scientific_articles(
        topic: list, 
        save_dir: str,  # ".../_ood_articles"
        articles_per_source: int = 10, 
        max_crawled_articles: int = 2500,
        start_date: str="2024-01-01",
        restrict_crawl_fields=False
    ):

    logging_file = save_dir.replace("_ood_articles", "") + "logging.txt"

    articles = []
    fields = None 
    if restrict_crawl_fields:
        fields = ['title', 'abstract']
    
    index_scientific_articles(start_date=start_date)

    # Get PubMed Articles
    os.makedirs(f"{save_dir}/metadata/pubmed", exist_ok=True)
    pubmed_index_path = f'{save_dir}/metadata/pubmed/index.jsonl'
    try:
        get_and_dump_pubmed_papers(
            topic, 
            output_filepath=pubmed_index_path,
            max_results=articles_per_source,
            start_date="2024/01/01"
        )
        log(colored(f"Finished crawling PubMed articles for topic: {topic[0]}!", "magenta"), logging_file)
    except Exception as e: 
        log(colored(f"Error when getting PubMed articles: {e}", "red"), logging_file)

    # Get arXiv Articles
    os.makedirs(f"{save_dir}/metadata/arxiv", exist_ok=True)
    arxiv_index_path = f'{save_dir}/metadata/arxiv/index.jsonl'
    try:
        get_and_dump_arxiv_papers(
            topic, 
            output_filepath=arxiv_index_path,
            max_results=articles_per_source,
            start_date="2024-01-01"
        )
        log(colored(f"Finished crawling arXiv articles for topic: {topic[0]}!", "magenta"), logging_file)
    except Exception as e: 
        log(colored(f"Error when getting arXiv articles: {e}", "red"), logging_file)

    # Get Bio/Chem/Med Articles (skip if dump not available)
    xrxiv_sources = [
        ("biorxiv", "./_database/biorxiv.jsonl"),
        ("chemrxiv", "./_database/chemrxiv.jsonl"),
        ("medrxiv", "./_database/medrxiv.jsonl"),
    ]
    xrxiv_index_paths = {}
    for source_name, dump_path in xrxiv_sources:
        index_dir = f"{save_dir}/metadata/{source_name}"
        index_path = f"{index_dir}/index.jsonl"
        xrxiv_index_paths[source_name] = index_path

        if not os.path.exists(dump_path):
            log(colored(f"Skipping {source_name} (dump not available at {dump_path})", "yellow"), logging_file)
            continue

        os.makedirs(index_dir, exist_ok=True)
        try:
            # Split topic into individual keywords for AND search
            # (XRXivQuery does substring match per keyword, AND across list)
            topic_str = topic[0] if isinstance(topic, list) else topic
            keywords = [w for w in topic_str.replace('-', ' ').split() if len(w) > 3]
            if not keywords:
                keywords = [topic_str]  # fallback

            querier = XRXivQuery(dump_path)
            querier.search_keywords(
                keywords,
                output_filepath=index_path,
                fields=fields,
            )
            # Cap xrxiv results (abstracts), but allow more than 1 so we get real coverage
            xrxiv_abstract_cap = 50
            n_papers = 0
            if os.path.exists(index_path) and os.path.getsize(index_path) > 10:
                df = pd.read_json(index_path, lines=True)
                n_papers = len(df)
                if n_papers > xrxiv_abstract_cap:
                    df.head(xrxiv_abstract_cap).to_json(index_path, orient='records', lines=True)
                    log(colored(f"Capped {source_name} from {n_papers} to {xrxiv_abstract_cap}", "yellow"), logging_file)
                    n_papers = xrxiv_abstract_cap
            log(colored(f"Finished crawling {source_name} for topic: {topic_str}! ({n_papers} papers)", "magenta"), logging_file)
        except Exception as e:
            log(colored(f"Error when getting {source_name} articles: {e}", "red"), logging_file)

    biorxiv_index_path = xrxiv_index_paths["biorxiv"]
    chemrxiv_index_path = xrxiv_index_paths["chemrxiv"]
    medrxiv_index_path = xrxiv_index_paths["medrxiv"]

    # Get Article PDFs — try save_pdf_from_dump first, then preprint fallback
    pdf_path = f"{save_dir}/pdfs/{topic[0]}"
    os.makedirs(pdf_path, exist_ok=True)
    all_index_paths = [
        pubmed_index_path,
        arxiv_index_path,
        biorxiv_index_path,
        chemrxiv_index_path,
        medrxiv_index_path,
    ]

    # First pass: try direct PDF download via DOI
    # Cap PDF download attempts to articles_per_source (PDF downloads are expensive / often blocked)
    pdf_download_cap = articles_per_source
    for dump_path in all_index_paths:
        if not os.path.exists(dump_path) or os.path.getsize(dump_path) < 10:
            continue
        try:
            df = pd.read_json(dump_path, lines=True)
            if len(df) > pdf_download_cap:
                capped_path = dump_path + '.pdfcap.jsonl'
                df.head(pdf_download_cap).to_json(capped_path, orient='records', lines=True)
                save_pdf_from_dump(
                    dump_path=capped_path,
                    pdf_path=pdf_path,
                    key_to_save='doi',
                )
                os.remove(capped_path)
            else:
                save_pdf_from_dump(
                    dump_path=dump_path,
                    pdf_path=pdf_path,
                    key_to_save='doi',
                )
        except Exception as e:
            log(colored(f"Warning: failed to save PDFs from {dump_path}: {e}", "yellow"), logging_file)

    # Second pass: for papers without PDFs, try preprint servers
    downloaded_dois = {f.replace(".pdf", "") for f in os.listdir(pdf_path) if f.endswith(".pdf")}
    preprint_fallback_count = 0
    for dump_path in all_index_paths:
        if not os.path.exists(dump_path):
            continue
        try:
            papers_df = pd.read_json(dump_path, lines=True)
            for _, row in papers_df.iterrows():
                doi = str(row.get('doi', ''))
                # Skip if we already have this PDF
                if doi.replace("/", "_") in downloaded_dois or doi in downloaded_dois:
                    continue
                title = str(row.get('title', ''))
                authors = str(row.get('authors', ''))
                if not title or len(title) < 10:
                    continue
                result = _try_preprint_pdf(title, authors, pdf_path)
                if result:
                    preprint_fallback_count += 1
        except Exception:
            pass
    log(colored(f"Finished saving PDFs for topic: {topic[0]}! ({preprint_fallback_count} via preprint fallback)", "magenta"), logging_file)

    # Get Article Texts & Port to Json
    count = 0
    for filename in os.listdir(pdf_path):
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(os.path.join(pdf_path, filename))
            articles.append({
                "text": text,
                'source': 'science', 
                'doc_id': filename.replace(".pdf", ""),
            })
            count += 1
    # Append abstracts from all xrxiv sources (biorxiv, medrxiv, chemrxiv)
    xrxiv_abstract_count = 0
    for source_name, index_path in [
        ("biorxiv", biorxiv_index_path),
        ("medrxiv", medrxiv_index_path),
        ("chemrxiv", chemrxiv_index_path),
    ]:
        if not os.path.exists(index_path) or os.path.getsize(index_path) < 10:
            continue
        try:
            df = pd.read_json(index_path, lines=True)
            if 'abstract' not in df.columns or 'doi' not in df.columns:
                continue
            papers = df['abstract'].to_list()
            dois = df['doi'].to_list()
            articles += [
                {"text": x, "source": "science", "doc_id": doi}
                for x, doi in zip(papers, dois)
                if x and len(str(x)) > 100
            ]
            xrxiv_abstract_count += len(papers)
            count += len(papers)
        except Exception as e:
            log(colored(f"Didn't add {source_name} abstracts: {e}", "yellow"), logging_file)
    log(colored(f"Appended {xrxiv_abstract_count} xrxiv abstracts for topic: {topic[0]}!", "magenta"), logging_file)
    log(colored(f"Finished converting PDFs to JSONs for topic: {topic[0]}!", "magenta"), logging_file)

    # Return List of Json's of Articles
    return articles


