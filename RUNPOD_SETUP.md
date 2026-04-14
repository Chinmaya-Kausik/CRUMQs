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

## Setup Steps

### 1. Clone and checkout

```bash
git clone https://github.com/pybeebee/CRUMQs.git
cd CRUMQs
git checkout runpod-fixes  # or whatever branch has our changes
```

### 2. Create Python environment

```bash
# Python 3.11 required
python3.11 -m venv unans_env
source ./unans_env/bin/activate

# Install dependencies
pip install -e .
pip install gnews googlenewsdecoder ddgs

# Pin versions to avoid conflicts
pip install langchain-core==0.3.12 langchain-openai==0.2.2 openai==1.51.2

# Patch ragas if needed (VertexAI metaclass fix)
# Check if ragas/llms/base.py has a VertexAI import error and comment it out
```

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

```bash
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=true
export LITELLM_API_KEY="<your-openrouter-key>"
export LITELLM_IP="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="<your-openai-key>"
```

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

- **"No module named 'gnews'"**: `pip install gnews googlenewsdecoder ddgs`
- **xrxiv "No dump found"**: Download dumps per step 3. The import-time warning from paperscraper is harmless.
- **429 rate limit errors**: Reduce NUM_WORKERS to 1
- **PDF download stuck on thousands of papers**: The xrxiv index cap fix should prevent this. If it recurs, delete the bloated `_ood_articles/metadata/<source>/index.jsonl` and re-run.
- **JSON decode errors**: Usually a transient OpenRouter issue. The script continues to the next request.
