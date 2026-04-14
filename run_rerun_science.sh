#!/bin/bash

source ./unans_env/bin/activate
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=true
export LITELLM_API_KEY="${LITELLM_API_KEY}"
export LITELLM_IP="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY}"

NEUCLIR_ARGS="--database_dir=./_database/neuclir_test --report_requests_path=./_database/test_requests_neuclir.jsonl --num_articles_to_crawl=25 --num_topics=25 --dataset_test_size=100 --save_dir=./generated_data/mh_neuclir --max_q_prompts_per_request=100 --max_unanswerable_qs_per_request=100 --max_node_dicts=100 --max_per_split=25"

echo "========== RERUN WITH SCIENCE START: $(date) =========="

# Run one at a time to avoid rate limit issues (TREC RAG running in parallel)
for id in 300 308 334 335; do
    echo "$(date): Starting rerun id=$id"
    python ./src/crumqs_generation/run.py $NEUCLIR_ARGS --exp_name=$id --ids_to_run $id \
        > ./generated_data/mh_neuclir/${id}/rerun.log 2>&1
    echo "$(date): Finished rerun id=$id"
done

echo "========== RERUN WITH SCIENCE DONE: $(date) =========="
