#!/bin/bash
set -e

source ./unans_env/bin/activate
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=true
export LITELLM_API_KEY="${LITELLM_API_KEY}"
export LITELLM_IP="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY}"

NEUCLIR_ARGS="--database_dir=./_database/neuclir_test --report_requests_path=./_database/test_requests_neuclir.jsonl --num_articles_to_crawl=25 --num_topics=25 --dataset_test_size=100 --save_dir=./generated_data/mh_neuclir --max_q_prompts_per_request=100 --max_unanswerable_qs_per_request=100 --max_node_dicts=100 --max_per_split=25"

echo "========== NEUCLIR REMAINING START: $(date) =========="
for id in 309 334 335 343 351 352 365 367 372 373 377 380 382 383 388; do
    echo "--- Running NeuCLIR id=$id: $(date) ---"
    python ./src/crumqs_generation/run.py $NEUCLIR_ARGS --exp_name=$id --ids_to_run $id || echo "*** FAILED id=$id ***"
done
echo "========== NEUCLIR DONE: $(date) =========="

echo "========== TRECRAG START: $(date) =========="
bash ./src/crumqs_generation/run_trecrag.sh
echo "========== TRECRAG DONE: $(date) =========="
