#!/usr/bin/env bash
# M2/M3: generate answers for a suite with every pipeline, then judge them.
#   scripts/m2_run.sh <suite.jsonl> <pipelines-template.yaml> <tag> [extra gen args]
# One GPU (the gpu QOS allows one per job): vLLM shares it with the in-process retrieval encoders
# and NLI verifier during generation; the judges run after the generator is stopped.
# With tag "dev" the two-hop splice is first tuned on dev and scored once on test and full.
set -euo pipefail
source slurm/serve.sh
suite=$1; pipes_tmpl=$2; tag=$3; shift 3
R=results/m2; M1=results/m1; mkdir -p "$R"
cli() { in_image python3 -m rageval.cli "$@"; }
EMB=$(in_image python3 -c "import json; print(json.load(open('$M1/selected.json'))['selected']['embedder_slug'])")
UTIL=${GEN_UTIL:-0.55}

start_vllm Qwen/Qwen3-8B 8000 0 --max-model-len 8192 --gpu-memory-utilization "$UTIL" --seed 0
LLM=(--llm-url http://127.0.0.1:8000 --llm-model Qwen/Qwen3-8B)

if [ "$tag" = dev ]; then
  cli retrieval materialize --selected "$M1/selected.json" --models-template configs/models.yaml \
    --pipelines-template "$pipes_tmpl" --out-dir "$R/${tag}_cfg0" > /dev/null
  cli retrieval multihop --suite data/hotpot100k/dev.jsonl --models "$R/${tag}_cfg0/models.yaml" \
    --selected "$M1/selected.json" "${LLM[@]}" --candidates "artifacts/cands/dev__$EMB.jsonl" \
    --plan-k 3 --plan-k 5 --keep 2 --keep 3 --keep 4 --keep 5 --keep 6 --take 1 --take 2 --take 3 --take 4 \
    --out "$M1/multihop_dev.json" --rows-out "$M1/multihop_dev.rows.jsonl"
  read -r PK KP TK < <(in_image python3 -c "import json; b=json.load(open('$M1/multihop_dev.json'))['best']; print(b['plan_k'], b['keep'], b['take'])")
  for s in test full; do
    cli retrieval multihop --suite "data/hotpot100k/$s.jsonl" --models "$R/${tag}_cfg0/models.yaml" \
      --selected "$M1/selected.json" "${LLM[@]}" --candidates "artifacts/cands/${s}__$EMB.jsonl" \
      --plan-k "$PK" --keep "$KP" --take "$TK" --out "$M1/multihop_$s.json" --rows-out "$M1/multihop_$s.rows.jsonl"
  done
fi

in_image python3 -m rageval.cli retrieval materialize --selected "$M1/selected.json" --multihop "$M1/multihop_dev.json" \
  --models-template configs/models.yaml --pipelines-template "$pipes_tmpl" --out-dir "$R/${tag}_cfg"
pipes=$R/${tag}_cfg/pipelines.yaml; models=$R/${tag}_cfg/models.yaml
cli gen run --suite "$suite" --pipelines "$pipes" --models "$models" --out "$R/${tag}_gen.jsonl" --concurrency 192 "$@"
stop_vllms

cli gen judge-nli --records "$R/${tag}_gen.jsonl" --out "$R/${tag}_nli.json"
start_vllm openai/gpt-oss-120b 8010 0 --max-model-len 16384 --gpu-memory-utilization 0.85
cli gen judge-llm --records "$R/${tag}_gen.jsonl" --out "$R/${tag}_llm.json" --llm-url http://127.0.0.1:8010
stop_vllms
cli gen merge --records "$R/${tag}_gen.jsonl" --nli "$R/${tag}_nli.json" --llm "$R/${tag}_llm.json" --out "$R/${tag}_judged.jsonl"
cli gen report --judged "$R/${tag}_judged.jsonl" --out "$R/${tag}_report.json"
