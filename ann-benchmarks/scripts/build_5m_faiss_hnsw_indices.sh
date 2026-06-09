#!/usr/bin/env bash
set -euo pipefail

# Build and save Faiss HNSW indexes for 5M OpenAI and LAION/ReLAION datasets.
# Override BUILD_THREADS, OPENAI_DATASET, LAION_DATASET, or INDEX_DIR if needed.

BENCH_DIR="${BENCH_DIR:-$HOME/cunzhe/faiss/ann-benchmarks}"
INDEX_DIR="${INDEX_DIR:-$HOME/cunzhe/faiss/index_cache/faiss_hnsw}"
BUILD_THREADS="${BUILD_THREADS:-32}"
CPU_END=$((BUILD_THREADS - 1))

OPENAI_DATASET="${OPENAI_DATASET:-openai-5m-1536-angular}"
LAION_DATASET="${LAION_DATASET:-relaion2b-natural-5m-angular}"

LOG_DIR="${BENCH_DIR}/logs"
LOG_FILE="${LOG_DIR}/build_5m_faiss_hnsw_$(date +%F_%H%M%S).log"

mkdir -p "${INDEX_DIR}" "${LOG_DIR}"
cd "${BENCH_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "== build started: $(date '+%F %T') =="
echo "BENCH_DIR=${BENCH_DIR}"
echo "INDEX_DIR=${INDEX_DIR}"
echo "BUILD_THREADS=${BUILD_THREADS}"
echo "CPU binding: 0-${CPU_END}, memory node 0"
echo "OPENAI_DATASET=${OPENAI_DATASET}"
echo "LAION_DATASET=${LAION_DATASET}"
echo

run_build() {
  local dataset="$1"
  local output_dir="results/${dataset}/10/build_${BUILD_THREADS}t_save"

  echo "== ${dataset}: build/save start $(date '+%F %T') =="
  OMP_NUM_THREADS="${BUILD_THREADS}" \
  numactl -C "0-${CPU_END}" -m 0 python3 run.py \
    --dataset "${dataset}" \
    --algorithm 'hnsw(faiss)' \
    --count 10 \
    --runs 1 \
    --local \
    --batch \
    --force \
    --index-dir "${INDEX_DIR}" \
    --index-mode auto \
    --output-dir "${output_dir}"

  echo "== ${dataset}: build/save done $(date '+%F %T') =="
  find "${INDEX_DIR}/${dataset}" -type f -name "*.index" \
    -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" | sort || true
  echo
}

run_build "${OPENAI_DATASET}"
run_build "${LAION_DATASET}"

echo "== build finished: $(date '+%F %T') =="
echo "log: ${LOG_FILE}"
