# Faiss HNSW Cache Commands

These commands assume:

```bash
cd ~/cunzhe/faiss/ann-benchmarks
```

Use `--index-dir ~/cunzhe/faiss/index_cache/faiss_hnsw`, not `./indices`.
The original ann-benchmarks startup path treats `./indices` as temporary state.

## Build And Save One Dataset

```bash
OMP_NUM_THREADS=32 \
numactl -C 0-31 -m 0 python3 run.py \
  --dataset openai-5m-1536-angular \
  --algorithm 'hnsw(faiss)' \
  --count 10 \
  --runs 1 \
  --local \
  --batch \
  --force \
  --index-dir ~/cunzhe/faiss/index_cache/faiss_hnsw \
  --index-mode auto \
  --output-dir results/openai-5m-1536-angular/10/build_32t_save
```

Check saved indexes:

```bash
find ~/cunzhe/faiss/index_cache/faiss_hnsw/openai-5m-1536-angular \
  -type f -name "*.index" \
  -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" | sort
```

Summarize incidental query results:

```bash
python3 scripts/summarize_ann_results.py \
  --results-dir results/openai-5m-1536-angular/10/build_32t_save \
  --dataset-hdf5 data/openai-5m-1536-angular.hdf5 \
  --count 10
```

## Load Existing Index And Test Search

```bash
OMP_NUM_THREADS=8 \
numactl -C 0-7 -m 0 python3 run.py \
  --dataset openai-5m-1536-angular \
  --algorithm 'hnsw(faiss)' \
  --count 10 \
  --runs 10 \
  --local \
  --batch \
  --force \
  --index-dir ~/cunzhe/faiss/index_cache/faiss_hnsw \
  --index-mode load \
  --output-dir results/openai-5m-1536-angular/10/load_8t_runs10
```

Summarize:

```bash
python3 scripts/summarize_ann_results.py \
  --results-dir results/openai-5m-1536-angular/10/load_8t_runs10 \
  --dataset-hdf5 data/openai-5m-1536-angular.hdf5 \
  --count 10
```

In load mode, `build_s` is index load/setup time, not HNSW build time.

## Search Thread Sweep

```bash
for t in 1 2 4 8; do
  OMP_NUM_THREADS=$t \
  numactl -C 0-$((t-1)) -m 0 python3 run.py \
    --dataset openai-5m-1536-angular \
    --algorithm 'hnsw(faiss)' \
    --count 10 \
    --runs 10 \
    --local \
    --batch \
    --force \
    --index-dir ~/cunzhe/faiss/index_cache/faiss_hnsw \
    --index-mode load \
    --output-dir results/openai-5m-1536-angular/10/load_${t}t_runs10
done
```

Summarize:

```bash
for t in 1 2 4 8; do
  echo "== openai-5m-1536-angular ${t}t runs10 =="
  python3 scripts/summarize_ann_results.py \
    --results-dir results/openai-5m-1536-angular/10/load_${t}t_runs10 \
    --dataset-hdf5 data/openai-5m-1536-angular.hdf5 \
    --count 10
done
```

## Build OpenAI 5M And ReLAION 5M Overnight

Foreground:

```bash
chmod +x scripts/build_5m_faiss_hnsw_indices.sh

BUILD_THREADS=32 \
OPENAI_DATASET=openai-5m-1536-angular \
LAION_DATASET=relaion2b-natural-5m-angular \
scripts/build_5m_faiss_hnsw_indices.sh
```

Background:

```bash
mkdir -p logs
chmod +x scripts/build_5m_faiss_hnsw_indices.sh

nohup env \
  BUILD_THREADS=32 \
  OPENAI_DATASET=openai-5m-1536-angular \
  LAION_DATASET=relaion2b-natural-5m-angular \
  scripts/build_5m_faiss_hnsw_indices.sh \
  > logs/nohup_build_5m_indices.log 2>&1 &

echo $!
```

Watch progress:

```bash
tail -f logs/nohup_build_5m_indices.log
pgrep -af "build_5m_faiss_hnsw_indices|python3 run.py"
```

Verify all 5M indexes:

```bash
find ~/cunzhe/faiss/index_cache/faiss_hnsw \
  -type f -name "*.index" \
  \( -path "*openai-5m-1536-angular*" -o -path "*relaion2b-natural-5m-angular*" \) \
  -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" | sort
```

## SIFT 128D Build Smoke

```bash
OMP_NUM_THREADS=32 \
numactl -C 0-31 -m 0 python3 run.py \
  --dataset sift-128-euclidean \
  --algorithm 'hnsw(faiss)' \
  --count 10 \
  --runs 1 \
  --local \
  --batch \
  --force \
  --index-dir ~/cunzhe/faiss/index_cache/faiss_hnsw \
  --index-mode auto \
  --output-dir results/sift-128-euclidean/10/build_32t_save
```

Check:

```bash
find ~/cunzhe/faiss/index_cache/faiss_hnsw/sift-128-euclidean \
  -type f -name "*.index" \
  -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" | sort
```

Summarize:

```bash
python3 scripts/summarize_ann_results.py \
  --results-dir results/sift-128-euclidean/10/build_32t_save \
  --dataset-hdf5 data/sift-128-euclidean.hdf5 \
  --count 10
```
