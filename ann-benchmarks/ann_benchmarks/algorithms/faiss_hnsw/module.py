import os

import faiss
import numpy as np

from ..faiss.module import Faiss


class FaissHNSW(Faiss):
    def __init__(self, metric, method_param):
        self._metric = metric
        self.method_param = method_param
        self.index_cache_dir = None
        self.index_cache_mode = "none"
        self.index_cache_dataset = None
        self.index_cache_path = None
        self.index_cache_action = "none"

    def set_index_cache(self, index_dir, index_mode="none", dataset_name=None):
        if index_mode != "none" and not index_dir:
            raise ValueError("--index-dir is required when --index-mode is not 'none'")
        self.index_cache_dir = os.path.expanduser(index_dir) if index_dir else None
        self.index_cache_mode = index_mode or "none"
        self.index_cache_dataset = dataset_name

    def _cache_path(self, X):
        if not self.index_cache_dir or self.index_cache_mode == "none":
            return None
        dataset = self.index_cache_dataset or "unknown-dataset"
        n, d = X.shape
        m = self.method_param["M"]
        efc = self.method_param["efConstruction"]
        filename = f"faiss_hnsw_{self._metric}_N{n}_d{d}_M{m}_efC{efc}.index"
        return os.path.join(self.index_cache_dir, dataset, filename)

    def fit(self, X):
        self.index_cache_action = "none"
        self.index_cache_path = self._cache_path(X)
        if self.index_cache_path and self.index_cache_mode in ("auto", "load") and os.path.exists(self.index_cache_path):
            print(f"Loading Faiss HNSW index from {self.index_cache_path}")
            self.index = faiss.read_index(self.index_cache_path)
            self.index_cache_action = "load"
            return

        if self.index_cache_path and self.index_cache_mode == "load":
            raise FileNotFoundError(f"cached index not found: {self.index_cache_path}")

        self.index = faiss.IndexHNSWFlat(len(X[0]), self.method_param["M"])
        self.index.hnsw.efConstruction = self.method_param["efConstruction"]
        self.index.verbose = True

        if self._metric == "angular":
            X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.index.add(X)

        if self.index_cache_path and self.index_cache_mode in ("auto", "save"):
            os.makedirs(os.path.dirname(self.index_cache_path), exist_ok=True)
            tmp_path = self.index_cache_path + ".tmp"
            print(f"Saving Faiss HNSW index to {self.index_cache_path}")
            faiss.write_index(self.index, tmp_path)
            os.replace(tmp_path, self.index_cache_path)
            self.index_cache_action = "build_save"
        elif self.index_cache_path:
            self.index_cache_action = "build"

    def set_query_arguments(self, ef):
        faiss.cvar.hnsw_stats.reset()
        self.index.hnsw.efSearch = ef

    def get_additional(self):
        return {
            "dist_comps": faiss.cvar.hnsw_stats.ndis,
            "index_cache_mode": self.index_cache_mode,
            "index_cache_action": self.index_cache_action,
            "index_cache_path": self.index_cache_path or "",
        }

    def __str__(self):
        return "faiss (%s, ef: %d)" % (self.method_param, self.index.hnsw.efSearch)

    def freeIndex(self):
        del self.p
