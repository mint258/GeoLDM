# dataset_lmdb.py
# 只读版 LmdbMoleculeDataset，与 MoleculeDataset 接口完全一致
import lmdb, pickle
import logging
from torch_geometric.data import Dataset

LOGGER = logging.getLogger(__name__)

class LmdbMoleculeDataset(Dataset):
    def __init__(self, lmdb_path, start_idx=None, end_idx=None,
                 limit=None, transform=None, pre_transform=None,
                 filter_missing_on_init: bool = False):
        super().__init__(None, transform, pre_transform)
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(lmdb_path, subdir=False, readonly=True,
                             lock=False, readahead=True)  # 打开时启用 readahead
        with self.env.begin() as txn:
            raw_len = txn.get(b'__len__')
            if raw_len is None:
                raise KeyError(
                    f"LMDB shard '{lmdb_path}' missing '__len__' key. "
                    "This file may be corrupted or not produced by the expected writer."
                )
            self._len = pickle.loads(raw_len)
            self._key_fmt = self._detect_key_fmt(txn)

        self.start = int(start_idx) if start_idx is not None else 0
        if self.start < 0:
            self.start = 0
        raw_end = int(end_idx) if end_idx is not None else (self._len - 1)
        self.end = min(raw_end, self._len - 1) if self._len > 0 else -1

        if self._len <= 0 or self.end < self.start:
            self.idx_list = []
            LOGGER.warning(
                "LMDB %s: empty index range (len=%s, start=%s, end=%s)",
                lmdb_path, self._len, self.start, self.end,
            )
            return
        
        self.idx_list = list(range(self.start, self.end + 1))
        if limit:
            self.idx_list = self.idx_list[:limit]

        # 可选：在 init 就做一次全过滤
        if filter_missing_on_init:
            valid = []
            with self.env.begin() as txn:
                for real in range(self.start, self.end + 1):
                    if txn.get(self._format_key(real)) is not None:
                        valid.append(real)
            self.idx_list = valid

        if limit:
            self.idx_list = self.idx_list[: int(limit)]

        if filter_missing_on_init and len(self.idx_list) == 0:
            LOGGER.warning(
                "LMDB %s: 0 valid samples after filtering (key_fmt=%s). "
                "Possible causes: corrupted LMDB, wrong key formatting, or start/end out of range.",
                lmdb_path, self._key_fmt,
            )

    def _format_key(self, real: int) -> bytes:
        if self._key_fmt == 'raw':
            return str(real).encode()
        return f"{real:{self._key_fmt}}".encode()

    def _detect_key_fmt(self, txn) -> str:
        candidates = ['09d', '08d', '010d', 'raw']
        probes = {0, 1, 2}
        if isinstance(getattr(self, '_len', None), int) and self._len > 0:
            probes.add(self._len - 1)
        probes = [p for p in probes if isinstance(p, int) and p >= 0 and (self._len is None or p < self._len)]

        best_fmt, best_hits = '09d', -1
        for fmt in candidates:
            hits = 0
            for i in probes:
                key = str(i).encode() if fmt == 'raw' else f"{i:{fmt}}".encode()
                if txn.get(key) is not None:
                    hits += 1
            if hits > best_hits:
                best_hits, best_fmt = hits, fmt

        if best_hits <= 0:
            LOGGER.warning(
                "LMDB %s: could not detect key format from probes=%s; fallback to 09d. "
                "If this shard uses a different key format, it may appear empty.",
                self.lmdb_path, probes,
            )
            best_fmt = '09d'
        return best_fmt

    def len(self): 
        return len(self.idx_list)

    def get(self, i):
        with self.env.begin() as txn:
            n = len(self.idx_list)
            if n == 0:
                raise IndexError("Empty LMDB dataset")
            real = self.idx_list[i]
            buf = txn.get(self._format_key(real))
            if buf is not None:
                return pickle.loads(buf)
            # 就近向前/向后探测一个可用样本，返回之（允许产生重复样本，但长度稳定）
            n = len(self.idx_list)
            # 向前探测
            for j in range(i + 1, n):
                buf = txn.get(self._format_key(self.idx_list[j]))
                if buf is not None:
                    # 缓存映射以加速下次命中
                    self.idx_list[i] = self.idx_list[j]
                    return pickle.loads(buf)
            # 向后探测
            for j in range(i - 1, -1, -1):
                buf = txn.get(self._format_key(self.idx_list[j]))
                if buf is not None:
                    self.idx_list[i] = self.idx_list[j]
                    return pickle.loads(buf)
        # 兜底：交给外层安全包装器处理
        raise IndexError(f"LMDB sample missing at logical index {i}")
