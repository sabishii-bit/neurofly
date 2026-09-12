"""Load the MaleCNS v1.0 flat connectome into a signed sparse weight matrix.

The result is a ``Connectome``: a neuron table (one row per neuron with the
annotation columns you need to pick populations) and a scipy CSR matrix
``W`` of shape (n, n) where ``W[post, pre]`` is the synapse count from
``pre`` to ``post``, multiplied by the sign of ``pre``'s neurotransmitter.
That is exactly the input a leaky integrate-and-fire simulation needs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import scipy.sparse as sp

from neurofly_training.data.download import FILES

# Sign convention from Shiu et al. (Nature 2024): acetylcholine excites,
# GABA and glutamate inhibit. Histamine (photoreceptors) inhibits. The
# monoamines are modulatory; treating them as excitatory is the usual
# simplification. 'unclear' neurons default to excitatory.
NT_SIGN = {
    "acetylcholine": +1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
    "dopamine": +1,
    "octopamine": +1,
    "serotonin": +1,
    "unclear": +1,
}

KEEP_COLS = ["bodyId", "type", "flywireType", "superclass", "class", "subclass",
             "somaSide", "rootSide", "somaNeuromere", "entryNerve", "exitNerve",
             "status", "assignedOlHex1", "assignedOlHex2"]
HEX_COLS = ["assignedOlHex1", "assignedOlHex2"]  # optic-lobe column of columnar neurons

# Named subsets of the brain. The full CNS is 165k neurons / 25.6M edges,
# which is slow to step; the optic lobes alone are ~95k neurons. The nerve
# cord plus the neurons entering and leaving it is enough for walking.
VNC_SUPERCLASSES = ["vnc_intrinsic", "vnc_sensory", "vnc_motor", "vnc_efferent",
                    "vnc_tbc", "vnc_sensory_tbc", "vnc_endocrine",
                    "descending_neuron", "ascending_neuron",
                    "sensory_ascending", "sensory_descending",
                    "efferent_ascending", "efferent_descending"]
OPTIC_SUPERCLASSES = ["ol_intrinsic", "ol_sensory"]


def _is_vnc(cx):
    return cx.neurons["superclass"].isin(VNC_SUPERCLASSES).values


def _is_cord(cx):
    """Nerve cord proper: the VNC superclasses minus the descending neurons,
    which the game subsets keep as their motor output."""
    sc = cx.neurons["superclass"]
    return (sc.isin(VNC_SUPERCLASSES) & (sc != "descending_neuron")).values


def _is_optic(cx):
    return cx.neurons["superclass"].isin(OPTIC_SUPERCLASSES).values


def _has_retina_column(cx):
    """Columnar optic-lobe neurons annotated with their hex column (L1, L2, Mi1, ...)."""
    if not all(c in cx.neurons.columns for c in HEX_COLS):
        return np.zeros(cx.n, dtype=bool)
    return cx.neurons[HEX_COLS].notna().all(axis=1).values


SUBSETS = {
    # body side
    "full": lambda cx: np.arange(cx.n),
    "no-optic": lambda cx: np.flatnonzero(~_is_optic(cx)),
    "vnc": lambda cx: np.flatnonzero(_is_vnc(cx)),
    # game side (frames in, key presses out): no nerve cord, descending neurons kept
    "brain": lambda cx: np.flatnonzero(~_is_cord(cx)),
    "central": lambda cx: np.flatnonzero(~_is_cord(cx) & ~_is_optic(cx)),
    "visual": lambda cx: np.flatnonzero(
        ~_is_cord(cx) & (~_is_optic(cx) | _has_retina_column(cx)
                         | (cx.neurons["superclass"] == "ol_sensory").values)),
}

LEGS = [("T1", "L"), ("T1", "R"), ("T2", "L"), ("T2", "R"), ("T3", "L"), ("T3", "R")]
LEG_NERVE = {"T1": "ProLN", "T2": "MesoLN", "T3": "MetaLN"}


@dataclass
class Connectome:
    neurons: pd.DataFrame
    W: sp.csr_matrix
    name: str = "malecns"
    _Wc: sp.csc_matrix | None = field(default=None, repr=False, compare=False)

    # --- basic properties ---------------------------------------------------

    @property
    def n(self) -> int:
        return len(self.neurons)

    @property
    def n_edges(self) -> int:
        return int(self.W.nnz)

    @property
    def n_synapses(self) -> int:
        return int(np.abs(self.W.data).sum())

    @property
    def sign(self) -> np.ndarray:
        return self.neurons["sign"].values

    def summary(self) -> str:
        lines = [f"{self.name}: {self.n:,} neurons, {self.n_edges:,} edges, "
                 f"{self.n_synapses:,} synapses"]
        counts = self.neurons["superclass"].fillna("None").value_counts()
        for k, v in counts.items():
            lines.append(f"  {k:22s} {v:7,}")
        nt = self.neurons["nt"].value_counts()
        lines.append("  neurotransmitters: " + ", ".join(f"{k} {v:,}" for k, v in nt.items()))
        return "\n".join(lines)

    # --- selection ------------------------------------------------------------

    def select(self, **criteria) -> np.ndarray:
        """Indices of neurons matching all criteria.

        ``select(superclass='vnc_motor', somaNeuromere='T1', somaSide='L')``.
        A list value matches any element; a key ending in ``_re`` is a regex
        on that column; ``class_`` is an alias for the ``class`` column.
        """
        mask = np.ones(self.n, dtype=bool)
        for key, value in criteria.items():
            if key == "class_":
                key = "class"
            if key.endswith("_re"):
                col = self.neurons[key[:-3]].fillna("").astype(str)
                mask &= col.str.contains(value, regex=True).values
            elif isinstance(value, (list, tuple, set, np.ndarray)):
                mask &= self.neurons[key].isin(list(value)).values
            else:
                mask &= (self.neurons[key] == value).values
        return np.flatnonzero(mask)

    def subgraph(self, keep: np.ndarray, name: str | None = None) -> "Connectome":
        keep = np.unique(np.asarray(keep, dtype=np.int64))
        neurons = self.neurons.iloc[keep].reset_index(drop=True)
        W = self.W[keep][:, keep].tocsr()
        W.sort_indices()
        return Connectome(neurons, W, name=name or f"{self.name}[{len(keep)}]")

    def subset(self, name: str) -> "Connectome":
        if name not in SUBSETS:
            raise ValueError(f"unknown subset {name!r}; choose from {sorted(SUBSETS)}")
        if name == "full":
            return self
        return self.subgraph(SUBSETS[name](self), name=f"{self.name}:{name}")

    def neighborhood(self, seeds, hops: int = 1, direction: str = "both") -> np.ndarray:
        """Seeds plus everything within ``hops`` synaptic steps of them."""
        if self._Wc is None:
            self._Wc = self.W.tocsc()
        visited = np.zeros(self.n, dtype=bool)
        frontier = np.unique(np.asarray(seeds, dtype=np.int64))
        visited[frontier] = True
        for _ in range(hops):
            new = []
            if direction in ("down", "both"):      # postsynaptic partners
                new.append(self._Wc[:, frontier].nonzero()[0])
            if direction in ("up", "both"):        # presynaptic partners
                new.append(self.W[frontier].nonzero()[1])
            frontier = np.unique(np.concatenate(new)) if new else np.array([], dtype=np.int64)
            frontier = frontier[~visited[frontier]]
            if len(frontier) == 0:
                break
            visited[frontier] = True
        return np.flatnonzero(visited)

    # --- construction ---------------------------------------------------------

    @classmethod
    def load(cls, data_dir: str, status: str | None = "Traced", min_weight: int = 1,
             use_cache: bool = True, verbose: bool = True) -> "Connectome":
        """Load from the downloaded feather files (cached as npz after first use)."""
        tag = f"{status or 'all'}-w{min_weight}"
        cache_dir = os.path.join(data_dir, "cache")
        w_path = os.path.join(cache_dir, f"W-{tag}.npz")
        n_path = os.path.join(cache_dir, f"neurons-{tag}.feather")

        def path(local):
            p = os.path.join(data_dir, local)
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{p} not found. Run `neurofly download` first.")
            return p

        if use_cache and os.path.exists(w_path) and os.path.exists(n_path):
            z = np.load(w_path)
            W = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
            neurons = feather.read_feather(n_path)
            missing = [c for c in KEEP_COLS if c not in neurons.columns]
            if missing:  # cache written by an older version: pull the new columns only
                extra = feather.read_feather(path("body-annotations.feather"),
                                             columns=["bodyId"] + missing)
                neurons = neurons.merge(extra, on="bodyId", how="left")
                neurons.to_feather(n_path)
            return cls(neurons, W)

        if verbose:
            print("reading annotations ...")
        ann = feather.read_feather(path("body-annotations.feather"), columns=KEEP_COLS)
        if status:
            ann = ann[ann["status"] == status]
        ann = ann.reset_index(drop=True)
        nt = feather.read_feather(path("body-neurotransmitters.feather"),
                                  columns=["body", "consensus_nt"])
        nt = nt.rename(columns={"body": "bodyId", "consensus_nt": "nt"})
        ann = ann.merge(nt, on="bodyId", how="left")
        ann["nt"] = ann["nt"].fillna("unclear")
        ann["sign"] = ann["nt"].map(NT_SIGN).fillna(1).astype(np.int8)

        if verbose:
            print("reading weights (this takes a minute) ...")
        w = feather.read_feather(path("connectome-weights-traced-only.feather"),
                                 columns=["body_pre", "body_post", "weight"])
        if min_weight > 1:
            w = w[w["weight"] >= min_weight]
        index = pd.Index(ann["bodyId"].values)
        pre = index.get_indexer(w["body_pre"].values)
        post = index.get_indexer(w["body_post"].values)
        ok = (pre >= 0) & (post >= 0)
        pre, post = pre[ok], post[ok]
        data = w["weight"].values[ok].astype(np.float32) * ann["sign"].values[pre]
        n = len(ann)
        W = sp.csr_matrix((data, (post, pre)), shape=(n, n), dtype=np.float32)
        W.sum_duplicates()
        W.sort_indices()

        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(w_path, data=W.data, indices=W.indices, indptr=W.indptr,
                     shape=np.array(W.shape))
            ann.to_feather(n_path)
        return cls(ann, W)

    @classmethod
    def toy(cls, n: int = 1500, seed: int = 0) -> "Connectome":
        """A small brain with a designed path, for meaningful tests and examples.

        Same annotation columns and populations as ``synthetic``, plus strong
        feed-forward wiring: each eye's retina columns drive their own half of the
        visual projection neurons, which drive their own half of the central
        interneurons, which drive their own half of the descending neurons. The
        descending readout therefore tells left from right, which a random graph
        cannot. The random background wiring of ``synthetic`` is kept underneath.
        """
        cx = cls.synthetic(n=n, seed=seed)
        rng = np.random.default_rng(seed + 7)
        df = cx.neurons
        sc = df["superclass"].values
        side = df["somaSide"].values
        halves = {}
        for name in ("visual_projection", "cb_intrinsic", "descending_neuron"):
            idx = np.flatnonzero(sc == name)
            halves[name] = (idx[: len(idx) // 2], idx[len(idx) // 2:])
        retina = {s: np.flatnonzero((sc == "ol_intrinsic") & (side == s)) for s in ("L", "R")}
        rows, cols, vals = [], [], []

        def connect(pre, post, fan_in=8, weight=40.0):
            for p_ in post:
                for q in rng.choice(pre, size=min(fan_in, len(pre)), replace=False):
                    rows.append(int(p_))
                    cols.append(int(q))
                    vals.append(weight * rng.uniform(0.7, 1.3))

        for k, s in enumerate(("L", "R")):
            connect(retina[s], halves["visual_projection"][k])
            connect(halves["visual_projection"][k], halves["cb_intrinsic"][k])
            connect(halves["cb_intrinsic"][k], halves["descending_neuron"][k])
        # the feed-forward path is excitatory: make its presynaptic neurons cholinergic
        pre_ids = np.unique(cols)
        df.loc[pre_ids, "nt"] = "acetylcholine"
        df.loc[pre_ids, "sign"] = 1
        W = cx.W.tolil()
        for p_, q, v in zip(rows, cols, vals):
            W[p_, q] = v
        # re-sign every outgoing edge to match the (possibly changed) transmitter
        W = W.tocsr()
        W.sort_indices()
        pre_of = np.repeat(np.arange(cx.n), np.diff(W.tocsc().indptr))  # placeholder (unused)
        Wc = W.tocsc()
        data = np.abs(Wc.data) * np.repeat(df["sign"].values, np.diff(Wc.indptr))
        W = sp.csc_matrix((data, Wc.indices, Wc.indptr), shape=Wc.shape).tocsr()
        W.sort_indices()
        del pre_of
        return cls(df, W, name=f"toy{n}")

    @classmethod
    def synthetic(cls, n: int = 3000, out_degree: float = 15.0, seed: int = 0) -> "Connectome":
        """A random connectome with the same annotation columns, for tests.

        Population proportions are loosely modelled on the nerve cord so that
        ``Populations`` finds motor, sensory and descending neurons.
        """
        rng = np.random.default_rng(seed)
        classes = ["vnc_intrinsic", "cb_intrinsic", "vnc_sensory", "vnc_motor",
                   "descending_neuron", "ascending_neuron", "cb_sensory",
                   "ol_intrinsic", "visual_projection"]
        probs = [0.30, 0.18, 0.13, 0.09, 0.06, 0.04, 0.06, 0.10, 0.04]
        superclass = rng.choice(classes, size=n, p=probs)
        df = pd.DataFrame({
            "bodyId": np.arange(n, dtype=np.int64) + 1,
            "type": "synthetic", "flywireType": None,
            "superclass": superclass, "class": None, "subclass": None,
            "somaSide": None, "rootSide": None, "somaNeuromere": None,
            "entryNerve": None, "exitNerve": None, "status": "Traced",
            "assignedOlHex1": np.nan, "assignedOlHex2": np.nan,
        })
        df = df.astype({c: "object" for c in ["type", "class", "subclass", "somaSide",
                                               "rootSide", "somaNeuromere", "entryNerve",
                                               "exitNerve"]})
        # a small retina: lamina/medulla columnar neurons on an 8x8 hex grid per eye
        retina_types = ["L1", "L2", "Mi1", "Tm1"]
        for k, i in enumerate(np.flatnonzero(superclass == "ol_intrinsic")):
            df.loc[i, ["type", "somaSide"]] = [retina_types[(k // 2) % 4], "LR"[k % 2]]
            col = k // 8
            df.loc[i, ["assignedOlHex1", "assignedOlHex2"]] = [1 + col % 8, 1 + (col // 8) % 8]
        # a few dopamine neurons of the PPL1 cluster (the punishment input)
        cb = np.flatnonzero(superclass == "cb_intrinsic")
        for k, i in enumerate(cb[:8]):
            df.loc[i, ["class", "type"]] = ["DAN", f"PPL10{1 + k % 8}"]
        motor = np.flatnonzero(superclass == "vnc_motor")
        for k, i in enumerate(motor):
            t, side = LEGS[k % 6]
            df.loc[i, ["somaNeuromere", "somaSide", "subclass"]] = [t, side, "fl"]
        sens = np.flatnonzero(superclass == "vnc_sensory")
        for k, i in enumerate(sens):
            t, side = LEGS[k % 6]
            cls_ = "mechanosensory_proprioceptive" if (k // 6) % 3 == 0 else "mechanosensory_tactile"
            df.loc[i, ["entryNerve", "rootSide", "class"]] = [LEG_NERVE[t], side, cls_]
        head = np.flatnonzero(superclass == "cb_sensory")
        for k, i in enumerate(head):
            sub = ["wind_gravity", "haltere", "auditory"][k % 3]
            df.loc[i, ["class", "subclass"]] = ["mechanosensory", sub]
        df.loc[superclass == "descending_neuron", "subclass"] = "xn"
        df["nt"] = rng.choice(["acetylcholine", "gaba", "glutamate"], size=n, p=[0.7, 0.15, 0.15])
        df["sign"] = df["nt"].map(NT_SIGN).astype(np.int8)

        k = rng.poisson(out_degree, size=n)
        pre = np.repeat(np.arange(n), k)
        post = rng.integers(0, n, size=len(pre))
        keep = pre != post
        pre, post = pre[keep], post[keep]
        weight = rng.geometric(0.3, size=len(pre)).astype(np.float32)
        data = weight * df["sign"].values[pre]
        W = sp.csr_matrix((data, (post, pre)), shape=(n, n), dtype=np.float32)
        W.sum_duplicates()
        W.sort_indices()
        return cls(df, W, name=f"synthetic{n}")
