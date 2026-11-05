"""Loaders for two FLamby tasks, without FLamby.

FLamby (du Terrail et al., NeurIPS 2022, https://github.com/owkin/FLamby, MIT)
defines the centres and the train/test split of each task. Nothing here is
FLamby's code. The files are fetched at run time into `benchmarks/out/flamby`
and checked against the SHA-256 below.

Fed-TCGA-BRCA. The preprocessed table `brca.csv` and the split file
`dataset_creation_scripts/train_test_split.csv` are read from the FLamby
repository at the pinned commit. The split is the one FLamby's
`fed_tcga_brca/dataset.py` applies: a patient belongs to the fold named in
column `fold2`, `train_<centre>` or `test_<centre>`, for six centres that are
regions of the tissue source sites. The underlying clinical data is TCGA-BRCA
open access data from the NCI Genomic Data Commons.

Fed-Heart-Disease. The four `processed.<centre>.data` files come from the UCI
repository (Janosi, Steinbrunn, Pfisterer and Detrano, 1988, CC BY 4.0). The
preparation follows FLamby's `fed_heart_disease/dataset.py`: columns 10 to 12
are dropped, then rows with a missing value; chest pain type and resting ECG
are one-hot encoded with the first level dropped; the label is disease or
none. FLamby splits each centre with scikit-learn's
`train_test_split(train_size=0.66, test_size=0.34, random_state=43)`,
stratified on the label where every class has more than two rows. `_split`
replays that call draw for draw on numpy's legacy generator, and the resulting
test rows are pinned by hash so a drift would be caught.
"""

import csv
import hashlib
import math
import urllib.request
from pathlib import Path

import numpy as np

FLAMBY_COMMIT = "edacf54d5211520583b0133d55ac39b6fda8324b"
FLAMBY_RAW = f"https://raw.githubusercontent.com/owkin/FLamby/{FLAMBY_COMMIT}/flamby/datasets"
UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"

FILES = {
    "brca.csv": (
        f"{FLAMBY_RAW}/fed_tcga_brca/brca.csv",
        "07cb53423f7d579d33cfdbf2cba57b2aee75318b85866c68ce39a75b1f1ba020",
    ),
    "train_test_split.csv": (
        f"{FLAMBY_RAW}/fed_tcga_brca/dataset_creation_scripts/train_test_split.csv",
        "1249c6543a4060ff0fdaf8510fc37017755836db6c9fc5e6af96057c36510990",
    ),
    "processed.cleveland.data": (
        f"{UCI}/processed.cleveland.data",
        "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8",
    ),
    "processed.hungarian.data": (
        f"{UCI}/processed.hungarian.data",
        "d1ad108f785768cd3d7e82dc522e6f5a61eea93cccfb3a46ee8076f73fc3d796",
    ),
    "processed.switzerland.data": (
        f"{UCI}/processed.switzerland.data",
        "834a405ccf5b66ab4056bb77794adc8df0b7125186454c0a1d002d33c6c3b314",
    ),
    "processed.va.data": (
        f"{UCI}/processed.va.data",
        "e7c93d8d0d2acdadfa4c5e8de768e2191e7f618b952e29623f1f0d5949ff6b8f",
    ),
}

HEART_CENTRES = ["cleveland", "hungarian", "switzerland", "va"]
# SHA-256 over each centre's sorted test row numbers, as scikit-learn 1.9.1 gives them.
HEART_SPLIT = "1f5bf32141238b97be64571a9293ad97a4e33dca1f5274222586fa58a40856a1"

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "out" / "flamby"


def fetch(name, directory=DEFAULT_DIR):
    """The path of a data file, downloaded if absent, with its hash checked."""
    url, want = FILES[name]
    path = Path(directory) / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            body = response.read()
        partial = path.with_suffix(path.suffix + ".partial")
        partial.write_bytes(body)
        partial.rename(path)
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        raise RuntimeError(f"{path} has SHA-256 {got}, expected {want}")
    return path


def sources():
    """What was read and from where, for a result file."""
    return {name: {"url": url, "sha256": digest} for name, (url, digest) in FILES.items()}


def tcga_brca(directory=DEFAULT_DIR):
    """Six centres of (x, event, time) for training, and the same for testing.

    x has FLamby's 39 covariates. Age is the only one that is not an
    indicator, and is rescaled by fixed constants, not by statistics of the data.
    """
    with open(fetch("brca.csv", directory), newline="") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    assert header[0] == "pid" and header[1] == "age_at_index" and header[40:42] == ["E", "T"]
    table = {row[0]: np.array(row[1:42], dtype=np.float64) for row in rows}
    with open(fetch("train_test_split.csv", directory), newline="") as f:
        folds = {row["pid"]: row["fold2"] for row in csv.DictReader(f)}

    def centre(fold):
        values = np.array([table[pid] for pid in table if folds.get(pid) == fold])
        x = values[:, :39].copy()
        x[:, 0] = (x[:, 0] - 60.0) / 15.0
        return x, values[:, 39], values[:, 40]

    train = [centre(f"train_{c}") for c in range(6)]
    test = [centre(f"test_{c}") for c in range(6)]
    return train, test


def _approximate_mode(counts, draws, rng):
    continuous = counts / counts.sum() * draws
    floored = np.floor(continuous)
    need = int(draws - floored.sum())
    if need > 0:
        remainder = continuous - floored
        for value in np.sort(np.unique(remainder))[::-1]:
            (where,) = np.where(remainder == value)
            now = min(len(where), need)
            floored[rng.choice(where, size=now, replace=False)] += 1
            need -= now
            if need == 0:
                break
    return floored.astype(int)


def _split(labels, seed=43, train_size=0.66):
    """The test rows scikit-learn's `train_test_split` picks for FLamby's call.

    FLamby puts every row that is not a test row in training.
    """
    n = len(labels)
    n_test = math.ceil((1.0 - train_size) * n)
    n_train = math.floor(train_size * n)
    rng = np.random.RandomState(seed)
    classes, index = np.unique(labels, return_inverse=True)
    counts = np.bincount(index)
    if len(classes) < 2 or counts.min() <= 2:
        return np.sort(rng.permutation(n)[:n_test])
    by_class = np.split(np.argsort(index, kind="mergesort"), np.cumsum(counts)[:-1])
    take_train = _approximate_mode(counts, n_train, rng)
    take_test = _approximate_mode(counts - take_train, n_test, rng)
    test = []
    for c in range(len(classes)):
        shuffled = by_class[c].take(rng.permutation(counts[c]), mode="clip")
        test.extend(shuffled[take_train[c] : take_train[c] + take_test[c]])
    return np.sort(np.array(test))


def _heart_rows(name, directory):
    kept = []
    with open(fetch(f"processed.{name}.data", directory), newline="") as f:
        for row in csv.reader(f):
            row = row[:10] + row[13:]
            if "?" not in row:
                kept.append([float(v) for v in row])
    return np.array(kept)


def heart_disease(directory=DEFAULT_DIR, pooled_statistics=False):
    """Four centres of (x, label) for training, and the same for testing.

    Each centre's 13 features are standardised with that centre's own training
    mean and deviation, as FLamby does for a federated run. With
    `pooled_statistics` the pooled training rows give them, FLamby's choice
    for its pooled baseline.
    """
    raw = [_heart_rows(name, directory) for name in HEART_CENTRES]
    tests = [_split((rows[:, -1] > 0).astype(int)) for rows in raw]
    digest = hashlib.sha256()
    for rows in tests:
        digest.update(np.asarray(rows, dtype="<i8").tobytes())
    if digest.hexdigest() != HEART_SPLIT:
        raise RuntimeError(f"the Heart-Disease split changed: {digest.hexdigest()}")

    def features(rows):
        plain = np.delete(rows[:, :10], [2, 6], axis=1)
        pain = [(rows[:, 2] == level).astype(float) for level in (2.0, 3.0, 4.0)]
        ecg = [(rows[:, 6] == level).astype(float) for level in (1.0, 2.0)]
        return np.column_stack([plain, *pain, *ecg])

    def halves(rows, test_rows):
        mask = np.zeros(len(rows), dtype=bool)
        mask[test_rows] = True
        x, y = features(rows), (rows[:, -1] > 0).astype(np.float64)
        return (x[~mask], y[~mask]), (x[mask], y[mask])

    parts = [halves(rows, test_rows) for rows, test_rows in zip(raw, tests)]
    pooled = np.concatenate([train[0] for train, _ in parts])
    train, test = [], []
    for part_train, part_test in parts:
        basis = pooled if pooled_statistics else part_train[0]
        mean, deviation = basis.mean(axis=0), basis.std(axis=0, ddof=1) + 1e-9
        train.append(((part_train[0] - mean) / deviation, part_train[1]))
        test.append(((part_test[0] - mean) / deviation, part_test[1]))
    return train, test
