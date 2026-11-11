"""Criteo Uplift Prediction dataset (Diemert et al., 2018/2021): 13.9M users from a randomised ad-incrementality
test. Columns: f0..f11 (anonymised features), treatment (1 = could be shown ads, ~85%), visit, conversion, exposure."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [f"f{i}" for i in range(12)]
HF_REPO = "criteo/criteo-uplift"                 # Criteo's own Hugging Face copy
KAGGLE_DATASET = "arashnic/uplift-modeling"     # Kaggle mirror (needs a Kaggle API token)
LEGACY_URLS = ["https://criteo-bucket.s3.eu-central-1.amazonaws.com/criteo10.csv.gz",
               "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"]
REQUIRED = FEATURES + ["treatment", "visit", "conversion", "exposure"]
DATA_EXT = (".csv.gz", ".csv", ".parquet", ".zip")


def _secret(name):
    try:
        from google.colab import userdata  # type: ignore
        return userdata.get(name)
    except Exception:  # noqa: BLE001
        return os.environ.get(name)


def _from_hf(dest: Path) -> list[Path]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
    from huggingface_hub import hf_hub_download, list_repo_files

    token = _secret("HF_TOKEN")
    files = [f for f in list_repo_files(HF_REPO, repo_type="dataset", token=token) if f.lower().endswith(DATA_EXT)]
    print("  files in repo:", files)
    if not files:
        raise RuntimeError("no data files found in the HF repo")
    return [Path(hf_hub_download(HF_REPO, f, repo_type="dataset", token=token, local_dir=str(dest / "hf"))) for f in files]


def _from_kaggle(dest: Path) -> list[Path]:
    token = _secret("KAGGLE_API_TOKEN")
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token
        kd = Path.home() / ".kaggle"; kd.mkdir(exist_ok=True)
        (kd / "access_token").write_text(token); os.chmod(kd / "access_token", 0o600)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle>=1.7"], check=True)
    out = dest / "kaggle"; out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(out), "--unzip"], check=True)
    return [p for p in out.rglob("*") if p.is_file() and p.name.lower().endswith(DATA_EXT)]


def _from_url(dest: Path) -> list[Path]:
    out = []
    for url in LEGACY_URLS:
        f = dest / url.rsplit("/", 1)[-1]
        r = subprocess.run(["wget", "-q", "-O", str(f), url])
        if r.returncode == 0 and f.stat().st_size > 1e6:
            out.append(f)
            break
        f.unlink(missing_ok=True)
    if not out:
        raise RuntimeError("legacy URLs unreachable")
    return out


def _read_sampled(path: Path, frac: float, seed: int) -> pd.DataFrame:
    """Read (possibly 14M rows) in chunks, keeping a treatment-stratified random `frac` of each chunk."""
    if path.name.endswith(".parquet"):
        chunks = [pd.read_parquet(path)]
    else:
        chunks = pd.read_csv(path, chunksize=2_000_000)
    parts = []
    for i, c in enumerate(chunks):
        c.columns = [str(x).strip().lower() for x in c.columns]
        missing = [x for x in REQUIRED if x not in c.columns]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}; has {list(c.columns)[:20]}")
        c = c[REQUIRED]
        if frac < 1:
            c = c.groupby("treatment", group_keys=False).sample(frac=frac, random_state=seed + i)
        parts.append(c)
        print(f"    chunk {i}: kept {sum(len(p) for p in parts):,} rows", end="\r")
    print()
    return pd.concat(parts, ignore_index=True)


def load_criteo(data_dir: str | Path, sample_frac: float = 0.10, seed: int = 0) -> pd.DataFrame:
    """Treatment-stratified random sample of the Criteo Uplift v2.1 data (default 10%, ~1.4M users).
    Tries Hugging Face, then Kaggle, then the legacy URLs; caches a parquet copy in data_dir."""
    data_dir = Path(data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / f"criteo_{int(sample_frac * 100)}pct.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    # a manually placed file wins
    manual = [p for p in data_dir.glob("*") if p.name.lower().endswith(DATA_EXT) and "criteo" in p.name.lower()
              and not p.name.startswith("criteo_")]
    sources = ([("file you placed in " + str(data_dir), lambda: manual)] if manual else []) + [
        ("Hugging Face (criteo/criteo-uplift)", lambda: _from_hf(data_dir)),
        ("Kaggle (arashnic/uplift-modeling)", lambda: _from_kaggle(data_dir)),
        ("legacy Criteo URLs", lambda: _from_url(data_dir))]
    errors = []
    for name, get in sources:
        try:
            print(f"Trying {name} ...")
            files = sorted(get(), key=lambda p: -p.stat().st_size)    # biggest file = the full table
            for f in files:
                if f.name.endswith(".zip"):
                    import zipfile
                    zipfile.ZipFile(f).extractall(f.parent)
                    files += [p for p in f.parent.rglob("*.csv")]
                    continue
                is_10pct_file = "criteo10" in f.name.lower()          # the scikit-uplift 10% file needs no sampling
                df = _read_sampled(f, 1.0 if is_10pct_file and sample_frac >= 0.10 else sample_frac, seed)
                df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)   # don't keep rows grouped by treatment
                if len(df) < 100_000:
                    raise ValueError(f"only {len(df)} rows in {f.name}")
                print(f"  OK: {len(df):,} rows from {f.name}")
                for c in ["treatment", "visit", "conversion", "exposure"]:
                    df[c] = df[c].astype("int8")
                for c in FEATURES:
                    df[c] = df[c].astype("float32")
                df.to_parquet(cache, index=False)
                return df
            raise RuntimeError("no usable data file")
        except Exception as e:  # noqa: BLE001 - try next source
            msg = f"{type(e).__name__}: {str(e)[:300]}"
            errors.append(f"{name}: {msg}"); print("  failed:", msg)
    raise RuntimeError(
        "Could not get the Criteo uplift data automatically.\n  " + "\n  ".join(errors) + "\n"
        "Fixes: (a) if Hugging Face says the dataset is gated: log in at huggingface.co, open "
        "https://huggingface.co/datasets/criteo/criteo-uplift, accept the terms, create a READ token and add it as a "
        "Colab secret named HF_TOKEN; or (b) make sure the Colab secret KAGGLE_API_TOKEN exists (from Project 1) and "
        "has notebook access; or (c) download the CSV from either page yourself and put it in " + str(data_dir))


def make_synthetic(n: int = 60_000, seed: int = 0) -> pd.DataFrame:
    """Criteo-shaped synthetic data with a known heterogeneous effect (for smoke tests only)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12)).astype("float32")
    t = (rng.random(n) < 0.85).astype("int8")
    base = -3.0 + 0.8 * X[:, 0] - 0.5 * X[:, 1]
    lift = 0.6 * (X[:, 2] > 0) - 0.3 * (X[:, 3] > 1)
    p = 1 / (1 + np.exp(-(base + t * lift)))
    visit = (rng.random(n) < p).astype("int8")
    conv = (visit * (rng.random(n) < 0.06)).astype("int8")
    expo = (t * (rng.random(n) < 0.04)).astype("int8")
    df = pd.DataFrame(X, columns=FEATURES)
    df["treatment"], df["visit"], df["conversion"], df["exposure"] = t, visit, conv, expo
    return df
