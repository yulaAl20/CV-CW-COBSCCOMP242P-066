
from __future__ import annotations

from pathlib import Path

import pandas as pd

# DDR encodes the sixth class as label 5.
UNGRADABLE_LABEL = 5

GRADE_NAMES = {
    0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR",
    3: "Severe NPDR", 4: "Proliferative DR",
}

# Quality head target: 0 = gradable, 1 = ungradable.
QUALITY_GRADABLE, QUALITY_UNGRADABLE = 0, 1


def inspect_layout(root: str, max_depth: int = 3, max_entries: int = 12) -> None:
    
    root = Path(root)
    if not root.exists():
        print(f"PATH DOES NOT EXIST: {root}")
        return

    for path in sorted(root.rglob("*"))[:400]:
        depth = len(path.relative_to(root).parts)
        if depth > max_depth:
            continue
        if path.is_dir():
            n = sum(1 for _ in path.iterdir())
            print(f"{'  ' * (depth - 1)}{path.name}/  ({n} entries)")
        elif depth <= 2 or path.suffix in {".csv", ".txt"}:
            print(f"{'  ' * (depth - 1)}{path.name}")


def _read_listing(path: Path) -> pd.DataFrame:
    """Parse a DDR listing file, whether it is whitespace- or comma-separated."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        name_col = next((c for c in df.columns
                         if c in {"image", "id_code", "filename", "name",
                                  "image_name", "img"}), df.columns[0])
        label_col = next((c for c in df.columns
                          if c in {"label", "level", "grade", "diagnosis",
                                   "dr_grade", "class"}), df.columns[-1])
        return pd.DataFrame({"filename": df[name_col].astype(str),
                             "raw_label": df[label_col].astype(int)})

    rows = []
    for line in path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            rows.append({"filename": parts[0], "raw_label": int(parts[1])})
    return pd.DataFrame(rows)


def _find_images(root: Path) -> dict:
    """Map bare filename -> path relative to root, for every image under root.

    Building this index once means the manifest does not have to encode any
    assumption about which subdirectory an image lives in. It costs one
    directory walk and removes an entire class of path bugs.
    """
    index = {}
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
        for path in root.rglob(ext):
            index.setdefault(path.name, str(path.relative_to(root)))
            index.setdefault(path.stem, str(path.relative_to(root)))
    return index


def load_ddr(root: str) -> pd.DataFrame:
    
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} does not exist. Run inspect_layout() on /kaggle/input "
            "to find the real mount path.")

    images = _find_images(root)
    if not images:
        raise FileNotFoundError(f"no image files found under {root}")

    frames, split_source = [], "official"

    # ---- Layout A: official per-split listings --------------------------
    listings = {"train": ["train.txt", "train.csv"],
                "val": ["valid.txt", "val.txt", "valid.csv", "val.csv"],
                "test": ["test.txt", "test.csv"]}
    found = {}
    for split, names in listings.items():
        for name in names:
            matches = list(root.rglob(name))
            if matches:
                found[split] = matches[0]
                break

    if len(found) >= 2:
        for split, path in found.items():
            frame = _read_listing(path)
            frame["split"] = split
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)

    else:
        # ---- Layout B: a single grading CSV -----------------------------
        csvs = [p for p in root.rglob("*.csv")
                if "grading" in p.name.lower() or "label" in p.name.lower()]
        if csvs:
            df = _read_listing(csvs[0])
            split_source = "generated"
        else:
            # ---- Layout C: per-class directories -------------------------
            rows = []
            for label_dir in sorted(root.rglob("*")):
                if label_dir.is_dir() and label_dir.name.isdigit():
                    for img in label_dir.iterdir():
                        if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                            rows.append({"filename": img.name,
                                         "raw_label": int(label_dir.name)})
            if not rows:
                raise RuntimeError(
                    f"could not recognise the DDR layout under {root}. "
                    "Run inspect_layout() and report the tree.")
            df = pd.DataFrame(rows)
            split_source = "generated"

        df = _assign_split(df)

    # Resolve each filename to a real path on disk; drop unmatched rows.
    df["image"] = df["filename"].map(
        lambda f: images.get(f) or images.get(Path(f).name)
        or images.get(Path(f).stem))
    missing = df["image"].isna().sum()
    if missing:
        print(f"[ddr] WARNING: {missing} listed images not found on disk, dropped")
    df = df.dropna(subset=["image"]).reset_index(drop=True)

    # Route the sixth class instead of deleting it.
    df["ungradable"] = df["raw_label"] == UNGRADABLE_LABEL
    df["quality"] = df["ungradable"].map(
        {True: QUALITY_UNGRADABLE, False: QUALITY_GRADABLE})
    df["level"] = df["raw_label"].where(~df["ungradable"], -1)
    df["source"] = "DDR"
    df["split_source"] = split_source

    print(f"[ddr] {len(df)} images, split_source={split_source}, "
          f"{int(df['ungradable'].sum())} ungradable retained")
    return df[["image", "level", "quality", "split", "ungradable", "source",
               "split_source", "raw_label"]]


def _assign_split(df: pd.DataFrame, val_frac: float = 0.15,
                  test_frac: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """Stratified split, used only when the mirror lacks the official one.

    Stratifying on ``raw_label`` keeps the ungradable class and the rare grades
    3 and 4 present in all three partitions. DDR ships no patient identifiers,
    so patient-level grouping is impossible here - that limitation is stated
    explicitly in the report rather than quietly ignored.
    """
    from sklearn.model_selection import train_test_split

    def safe_strata(labels: pd.Series) -> pd.Series | None:
        """Collapse classes too rare to stratify into a single bucket.

        ``train_test_split`` raises when any stratum has fewer than two
        members. On a severely imbalanced dataset that can happen for grade 3
        after the first split, which would abort the whole pipeline over a
        handful of images. Rare classes are merged into one stratum so the
        common classes still get proportional representation; if even that
        fails, stratification is dropped entirely rather than crashing.
        """
        counts = labels.value_counts()
        rare = set(counts[counts < 2].index)
        collapsed = labels.where(~labels.isin(rare), -99)
        return None if collapsed.value_counts().min() < 2 else collapsed

    train_idx, hold_idx = train_test_split(
        df.index, test_size=val_frac + test_frac, random_state=seed,
        stratify=safe_strata(df["raw_label"]))
    val_idx, test_idx = train_test_split(
        hold_idx, test_size=test_frac / (val_frac + test_frac),
        random_state=seed,
        stratify=safe_strata(df.loc[hold_idx, "raw_label"]))

    df = df.copy()
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def load_aptos(root: str, labels_csv: str = "train.csv",
               image_dir: str = "train_images") -> pd.DataFrame:
    """Build the APTOS 2019 manifest as a pure external test set.

    Every row is marked ``split="external"``. Nothing in the training or
    threshold-selection code ever reads this split - that separation is what
    makes the APTOS number an honest generalisation estimate rather than a
    second validation set in disguise.

    APTOS carries no quality annotation, so ``quality`` is -1 throughout and
    the quality head is simply not evaluated on it.
    """
    root = Path(root)
    df = pd.read_csv(root / labels_csv)
    df.columns = [c.strip().lower() for c in df.columns]

    id_col = "id_code" if "id_code" in df.columns else df.columns[0]
    label_col = "diagnosis" if "diagnosis" in df.columns else df.columns[1]

    return pd.DataFrame({
        "image": df[id_col].astype(str).apply(
            lambda x: str(Path(image_dir) / f"{x}.png")),
        "level": df[label_col].astype(int),
        "quality": -1,
        "ungradable": False,
        "split": "external",
        "source": "APTOS",
    })


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Per-split, per-grade summary table for the dataset section of the report."""
    graded = df[~df["ungradable"]]
    table = pd.crosstab(graded["split"], graded["level"])
    table.columns = [f"{g} ({GRADE_NAMES.get(g, g)})" for g in table.columns]
    table["Gradable total"] = table.sum(axis=1)
    table["Ungradable"] = df[df["ungradable"]].groupby("split").size()
    table["Ungradable"] = table["Ungradable"].fillna(0).astype(int)
    table["All images"] = table["Gradable total"] + table["Ungradable"]

    referable = (graded["level"] >= 2).groupby(graded["split"]).mean()
    table["Referable %"] = (referable * 100).round(1)
    return table


def imbalance_report(df: pd.DataFrame) -> pd.DataFrame:
    """Grade counts, percentages and the imbalance ratio driving the sampler."""
    graded = df[~df["ungradable"]]
    counts = graded["level"].value_counts().sort_index()
    out = pd.DataFrame({
        "grade": counts.index,
        "name": [GRADE_NAMES.get(g, str(g)) for g in counts.index],
        "count": counts.values,
    })
    out["percent"] = (out["count"] / out["count"].sum() * 100).round(2)
    out["ratio_to_majority"] = (out["count"].max() / out["count"]).round(1)
    return out


def build_combined_manifest(ddr_root: str, aptos_root: str | None = None,
                            out_csv: str | None = None) -> pd.DataFrame:
    """Assemble the full manifest used by every downstream script."""
    frames = [load_ddr(ddr_root)]
    if aptos_root:
        frames.append(load_aptos(aptos_root))
    df = pd.concat(frames, ignore_index=True)

    if out_csv:
        df.to_csv(out_csv, index=False)

    print(f"[manifest] {len(df)} rows "
          f"({(df['source'] == 'DDR').sum()} DDR, "
          f"{(df['source'] == 'APTOS').sum()} APTOS)")
    print(f"[manifest] {int(df['ungradable'].sum())} ungradable images "
          "retained for the quality head "
          "(standard practice in the literature is to delete these)")
    return df
