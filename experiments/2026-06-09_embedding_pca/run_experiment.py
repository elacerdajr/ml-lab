"""Run PCA reconstruction analysis on sentence embeddings of site descriptions.

The preferred path uses Sentence Transformers plus the scientific Python stack.  A
minimal, deterministic fallback is included so the experiment can still generate
artifacts in constrained/offline containers where those optional packages are not
installed.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

INSTALL_HINT = "pip install numpy pandas scikit-learn matplotlib pyyaml sentence-transformers"
REQUIRED_PACKAGES = ["numpy", "pandas", "sklearn", "matplotlib", "yaml", "sentence_transformers"]
TOKEN_RE = re.compile(r"[a-z0-9]+")


DEFAULT_CONFIG = {
    "embedding": {
        "provider": "sentence-transformers",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "normalize_embeddings": True,
        "batch_size": 32,
        "fallback_provider": "local-hashing-sentence-embedding",
        "fallback_dimension": 384,
    },
    "pca": {
        "max_components_to_report": 50,
        "variance_thresholds": [0.80, 0.90, 0.95, 0.99],
    },
    "paths": {
        "sites_csv": "site_descriptions.csv",
        "output_dir": "outputs",
    },
}


def available_runtime_dependencies() -> dict[str, bool]:
    """Return availability for the optional full experiment dependencies."""

    return {name: importlib.util.find_spec(name) is not None for name in REQUIRED_PACKAGES}


def load_config(path: Path) -> dict:
    """Load experiment configuration from YAML, falling back to defaults."""

    if importlib.util.find_spec("yaml") is None:
        return DEFAULT_CONFIG
    import yaml

    loaded = yaml.safe_load(path.read_text())
    return loaded or DEFAULT_CONFIG


def read_site_descriptions(csv_path: Path) -> list[dict[str, str]]:
    """Read and validate site descriptions with the standard library."""

    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 100:
        raise ValueError("Embedding PCA experiment requires at least 100 site descriptions.")
    for row in rows:
        if not row.get("site") or not row.get("description"):
            raise ValueError("Every row must include populated site and description fields.")
    return rows


def embedding_texts(rows: Iterable[dict[str, str]]) -> list[str]:
    """Combine site name and site description into a single embedding phrase."""

    return [f"{row['site']}: {row['description']}" for row in rows]


def _signed_bucket(term: str, dimensions: int) -> tuple[int, float]:
    digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    sign = 1.0 if value & 1 else -1.0
    return value % dimensions, sign


def local_sentence_embeddings(texts: list[str], dimensions: int) -> list[list[float]]:
    """Create deterministic medium-sized sentence embeddings without dependencies.

    The fallback embedding mixes token, token-bigram, and character-trigram
    features into a signed hashing vector and L2-normalizes each row.  It is not a
    replacement for `all-MiniLM-L6-v2`, but it preserves the experiment workflow
    and makes the PCA artifacts reproducible in dependency-limited environments.
    """

    embeddings: list[list[float]] = []
    for text in texts:
        tokens = TOKEN_RE.findall(text.lower())
        vector = [0.0] * dimensions
        features: list[tuple[str, float]] = []
        features.extend((f"tok:{token}", 1.0) for token in tokens)
        features.extend(
            (f"bigram:{left}_{right}", 1.3)
            for left, right in zip(tokens, tokens[1:], strict=False)
        )
        compact = " ".join(tokens)
        features.extend((f"char3:{compact[i:i + 3]}", 0.35) for i in range(max(0, len(compact) - 2)))

        for feature, weight in features:
            bucket, sign = _signed_bucket(feature, dimensions)
            vector[bucket] += sign * weight
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        embeddings.append([value / norm for value in vector])
    return embeddings


def center_matrix(matrix: list[list[float]]) -> tuple[list[list[float]], list[float]]:
    """Mean-center an embedding matrix."""

    n_rows = len(matrix)
    n_cols = len(matrix[0])
    means = [sum(row[col] for row in matrix) / n_rows for col in range(n_cols)]
    centered = [[row[col] - means[col] for col in range(n_cols)] for row in matrix]
    return centered, means


def gram_matrix(centered: list[list[float]]) -> list[list[float]]:
    """Compute sample covariance Gram matrix X X^T / (n - 1)."""

    n_rows = len(centered)
    denominator = max(1, n_rows - 1)
    gram = [[0.0] * n_rows for _ in range(n_rows)]
    for i in range(n_rows):
        for j in range(i, n_rows):
            value = sum(a * b for a, b in zip(centered[i], centered[j], strict=True)) / denominator
            gram[i][j] = value
            gram[j][i] = value
    return gram


def jacobi_eigh(matrix: list[list[float]], max_sweeps: int = 120, tolerance: float = 1e-12) -> tuple[list[float], list[list[float]]]:
    """Eigen-decompose a small symmetric matrix using Jacobi rotations."""

    n = len(matrix)
    a = [row[:] for row in matrix]
    vectors = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(max_sweeps):
        p, q = 0, 1
        max_offdiag = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                candidate = abs(a[i][j])
                if candidate > max_offdiag:
                    max_offdiag = candidate
                    p, q = i, j
        if max_offdiag < tolerance:
            break

        if abs(a[p][q]) < tolerance:
            continue
        tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
        t = math.copysign(1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau)), tau)
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c
        app = a[p][p]
        aqq = a[q][q]
        apq = a[p][q]

        a[p][p] = app - t * apq
        a[q][q] = aqq + t * apq
        a[p][q] = 0.0
        a[q][p] = 0.0

        for k in range(n):
            if k not in (p, q):
                akp = a[k][p]
                akq = a[k][q]
                a[k][p] = c * akp - s * akq
                a[p][k] = a[k][p]
                a[k][q] = s * akp + c * akq
                a[q][k] = a[k][q]

        for k in range(n):
            vkp = vectors[k][p]
            vkq = vectors[k][q]
            vectors[k][p] = c * vkp - s * vkq
            vectors[k][q] = s * vkp + c * vkq

    eigenvalues = [max(0.0, a[i][i]) for i in range(n)]
    order = sorted(range(n), key=lambda idx: eigenvalues[idx], reverse=True)
    sorted_values = [eigenvalues[idx] for idx in order]
    sorted_vectors = [[vectors[row][idx] for idx in order] for row in range(n)]
    return sorted_values, sorted_vectors


def fit_pca_builtin(rows: list[dict[str, str]], embeddings: list[list[float]], thresholds: list[float], max_components: int) -> dict[str, object]:
    """Fit PCA with the standard library and return serializable tables."""

    centered, _means = center_matrix(embeddings)
    eigenvalues, eigenvectors = jacobi_eigh(gram_matrix(centered))
    positive_count = max(1, min(len(rows) - 1, len(embeddings[0])))
    eigenvalues = eigenvalues[:positive_count]
    total_variance = sum(eigenvalues) or 1.0
    cumulative = []
    running = 0.0
    for value in eigenvalues:
        running += value / total_variance
        cumulative.append(min(1.0, running))

    variance_rows = []
    for idx, value in enumerate(eigenvalues[:max_components], start=1):
        variance_rows.append(
            {
                "pca_dimension": idx,
                "explained_variance_ratio": value / total_variance,
                "cumulative_explained_variance": cumulative[idx - 1],
                "relative_reconstruction_error": max(0.0, 1.0 - cumulative[idx - 1]),
            }
        )

    threshold_rows = []
    for threshold in thresholds:
        required = next((idx + 1 for idx, value in enumerate(cumulative) if value >= threshold), len(cumulative))
        threshold_rows.append(
            {
                "variance_threshold": threshold,
                "pca_dimensions_required": required,
                "cumulative_explained_variance": cumulative[required - 1],
                "relative_reconstruction_error": max(0.0, 1.0 - cumulative[required - 1]),
            }
        )

    coordinate_rows = []
    for row_idx, row in enumerate(rows):
        coord_row = {"site": row["site"], "description": row["description"]}
        for component in range(3):
            scale = math.sqrt(max(0.0, eigenvalues[component]) * max(1, len(rows) - 1))
            coord_row[f"pca_{component + 1}"] = eigenvectors[row_idx][component] * scale
        coordinate_rows.append(coord_row)

    return {
        "variance": variance_rows,
        "thresholds": threshold_rows,
        "coordinates": coordinate_rows,
        "embedding_dimension": len(embeddings[0]),
        "n_items": len(rows),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a list of dictionaries to CSV."""

    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float) -> str:
    """Format numeric report values consistently."""

    return f"{value:.4f}"


def make_svg_variance(path: Path, variance_rows: list[dict[str, object]]) -> None:
    """Create a dependency-free SVG variance plot."""

    width, height = 960, 520
    margin_left, margin_bottom, margin_top, margin_right = 70, 60, 45, 35
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    rows = variance_rows
    max_dim = max(1, int(rows[-1]["pca_dimension"]))
    bar_width = plot_width / max_dim * 0.75

    def x_pos(dim: int) -> float:
        return margin_left + (dim - 0.5) * plot_width / max_dim

    def y_pos(value: float) -> float:
        return margin_top + (1.0 - value) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="28" font-family="Arial" font-size="20" font-weight="bold">PCA explained variance by dimension</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#333"/>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = y_pos(tick)
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#ddd"/>')
        parts.append(f'<text x="18" y="{y + 4:.2f}" font-family="Arial" font-size="12">{tick:.2f}</text>')
    for row in rows:
        dim = int(row["pca_dimension"])
        explained = float(row["explained_variance_ratio"])
        cumulative_value = float(row["cumulative_explained_variance"])
        x = x_pos(dim)
        y = y_pos(explained)
        parts.append(
            f'<rect x="{x - bar_width / 2:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{margin_top + plot_height - y:.2f}" fill="#4682b4" opacity="0.65"/>'
        )
        parts.append(f'<circle cx="{x:.2f}" cy="{y_pos(cumulative_value):.2f}" r="3" fill="#d97706"/>')
        if dim > 1:
            previous = rows[dim - 2]
            parts.append(
                f'<line x1="{x_pos(dim - 1):.2f}" y1="{y_pos(float(previous["cumulative_explained_variance"])):.2f}" x2="{x:.2f}" y2="{y_pos(cumulative_value):.2f}" stroke="#d97706" stroke-width="2"/>'
            )
    parts.extend(
        [
            f'<text x="{margin_left + plot_width / 2 - 65:.2f}" y="{height - 18}" font-family="Arial" font-size="14">PCA dimension</text>',
            '<text x="18" y="250" transform="rotate(-90 18 250)" font-family="Arial" font-size="14">Explained variance</text>',
            '<rect x="705" y="54" width="16" height="10" fill="#4682b4" opacity="0.65"/><text x="728" y="64" font-family="Arial" font-size="13">Per-component variance</text>',
            '<circle cx="713" cy="84" r="4" fill="#d97706"/><text x="728" y="88" font-family="Arial" font-size="13">Cumulative variance</text>',
            '</svg>',
        ]
    )
    path.write_text("\n".join(parts))


def make_svg_pca123(path: Path, coordinate_rows: list[dict[str, object]]) -> None:
    """Create dependency-free pairwise scatter plots for PCA 1, 2, and 3."""

    width, height = 1200, 430
    panel_width, panel_height = 350, 300
    top = 70
    lefts = [55, 435, 815]
    pairs = [("pca_1", "pca_2"), ("pca_1", "pca_3"), ("pca_2", "pca_3")]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="32" font-family="Arial" font-size="20" font-weight="bold">Website descriptions in first three PCA dimensions</text>',
    ]
    for left, (x_col, y_col) in zip(lefts, pairs, strict=True):
        xs = [float(row[x_col]) for row in coordinate_rows]
        ys = [float(row[y_col]) for row in coordinate_rows]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad_x = (max_x - min_x) * 0.08 or 1.0
        pad_y = (max_y - min_y) * 0.08 or 1.0
        min_x -= pad_x
        max_x += pad_x
        min_y -= pad_y
        max_y += pad_y

        def sx(value: float) -> float:
            return left + (value - min_x) / (max_x - min_x) * panel_width

        def sy(value: float) -> float:
            return top + (1.0 - (value - min_y) / (max_y - min_y)) * panel_height

        parts.append(f'<rect x="{left}" y="{top}" width="{panel_width}" height="{panel_height}" fill="#fbfbfb" stroke="#333"/>')
        if min_x <= 0 <= max_x:
            zero_x = sx(0.0)
            parts.append(f'<line x1="{zero_x:.2f}" y1="{top}" x2="{zero_x:.2f}" y2="{top + panel_height}" stroke="#aaa"/>')
        if min_y <= 0 <= max_y:
            zero_y = sy(0.0)
            parts.append(f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + panel_width}" y2="{zero_y:.2f}" stroke="#aaa"/>')
        for row in coordinate_rows:
            parts.append(f'<circle cx="{sx(float(row[x_col])):.2f}" cy="{sy(float(row[y_col])):.2f}" r="4" fill="#6a5acd" opacity="0.75"/>')
        parts.append(f'<text x="{left + panel_width / 2 - 32:.2f}" y="{top + panel_height + 36}" font-family="Arial" font-size="13">{x_col.upper()}</text>')
        parts.append(f'<text x="{left + 10}" y="{top - 12}" font-family="Arial" font-size="13">{y_col.upper()} vs {x_col.upper()}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts))


def write_report(path: Path, metadata: dict[str, object], threshold_rows: list[dict[str, object]], variance_rows: list[dict[str, object]]) -> None:
    """Write a Markdown report summarizing the PCA reconstruction experiment."""

    first = variance_rows[0]
    third = variance_rows[2]
    threshold_lines = [
        f"| {row['variance_threshold']:.0%} | {row['pca_dimensions_required']} | {format_float(float(row['cumulative_explained_variance']))} | {format_float(float(row['relative_reconstruction_error']))} |"
        for row in threshold_rows
    ]
    report = f"""# Embedding PCA reconstruction report

## Setup

- Embedding provider: `{metadata['embedding_provider']}`
- Embedding model: `{metadata['embedding_model']}`
- Number of site descriptions: {metadata['n_items']}
- Embedding dimension: {metadata['embedding_dimension']}
- PCA dimensions reported: {len(variance_rows)}

## Key results

- PCA dimension 1 explains {format_float(float(first['explained_variance_ratio']))} of variance, with cumulative explained variance {format_float(float(first['cumulative_explained_variance']))}.
- PCA dimensions 1-3 explain {format_float(float(third['cumulative_explained_variance']))} of variance, leaving relative reconstruction error {format_float(float(third['relative_reconstruction_error']))}.
- The configured variance thresholds require the following PCA dimensions:

| Variance threshold | PCA dimensions required | Cumulative explained variance | Relative reconstruction error |
|---:|---:|---:|---:|
{chr(10).join(threshold_lines)}

## Artifacts

- `pca_variance.csv` contains explained variance, cumulative variance, and reconstruction error by PCA dimension.
- `pca_thresholds.csv` contains the PCA dimension count required for each configured variance threshold.
- `site_pca_coordinates.csv` contains each site projected onto PCA dimensions 1, 2, and 3.
- `embedding_metadata.json` records the embedding provider, model, dimensions, row count, and thresholds.
- `fig_pca_variance.svg` plots per-component and cumulative explained variance.
- `fig_pca_123.svg` plots pairwise views of PCA dimensions 1, 2, and 3.
"""
    path.write_text(report)


def run_fallback(cfg: dict, missing: list[str]) -> None:
    """Run the dependency-free fallback implementation."""

    paths = cfg["paths"]
    embedding_cfg = cfg["embedding"]
    pca_cfg = cfg["pca"]
    rows = read_site_descriptions(HERE / paths["sites_csv"])
    texts = embedding_texts(rows)
    dimensions = int(embedding_cfg.get("fallback_dimension", 384))
    embeddings = local_sentence_embeddings(texts, dimensions)
    result = fit_pca_builtin(
        rows,
        embeddings,
        [float(value) for value in pca_cfg.get("variance_thresholds", [0.80, 0.90, 0.95, 0.99])],
        int(pca_cfg.get("max_components_to_report", 50)),
    )

    out = HERE / paths["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "pca_variance.csv", result["variance"])
    write_csv(out / "pca_thresholds.csv", result["thresholds"])
    write_csv(out / "site_pca_coordinates.csv", result["coordinates"])

    metadata = {
        "embedding_provider": embedding_cfg.get("fallback_provider", "local-hashing-sentence-embedding"),
        "embedding_model": f"signed token/ngram hashing fallback ({dimensions} dimensions)",
        "preferred_embedding_provider": embedding_cfg["provider"],
        "preferred_embedding_model": embedding_cfg["model_name"],
        "fallback_reason": f"Missing runtime dependencies: {', '.join(missing)}. Install with: {INSTALL_HINT}",
        "normalize_embeddings": True,
        "n_items": result["n_items"],
        "embedding_dimension": result["embedding_dimension"],
        "variance_thresholds": result["thresholds"],
    }
    (out / "embedding_metadata.json").write_text(json.dumps(metadata, indent=2))
    make_svg_variance(out / "fig_pca_variance.svg", result["variance"])
    make_svg_pca123(out / "fig_pca_123.svg", result["coordinates"])
    write_report(out / "report.md", metadata, result["thresholds"], result["variance"])
    print(f"done -> {out} (fallback mode; missing: {', '.join(missing)})")


def run_sentence_transformer(cfg: dict) -> None:
    """Run the preferred Sentence Transformers implementation."""

    from ml_elements import (
        build_embedding_texts,
        encode_with_sentence_transformer,
        fit_embedding_pca,
        load_site_descriptions,
        plot_first_three_pcs,
        plot_pca_variance,
    )
    from ml_elements.embedding_pca import require_package

    paths = cfg["paths"]
    embedding_cfg = cfg["embedding"]
    pca_cfg = cfg["pca"]
    site_data = load_site_descriptions(HERE / paths["sites_csv"])
    texts = build_embedding_texts(site_data)
    embeddings = encode_with_sentence_transformer(
        texts,
        model_name=embedding_cfg["model_name"],
        batch_size=int(embedding_cfg.get("batch_size", 32)),
        normalize_embeddings=bool(embedding_cfg.get("normalize_embeddings", True)),
    )

    result = fit_embedding_pca(
        site_data=site_data,
        embeddings=embeddings,
        variance_thresholds=pca_cfg.get("variance_thresholds", [0.80, 0.90, 0.95, 0.99]),
        max_components_to_report=int(pca_cfg.get("max_components_to_report", 50)),
    )

    out = HERE / paths["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    result.variance.to_csv(out / "pca_variance.csv", index=False)
    result.thresholds.to_csv(out / "pca_thresholds.csv", index=False)
    result.coordinates.to_csv(out / "site_pca_coordinates.csv", index=False)

    metadata = {
        "embedding_provider": embedding_cfg["provider"],
        "embedding_model": embedding_cfg["model_name"],
        "normalize_embeddings": bool(embedding_cfg.get("normalize_embeddings", True)),
        "n_items": result.n_items,
        "embedding_dimension": result.embedding_dimension,
        "variance_thresholds": result.thresholds.to_dict(orient="records"),
    }
    (out / "embedding_metadata.json").write_text(json.dumps(metadata, indent=2))

    matplotlib = require_package("matplotlib", "pip install matplotlib")
    matplotlib.use("Agg")
    fig_variance = plot_pca_variance(result.variance)
    fig_variance.savefig(out / "fig_pca_variance.png", dpi=160)
    fig_pcs = plot_first_three_pcs(result.coordinates)
    fig_pcs.savefig(out / "fig_pca_123.png", dpi=160)

    variance_rows = result.variance.to_dict(orient="records")
    threshold_rows = result.thresholds.to_dict(orient="records")
    make_svg_variance(out / "fig_pca_variance.svg", variance_rows)
    make_svg_pca123(out / "fig_pca_123.svg", result.coordinates.to_dict(orient="records"))
    write_report(out / "report.md", metadata, threshold_rows, variance_rows)
    print(f"done -> {out}")


def main() -> None:
    cfg = load_config(HERE / "config.yaml")
    availability = available_runtime_dependencies()
    missing = [name for name, present in availability.items() if not present]
    if missing:
        run_fallback(cfg, missing)
    else:
        run_sentence_transformer(cfg)


if __name__ == "__main__":
    main()
