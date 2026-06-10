"""Utilities for PCA experiments on sentence embedding matrices."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class EmbeddingPCAResult:
    """Container for PCA outputs that are saved by embedding experiments."""

    coordinates: object
    variance: object
    thresholds: object
    embedding_dimension: int
    n_items: int


def require_package(module_name: str, install_hint: str) -> object:
    """Import an optional runtime dependency with a helpful error message."""

    if importlib.util.find_spec(module_name) is None:
        raise ImportError(
            f"Missing optional dependency '{module_name}'. Install it with: {install_hint}"
        )
    return importlib.import_module(module_name)


def load_site_descriptions(csv_path: str | Path) -> object:
    """Load site names and one-sentence descriptions from a CSV file."""

    pd = require_package("pandas", "pip install pandas")
    data = pd.read_csv(csv_path)
    expected = {"site", "description"}
    missing = expected.difference(data.columns)
    if missing:
        raise ValueError(f"Site description CSV is missing columns: {sorted(missing)}")
    if len(data) < 100:
        raise ValueError("Embedding PCA experiment requires at least 100 site descriptions.")
    return data.loc[:, ["site", "description"]].copy()


def build_embedding_texts(data: object) -> list[str]:
    """Combine site name and description into one phrase for embedding."""

    return [
        f"{row.site}: {row.description}"
        for row in data.itertuples(index=False)
    ]


def encode_with_sentence_transformer(
    texts: Sequence[str],
    model_name: str,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
) -> object:
    """Compute sentence embeddings using a Sentence Transformers model."""

    st = require_package(
        "sentence_transformers",
        "pip install sentence-transformers",
    )
    model = st.SentenceTransformer(model_name)
    return model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=True,
    )


def fit_embedding_pca(
    site_data: object,
    embeddings: object,
    variance_thresholds: Iterable[float] = (0.80, 0.90, 0.95, 0.99),
    max_components_to_report: int | None = None,
) -> EmbeddingPCAResult:
    """Fit PCA and compute coordinates, variance, and reconstruction summaries."""

    np = require_package("numpy", "pip install numpy")
    pd = require_package("pandas", "pip install pandas")
    decomposition = require_package("sklearn.decomposition", "pip install scikit-learn")

    embedding_matrix = np.asarray(embeddings, dtype=float)
    if embedding_matrix.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional matrix.")

    n_items, embedding_dimension = embedding_matrix.shape
    n_components = min(n_items, embedding_dimension)
    pca = decomposition.PCA(n_components=n_components, random_state=0)
    transformed = pca.fit_transform(embedding_matrix)

    centered = embedding_matrix - pca.mean_
    total_energy = float(np.sum(centered ** 2))
    reconstructed_energy = []
    for k in range(1, n_components + 1):
        reconstructed = transformed[:, :k] @ pca.components_[:k, :] + pca.mean_
        error = float(np.sum((embedding_matrix - reconstructed) ** 2))
        reconstructed_energy.append(error / total_energy if total_energy else 0.0)

    variance = pd.DataFrame(
        {
            "pca_dimension": np.arange(1, n_components + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
            "relative_reconstruction_error": reconstructed_energy,
        }
    )
    if max_components_to_report is not None:
        variance = variance.head(max_components_to_report).copy()

    full_cumulative = np.cumsum(pca.explained_variance_ratio_)
    threshold_rows = []
    for threshold in variance_thresholds:
        required = int(np.searchsorted(full_cumulative, threshold, side="left") + 1)
        required = min(required, n_components)
        threshold_rows.append(
            {
                "variance_threshold": float(threshold),
                "pca_dimensions_required": required,
                "cumulative_explained_variance": float(full_cumulative[required - 1]),
                "relative_reconstruction_error": float(reconstructed_energy[required - 1]),
            }
        )
    thresholds = pd.DataFrame(threshold_rows)

    coordinates = site_data.loc[:, ["site", "description"]].copy()
    for component in range(min(3, n_components)):
        coordinates[f"pca_{component + 1}"] = transformed[:, component]

    return EmbeddingPCAResult(
        coordinates=coordinates,
        variance=variance,
        thresholds=thresholds,
        embedding_dimension=embedding_dimension,
        n_items=n_items,
    )


def plot_pca_variance(variance: object) -> object:
    """Plot per-dimension and cumulative explained variance for PCA components."""

    plt = require_package("matplotlib.pyplot", "pip install matplotlib")
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(
        variance["pca_dimension"],
        variance["explained_variance_ratio"],
        color="steelblue",
        alpha=0.65,
        label="Per-component variance",
    )
    ax1.set_xlabel("PCA dimension")
    ax1.set_ylabel("Explained variance ratio")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = ax1.twinx()
    ax2.plot(
        variance["pca_dimension"],
        variance["cumulative_explained_variance"],
        color="darkorange",
        marker="o",
        linewidth=2,
        label="Cumulative variance",
    )
    ax2.set_ylabel("Cumulative explained variance")
    ax2.set_ylim(0, 1.02)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="center right")
    fig.tight_layout()
    return fig


def plot_first_three_pcs(coordinates: object) -> object:
    """Plot the first three PCA dimensions as pairwise scatter plots."""

    plt = require_package("matplotlib.pyplot", "pip install matplotlib")
    pairs = [("pca_1", "pca_2"), ("pca_1", "pca_3"), ("pca_2", "pca_3")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (x_col, y_col) in zip(axes, pairs, strict=True):
        ax.scatter(coordinates[x_col], coordinates[y_col], s=24, alpha=0.75, color="slateblue")
        ax.axhline(0, color="gray", linewidth=0.8, alpha=0.35)
        ax.axvline(0, color="gray", linewidth=0.8, alpha=0.35)
        ax.set_xlabel(x_col.upper())
        ax.set_ylabel(y_col.upper())
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
