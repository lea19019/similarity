"""
3D interactive visualizations using Plotly.

Generates standalone HTML files viewable in any browser.

Usage:
    uv run python -m circuits.viz3d --results-dir results --out-dir results/figures
"""
import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from circuits.config import ALL_LANGS, LANG_CONFIGS, RESULTS_DIR, FIGURES_DIR


# Language colors for consistent cross-plot styling
LANG_COLORS = {
    "en": "#1f77b4",   # blue
    "es": "#d62728",   # red
    "tr": "#2ca02c",   # green
    "sw": "#9467bd",   # purple
}
LANG_COLORSCALES = {
    "en": "Blues",
    "es": "Reds",
    "tr": "Greens",
    "sw": "Purples",
}


def plot_3d_importance_surface(
    importance_maps: dict,
    out_path: str,
    title: str = "Weight Importance Map — per-language surfaces",
) -> None:
    """
    3D surface plot: X=layer, Y=head, Z=importance.
    One semi-transparent surface per language, overlaid for comparison.

    Args:
        importance_maps: {lang: (n_layers, n_heads)} numpy arrays
        out_path: path for HTML output
    """
    fig = go.Figure()

    for lang, data in importance_maps.items():
        n_layers, n_heads = data.shape
        name = LANG_CONFIGS.get(lang, {}).get("name", lang.upper())
        fig.add_trace(go.Surface(
            z=data,
            x=list(range(n_heads)),
            y=list(range(n_layers)),
            name=name,
            colorscale=LANG_COLORSCALES.get(lang, "Viridis"),
            opacity=0.6,
            showscale=False,
            hovertemplate=(
                f"{name}<br>"
                "Layer: %{y}<br>"
                "Head: %{x}<br>"
                "Importance: %{z:.4f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Head",
            yaxis_title="Layer",
            zaxis_title="Weight Importance",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
        ),
        width=1000,
        height=700,
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved 3D surface → {out_path}")


def plot_3d_circuit_graph(
    node_scores: dict,
    component_labels: np.ndarray,
    out_path: str,
    n_layers: int = 18,
    n_heads: int = 18,
    top_k: int = 30,
    title: str = "Circuit Graph — top components per language",
) -> None:
    """
    3D scatter of circuit components: X=layer, Y=head/type, Z=0 (flat).
    Node size = importance, color = language.
    One trace per language, showing top-K most important components.

    Args:
        node_scores: {lang: (n_components,)} from edge_patching
        component_labels: array of "L0H0", ..., "MLP0", ... labels
        out_path: path for HTML output
    """
    fig = go.Figure()

    for lang, scores in node_scores.items():
        top_idx = np.argsort(scores)[::-1][:top_k]
        name = LANG_CONFIGS.get(lang, {}).get("name", lang.upper())

        xs, ys, zs, texts, sizes = [], [], [], [], []
        for idx in top_idx:
            label = str(component_labels[idx])
            if label.startswith("L"):
                # Parse "L{layer}H{head}"
                parts = label[1:].split("H")
                layer, head = int(parts[0]), int(parts[1])
                y_pos = head
                z_pos = 0
            else:
                # MLP{layer}
                layer = int(label[3:])
                y_pos = n_heads + 1  # offset MLPs to the side
                z_pos = 0

            xs.append(layer)
            ys.append(y_pos)
            zs.append(scores[idx])
            texts.append(f"{label}: {scores[idx]:.4f}")
            sizes.append(max(5, scores[idx] / scores.max() * 30))

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=LANG_COLORS.get(lang, "gray"),
                opacity=0.7,
            ),
            text=[str(component_labels[i]) for i in top_idx],
            textposition="top center",
            textfont=dict(size=7),
            name=name,
            hovertext=texts,
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Layer",
            yaxis_title="Head (MLPs offset)",
            zaxis_title="EAP Score",
        ),
        width=1100,
        height=700,
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved 3D circuit graph → {out_path}")


def plot_cka_heatmap_animated(
    cka_per_layer: np.ndarray,
    lang_labels: list,
    out_path: str,
    title: str = "Cross-lingual CKA similarity across layers",
) -> None:
    """
    Animated heatmap showing CKA similarity evolving through layers.
    Uses a slider to step through layers.

    Args:
        cka_per_layer: (n_layers, n_langs, n_langs)
        lang_labels: list of language names for axes
        out_path: path for HTML output
    """
    n_layers = cka_per_layer.shape[0]

    frames = []
    for layer in range(n_layers):
        frames.append(go.Frame(
            data=[go.Heatmap(
                z=cka_per_layer[layer],
                x=lang_labels,
                y=lang_labels,
                colorscale="RdBu_r",
                zmin=0, zmax=1,
                text=np.round(cka_per_layer[layer], 3).astype(str),
                texttemplate="%{text}",
                hovertemplate="CKA(%{x}, %{y}) = %{z:.3f}<extra></extra>",
            )],
            name=str(layer),
        ))

    fig = go.Figure(
        data=[go.Heatmap(
            z=cka_per_layer[0],
            x=lang_labels,
            y=lang_labels,
            colorscale="RdBu_r",
            zmin=0, zmax=1,
            text=np.round(cka_per_layer[0], 3).astype(str),
            texttemplate="%{text}",
        )],
        frames=frames,
    )

    sliders = [dict(
        active=0,
        steps=[dict(
            method="animate",
            args=[[str(l)], dict(mode="immediate", frame=dict(duration=300))],
            label=f"L{l}",
        ) for l in range(n_layers)],
        currentvalue=dict(prefix="Layer: "),
        pad=dict(t=50),
    )]

    fig.update_layout(
        title=title,
        sliders=sliders,
        width=700,
        height=600,
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved animated CKA heatmap → {out_path}")


def plot_convergence_3d(
    geometry_data: dict,
    out_path: str,
    title: str = "Cross-lingual convergence across layers",
) -> None:
    """
    3D line plot showing multiple geometry metrics across layers for each pair.
    X=layer, Y=metric value, Z=pair index.

    Args:
        geometry_data: loaded geometry.npz with cka_per_layer, svcca_per_layer, etc.
        out_path: path for HTML output
    """
    fig = go.Figure()

    pair_labels = list(geometry_data["pair_labels"])
    n_layers = geometry_data["cka_per_layer"].shape[0]
    layers = list(range(n_layers))

    metrics = {
        "CKA": geometry_data["cka_per_layer"],
        "SVCCA": geometry_data["svcca_per_layer"],
        "RSA": geometry_data["rsa_per_layer"],
    }

    for mi, (metric_name, values) in enumerate(metrics.items()):
        for pi, pair_label in enumerate(pair_labels):
            fig.add_trace(go.Scatter3d(
                x=layers,
                y=values[:, pi],
                z=[mi] * n_layers,
                mode="lines+markers",
                marker=dict(size=3),
                line=dict(width=3),
                name=f"{metric_name}: {pair_label}",
                hovertemplate=(
                    f"{metric_name} ({pair_label})<br>"
                    "Layer %{x}<br>"
                    "Score: %{y:.3f}<extra></extra>"
                ),
            ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Layer",
            yaxis_title="Similarity Score",
            zaxis_title="Metric",
            zaxis=dict(
                ticktext=list(metrics.keys()),
                tickvals=list(range(len(metrics))),
            ),
        ),
        width=1100,
        height=700,
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved 3D convergence → {out_path}")


def plot_svd_spectrum_3d(
    svd_spectra: dict,
    out_path: str,
    title: str = "SVD spectra across heads — OV matrix functional rank",
) -> None:
    """
    3D bar chart: for each language, show the top singular values per head
    at a specified layer. X=head, Y=singular value index, Z=magnitude.

    Args:
        svd_spectra: {lang: (n_layers, n_heads, top_k)} from circuit_map
        out_path: path for HTML output
    """
    fig = go.Figure()

    # Show layer 13 (key SVA layer) by default
    target_layer = 13

    for lang, spectra in svd_spectra.items():
        n_layers = spectra.shape[0]
        layer = min(target_layer, n_layers - 1)
        data = spectra[layer]  # (n_heads, top_k)
        n_heads, top_k = data.shape
        name = LANG_CONFIGS.get(lang, {}).get("name", lang.upper())

        for head in range(n_heads):
            fig.add_trace(go.Bar(
                x=[f"H{head}"] * top_k,
                y=data[head],
                name=f"{name} H{head}" if head == 0 else None,
                showlegend=(head == 0),
                marker_color=LANG_COLORS.get(lang, "gray"),
                opacity=0.7,
                legendgroup=lang,
            ))

    fig.update_layout(
        title=f"{title} (Layer {target_layer})",
        xaxis_title="Head",
        yaxis_title="Singular Value",
        barmode="group",
        width=1200,
        height=500,
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved SVD spectrum → {out_path}")


def plot_task_cosine_heatmap(
    task_cosine: np.ndarray,
    pair_labels: list,
    out_path: str,
    title: str = "Task projection cosine similarity across heads",
) -> None:
    """
    Heatmap of per-head task projection cosine similarity for each language pair.
    Subplots: one per pair. X=head, Y=layer, color=cosine similarity.

    Args:
        task_cosine: (n_pairs, n_layers, n_heads)
        pair_labels: list of "lang_a-lang_b" strings
        out_path: path for HTML output
    """
    n_pairs = task_cosine.shape[0]
    cols = min(n_pairs, 3)
    rows = (n_pairs + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[str(l) for l in pair_labels],
        shared_xaxes=True, shared_yaxes=True,
    )

    for pi, label in enumerate(pair_labels):
        r = pi // cols + 1
        c = pi % cols + 1
        fig.add_trace(go.Heatmap(
            z=task_cosine[pi],
            colorscale="RdBu_r",
            zmid=0, zmin=-1, zmax=1,
            showscale=(pi == 0),
            hovertemplate=f"{label}<br>Layer %{{y}}, Head %{{x}}<br>Cosine: %{{z:.3f}}<extra></extra>",
        ), row=r, col=c)

    fig.update_layout(
        title=title,
        width=400 * cols,
        height=350 * rows,
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved task cosine heatmap → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D interactive visualizations"
    )
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    parser.add_argument("--langs", nargs="+", default=ALL_LANGS)
    args = parser.parse_args()

    r = Path(args.results_dir)
    o = Path(args.out_dir)
    o.mkdir(parents=True, exist_ok=True)

    # 1. Importance surfaces from circuit_map
    importance_maps = {}
    svd_spectra = {}
    for lang in args.langs:
        cm_path = r / f"circuit_map_{lang}.npz"
        if cm_path.exists():
            data = np.load(cm_path)
            importance_maps[lang] = data["head_importance"]
            svd_spectra[lang] = data["svd_spectra"]

    if importance_maps:
        plot_3d_importance_surface(importance_maps, str(o / "viz_importance_3d.html"))

    if svd_spectra:
        plot_svd_spectrum_3d(svd_spectra, str(o / "viz_svd_spectrum.html"))

    # 2. Circuit graph from edge_patching
    node_scores = {}
    component_labels = None
    for lang in args.langs:
        ep_path = r / f"edge_patching_{lang}.npz"
        if ep_path.exists():
            data = np.load(ep_path, allow_pickle=True)
            node_scores[lang] = data["node_scores"]
            if component_labels is None:
                component_labels = data["component_labels"]

    if node_scores and component_labels is not None:
        plot_3d_circuit_graph(node_scores, component_labels, str(o / "viz_circuit_graph_3d.html"))

    # 3. Geometry metrics
    geom_path = r / "geometry.npz"
    if geom_path.exists():
        geom = np.load(geom_path, allow_pickle=True)

        # Convergence plot
        plot_convergence_3d(geom, str(o / "viz_convergence_3d.html"))

        # Task cosine heatmap
        if "task_cosine" in geom:
            plot_task_cosine_heatmap(
                geom["task_cosine"],
                list(geom["pair_labels"]),
                str(o / "viz_task_cosine.html"),
            )

        # Animated CKA heatmap — need to reconstruct per-layer NxN matrices
        cka_per_layer = geom["cka_per_layer"]  # (n_layers, n_pairs)
        pair_labels = list(geom["pair_labels"])
        n_layers = cka_per_layer.shape[0]
        langs_used = sorted(set(l for p in pair_labels for l in p.split("-")))
        n_langs = len(langs_used)

        # Reconstruct (n_layers, n_langs, n_langs) matrix from pairs
        cka_matrices = np.zeros((n_layers, n_langs, n_langs))
        for layer in range(n_layers):
            np.fill_diagonal(cka_matrices[layer], 1.0)
            for pi, pair in enumerate(pair_labels):
                la, lb = pair.split("-")
                ai, bi = langs_used.index(la), langs_used.index(lb)
                cka_matrices[layer, ai, bi] = cka_per_layer[layer, pi]
                cka_matrices[layer, bi, ai] = cka_per_layer[layer, pi]

        lang_names = [LANG_CONFIGS.get(l, {}).get("name", l) for l in langs_used]
        plot_cka_heatmap_animated(cka_matrices, lang_names, str(o / "viz_cka_animated.html"))

    print(f"\nAll 3D visualizations saved to {o}")


if __name__ == "__main__":
    main()
