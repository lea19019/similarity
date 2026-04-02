"""Tests for circuits.viz3d — 3D interactive Plotly visualizations."""
import numpy as np
import pytest

from circuits.viz3d import (
    plot_3d_importance_surface,
    plot_3d_circuit_graph,
    plot_cka_heatmap_animated,
    plot_convergence_3d,
    plot_task_cosine_heatmap,
    plot_svd_spectrum_3d,
)


class TestPlot3DImportanceSurface:
    def test_produces_html_file(self, tmp_path):
        importance_maps = {
            "en": np.random.rand(4, 4),
            "es": np.random.rand(4, 4),
        }
        out = tmp_path / "surface.html"
        plot_3d_importance_surface(importance_maps, str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_html_contains_plotly(self, tmp_path):
        importance_maps = {"en": np.random.rand(4, 4)}
        out = tmp_path / "surface.html"
        plot_3d_importance_surface(importance_maps, str(out))
        content = out.read_text()
        assert "plotly" in content.lower()

    def test_four_languages(self, tmp_path):
        importance_maps = {l: np.random.rand(4, 4) for l in ["en", "es", "tr", "sw"]}
        out = tmp_path / "surface4.html"
        plot_3d_importance_surface(importance_maps, str(out))
        assert out.exists()


class TestPlot3DCircuitGraph:
    def test_produces_html_file(self, tmp_path):
        n_comp = 4 * 4 + 4
        labels = np.array(
            [f"L{l}H{h}" for l in range(4) for h in range(4)]
            + [f"MLP{l}" for l in range(4)]
        )
        node_scores = {"en": np.random.rand(n_comp)}
        out = tmp_path / "graph.html"
        plot_3d_circuit_graph(node_scores, labels, str(out), n_layers=4, n_heads=4)
        assert out.exists()
        assert out.stat().st_size > 0


class TestPlotCKAHeatmapAnimated:
    def test_produces_html_file(self, tmp_path):
        cka = np.random.rand(4, 3, 3)
        # Fill diagonal with 1s
        for l in range(4):
            np.fill_diagonal(cka[l], 1.0)
        out = tmp_path / "cka_anim.html"
        plot_cka_heatmap_animated(cka, ["EN", "ES", "TR"], str(out))
        assert out.exists()
        assert out.stat().st_size > 0


class TestPlotConvergence3D:
    def test_produces_html_file(self, tmp_path):
        geom = {
            "cka_per_layer": np.random.rand(4, 2),
            "svcca_per_layer": np.random.rand(4, 2),
            "rsa_per_layer": np.random.rand(4, 2),
            "pair_labels": np.array(["en-es", "en-tr"]),
        }
        out = tmp_path / "convergence.html"
        plot_convergence_3d(geom, str(out))
        assert out.exists()


class TestPlotTaskCosineHeatmap:
    def test_produces_html_file(self, tmp_path):
        task_cosine = np.random.randn(2, 4, 4)
        pair_labels = ["en-es", "en-tr"]
        out = tmp_path / "cosine.html"
        plot_task_cosine_heatmap(task_cosine, pair_labels, str(out))
        assert out.exists()
        assert out.stat().st_size > 0


class TestPlotSVDSpectrum3D:
    def test_produces_html_file(self, tmp_path):
        spectra = {"en": np.random.rand(4, 4, 5)}
        out = tmp_path / "svd.html"
        plot_svd_spectrum_3d(spectra, str(out))
        assert out.exists()
