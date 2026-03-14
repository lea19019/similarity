"""Tests for circuits.plotting — plot functions produce files without error."""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no display needed

import pytest

from circuits.plotting import (
    plot_dla,
    plot_head_patching,
    plot_pca_scatter,
    plot_steering,
)


class TestPlotHeadPatching:
    def test_produces_file(self, patching_npz, tmp_path):
        npz_path, _ = patching_npz
        out = tmp_path / "fig_patching.png"
        plot_head_patching(str(npz_path), str(out), title="Test Patching")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_default_title(self, patching_npz, tmp_path):
        npz_path, _ = patching_npz
        out = tmp_path / "fig_patching_default.png"
        plot_head_patching(str(npz_path), str(out))
        assert out.exists()


class TestPlotDla:
    def test_produces_file(self, dla_npz, tmp_path):
        npz_path, _ = dla_npz
        out = tmp_path / "fig_dla.png"
        plot_dla(str(npz_path), str(out), title="Test DLA")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_custom_top_k(self, dla_npz, tmp_path):
        npz_path, _ = dla_npz
        out = tmp_path / "fig_dla_topk.png"
        plot_dla(str(npz_path), str(out), top_k=5)
        assert out.exists()


class TestPlotPcaScatter:
    def test_produces_file(self, pca_npz, tmp_path):
        npz_path, _ = pca_npz
        out = tmp_path / "fig_pca.png"
        plot_pca_scatter(str(npz_path), str(out))
        assert out.exists()
        assert out.stat().st_size > 0


class TestPlotSteering:
    def test_produces_file(self, steering_npz, tmp_path):
        npz_path, _ = steering_npz
        out = tmp_path / "fig_steering.png"
        plot_steering(str(npz_path), str(out))
        assert out.exists()
        assert out.stat().st_size > 0
