"""Tests verifying correct TransformerLens hook names and tensor safety patterns."""
from circuits.patching import _patch_head, _patch_hook
from circuits.pca import collect_head_outputs


class TestHookNames:
    """Verify we use hook_z (not hook_result) for per-head attention output."""

    def test_patch_head_uses_hook_z(self):
        """_patch_head should reference hook_z, not hook_result."""
        import inspect
        src = inspect.getsource(_patch_head)
        assert "hook_z" in src
        assert "hook_result" not in src

    def test_patch_hook_no_hook_result(self):
        import inspect
        src = inspect.getsource(_patch_hook)
        assert "hook_result" not in src

    def test_collect_head_outputs_uses_hook_z(self):
        import inspect
        src = inspect.getsource(collect_head_outputs)
        assert "hook_z" in src
        assert "hook_result" not in src


class TestDetachSafety:
    """Verify .detach() is called before .numpy() to avoid grad errors."""

    def test_neurons_detach_before_numpy(self):
        from circuits.neurons import compute_neuron_dla
        import inspect
        src = inspect.getsource(compute_neuron_dla)
        assert ".detach().cpu().numpy()" in src or ".detach().numpy()" in src

    def test_pca_detach_before_numpy(self):
        from circuits.pca import collect_head_outputs
        import inspect
        src = inspect.getsource(collect_head_outputs)
        assert ".detach().cpu().numpy()" in src or ".detach().numpy()" in src
