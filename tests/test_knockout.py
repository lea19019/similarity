"""Tests for circuits.knockout — circuit knockout validation."""
import numpy as np
import pytest

from circuits.knockout import identify_circuit_heads


class TestIdentifyCircuitHeads:
    def test_returns_list(self, tmp_path):
        head_out = np.array([
            [0.5, 0.02, 0.8],
            [0.01, 0.3, 0.05],
        ])
        p = tmp_path / "patching_test.npz"
        np.savez(p, head_out=head_out)

        circuit = identify_circuit_heads(str(p), threshold=0.1)
        assert isinstance(circuit, list)

    def test_threshold_filtering(self, tmp_path):
        head_out = np.array([
            [0.5, 0.02, 0.8],
            [0.01, 0.3, 0.05],
        ])
        p = tmp_path / "patching_test.npz"
        np.savez(p, head_out=head_out)

        circuit = identify_circuit_heads(str(p), threshold=0.1)
        # Should include (0,0)=0.5, (0,2)=0.8, (1,1)=0.3
        assert len(circuit) == 3
        assert (0, 0) in circuit
        assert (0, 2) in circuit
        assert (1, 1) in circuit

    def test_high_threshold(self, tmp_path):
        head_out = np.array([
            [0.5, 0.02, 0.8],
            [0.01, 0.3, 0.05],
        ])
        p = tmp_path / "patching_test.npz"
        np.savez(p, head_out=head_out)

        circuit = identify_circuit_heads(str(p), threshold=0.6)
        assert len(circuit) == 1
        assert (0, 2) in circuit

    def test_zero_threshold(self, tmp_path):
        head_out = np.array([
            [0.5, 0.02, 0.8],
            [0.01, 0.3, 0.05],
        ])
        p = tmp_path / "patching_test.npz"
        np.savez(p, head_out=head_out)

        circuit = identify_circuit_heads(str(p), threshold=0.0)
        # All nonzero heads
        assert len(circuit) == 6

    def test_empty_circuit(self, tmp_path):
        head_out = np.zeros((2, 3))
        p = tmp_path / "patching_test.npz"
        np.savez(p, head_out=head_out)

        circuit = identify_circuit_heads(str(p), threshold=0.1)
        assert len(circuit) == 0
