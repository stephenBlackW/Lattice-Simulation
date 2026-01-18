"""Tests for 4D lattice geometry and periodic boundary conditions."""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lattice import Lattice4D


class TestLatticeConstruction:
    """Test lattice construction and basic properties."""

    def test_lattice_size(self):
        """Verify lattice has correct number of sites."""
        for N in [4, 8]:
            lattice = Lattice4D(N)
            assert len(lattice.displacements) == N ** 4
            assert lattice.N == N

    def test_initial_displacements_zero(self):
        """Verify all initial displacements are zero."""
        lattice = Lattice4D(4)
        for disp in lattice.displacements.values():
            assert np.allclose(disp, np.zeros(4))

    def test_all_positions_present(self):
        """Verify all expected positions exist."""
        N = 4
        lattice = Lattice4D(N)
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for w in range(N):
                        assert (x, y, z, w) in lattice.displacements


class TestPeriodicBoundary:
    """Test periodic boundary condition wrapping."""

    def test_wrap_positive(self):
        """Test wrapping of positive coordinates."""
        lattice = Lattice4D(8)
        assert lattice.wrap(8) == 0
        assert lattice.wrap(9) == 1
        assert lattice.wrap(15) == 7
        assert lattice.wrap(16) == 0

    def test_wrap_negative(self):
        """Test wrapping of negative coordinates."""
        lattice = Lattice4D(8)
        assert lattice.wrap(-1) == 7
        assert lattice.wrap(-8) == 0
        assert lattice.wrap(-9) == 7

    def test_wrap_in_range(self):
        """Test that in-range coordinates are unchanged."""
        lattice = Lattice4D(8)
        for i in range(8):
            assert lattice.wrap(i) == i

    def test_wrap_position_tuple(self):
        """Test wrapping of full position tuples."""
        lattice = Lattice4D(8)
        assert lattice.wrap_position((8, -1, 0, 9)) == (0, 7, 0, 1)
        assert lattice.wrap_position((0, 0, 0, 0)) == (0, 0, 0, 0)
        assert lattice.wrap_position((7, 7, 7, 7)) == (7, 7, 7, 7)


class TestNeighborFinding:
    """Test neighbor finding with periodic boundaries."""

    def test_neighbor_count(self):
        """Verify all sites have exactly 8 neighbors."""
        lattice = Lattice4D(8)
        for pos in lattice.displacements:
            neighbors = lattice.get_neighbors(pos)
            assert len(neighbors) == 8, f"Site {pos} has {len(neighbors)} neighbors, expected 8"

    def test_neighbors_unique(self):
        """Verify all neighbors are unique."""
        lattice = Lattice4D(8)
        for pos in lattice.displacements:
            neighbors = lattice.get_neighbors(pos)
            assert len(neighbors) == len(set(neighbors)), f"Duplicate neighbors for {pos}"

    def test_interior_neighbors(self):
        """Test neighbors for an interior point (no wrapping needed)."""
        lattice = Lattice4D(8)
        pos = (4, 4, 4, 4)
        neighbors = lattice.get_neighbors(pos)

        expected = [
            (5, 4, 4, 4), (3, 4, 4, 4),  # x
            (4, 5, 4, 4), (4, 3, 4, 4),  # y
            (4, 4, 5, 4), (4, 4, 3, 4),  # z
            (4, 4, 4, 5), (4, 4, 4, 3),  # w
        ]
        assert set(neighbors) == set(expected)

    def test_edge_neighbors_wrap(self):
        """Test neighbors at boundary wrap correctly."""
        lattice = Lattice4D(8)

        # Test at origin
        neighbors = lattice.get_neighbors((0, 0, 0, 0))
        expected = [
            (1, 0, 0, 0), (7, 0, 0, 0),  # x wraps to 7
            (0, 1, 0, 0), (0, 7, 0, 0),  # y wraps to 7
            (0, 0, 1, 0), (0, 0, 7, 0),  # z wraps to 7
            (0, 0, 0, 1), (0, 0, 0, 7),  # w wraps to 7
        ]
        assert set(neighbors) == set(expected)

        # Test at far corner
        neighbors = lattice.get_neighbors((7, 7, 7, 7))
        expected = [
            (0, 7, 7, 7), (6, 7, 7, 7),  # x wraps to 0
            (7, 0, 7, 7), (7, 6, 7, 7),  # y wraps to 0
            (7, 7, 0, 7), (7, 7, 6, 7),  # z wraps to 0
            (7, 7, 7, 0), (7, 7, 7, 6),  # w wraps to 0
        ]
        assert set(neighbors) == set(expected)

    def test_neighbors_are_valid_positions(self):
        """Verify all neighbors are valid lattice positions."""
        lattice = Lattice4D(8)
        for pos in lattice.displacements:
            for neighbor in lattice.get_neighbors(pos):
                assert neighbor in lattice.displacements


class TestNeighborVectors:
    """Test neighbor vector calculations."""

    def test_ideal_neighbor_vectors_interior(self):
        """Test ideal neighbor vectors for interior point."""
        lattice = Lattice4D(8)
        pos = (4, 4, 4, 4)

        for neighbor in lattice.get_neighbors(pos):
            vec = lattice.get_ideal_neighbor_vector(pos, neighbor)
            # Should be unit vector along one axis
            assert np.linalg.norm(vec) == pytest.approx(1.0)
            assert sum(abs(v) for v in vec) == 1  # Only one component nonzero

    def test_ideal_neighbor_vectors_wrapped(self):
        """Test ideal neighbor vectors respect periodic boundaries."""
        lattice = Lattice4D(8)

        # Origin to (7, 0, 0, 0) should be (-1, 0, 0, 0) not (7, 0, 0, 0)
        vec = lattice.get_ideal_neighbor_vector((0, 0, 0, 0), (7, 0, 0, 0))
        assert np.allclose(vec, [-1, 0, 0, 0])

        # (7, 0, 0, 0) to (0, 0, 0, 0) should be (1, 0, 0, 0) not (-7, 0, 0, 0)
        vec = lattice.get_ideal_neighbor_vector((7, 0, 0, 0), (0, 0, 0, 0))
        assert np.allclose(vec, [1, 0, 0, 0])

    def test_all_ideal_neighbor_distances_are_one(self):
        """Verify all ideal neighbor distances are exactly 1."""
        lattice = Lattice4D(8)
        for pos in lattice.displacements:
            for neighbor in lattice.get_neighbors(pos):
                vec = lattice.get_ideal_neighbor_vector(pos, neighbor)
                dist = np.linalg.norm(vec)
                assert dist == pytest.approx(1.0), f"Distance {pos} to {neighbor} is {dist}"

    def test_actual_equals_ideal_without_displacement(self):
        """With zero displacements, actual vectors equal ideal vectors."""
        lattice = Lattice4D(8)
        pos = (3, 3, 3, 3)
        for neighbor in lattice.get_neighbors(pos):
            ideal = lattice.get_ideal_neighbor_vector(pos, neighbor)
            actual = lattice.get_actual_neighbor_vector(pos, neighbor)
            assert np.allclose(ideal, actual)

    def test_actual_vectors_with_displacement(self):
        """Test that displacements correctly modify neighbor vectors."""
        lattice = Lattice4D(8)
        pos = (3, 3, 3, 3)
        neighbor = (4, 3, 3, 3)

        # Set displacement on neighbor
        lattice.displacements[neighbor] = np.array([0.1, 0.0, 0.0, 0.0])

        ideal = lattice.get_ideal_neighbor_vector(pos, neighbor)
        actual = lattice.get_actual_neighbor_vector(pos, neighbor)

        # Actual should be ideal + displacement difference
        assert np.allclose(actual, [1.1, 0.0, 0.0, 0.0])


class TestEnergy:
    """Test energy calculations."""

    def test_zero_energy_undeformed(self):
        """Undeformed lattice should have zero energy."""
        lattice = Lattice4D(4)
        total_energy = lattice.compute_total_energy()
        assert total_energy == pytest.approx(0.0)

    def test_site_energy_undeformed(self):
        """Site energy should be zero for undeformed lattice."""
        lattice = Lattice4D(4)
        for pos in lattice.displacements:
            energy = lattice.compute_site_energy(pos)
            assert energy == pytest.approx(0.0)

    def test_energy_increases_with_displacement(self):
        """Deforming the lattice should increase energy."""
        lattice = Lattice4D(4)

        # Add some displacement
        lattice.displacements[(2, 2, 2, 2)] = np.array([0.1, 0.0, 0.0, 0.0])

        total_energy = lattice.compute_total_energy()
        assert total_energy > 0


class TestStatistics:
    """Test lattice statistics."""

    def test_statistics_initial(self):
        """Test statistics for initial lattice."""
        lattice = Lattice4D(8)
        stats = lattice.get_statistics()

        assert stats['N'] == 8
        assert stats['num_sites'] == 8 ** 4
        assert stats['expected_sites'] == 8 ** 4
        assert stats['min_neighbors'] == 8
        assert stats['max_neighbors'] == 8
        assert stats['max_displacement'] == pytest.approx(0.0)
        assert stats['mean_displacement'] == pytest.approx(0.0)


class TestCheckpoint:
    """Test save/load functionality."""

    def test_save_load_roundtrip(self, tmp_path):
        """Test that save and load preserves lattice state."""
        lattice = Lattice4D(4)

        # Add some displacements
        lattice.displacements[(1, 1, 1, 1)] = np.array([0.1, 0.2, 0.3, 0.4])
        lattice.displacements[(2, 2, 2, 2)] = np.array([-0.1, 0.0, 0.1, 0.0])

        # Save (np.savez automatically appends .npz)
        filepath = str(tmp_path / "test_checkpoint")
        lattice.save_checkpoint(filepath)

        # Load (file will be filepath + .npz)
        loaded = Lattice4D.load_checkpoint(filepath + ".npz")

        # Verify
        assert loaded.N == lattice.N
        assert len(loaded.displacements) == len(lattice.displacements)

        for pos in lattice.displacements:
            assert np.allclose(loaded.displacements[pos], lattice.displacements[pos])


class TestCopy:
    """Test copy functionality."""

    def test_copy_is_independent(self):
        """Test that copy creates independent lattice."""
        original = Lattice4D(4)
        original.displacements[(1, 1, 1, 1)] = np.array([0.1, 0.2, 0.3, 0.4])

        copy = original.copy()

        # Modify original
        original.displacements[(1, 1, 1, 1)] = np.array([9.9, 9.9, 9.9, 9.9])

        # Copy should be unchanged
        assert np.allclose(copy.displacements[(1, 1, 1, 1)], [0.1, 0.2, 0.3, 0.4])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
