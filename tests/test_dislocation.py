"""Tests for dislocation introduction."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lattice import Lattice4D
from src.dislocation import (
    introduce_simple_dislocation,
    get_burgers_vector,
    verify_discontinuity,
    compute_initial_strain_energy,
)


class TestBurgersVector:
    """Test Burgers vector generation."""

    def test_burgers_x(self):
        """Test x-direction Burgers vector."""
        b = get_burgers_vector('x')
        assert b == (1.0, 0.0, 0.0, 0.0)

    def test_burgers_y(self):
        """Test y-direction Burgers vector."""
        b = get_burgers_vector('y')
        assert b == (0.0, 1.0, 0.0, 0.0)

    def test_burgers_z(self):
        """Test z-direction Burgers vector."""
        b = get_burgers_vector('z')
        assert b == (0.0, 0.0, 1.0, 0.0)

    def test_burgers_w(self):
        """Test w-direction Burgers vector."""
        b = get_burgers_vector('w')
        assert b == (0.0, 0.0, 0.0, 1.0)


class TestDislocationIntroduction:
    """Test dislocation creation."""

    def test_dislocation_creates_displacement(self):
        """Verify dislocation creates non-zero displacements."""
        lattice = Lattice4D(8)
        b = get_burgers_vector('x')

        # Before: all displacements zero
        assert all(np.allclose(d, 0) for d in lattice.displacements.values())

        introduce_simple_dislocation(lattice, b)

        # After: some displacements non-zero
        max_disp = max(np.linalg.norm(d) for d in lattice.displacements.values())
        assert max_disp > 0

    def test_dislocation_preserves_lattice_size(self):
        """Verify dislocation doesn't change number of sites."""
        lattice = Lattice4D(8)
        original_size = len(lattice.displacements)

        introduce_simple_dislocation(lattice, get_burgers_vector('x'))

        assert len(lattice.displacements) == original_size

    def test_displacement_magnitude(self):
        """Verify displacement magnitudes are b/2."""
        lattice = Lattice4D(8)
        b = get_burgers_vector('x')
        b_mag = np.linalg.norm(b) / 2  # Expected magnitude

        introduce_simple_dislocation(lattice, b)

        for pos, disp in lattice.displacements.items():
            mag = np.linalg.norm(disp)
            assert mag == pytest.approx(b_mag), f"Unexpected magnitude {mag} at {pos}"

    def test_inplace_false_preserves_original(self):
        """Verify inplace=False doesn't modify original lattice."""
        lattice = Lattice4D(4)
        original_disp = lattice.displacements[(0, 0, 0, 0)].copy()

        new_lattice = introduce_simple_dislocation(
            lattice, get_burgers_vector('x'), inplace=False
        )

        # Original unchanged
        assert np.allclose(lattice.displacements[(0, 0, 0, 0)], original_disp)

        # New lattice modified
        assert not np.allclose(new_lattice.displacements[(0, 0, 0, 0)], original_disp)

    def test_discontinuity_exists(self):
        """Verify discontinuity across the cut plane."""
        lattice = Lattice4D(8)
        b = get_burgers_vector('x')

        introduce_simple_dislocation(lattice, b)

        result = verify_discontinuity(lattice, axis=0)

        # The discontinuity should be approximately |b| = 1
        assert result['mean_discontinuity'] == pytest.approx(1.0)
        assert result['num_pairs_checked'] > 0

    def test_different_burgers_vectors(self):
        """Test all four Burgers vector orientations."""
        for direction in ['x', 'y', 'z', 'w']:
            lattice = Lattice4D(8)
            b = get_burgers_vector(direction)

            introduce_simple_dislocation(lattice, b)

            # All should create non-zero energy
            energy = compute_initial_strain_energy(lattice)
            assert energy > 0, f"Burgers vector {direction} created zero energy"


class TestStrainEnergy:
    """Test strain energy calculations for dislocations."""

    def test_undeformed_zero_energy(self):
        """Undeformed lattice has zero energy."""
        lattice = Lattice4D(4)
        energy = compute_initial_strain_energy(lattice)
        assert energy == pytest.approx(0.0)

    def test_dislocation_positive_energy(self):
        """Dislocation creates positive strain energy."""
        lattice = Lattice4D(8)
        introduce_simple_dislocation(lattice, get_burgers_vector('x'))

        energy = compute_initial_strain_energy(lattice)
        assert energy > 0

    def test_energy_comparison_burgers(self):
        """Compare energies for different Burgers vectors."""
        energies = {}

        for direction in ['x', 'y', 'z', 'w']:
            lattice = Lattice4D(8)
            b = get_burgers_vector(direction)
            introduce_simple_dislocation(lattice, b)
            energies[direction] = compute_initial_strain_energy(lattice)

        # Print for inspection
        print("\nBurgers vector energies:")
        for d, e in energies.items():
            print(f"  b={d}: E={e:.4f}")

        # All energies should be positive and finite
        for d, e in energies.items():
            assert e > 0, f"Energy for b={d} should be positive"
            assert np.isfinite(e), f"Energy for b={d} should be finite"

        # The y, z, w energies should be similar (same geometry perpendicular to x-cut)
        # x energy is different because cut is along x
        perpendicular_energies = [energies['y'], energies['z'], energies['w']]
        assert max(perpendicular_energies) / min(perpendicular_energies) < 1.1


class TestDiscontinuityVerification:
    """Test discontinuity verification function."""

    def test_no_discontinuity_undeformed(self):
        """Undeformed lattice has no discontinuity."""
        lattice = Lattice4D(8)
        result = verify_discontinuity(lattice, axis=0)

        assert result['mean_discontinuity'] == pytest.approx(0.0)

    def test_discontinuity_magnitude(self):
        """Discontinuity magnitude matches Burgers vector."""
        lattice = Lattice4D(8)
        b = get_burgers_vector('x')
        b_mag = np.linalg.norm(b)

        introduce_simple_dislocation(lattice, b)
        result = verify_discontinuity(lattice, axis=0)

        # Discontinuity should be |b|
        assert result['mean_discontinuity'] == pytest.approx(b_mag)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
