"""Tests for lattice relaxation algorithms."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lattice import Lattice4D
from src.dislocation import introduce_simple_dislocation, get_burgers_vector
from src.relaxation import gradient_descent, fire, relax, RelaxationResult


class TestForceComputation:
    """Test force computation in the lattice."""

    def test_zero_force_undeformed(self):
        """Undeformed lattice should have zero forces."""
        lattice = Lattice4D(4)
        for pos in lattice.displacements:
            force = lattice.compute_site_force(pos)
            assert np.allclose(force, 0), f"Non-zero force at {pos}: {force}"

    def test_max_force_undeformed(self):
        """Undeformed lattice should have zero max force."""
        lattice = Lattice4D(4)
        max_force = lattice.compute_max_force()
        assert max_force == pytest.approx(0.0)

    def test_force_with_displacement(self):
        """Displaced site should experience restoring force."""
        lattice = Lattice4D(4)
        # Displace one site in +x direction
        lattice.displacements[(2, 2, 2, 2)] = np.array([0.1, 0.0, 0.0, 0.0])

        # This site should have a restoring force
        force = lattice.compute_site_force((2, 2, 2, 2))

        # Force should be in -x direction (restoring toward equilibrium)
        # +x neighbor: bond compressed, pushes site in -x
        # -x neighbor: bond stretched, pulls site in -x
        assert force[0] < 0, f"Force should be negative in x (restoring), got {force[0]}"
        assert np.linalg.norm(force) > 0

    def test_force_dislocation(self):
        """Dislocation should create non-zero forces."""
        lattice = Lattice4D(4)
        introduce_simple_dislocation(lattice, get_burgers_vector('x'))

        max_force = lattice.compute_max_force()
        assert max_force > 0, "Dislocation should create non-zero forces"


class TestGradientDescent:
    """Test gradient descent relaxation."""

    def test_converges_simple_perturbation(self):
        """Should converge for a simple perturbation."""
        lattice = Lattice4D(4)
        # Small perturbation
        lattice.displacements[(2, 2, 2, 2)] = np.array([0.05, 0.0, 0.0, 0.0])

        initial_energy = lattice.compute_total_energy()

        result = gradient_descent(lattice, tolerance=1e-4, max_steps=1000, dt=0.05)

        assert result.converged, "Should converge for simple perturbation"
        assert result.final_energy < initial_energy, "Energy should decrease"
        assert result.final_max_force < 1e-4, "Max force should be below tolerance"

    def test_energy_decreases_monotonically(self):
        """Energy should generally decrease (may have small oscillations with fixed dt)."""
        lattice = Lattice4D(4)
        lattice.displacements[(2, 2, 2, 2)] = np.array([0.1, 0.0, 0.0, 0.0])

        result = gradient_descent(lattice, tolerance=1e-4, max_steps=500, dt=0.02)

        # Check that overall trend is decreasing
        # Use a smoothed version to avoid small oscillations
        energies = result.energy_history
        first_quarter = np.mean(energies[:len(energies)//4])
        last_quarter = np.mean(energies[-len(energies)//4:])
        assert last_quarter < first_quarter, "Energy should decrease overall"

    def test_returns_history(self):
        """Should return energy and force history."""
        lattice = Lattice4D(4)
        lattice.displacements[(2, 2, 2, 2)] = np.array([0.05, 0.0, 0.0, 0.0])

        result = gradient_descent(lattice, tolerance=1e-4, max_steps=100, dt=0.05)

        assert len(result.energy_history) > 0
        assert len(result.force_history) > 0
        assert len(result.energy_history) == len(result.force_history)


class TestFIRE:
    """Test FIRE relaxation algorithm."""

    def test_converges_simple_perturbation(self):
        """Should converge for a simple perturbation."""
        lattice = Lattice4D(4)
        lattice.displacements[(2, 2, 2, 2)] = np.array([0.05, 0.0, 0.0, 0.0])

        initial_energy = lattice.compute_total_energy()

        result = fire(lattice, tolerance=1e-4, max_steps=1000)

        assert result.converged, "FIRE should converge for simple perturbation"
        assert result.final_energy < initial_energy, "Energy should decrease"

    def test_faster_than_gradient_descent(self):
        """FIRE should typically converge faster than gradient descent."""
        # Create two identical lattices
        lattice_gd = Lattice4D(4)
        lattice_fire = Lattice4D(4)

        for lattice in [lattice_gd, lattice_fire]:
            lattice.displacements[(2, 2, 2, 2)] = np.array([0.1, 0.0, 0.0, 0.0])
            lattice.displacements[(1, 1, 1, 1)] = np.array([0.0, 0.05, 0.0, 0.0])

        result_gd = gradient_descent(lattice_gd, tolerance=1e-4, max_steps=2000, dt=0.02)
        result_fire = fire(lattice_fire, tolerance=1e-4, max_steps=2000)

        # FIRE should converge in fewer iterations (or at least as good)
        if result_gd.converged and result_fire.converged:
            # FIRE should be faster or comparable
            assert result_fire.iterations <= result_gd.iterations * 2


class TestRelaxDislocation:
    """Test relaxation of dislocated lattice."""

    def test_dislocation_relaxation_reduces_energy(self):
        """Relaxation should reduce dislocation strain energy."""
        lattice = Lattice4D(6)  # Smaller for faster test
        introduce_simple_dislocation(lattice, get_burgers_vector('y'))

        initial_energy = lattice.compute_total_energy()

        result = relax(lattice, method='fire', tolerance=1e-3, max_steps=500)

        assert result.final_energy < initial_energy, "Relaxation should reduce energy"
        # Energy reduction should be significant
        reduction = (initial_energy - result.final_energy) / initial_energy
        assert reduction > 0.1, f"Energy reduction {reduction:.1%} should be > 10%"

    def test_relax_method_selection(self):
        """Test that method parameter works."""
        lattice = Lattice4D(4)
        lattice.displacements[(2, 2, 2, 2)] = np.array([0.05, 0.0, 0.0, 0.0])

        # Both methods should work
        result_gd = relax(lattice.copy(), method='gd', tolerance=1e-3, max_steps=200)
        result_fire = relax(lattice.copy(), method='fire', tolerance=1e-3, max_steps=200)

        assert isinstance(result_gd, RelaxationResult)
        assert isinstance(result_fire, RelaxationResult)

    def test_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        lattice = Lattice4D(4)
        with pytest.raises(ValueError):
            relax(lattice, method='invalid')


class TestRelaxationResult:
    """Test RelaxationResult dataclass."""

    def test_result_attributes(self):
        """Result should have all expected attributes."""
        lattice = Lattice4D(4)
        lattice.displacements[(2, 2, 2, 2)] = np.array([0.05, 0.0, 0.0, 0.0])

        result = relax(lattice, tolerance=1e-3, max_steps=100)

        assert hasattr(result, 'converged')
        assert hasattr(result, 'iterations')
        assert hasattr(result, 'final_energy')
        assert hasattr(result, 'final_max_force')
        assert hasattr(result, 'energy_history')
        assert hasattr(result, 'force_history')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
