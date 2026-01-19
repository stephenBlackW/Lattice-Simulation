"""Simple 3D cubic lattice with shear dislocation visualization.

This module focuses on clearly showing atom positions as they move
during shearing and relaxation processes.

Key concept: Bonds are defined by the IDEAL lattice structure and
remain fixed. Forces arise when bonds are stretched or compressed
from their equilibrium length.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
import copy


@dataclass
class Atom:
    """Represents an atom in the lattice."""
    ideal_pos: np.ndarray      # Where atom "should" be in perfect lattice
    actual_pos: np.ndarray     # Where atom actually is (displaced)
    index: int = 0             # Index in the lattice

    @property
    def displacement(self) -> np.ndarray:
        return self.actual_pos - self.ideal_pos


class Lattice3D:
    """Simple 3D cubic lattice for dislocation visualization.

    Atoms are connected by bonds based on their ideal positions.
    Forces arise when bonds are stretched/compressed from equilibrium.
    """

    def __init__(self, nx: int = 8, ny: int = 8, nz: int = 8,
                 lattice_constant: float = 1.0):
        """Create a 3D cubic lattice.

        Args:
            nx, ny, nz: Number of atoms in each direction
            lattice_constant: Spacing between atoms
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.a = lattice_constant
        self.atoms: List[Atom] = []
        self.bonds: List[Tuple[int, int]] = []  # Pairs of atom indices

        # Create perfect lattice
        idx = 0
        self._index_map = {}  # (ix, iy, iz) -> atom index
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    pos = np.array([ix * self.a, iy * self.a, iz * self.a],
                                   dtype=float)
                    self.atoms.append(Atom(
                        ideal_pos=pos.copy(),
                        actual_pos=pos.copy(),
                        index=idx
                    ))
                    self._index_map[(ix, iy, iz)] = idx
                    idx += 1

        # Create bonds (nearest neighbors in cubic lattice)
        self._create_bonds()

    def _create_bonds(self) -> None:
        """Create bonds between nearest neighbors based on ideal positions."""
        self.bonds = []
        added = set()

        # Each atom connects to neighbors in +x, +y, +z directions
        for ix in range(self.nx):
            for iy in range(self.ny):
                for iz in range(self.nz):
                    idx = self._index_map[(ix, iy, iz)]

                    # Check +x neighbor
                    if ix + 1 < self.nx:
                        j = self._index_map[(ix + 1, iy, iz)]
                        bond = (min(idx, j), max(idx, j))
                        if bond not in added:
                            added.add(bond)
                            self.bonds.append(bond)

                    # Check +y neighbor
                    if iy + 1 < self.ny:
                        j = self._index_map[(ix, iy + 1, iz)]
                        bond = (min(idx, j), max(idx, j))
                        if bond not in added:
                            added.add(bond)
                            self.bonds.append(bond)

                    # Check +z neighbor
                    if iz + 1 < self.nz:
                        j = self._index_map[(ix, iy, iz + 1)]
                        bond = (min(idx, j), max(idx, j))
                        if bond not in added:
                            added.add(bond)
                            self.bonds.append(bond)

    def copy(self) -> 'Lattice3D':
        """Create a deep copy of the lattice."""
        new_lattice = Lattice3D.__new__(Lattice3D)
        new_lattice.nx = self.nx
        new_lattice.ny = self.ny
        new_lattice.nz = self.nz
        new_lattice.a = self.a
        new_lattice.atoms = [
            Atom(ideal_pos=a.ideal_pos.copy(),
                 actual_pos=a.actual_pos.copy(),
                 index=a.index)
            for a in self.atoms
        ]
        new_lattice.bonds = list(self.bonds)
        new_lattice._index_map = dict(self._index_map)
        return new_lattice

    def get_positions(self) -> np.ndarray:
        """Get all actual atom positions as Nx3 array."""
        return np.array([a.actual_pos for a in self.atoms])

    def get_ideal_positions(self) -> np.ndarray:
        """Get all ideal atom positions as Nx3 array."""
        return np.array([a.ideal_pos for a in self.atoms])

    def get_displacements(self) -> np.ndarray:
        """Get all displacement vectors as Nx3 array."""
        return np.array([a.displacement for a in self.atoms])

    def apply_shear(self, cut_plane_z: float, shear_vector: np.ndarray,
                    fraction: float = 1.0) -> None:
        """Apply a shear displacement to atoms above cut plane.

        Args:
            cut_plane_z: Z coordinate of the cut plane
            shear_vector: Direction and magnitude of shear (e.g., [1, 0, 0])
            fraction: Fraction of full shear to apply (0 to 1, for animation)
        """
        shear = np.array(shear_vector, dtype=float) * fraction * self.a

        for atom in self.atoms:
            if atom.ideal_pos[2] >= cut_plane_z:
                # Atom is above cut plane - apply shear
                atom.actual_pos = atom.ideal_pos + shear
            else:
                # Atom is below cut plane - stays at ideal position
                atom.actual_pos = atom.ideal_pos.copy()

    def compute_energy(self, k: float = 1.0) -> float:
        """Compute total spring potential energy.

        E = sum over bonds of (1/2) * k * (r - r0)^2

        Args:
            k: Spring constant

        Returns:
            Total energy
        """
        energy = 0.0
        r0 = self.a  # Equilibrium bond length

        for i, j in self.bonds:
            r_vec = self.atoms[j].actual_pos - self.atoms[i].actual_pos
            r = np.linalg.norm(r_vec)
            energy += 0.5 * k * (r - r0) ** 2

        return energy

    def compute_forces(self, k: float = 1.0) -> np.ndarray:
        """Compute spring forces on each atom.

        F_i = sum over bonds to i of -k * (r - r0) * r_hat

        Args:
            k: Spring constant

        Returns:
            Nx3 array of force vectors
        """
        forces = np.zeros((len(self.atoms), 3))
        r0 = self.a  # Equilibrium bond length

        for i, j in self.bonds:
            r_vec = self.atoms[j].actual_pos - self.atoms[i].actual_pos
            r = np.linalg.norm(r_vec)

            if r > 1e-10:
                r_hat = r_vec / r
                # Spring force magnitude
                f_mag = k * (r - r0)
                # Force on i points toward j if r > r0 (stretched)
                forces[i] += f_mag * r_hat
                forces[j] -= f_mag * r_hat

        return forces

    def relax_step(self, dt: float = 0.01, k: float = 1.0) -> Tuple[float, float]:
        """Perform one relaxation step using steepest descent.

        Args:
            dt: Step size
            k: Spring constant

        Returns:
            (max_force, energy) tuple
        """
        forces = self.compute_forces(k)
        max_force = np.max(np.linalg.norm(forces, axis=1))
        energy = self.compute_energy(k)

        # Move atoms in direction of force
        for i, atom in enumerate(self.atoms):
            atom.actual_pos = atom.actual_pos + dt * forces[i]

        return max_force, energy

    def relax(self, max_steps: int = 1000, dt: float = 0.01, k: float = 1.0,
              tol: float = 0.001, verbose: bool = False) -> List[Tuple[float, float]]:
        """Relax the lattice to minimize energy.

        Uses adaptive step size: reduces dt if energy increases.

        Args:
            max_steps: Maximum number of steps
            dt: Initial step size
            k: Spring constant
            tol: Force tolerance for convergence
            verbose: Print progress

        Returns:
            List of (step, energy) tuples
        """
        history = []
        energy = self.compute_energy(k)
        history.append((0, energy))

        current_dt = dt

        for step in range(max_steps):
            # Save current state
            old_positions = [a.actual_pos.copy() for a in self.atoms]
            old_energy = energy

            # Try a step
            forces = self.compute_forces(k)
            max_force = np.max(np.linalg.norm(forces, axis=1))

            # Move atoms
            for i, atom in enumerate(self.atoms):
                atom.actual_pos = atom.actual_pos + current_dt * forces[i]

            energy = self.compute_energy(k)

            # Adaptive step size
            if energy > old_energy:
                # Step increased energy - revert and reduce step size
                for i, atom in enumerate(self.atoms):
                    atom.actual_pos = old_positions[i]
                current_dt *= 0.5
                energy = old_energy
                if verbose and step % 100 == 0:
                    print(f"  Step {step}: reducing dt to {current_dt:.6f}")
            else:
                # Good step - maybe increase step size
                current_dt = min(current_dt * 1.1, dt * 2)

            history.append((step + 1, energy))

            if max_force < tol:
                if verbose:
                    print(f"  Converged at step {step + 1}, E={energy:.6f}")
                break

            if verbose and step % 100 == 0:
                print(f"  Step {step}: E={energy:.4f}, max_F={max_force:.4f}")

        return history


def create_shear_sequence(lattice: Lattice3D,
                          cut_z: float,
                          shear_vec: np.ndarray,
                          n_steps: int = 10) -> List[Lattice3D]:
    """Create a sequence of lattice states during shearing.

    Args:
        lattice: Initial perfect lattice
        cut_z: Z coordinate of cut plane
        shear_vec: Shear direction and magnitude
        n_steps: Number of intermediate steps

    Returns:
        List of lattice states from no shear to full shear
    """
    states = []

    for i in range(n_steps + 1):
        fraction = i / n_steps
        state = lattice.copy()
        state.apply_shear(cut_z, shear_vec, fraction)
        states.append(state)

    return states


def create_relaxation_sequence(lattice: Lattice3D,
                               max_steps: int = 500,
                               dt: float = 0.02,
                               k: float = 1.0,
                               tol: float = 0.001,
                               save_every: int = 10) -> List[Tuple[Lattice3D, float]]:
    """Create a sequence of lattice states during relaxation.

    Args:
        lattice: Initial sheared lattice
        max_steps: Maximum relaxation steps
        dt: Initial step size
        k: Spring constant
        tol: Force tolerance for convergence
        save_every: Save state every N steps

    Returns:
        List of (lattice_state, energy) tuples
    """
    states = []
    current = lattice.copy()

    # Save initial state
    energy = current.compute_energy(k)
    states.append((current.copy(), energy))
    print(f"  Initial energy: {energy:.4f}")

    current_dt = dt

    for step in range(max_steps):
        # Save current state
        old_positions = [a.actual_pos.copy() for a in current.atoms]
        old_energy = energy

        # Compute and apply forces
        forces = current.compute_forces(k)
        max_force = np.max(np.linalg.norm(forces, axis=1))

        for i, atom in enumerate(current.atoms):
            atom.actual_pos = atom.actual_pos + current_dt * forces[i]

        energy = current.compute_energy(k)

        # Adaptive step size
        if energy > old_energy * 1.001:  # Allow tiny increases
            # Revert
            for i, atom in enumerate(current.atoms):
                atom.actual_pos = old_positions[i]
            current_dt *= 0.5
            energy = old_energy
        else:
            current_dt = min(current_dt * 1.05, dt * 2)

        # Save state periodically
        if (step + 1) % save_every == 0:
            states.append((current.copy(), energy))
            print(f"  Step {step + 1}: E={energy:.4f}, max_F={max_force:.4f}")

        # Check convergence
        if max_force < tol:
            states.append((current.copy(), energy))
            print(f"  Converged at step {step + 1}: E={energy:.4f}")
            break

    return states
