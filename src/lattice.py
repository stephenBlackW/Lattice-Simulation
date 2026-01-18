"""4D cubic lattice with periodic boundary conditions."""

import numpy as np
from typing import Tuple, List, Dict
import json


class Lattice4D:
    """A 4D cubic lattice with periodic boundary conditions.

    Attributes:
        N: Size of lattice (N x N x N x N)
        displacements: Dict mapping (x, y, z, w) tuples to [dx, dy, dz, dw] arrays
    """

    def __init__(self, N: int = 8):
        """Initialize an N x N x N x N lattice with zero displacements.

        Args:
            N: Size of lattice in each dimension
        """
        self.N = N
        self.displacements: Dict[Tuple[int, int, int, int], np.ndarray] = {}

        # Initialize all sites with zero displacement
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for w in range(N):
                        self.displacements[(x, y, z, w)] = np.zeros(4)

    def wrap(self, coord: int) -> int:
        """Apply periodic boundary condition to a single coordinate.

        Args:
            coord: Coordinate value (may be outside [0, N))

        Returns:
            Wrapped coordinate in [0, N)
        """
        return coord % self.N

    def wrap_position(self, pos: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Apply periodic boundary conditions to a 4D position.

        Args:
            pos: (x, y, z, w) position tuple

        Returns:
            Wrapped position tuple
        """
        return (self.wrap(pos[0]), self.wrap(pos[1]),
                self.wrap(pos[2]), self.wrap(pos[3]))

    def get_neighbors(self, pos: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        """Get all 8 nearest neighbors of a site (respecting periodic boundaries).

        In 4D, each site has 2 neighbors per dimension (+ and -), giving 8 total.

        Args:
            pos: (x, y, z, w) position tuple

        Returns:
            List of 8 neighbor positions
        """
        x, y, z, w = pos
        neighbors = []

        # Neighbors along each axis (+ and -)
        # x-axis
        neighbors.append(self.wrap_position((x + 1, y, z, w)))
        neighbors.append(self.wrap_position((x - 1, y, z, w)))
        # y-axis
        neighbors.append(self.wrap_position((x, y + 1, z, w)))
        neighbors.append(self.wrap_position((x, y - 1, z, w)))
        # z-axis
        neighbors.append(self.wrap_position((x, y, z + 1, w)))
        neighbors.append(self.wrap_position((x, y, z - 1, w)))
        # w-axis
        neighbors.append(self.wrap_position((x, y, z, w + 1)))
        neighbors.append(self.wrap_position((x, y, z, w - 1)))

        return neighbors

    def get_ideal_neighbor_vector(self, pos: Tuple[int, int, int, int],
                                   neighbor: Tuple[int, int, int, int]) -> np.ndarray:
        """Get the ideal (undeformed) vector from pos to neighbor.

        This respects periodic boundaries - if a neighbor is across the boundary,
        the vector points in the short direction.

        Args:
            pos: Source position
            neighbor: Target neighbor position

        Returns:
            4D vector from pos to neighbor (accounting for periodicity)
        """
        diff = np.array(neighbor) - np.array(pos)

        # Apply minimum image convention
        for i in range(4):
            if diff[i] > self.N / 2:
                diff[i] -= self.N
            elif diff[i] < -self.N / 2:
                diff[i] += self.N

        return diff.astype(float)

    def get_actual_neighbor_vector(self, pos: Tuple[int, int, int, int],
                                    neighbor: Tuple[int, int, int, int]) -> np.ndarray:
        """Get the actual (deformed) vector from pos to neighbor.

        This includes displacements and respects periodic boundaries.

        Args:
            pos: Source position
            neighbor: Target neighbor position

        Returns:
            4D vector from actual pos to actual neighbor position
        """
        # Get ideal vector
        ideal = self.get_ideal_neighbor_vector(pos, neighbor)

        # Add displacement difference
        disp_diff = self.displacements[neighbor] - self.displacements[pos]

        return ideal + disp_diff

    def get_actual_position(self, pos: Tuple[int, int, int, int]) -> np.ndarray:
        """Get the actual position (lattice position + displacement).

        Args:
            pos: Lattice site position

        Returns:
            Actual position as numpy array
        """
        return np.array(pos) + self.displacements[pos]

    def compute_bond_energy(self, pos: Tuple[int, int, int, int],
                            neighbor: Tuple[int, int, int, int],
                            k: float = 1.0, a0: float = 1.0) -> float:
        """Compute energy of a single bond.

        E = (k/2) * (|r| - a0)^2

        Args:
            pos: First site
            neighbor: Second site (must be a neighbor)
            k: Spring constant
            a0: Ideal bond length

        Returns:
            Bond energy
        """
        r = self.get_actual_neighbor_vector(pos, neighbor)
        distance = np.linalg.norm(r)
        return 0.5 * k * (distance - a0) ** 2

    def compute_site_energy(self, pos: Tuple[int, int, int, int],
                            k: float = 1.0, a0: float = 1.0) -> float:
        """Compute local energy at a site (half the sum of bond energies).

        Factor of 1/2 avoids double counting.

        Args:
            pos: Site position
            k: Spring constant
            a0: Ideal bond length

        Returns:
            Local energy at site
        """
        energy = 0.0
        for neighbor in self.get_neighbors(pos):
            energy += self.compute_bond_energy(pos, neighbor, k, a0)
        return 0.5 * energy  # Factor of 1/2 for double counting

    def compute_total_energy(self, k: float = 1.0, a0: float = 1.0) -> float:
        """Compute total elastic energy of the lattice.

        Args:
            k: Spring constant
            a0: Ideal bond length

        Returns:
            Total energy
        """
        total = 0.0
        for pos in self.displacements:
            total += self.compute_site_energy(pos, k, a0)
        return total

    def compute_site_force(self, pos: Tuple[int, int, int, int],
                           k: float = 1.0, a0: float = 1.0) -> np.ndarray:
        """Compute force on a site from all its neighbors.

        For a spring connecting site i to neighbor j:
        - r_ij = vector from i to j
        - Force on i: F_i = k(|r_ij| - a0) * (r_ij/|r_ij|)
        - Stretched spring (|r| > a0): pulls i toward j
        - Compressed spring (|r| < a0): pushes i away from j

        Args:
            pos: Site position
            k: Spring constant
            a0: Ideal bond length

        Returns:
            4D force vector on this site
        """
        force = np.zeros(4)
        for neighbor in self.get_neighbors(pos):
            r = self.get_actual_neighbor_vector(pos, neighbor)
            distance = np.linalg.norm(r)
            if distance > 1e-10:  # Avoid division by zero
                # Force pulls/pushes site toward equilibrium distance
                force_magnitude = k * (distance - a0)
                force += force_magnitude * (r / distance)
        return force

    def compute_all_forces(self, k: float = 1.0, a0: float = 1.0) -> Dict[Tuple[int, int, int, int], np.ndarray]:
        """Compute forces on all sites.

        Args:
            k: Spring constant
            a0: Ideal bond length

        Returns:
            Dictionary mapping positions to force vectors
        """
        forces = {}
        for pos in self.displacements:
            forces[pos] = self.compute_site_force(pos, k, a0)
        return forces

    def compute_max_force(self, k: float = 1.0, a0: float = 1.0) -> float:
        """Compute maximum force magnitude in the lattice.

        Args:
            k: Spring constant
            a0: Ideal bond length

        Returns:
            Maximum force magnitude
        """
        max_force = 0.0
        for pos in self.displacements:
            force = self.compute_site_force(pos, k, a0)
            force_mag = np.linalg.norm(force)
            if force_mag > max_force:
                max_force = force_mag
        return max_force

    def get_statistics(self) -> dict:
        """Get basic statistics about the lattice.

        Returns:
            Dictionary with lattice statistics
        """
        num_sites = len(self.displacements)

        # Count neighbors for verification
        neighbor_counts = [len(self.get_neighbors(pos)) for pos in self.displacements]

        # Compute displacement statistics
        displacements_array = np.array(list(self.displacements.values()))
        max_disp = np.max(np.linalg.norm(displacements_array, axis=1))
        mean_disp = np.mean(np.linalg.norm(displacements_array, axis=1))

        return {
            'N': self.N,
            'num_sites': num_sites,
            'expected_sites': self.N ** 4,
            'min_neighbors': min(neighbor_counts),
            'max_neighbors': max(neighbor_counts),
            'expected_neighbors': 8,
            'max_displacement': max_disp,
            'mean_displacement': mean_disp,
        }

    def save_checkpoint(self, filepath: str):
        """Save lattice state to file.

        Args:
            filepath: Path to save checkpoint (will append .npz)
        """
        # Convert displacements to arrays for saving
        positions = np.array(list(self.displacements.keys()))
        displacements = np.array(list(self.displacements.values()))

        np.savez(filepath,
                 N=self.N,
                 positions=positions,
                 displacements=displacements)

    @classmethod
    def load_checkpoint(cls, filepath: str) -> 'Lattice4D':
        """Load lattice state from file.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Loaded Lattice4D instance
        """
        data = np.load(filepath)
        N = int(data['N'])
        positions = data['positions']
        displacements = data['displacements']

        lattice = cls(N)
        for i, pos in enumerate(positions):
            lattice.displacements[tuple(pos)] = displacements[i]

        return lattice

    def copy(self) -> 'Lattice4D':
        """Create a deep copy of the lattice.

        Returns:
            New Lattice4D with copied data
        """
        new_lattice = Lattice4D(self.N)
        for pos, disp in self.displacements.items():
            new_lattice.displacements[pos] = disp.copy()
        return new_lattice
