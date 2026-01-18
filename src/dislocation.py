"""Dislocation creation using cut-and-shift operations in 4D."""

import numpy as np
from typing import Tuple, Optional
from .lattice import Lattice4D


def introduce_dislocation(
    lattice: Lattice4D,
    burgers_vector: Tuple[float, float, float, float] = (1, 0, 0, 0),
    cut_plane_axis: str = 'w',
    cut_plane_value: int = 0,
    cut_direction_axis: str = 'x',
    inplace: bool = True
) -> Lattice4D:
    """Introduce an edge dislocation into the lattice using cut-and-shift.

    The dislocation geometry:
    - Cut hyperplane: 3D surface defined by cut_plane_axis = cut_plane_value
                      and cut_direction_axis > center
    - Dislocation core: 2D surface at cut_plane_axis = cut_plane_value,
                        cut_direction_axis = center
    - Burgers vector: defines the displacement discontinuity

    Implementation:
    - For sites on positive side of cut (cut_direction > center, cut_plane >= value),
      add displacement +b/2
    - For sites on negative side (cut_direction <= center, cut_plane >= value),
      add displacement -b/2

    Args:
        lattice: The Lattice4D to modify
        burgers_vector: 4D Burgers vector (dx, dy, dz, dw)
        cut_plane_axis: Axis normal to the cut extension ('x', 'y', 'z', or 'w')
        cut_plane_value: Position along cut_plane_axis where cut starts
        cut_direction_axis: Axis along which the cut extends ('x', 'y', 'z', or 'w')
        inplace: If True, modify lattice in place; otherwise create a copy

    Returns:
        Modified lattice (same object if inplace=True)
    """
    if not inplace:
        lattice = lattice.copy()

    b = np.array(burgers_vector)
    N = lattice.N
    center = N // 2  # Center of lattice for cut direction

    axis_map = {'x': 0, 'y': 1, 'z': 2, 'w': 3}
    cut_plane_idx = axis_map[cut_plane_axis]
    cut_dir_idx = axis_map[cut_direction_axis]

    for pos in lattice.displacements:
        cut_plane_coord = pos[cut_plane_idx]
        cut_dir_coord = pos[cut_dir_idx]

        # Check if this site is affected by the cut
        # The cut applies to the half-space where cut_plane_coord >= cut_plane_value
        # We use periodic coordinates, so we need to check the "upper half"
        # relative to the cut plane value

        # Determine if site is in the cut half-space
        # For periodic boundaries, we consider cut_plane_value to N-1 and 0 to cut_plane_value-1
        # as the two half-spaces, but we only modify one half
        if cut_plane_value == 0:
            in_cut_region = cut_plane_coord >= 0  # This is always true, so we use a different approach
            # For w=0 case, we apply to w >= 0 (which is all sites since w is [0, N))
            # But we want to create a discontinuity, so we only apply to w >= 0 and w < N/2
            # Actually, re-reading the spec: apply to all sites with w >= 0
            # Since all sites have w in [0, N), this means all sites
            in_cut_region = True
        else:
            in_cut_region = cut_plane_coord >= cut_plane_value

        if in_cut_region:
            # Apply displacement based on position relative to cut direction center
            if cut_dir_coord > center:
                # Positive side of cut
                lattice.displacements[pos] = lattice.displacements[pos] + b / 2
            else:
                # Negative side (or at center)
                lattice.displacements[pos] = lattice.displacements[pos] - b / 2

    return lattice


def introduce_simple_dislocation(
    lattice: Lattice4D,
    burgers_vector: Tuple[float, float, float, float] = (1, 0, 0, 0),
    inplace: bool = True
) -> Lattice4D:
    """Introduce a simple edge dislocation with standard geometry.

    This uses the geometry from the spec:
    - Cut hyperplane: all points where w >= 0 and x > N/2
    - Dislocation core: the 2D surface at x = N/2 (a plane in yz directions)

    Args:
        lattice: The Lattice4D to modify
        burgers_vector: 4D Burgers vector (dx, dy, dz, dw)
        inplace: If True, modify lattice in place; otherwise create a copy

    Returns:
        Modified lattice
    """
    if not inplace:
        lattice = lattice.copy()

    b = np.array(burgers_vector)
    N = lattice.N
    center = N // 2

    for pos in lattice.displacements:
        x, y, z, w = pos

        # All sites are affected (w >= 0 is always true for our coordinates)
        if x > center:
            # Positive side of cut (x > center)
            lattice.displacements[pos] = lattice.displacements[pos] + b / 2
        else:
            # Negative side of cut (x <= center)
            lattice.displacements[pos] = lattice.displacements[pos] - b / 2

    return lattice


def get_burgers_vector(name: str) -> Tuple[float, float, float, float]:
    """Get a named Burgers vector.

    Args:
        name: One of 'x', 'y', 'z', 'w' for unit vectors along each axis

    Returns:
        Burgers vector tuple
    """
    vectors = {
        'x': (1.0, 0.0, 0.0, 0.0),
        'y': (0.0, 1.0, 0.0, 0.0),
        'z': (0.0, 0.0, 1.0, 0.0),
        'w': (0.0, 0.0, 0.0, 1.0),
    }
    return vectors[name]


def verify_discontinuity(lattice: Lattice4D, axis: int = 0) -> dict:
    """Verify that a displacement discontinuity exists across the cut plane.

    Checks for displacement discontinuity by comparing sites on either
    side of x = N/2.

    Args:
        lattice: The lattice to check
        axis: Axis index to check discontinuity along (0=x, 1=y, 2=z, 3=w)

    Returns:
        Dictionary with discontinuity statistics
    """
    N = lattice.N
    center = N // 2

    # Sample some pairs of sites across the discontinuity
    discontinuities = []

    # Check at various y, z, w positions
    for y in range(N):
        for z in range(N):
            for w in range(N):
                if axis == 0:  # Check across x = center
                    pos_minus = (center, y, z, w)
                    pos_plus = (center + 1, y, z, w)
                elif axis == 1:
                    pos_minus = (y, center, z, w)
                    pos_plus = (y, center + 1, z, w)
                elif axis == 2:
                    pos_minus = (y, z, center, w)
                    pos_plus = (y, z, center + 1, w)
                else:  # axis == 3
                    pos_minus = (y, z, w, center)
                    pos_plus = (y, z, w, center + 1)

                if pos_minus in lattice.displacements and pos_plus in lattice.displacements:
                    diff = lattice.displacements[pos_plus] - lattice.displacements[pos_minus]
                    discontinuities.append(np.linalg.norm(diff))

    return {
        'num_pairs_checked': len(discontinuities),
        'mean_discontinuity': np.mean(discontinuities) if discontinuities else 0,
        'min_discontinuity': np.min(discontinuities) if discontinuities else 0,
        'max_discontinuity': np.max(discontinuities) if discontinuities else 0,
    }


def compute_initial_strain_energy(lattice: Lattice4D, k: float = 1.0, a0: float = 1.0) -> float:
    """Compute initial strain energy of the lattice.

    Args:
        lattice: The lattice
        k: Spring constant
        a0: Ideal bond length

    Returns:
        Total strain energy
    """
    return lattice.compute_total_energy(k, a0)
