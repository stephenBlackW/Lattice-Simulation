"""Main script for 4D lattice dislocation simulation - Phases 1, 2 & 3."""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

from config import (
    LATTICE_SIZE,
    SPRING_CONSTANT,
    IDEAL_BOND_LENGTH,
    RELAXATION_TOLERANCE,
    MAX_RELAXATION_STEPS,
)
from src.lattice import Lattice4D
from src.dislocation import (
    introduce_simple_dislocation,
    get_burgers_vector,
    verify_discontinuity,
    compute_initial_strain_energy,
)
from src.visualization import (
    plot_2d_slice,
    plot_displacement_field,
    plot_strain_energy_field,
    plot_comparison,
    plot_convergence,
    plot_relaxed_comparison,
)
from src.relaxation import relax


def ensure_directories():
    """Create output directories if they don't exist."""
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('output', exist_ok=True)


def phase1_lattice_construction(N=LATTICE_SIZE):
    """Phase 1: Construct and validate the 4D cubic lattice.

    Args:
        N: Lattice size

    Returns:
        Lattice4D instance
    """
    print("=" * 60)
    print("PHASE 1: Lattice Construction")
    print("=" * 60)

    print(f"\nCreating {N}x{N}x{N}x{N} lattice...")
    lattice = Lattice4D(N)

    # Get and print statistics
    stats = lattice.get_statistics()

    print("\nLattice Statistics:")
    print(f"  N = {stats['N']}")
    print(f"  Number of sites: {stats['num_sites']} (expected: {stats['expected_sites']})")
    print(f"  Neighbors per site: {stats['min_neighbors']}-{stats['max_neighbors']} (expected: {stats['expected_neighbors']})")
    print(f"  Max displacement: {stats['max_displacement']:.6f}")
    print(f"  Mean displacement: {stats['mean_displacement']:.6f}")

    # Verify all sites have correct neighbor count
    print("\nVerifying neighbor counts...")
    all_correct = True
    for pos in lattice.displacements:
        n_neighbors = len(lattice.get_neighbors(pos))
        if n_neighbors != 8:
            print(f"  ERROR: Site {pos} has {n_neighbors} neighbors!")
            all_correct = False

    if all_correct:
        print("  All sites have exactly 8 neighbors.")

    # Test periodic boundary distances
    print("\nVerifying periodic boundary distances...")
    test_cases = [
        ((0, 0, 0, 0), (N-1, 0, 0, 0)),  # Wrap in x
        ((0, 0, 0, 0), (0, N-1, 0, 0)),  # Wrap in y
        ((0, 0, 0, 0), (0, 0, N-1, 0)),  # Wrap in z
        ((0, 0, 0, 0), (0, 0, 0, N-1)),  # Wrap in w
    ]

    for pos, neighbor in test_cases:
        vec = lattice.get_ideal_neighbor_vector(pos, neighbor)
        dist = np.linalg.norm(vec)
        print(f"  {pos} -> {neighbor}: distance = {dist:.4f}, vector = {vec}")
        assert dist == 1.0, f"Expected distance 1.0, got {dist}"

    print("\n  All periodic boundary distances correct!")

    # Compute initial energy (should be zero)
    initial_energy = lattice.compute_total_energy(SPRING_CONSTANT, IDEAL_BOND_LENGTH)
    print(f"\nInitial total energy: {initial_energy:.6f} (expected: 0.0)")

    # Save checkpoint
    checkpoint_path = 'checkpoints/lattice_initial'
    lattice.save_checkpoint(checkpoint_path)
    print(f"\nSaved initial lattice to {checkpoint_path}.npz")

    print("\nPhase 1 COMPLETE")
    return lattice


def phase2_dislocation_introduction(lattice=None, N=LATTICE_SIZE):
    """Phase 2: Introduce dislocation and visualize.

    Args:
        lattice: Optional existing lattice (creates new if None)
        N: Lattice size if creating new

    Returns:
        Dictionary of lattices with different Burgers vectors
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Dislocation Introduction")
    print("=" * 60)

    if lattice is None:
        lattice = Lattice4D(N)

    results = {}

    # Test all four Burgers vector orientations
    for direction in ['x', 'y', 'z', 'w']:
        print(f"\n--- Burgers vector b = {direction} ---")

        # Create fresh lattice for each test
        test_lattice = Lattice4D(N)
        b = get_burgers_vector(direction)

        print(f"  Introducing dislocation with b = {b}")
        introduce_simple_dislocation(test_lattice, b)

        # Verify discontinuity
        disc = verify_discontinuity(test_lattice, axis=0)
        print(f"  Discontinuity verification:")
        print(f"    Pairs checked: {disc['num_pairs_checked']}")
        print(f"    Mean discontinuity: {disc['mean_discontinuity']:.4f} (expected: 1.0)")

        # Compute strain energy
        energy = compute_initial_strain_energy(test_lattice, SPRING_CONSTANT, IDEAL_BOND_LENGTH)
        print(f"  Initial strain energy: {energy:.4f}")

        # Get statistics
        stats = test_lattice.get_statistics()
        print(f"  Max displacement: {stats['max_displacement']:.4f}")
        print(f"  Mean displacement: {stats['mean_displacement']:.4f}")

        # Save checkpoint
        checkpoint_path = f'checkpoints/lattice_dislocation_b{direction}'
        test_lattice.save_checkpoint(checkpoint_path)
        print(f"  Saved to {checkpoint_path}.npz")

        results[direction] = test_lattice

    # Generate visualizations
    print("\n--- Generating Visualizations ---")

    # Plot each Burgers vector orientation
    for direction, test_lattice in results.items():
        # 2D slice at z=0, w=0
        save_path = f'output/slice_unrelaxed_b{direction}.png'
        plot_2d_slice(
            test_lattice,
            z_value=0,
            w_value=0,
            title=f'Unrelaxed Dislocation (b = {direction})',
            save_path=save_path,
        )
        plt.close()

        # Displacement field
        save_path = f'output/displacement_b{direction}.png'
        component = {'x': 0, 'y': 1, 'z': 2, 'w': 3}[direction]
        plot_displacement_field(
            test_lattice,
            z_value=0,
            w_value=0,
            component=component,
            title=f'Displacement Field d{direction} (b = {direction})',
            save_path=save_path,
        )
        plt.close()

        # Strain energy field
        save_path = f'output/strain_energy_b{direction}.png'
        plot_strain_energy_field(
            test_lattice,
            z_value=0,
            w_value=0,
            title=f'Strain Energy (b = {direction})',
            save_path=save_path,
        )
        plt.close()

    # Side-by-side comparison
    save_path = 'output/comparison_all_burgers.png'
    plot_comparison(results, z_value=0, w_value=0, save_path=save_path)
    plt.close()

    # Summary comparison
    print("\n--- Energy Comparison ---")
    print("Burgers Vector | Initial Energy")
    print("-" * 30)
    for direction, test_lattice in results.items():
        energy = compute_initial_strain_energy(test_lattice)
        print(f"  b = {direction}        | {energy:.4f}")

    print("\nPhase 2 COMPLETE")
    print(f"\nOutput files saved to output/ directory")
    print(f"Checkpoints saved to checkpoints/ directory")

    return results


def phase3_relaxation(unrelaxed_lattices: dict, N=LATTICE_SIZE):
    """Phase 3: Relax dislocated lattices and compare.

    Args:
        unrelaxed_lattices: Dictionary of unrelaxed lattices from Phase 2
        N: Lattice size

    Returns:
        Dictionary of relaxed lattices
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Lattice Relaxation")
    print("=" * 60)

    print(f"\nRelaxation parameters:")
    print(f"  Method: FIRE")
    print(f"  Tolerance: {RELAXATION_TOLERANCE}")
    print(f"  Max steps: {MAX_RELAXATION_STEPS}")

    relaxed_results = {}

    for direction, unrelaxed in unrelaxed_lattices.items():
        print(f"\n--- Relaxing b = {direction} ---")

        # Make a copy for relaxation
        lattice = unrelaxed.copy()
        initial_energy = lattice.compute_total_energy(SPRING_CONSTANT, IDEAL_BOND_LENGTH)
        initial_max_force = lattice.compute_max_force(SPRING_CONSTANT, IDEAL_BOND_LENGTH)

        print(f"  Initial energy: {initial_energy:.4f}")
        print(f"  Initial max force: {initial_max_force:.4f}")

        # Relax
        print(f"  Relaxing...")
        result = relax(
            lattice,
            method='fire',
            k=SPRING_CONSTANT,
            a0=IDEAL_BOND_LENGTH,
            tolerance=RELAXATION_TOLERANCE,
            max_steps=MAX_RELAXATION_STEPS,
            verbose=True,
            log_interval=500,
        )

        print(f"  Converged: {result.converged}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Final energy: {result.final_energy:.4f}")
        print(f"  Final max force: {result.final_max_force:.6f}")

        # Energy reduction
        reduction = (initial_energy - result.final_energy) / initial_energy * 100
        print(f"  Energy reduction: {reduction:.1f}%")

        # Save relaxed checkpoint
        checkpoint_path = f'checkpoints/lattice_relaxed_b{direction}'
        lattice.save_checkpoint(checkpoint_path)
        print(f"  Saved to {checkpoint_path}.npz")

        # Store results
        relaxed_results[direction] = {
            'lattice': lattice,
            'unrelaxed': unrelaxed,
            'result': result,
            'initial_energy': initial_energy,
        }

        # Plot convergence
        save_path = f'output/convergence_b{direction}.png'
        plot_convergence(
            result.energy_history,
            result.force_history,
            title=f'Relaxation Convergence (b = {direction})',
            save_path=save_path,
        )
        plt.close()

        # Plot relaxed vs unrelaxed comparison
        save_path = f'output/relaxation_comparison_b{direction}.png'
        plot_relaxed_comparison(
            unrelaxed,
            lattice,
            z_value=0,
            w_value=0,
            title=f'Relaxation Comparison (b = {direction})',
            save_path=save_path,
        )
        plt.close()

        # Plot relaxed strain energy
        save_path = f'output/strain_energy_relaxed_b{direction}.png'
        plot_strain_energy_field(
            lattice,
            z_value=0,
            w_value=0,
            title=f'Relaxed Strain Energy (b = {direction})',
            save_path=save_path,
        )
        plt.close()

    # Summary comparison
    print("\n--- Relaxation Summary ---")
    print("Burgers | Initial E | Final E   | Reduction | Converged | Iterations")
    print("-" * 70)
    for direction, data in relaxed_results.items():
        result = data['result']
        initial = data['initial_energy']
        final = result.final_energy
        reduction = (initial - final) / initial * 100
        print(f"  b={direction}   | {initial:9.2f} | {final:9.2f} | {reduction:7.1f}%  | "
              f"{'Yes' if result.converged else 'No':9s} | {result.iterations}")

    print("\nPhase 3 COMPLETE")
    return relaxed_results


def main():
    """Run Phases 1, 2, and 3 of the simulation."""
    ensure_directories()

    print("\n" + "=" * 60)
    print("4D LATTICE DISLOCATION SIMULATION")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Lattice size: {LATTICE_SIZE}")
    print(f"  Spring constant: {SPRING_CONSTANT}")
    print(f"  Ideal bond length: {IDEAL_BOND_LENGTH}")
    print(f"  Relaxation tolerance: {RELAXATION_TOLERANCE}")
    print(f"  Max relaxation steps: {MAX_RELAXATION_STEPS}")

    # Phase 1
    lattice = phase1_lattice_construction()

    # Phase 2
    unrelaxed_results = phase2_dislocation_introduction(lattice)

    # Phase 3
    relaxed_results = phase3_relaxation(unrelaxed_results)

    print("\n" + "=" * 60)
    print("PHASES 1, 2 & 3 COMPLETE")
    print("=" * 60)
    print("\nReady for Phase 4 (strain computation) and Phase 5 (3D visualization)")

    return lattice, unrelaxed_results, relaxed_results


if __name__ == "__main__":
    main()
