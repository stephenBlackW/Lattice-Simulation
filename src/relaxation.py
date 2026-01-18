"""Lattice relaxation algorithms for energy minimization."""

import numpy as np
import sys
from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass
from .lattice import Lattice4D

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


@dataclass
class RelaxationResult:
    """Results from a relaxation run."""
    converged: bool
    iterations: int
    final_energy: float
    final_max_force: float
    energy_history: List[float]
    force_history: List[float]


def estimate_progress(initial_force: float, current_force: float, tolerance: float) -> float:
    """Estimate relaxation progress based on force reduction (log scale).

    Args:
        initial_force: Starting max force
        current_force: Current max force
        tolerance: Target tolerance

    Returns:
        Progress percentage (0-100)
    """
    if current_force <= tolerance:
        return 100.0
    if initial_force <= tolerance:
        return 100.0

    # Use log scale since force decreases exponentially
    log_initial = np.log10(initial_force)
    log_current = np.log10(max(current_force, tolerance))
    log_target = np.log10(tolerance)

    progress = (log_initial - log_current) / (log_initial - log_target) * 100
    return min(max(progress, 0), 100)


def gradient_descent(
    lattice: Lattice4D,
    k: float = 1.0,
    a0: float = 1.0,
    tolerance: float = 1e-4,
    max_steps: int = 10000,
    dt: float = 0.1,
    callback: Optional[Callable[[int, float, float], None]] = None,
) -> RelaxationResult:
    """Relax lattice using gradient descent.

    Updates displacements according to:
        u_new = u_old + dt * force

    Args:
        lattice: The Lattice4D to relax (modified in place)
        k: Spring constant
        a0: Ideal bond length
        tolerance: Convergence criterion for max force
        max_steps: Maximum number of iterations
        dt: Time step (learning rate)
        callback: Optional function called each step with (iteration, energy, max_force)

    Returns:
        RelaxationResult with convergence info and history
    """
    energy_history = []
    force_history = []

    for step in range(max_steps):
        # Compute current state
        energy = lattice.compute_total_energy(k, a0)
        forces = lattice.compute_all_forces(k, a0)
        max_force = max(np.linalg.norm(f) for f in forces.values())

        energy_history.append(energy)
        force_history.append(max_force)

        if callback:
            callback(step, energy, max_force)

        # Check convergence
        if max_force < tolerance:
            return RelaxationResult(
                converged=True,
                iterations=step + 1,
                final_energy=energy,
                final_max_force=max_force,
                energy_history=energy_history,
                force_history=force_history,
            )

        # Update positions
        for pos in lattice.displacements:
            lattice.displacements[pos] = lattice.displacements[pos] + dt * forces[pos]

    # Did not converge
    final_energy = lattice.compute_total_energy(k, a0)
    final_max_force = lattice.compute_max_force(k, a0)
    energy_history.append(final_energy)
    force_history.append(final_max_force)

    return RelaxationResult(
        converged=False,
        iterations=max_steps,
        final_energy=final_energy,
        final_max_force=final_max_force,
        energy_history=energy_history,
        force_history=force_history,
    )


def fire(
    lattice: Lattice4D,
    k: float = 1.0,
    a0: float = 1.0,
    tolerance: float = 1e-4,
    max_steps: int = 10000,
    dt_start: float = 0.1,
    dt_max: float = 1.0,
    N_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.1,
    f_alpha: float = 0.99,
    callback: Optional[Callable[[int, float, float], None]] = None,
) -> RelaxationResult:
    """Relax lattice using FIRE algorithm (Fast Inertial Relaxation Engine).

    FIRE is a molecular dynamics-based optimizer that adapts velocity
    and time step based on the angle between velocity and force.

    Reference: Bitzek et al., PRL 97, 170201 (2006)

    Args:
        lattice: The Lattice4D to relax (modified in place)
        k: Spring constant
        a0: Ideal bond length
        tolerance: Convergence criterion for max force
        max_steps: Maximum number of iterations
        dt_start: Initial time step
        dt_max: Maximum time step
        N_min: Minimum steps before increasing dt
        f_inc: Factor to increase dt
        f_dec: Factor to decrease dt
        alpha_start: Initial mixing parameter
        f_alpha: Factor to decrease alpha
        callback: Optional function called each step with (iteration, energy, max_force)

    Returns:
        RelaxationResult with convergence info and history
    """
    energy_history = []
    force_history = []

    # Initialize velocities to zero
    velocities: Dict[Tuple[int, int, int, int], np.ndarray] = {}
    for pos in lattice.displacements:
        velocities[pos] = np.zeros(4)

    dt = dt_start
    alpha = alpha_start
    N_positive = 0

    for step in range(max_steps):
        # Compute current state
        energy = lattice.compute_total_energy(k, a0)
        forces = lattice.compute_all_forces(k, a0)
        max_force = max(np.linalg.norm(f) for f in forces.values())

        energy_history.append(energy)
        force_history.append(max_force)

        if callback:
            callback(step, energy, max_force)

        # Check convergence
        if max_force < tolerance:
            return RelaxationResult(
                converged=True,
                iterations=step + 1,
                final_energy=energy,
                final_max_force=max_force,
                energy_history=energy_history,
                force_history=force_history,
            )

        # Compute P = F . v (power)
        P = sum(np.dot(forces[pos], velocities[pos]) for pos in lattice.displacements)

        # Compute |F| and |v|
        F_norm = np.sqrt(sum(np.dot(f, f) for f in forces.values()))
        v_norm = np.sqrt(sum(np.dot(v, v) for v in velocities.values()))

        # FIRE velocity update: v = (1-alpha)*v + alpha*|v|*F_hat
        if F_norm > 1e-10:
            for pos in lattice.displacements:
                velocities[pos] = (1 - alpha) * velocities[pos] + alpha * v_norm * forces[pos] / F_norm

        if P > 0:
            # Going downhill
            N_positive += 1
            if N_positive > N_min:
                dt = min(dt * f_inc, dt_max)
                alpha = alpha * f_alpha
        else:
            # Going uphill - reset
            N_positive = 0
            dt = dt * f_dec
            alpha = alpha_start
            # Reset velocities
            for pos in velocities:
                velocities[pos] = np.zeros(4)

        # Velocity Verlet integration (simplified - just Euler for velocity)
        # Update velocities with forces
        for pos in lattice.displacements:
            velocities[pos] = velocities[pos] + dt * forces[pos]

        # Update positions
        for pos in lattice.displacements:
            lattice.displacements[pos] = lattice.displacements[pos] + dt * velocities[pos]

    # Did not converge
    final_energy = lattice.compute_total_energy(k, a0)
    final_max_force = lattice.compute_max_force(k, a0)
    energy_history.append(final_energy)
    force_history.append(final_max_force)

    return RelaxationResult(
        converged=False,
        iterations=max_steps,
        final_energy=final_energy,
        final_max_force=final_max_force,
        energy_history=energy_history,
        force_history=force_history,
    )


def relax(
    lattice: Lattice4D,
    method: str = 'fire',
    k: float = 1.0,
    a0: float = 1.0,
    tolerance: float = 1e-4,
    max_steps: int = 10000,
    verbose: bool = False,
    log_interval: int = 100,
    show_progress: bool = True,
) -> RelaxationResult:
    """Relax lattice using specified method.

    Args:
        lattice: The Lattice4D to relax (modified in place)
        method: 'fire' or 'gd' (gradient descent)
        k: Spring constant
        a0: Ideal bond length
        tolerance: Convergence criterion for max force
        max_steps: Maximum number of iterations
        verbose: If True, print detailed progress
        log_interval: Steps between progress messages
        show_progress: If True, show progress indicator

    Returns:
        RelaxationResult with convergence info and history
    """
    # Get initial force for progress estimation
    initial_force = lattice.compute_max_force(k, a0)
    last_progress_pct = 0

    def callback(step, energy, max_force):
        nonlocal last_progress_pct

        if verbose and step % log_interval == 0:
            progress = estimate_progress(initial_force, max_force, tolerance)
            print(f"  Step {step:5d}: E = {energy:.6f}, max|F| = {max_force:.6f} ({progress:.0f}% complete)")
        elif show_progress and not verbose:
            # Show progress milestones at 25%, 50%, 75%
            progress = estimate_progress(initial_force, max_force, tolerance)
            if progress >= last_progress_pct + 25:
                last_progress_pct = int(progress // 25) * 25
                print(f"  ... {last_progress_pct}% complete (step {step}, max|F|={max_force:.4f})")
                sys.stdout.flush()

    if method == 'fire':
        return fire(lattice, k, a0, tolerance, max_steps, callback=callback)
    elif method == 'gd':
        return gradient_descent(lattice, k, a0, tolerance, max_steps, callback=callback)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'fire' or 'gd'.")
