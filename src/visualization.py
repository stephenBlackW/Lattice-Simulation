"""Visualization functions for 4D lattice slices."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from .lattice import Lattice4D


def plot_2d_slice(
    lattice: Lattice4D,
    z_value: int = 0,
    w_value: int = 0,
    show_displacements: bool = True,
    color_by: str = 'strain',
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    show_arrows: bool = True,
    arrow_scale: float = 1.0,
) -> plt.Figure:
    """Plot a 2D slice of the lattice at fixed z and w.

    Args:
        lattice: The Lattice4D to visualize
        z_value: Fixed z coordinate
        w_value: Fixed w coordinate
        show_displacements: If True, show displaced positions
        color_by: 'strain' for local strain energy, 'displacement' for displacement magnitude
        title: Plot title
        save_path: If provided, save figure to this path
        figsize: Figure size
        show_arrows: If True, show displacement arrows
        arrow_scale: Scale factor for displacement arrows

    Returns:
        matplotlib Figure
    """
    N = lattice.N

    # Collect data for the slice
    x_coords = []
    y_coords = []
    x_displaced = []
    y_displaced = []
    colors = []

    for x in range(N):
        for y in range(N):
            pos = (x, y, z_value, w_value)
            if pos in lattice.displacements:
                disp = lattice.displacements[pos]

                x_coords.append(x)
                y_coords.append(y)
                x_displaced.append(x + disp[0])
                y_displaced.append(y + disp[1])

                if color_by == 'strain':
                    # Local strain energy
                    energy = lattice.compute_site_energy(pos)
                    colors.append(energy)
                else:
                    # Displacement magnitude
                    colors.append(np.linalg.norm(disp))

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    x_displaced = np.array(x_displaced)
    y_displaced = np.array(y_displaced)
    colors = np.array(colors)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot displaced positions colored by strain/displacement
    scatter = ax.scatter(
        x_displaced if show_displacements else x_coords,
        y_displaced if show_displacements else y_coords,
        c=colors,
        cmap='hot',
        s=100,
        alpha=0.8,
        edgecolors='black',
        linewidths=0.5,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Local Strain Energy' if color_by == 'strain' else 'Displacement Magnitude')

    # Show displacement arrows
    if show_arrows and show_displacements:
        dx = (x_displaced - x_coords) * arrow_scale
        dy = (y_displaced - y_coords) * arrow_scale

        # Only show arrows where displacement is significant
        mask = np.sqrt(dx**2 + dy**2) > 0.01
        if np.any(mask):
            ax.quiver(
                x_coords[mask], y_coords[mask],
                dx[mask], dy[mask],
                angles='xy', scale_units='xy', scale=1,
                color='blue', alpha=0.5, width=0.02
            )

    # Mark the dislocation core region
    center = N // 2
    ax.axvline(x=center + 0.5, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Cut plane')

    # Labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'2D Slice at z={z_value}, w={w_value}')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_displacement_field(
    lattice: Lattice4D,
    z_value: int = 0,
    w_value: int = 0,
    component: int = 0,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot displacement field component as a heatmap.

    Args:
        lattice: The Lattice4D to visualize
        z_value: Fixed z coordinate
        w_value: Fixed w coordinate
        component: Which component to plot (0=x, 1=y, 2=z, 3=w)
        title: Plot title
        save_path: If provided, save figure to this path
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    N = lattice.N
    component_names = ['x', 'y', 'z', 'w']

    # Create displacement field array
    field = np.zeros((N, N))

    for x in range(N):
        for y in range(N):
            pos = (x, y, z_value, w_value)
            if pos in lattice.displacements:
                field[y, x] = lattice.displacements[pos][component]  # Note: y, x for imshow

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Use diverging colormap centered at 0
    vmax = max(abs(field.min()), abs(field.max()))
    im = ax.imshow(
        field,
        cmap='RdBu_r',
        vmin=-vmax,
        vmax=vmax,
        origin='lower',
        extent=[-0.5, N - 0.5, -0.5, N - 0.5],
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'Displacement d{component_names[component]}')

    # Mark cut plane
    center = N // 2
    ax.axvline(x=center + 0.5, color='black', linestyle='--', linewidth=2, alpha=0.7)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'd{component_names[component]} at z={z_value}, w={w_value}')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_strain_energy_field(
    lattice: Lattice4D,
    z_value: int = 0,
    w_value: int = 0,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot local strain energy as a heatmap.

    Args:
        lattice: The Lattice4D to visualize
        z_value: Fixed z coordinate
        w_value: Fixed w coordinate
        title: Plot title
        save_path: If provided, save figure to this path
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    N = lattice.N

    # Create strain energy field array
    field = np.zeros((N, N))

    for x in range(N):
        for y in range(N):
            pos = (x, y, z_value, w_value)
            if pos in lattice.displacements:
                field[y, x] = lattice.compute_site_energy(pos)  # Note: y, x for imshow

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        field,
        cmap='hot',
        origin='lower',
        extent=[-0.5, N - 0.5, -0.5, N - 0.5],
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Local Strain Energy')

    # Mark cut plane
    center = N // 2
    ax.axvline(x=center + 0.5, color='green', linestyle='--', linewidth=2, alpha=0.7)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Local Strain Energy at z={z_value}, w={w_value}')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_comparison(
    lattices: dict,
    z_value: int = 0,
    w_value: int = 0,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 4),
) -> plt.Figure:
    """Plot side-by-side comparison of multiple lattices.

    Args:
        lattices: Dictionary mapping names to Lattice4D objects
        z_value: Fixed z coordinate
        w_value: Fixed w coordinate
        save_path: If provided, save figure to this path
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    n_lattices = len(lattices)
    fig, axes = plt.subplots(1, n_lattices, figsize=figsize)

    if n_lattices == 1:
        axes = [axes]

    # Find common color scale
    all_energies = []
    for lattice in lattices.values():
        N = lattice.N
        for x in range(N):
            for y in range(N):
                pos = (x, y, z_value, w_value)
                if pos in lattice.displacements:
                    all_energies.append(lattice.compute_site_energy(pos))

    vmin, vmax = min(all_energies), max(all_energies)

    for ax, (name, lattice) in zip(axes, lattices.items()):
        N = lattice.N
        field = np.zeros((N, N))

        for x in range(N):
            for y in range(N):
                pos = (x, y, z_value, w_value)
                if pos in lattice.displacements:
                    field[y, x] = lattice.compute_site_energy(pos)

        im = ax.imshow(
            field,
            cmap='hot',
            origin='lower',
            extent=[-0.5, N - 0.5, -0.5, N - 0.5],
            vmin=vmin,
            vmax=vmax,
        )

        center = N // 2
        ax.axvline(x=center + 0.5, color='green', linestyle='--', linewidth=1, alpha=0.7)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'b = {name}')
        ax.set_aspect('equal')

    # Add shared colorbar
    fig.colorbar(im, ax=axes, label='Local Strain Energy', shrink=0.8)

    plt.suptitle(f'Burgers Vector Comparison at z={z_value}, w={w_value}')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_convergence(
    energy_history: List[float],
    force_history: Optional[List[float]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Plot relaxation convergence curves.

    Args:
        energy_history: List of energy values at each iteration
        force_history: Optional list of max force values at each iteration
        title: Plot title
        save_path: If provided, save figure to this path
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    n_plots = 2 if force_history else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    iterations = range(len(energy_history))

    # Energy plot
    axes[0].semilogy(iterations, energy_history, 'b-', linewidth=1)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Total Energy')
    axes[0].set_title('Energy vs Iteration')
    axes[0].grid(True, alpha=0.3)

    # Force plot
    if force_history:
        axes[1].semilogy(iterations[:len(force_history)], force_history, 'r-', linewidth=1)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Max Force')
        axes[1].set_title('Max Force vs Iteration')
        axes[1].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_relaxed_comparison(
    unrelaxed: Lattice4D,
    relaxed: Lattice4D,
    z_value: int = 0,
    w_value: int = 0,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """Plot side-by-side comparison of unrelaxed and relaxed lattice.

    Args:
        unrelaxed: Lattice before relaxation
        relaxed: Lattice after relaxation
        z_value: Fixed z coordinate
        w_value: Fixed w coordinate
        title: Plot title
        save_path: If provided, save figure to this path
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    N = unrelaxed.N

    # Compute strain energy fields
    field_unrelaxed = np.zeros((N, N))
    field_relaxed = np.zeros((N, N))

    for x in range(N):
        for y in range(N):
            pos = (x, y, z_value, w_value)
            if pos in unrelaxed.displacements:
                field_unrelaxed[y, x] = unrelaxed.compute_site_energy(pos)
            if pos in relaxed.displacements:
                field_relaxed[y, x] = relaxed.compute_site_energy(pos)

    # Use same color scale for both
    vmax = max(field_unrelaxed.max(), field_relaxed.max())
    vmin = 0

    # Unrelaxed
    im1 = axes[0].imshow(
        field_unrelaxed,
        cmap='hot',
        origin='lower',
        extent=[-0.5, N - 0.5, -0.5, N - 0.5],
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title(f'Unrelaxed (E={field_unrelaxed.sum():.2f})')
    axes[0].set_aspect('equal')

    # Relaxed
    im2 = axes[1].imshow(
        field_relaxed,
        cmap='hot',
        origin='lower',
        extent=[-0.5, N - 0.5, -0.5, N - 0.5],
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title(f'Relaxed (E={field_relaxed.sum():.2f})')
    axes[1].set_aspect('equal')

    # Difference
    field_diff = field_unrelaxed - field_relaxed
    max_diff = max(abs(field_diff.min()), abs(field_diff.max()))
    if max_diff == 0:
        max_diff = 1  # Avoid division by zero
    im3 = axes[2].imshow(
        field_diff,
        cmap='RdBu_r',
        origin='lower',
        extent=[-0.5, N - 0.5, -0.5, N - 0.5],
        vmin=-max_diff,
        vmax=max_diff,
    )
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_title('Difference (Unrelaxed - Relaxed)')
    axes[2].set_aspect('equal')

    # Add colorbars
    fig.colorbar(im2, ax=axes[:2], label='Local Strain Energy', shrink=0.8)
    fig.colorbar(im3, ax=axes[2], label='Energy Reduction', shrink=0.8)

    if title:
        fig.suptitle(title)
    else:
        fig.suptitle(f'Relaxation Comparison at z={z_value}, w={w_value}')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig
