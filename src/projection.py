"""3D projection and visualization of 4D lattice data."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, List, Callable
import sys
import os

from .lattice import Lattice4D


def print_progress(message: str, flush: bool = True):
    """Print a progress message."""
    print(f"  {message}")
    if flush:
        sys.stdout.flush()


def get_3d_slice(
    lattice: Lattice4D,
    w_value: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract a 3D slice at fixed w.

    Args:
        lattice: The Lattice4D
        w_value: Fixed w coordinate

    Returns:
        Tuple of (x, y, z, strain_energy) arrays
    """
    N = lattice.N
    positions = []
    energies = []

    for x in range(N):
        for y in range(N):
            for z in range(N):
                pos = (x, y, z, w_value)
                if pos in lattice.displacements:
                    # Get actual position with displacement
                    disp = lattice.displacements[pos]
                    actual_pos = np.array([x, y, z]) + disp[:3]
                    positions.append(actual_pos)
                    energies.append(lattice.compute_site_energy(pos))

    positions = np.array(positions)
    energies = np.array(energies)

    return positions[:, 0], positions[:, 1], positions[:, 2], energies


def get_3d_shadow(
    lattice: Lattice4D,
    aggregation: str = 'sum',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project 4D lattice to 3D by aggregating over w dimension.

    Args:
        lattice: The Lattice4D
        aggregation: 'sum', 'max', or 'mean'

    Returns:
        Tuple of (x, y, z, aggregated_strain) arrays
    """
    N = lattice.N
    # Accumulate strain for each (x, y, z) position
    strain_grid = {}

    print_progress(f"Computing {aggregation} projection over w dimension...")

    for x in range(N):
        for y in range(N):
            for z in range(N):
                key = (x, y, z)
                values = []
                for w in range(N):
                    pos = (x, y, z, w)
                    if pos in lattice.displacements:
                        values.append(lattice.compute_site_energy(pos))

                if values:
                    if aggregation == 'sum':
                        strain_grid[key] = sum(values)
                    elif aggregation == 'max':
                        strain_grid[key] = max(values)
                    elif aggregation == 'mean':
                        strain_grid[key] = np.mean(values)

    # Convert to arrays
    positions = []
    energies = []
    for (x, y, z), energy in strain_grid.items():
        positions.append([x, y, z])
        energies.append(energy)

    positions = np.array(positions)
    energies = np.array(energies)

    return positions[:, 0], positions[:, 1], positions[:, 2], energies


def plot_3d_scatter(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'hot',
    alpha: float = 0.6,
    size_scale: float = 50,
    elev: float = 20,
    azim: float = 45,
    show_colorbar: bool = True,
) -> plt.Figure:
    """Create a 3D scatter plot colored by values.

    Args:
        x, y, z: Coordinate arrays
        values: Values for coloring
        title: Plot title
        save_path: If provided, save figure to this path
        figsize: Figure size
        cmap: Colormap name
        alpha: Point transparency
        size_scale: Base size for points
        elev: Elevation angle
        azim: Azimuth angle
        show_colorbar: Whether to show colorbar

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Normalize values for coloring
    if values.max() > values.min():
        norm_values = (values - values.min()) / (values.max() - values.min())
    else:
        norm_values = np.zeros_like(values)

    # Size based on value (larger = more strain)
    sizes = size_scale * (0.5 + norm_values)

    scatter = ax.scatter(
        x, y, z,
        c=values,
        cmap=cmap,
        s=sizes,
        alpha=alpha,
        edgecolors='none',
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)

    if show_colorbar:
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Local Strain Energy')

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print_progress(f"Saved 3D plot to {save_path}")

    return fig


def plot_3d_slice(
    lattice: Lattice4D,
    w_value: int = 0,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """Plot a 3D slice of the lattice at fixed w.

    Args:
        lattice: The Lattice4D
        w_value: Fixed w coordinate
        title: Plot title
        save_path: If provided, save figure to this path
        **kwargs: Additional arguments for plot_3d_scatter

    Returns:
        matplotlib Figure
    """
    print_progress(f"Extracting 3D slice at w={w_value}...")
    x, y, z, energies = get_3d_slice(lattice, w_value)

    if title is None:
        title = f'3D Slice at w={w_value}'

    return plot_3d_scatter(x, y, z, energies, title=title, save_path=save_path, **kwargs)


def plot_3d_shadow(
    lattice: Lattice4D,
    aggregation: str = 'sum',
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """Plot 3D shadow projection (aggregated over w).

    Args:
        lattice: The Lattice4D
        aggregation: 'sum', 'max', or 'mean'
        title: Plot title
        save_path: If provided, save figure to this path
        **kwargs: Additional arguments for plot_3d_scatter

    Returns:
        matplotlib Figure
    """
    x, y, z, energies = get_3d_shadow(lattice, aggregation)

    if title is None:
        title = f'3D Shadow Projection ({aggregation} over w)'

    return plot_3d_scatter(x, y, z, energies, title=title, save_path=save_path, **kwargs)


def create_w_slice_animation(
    lattice: Lattice4D,
    output_path: str = 'output/w_slice_animation.gif',
    fps: int = 4,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'hot',
    elev: float = 20,
    azim: float = 45,
) -> str:
    """Create an animated GIF sweeping through w slices.

    Args:
        lattice: The Lattice4D
        output_path: Path for output GIF
        fps: Frames per second
        figsize: Figure size
        cmap: Colormap name
        elev: Elevation angle
        azim: Azimuth angle

    Returns:
        Path to saved animation
    """
    try:
        import imageio
    except ImportError:
        print("  Warning: imageio not available, saving individual frames instead")
        return create_w_slice_frames(lattice, output_path.replace('.gif', ''), figsize, cmap, elev, azim)

    N = lattice.N
    frames = []

    # Find global min/max for consistent color scale
    print_progress("Computing global color scale...")
    all_energies = []
    for w in range(N):
        _, _, _, energies = get_3d_slice(lattice, w)
        all_energies.extend(energies)
    vmin, vmax = min(all_energies), max(all_energies)

    print_progress(f"Generating {N} frames for w-slice animation...")

    for w in range(N):
        print_progress(f"  Frame {w+1}/{N} (w={w})...")
        x, y, z, energies = get_3d_slice(lattice, w)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        sizes = 50 * (0.5 + (energies - vmin) / (vmax - vmin + 1e-10))

        scatter = ax.scatter(
            x, y, z,
            c=energies,
            cmap=cmap,
            s=sizes,
            alpha=0.6,
            vmin=vmin,
            vmax=vmax,
            edgecolors='none',
        )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-0.5, N - 0.5)
        ax.set_ylim(-0.5, N - 0.5)
        ax.set_zlim(-0.5, N - 0.5)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'w = {w}')

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Local Strain Energy')

        # Convert figure to image array
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

        plt.close(fig)

    # Save as GIF
    print_progress(f"Saving animation to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print_progress(f"Animation saved to {output_path}")

    return output_path


def create_w_slice_frames(
    lattice: Lattice4D,
    output_dir: str = 'output/w_slices',
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'hot',
    elev: float = 20,
    azim: float = 45,
) -> str:
    """Create individual frame images for w slices.

    Args:
        lattice: The Lattice4D
        output_dir: Directory for output frames
        figsize: Figure size
        cmap: Colormap name
        elev: Elevation angle
        azim: Azimuth angle

    Returns:
        Path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    N = lattice.N

    # Find global min/max for consistent color scale
    print_progress("Computing global color scale...")
    all_energies = []
    for w in range(N):
        _, _, _, energies = get_3d_slice(lattice, w)
        all_energies.extend(energies)
    vmin, vmax = min(all_energies), max(all_energies)

    print_progress(f"Generating {N} frames...")

    for w in range(N):
        print_progress(f"  Frame {w+1}/{N} (w={w})...")
        x, y, z, energies = get_3d_slice(lattice, w)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        sizes = 50 * (0.5 + (energies - vmin) / (vmax - vmin + 1e-10))

        scatter = ax.scatter(
            x, y, z,
            c=energies,
            cmap=cmap,
            s=sizes,
            alpha=0.6,
            vmin=vmin,
            vmax=vmax,
            edgecolors='none',
        )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-0.5, N - 0.5)
        ax.set_ylim(-0.5, N - 0.5)
        ax.set_zlim(-0.5, N - 0.5)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'w = {w}')

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Local Strain Energy')

        frame_path = os.path.join(output_dir, f'w_slice_{w:03d}.png')
        fig.savefig(frame_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print_progress(f"Frames saved to {output_dir}/")
    return output_dir


def create_rotating_view(
    lattice: Lattice4D,
    w_value: int = 0,
    output_path: str = 'output/rotating_view.gif',
    n_frames: int = 36,
    fps: int = 10,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'hot',
    elev: float = 20,
) -> str:
    """Create a rotating 3D view animation.

    Args:
        lattice: The Lattice4D
        w_value: Fixed w coordinate for the slice
        output_path: Path for output GIF
        n_frames: Number of frames (rotation angles)
        fps: Frames per second
        figsize: Figure size
        cmap: Colormap name
        elev: Elevation angle

    Returns:
        Path to saved animation
    """
    try:
        import imageio
    except ImportError:
        print("  Warning: imageio not available, skipping rotating view")
        return ""

    N = lattice.N
    frames = []

    print_progress(f"Extracting slice at w={w_value}...")
    x, y, z, energies = get_3d_slice(lattice, w_value)
    vmin, vmax = energies.min(), energies.max()

    print_progress(f"Generating {n_frames} rotation frames...")

    for i, azim in enumerate(np.linspace(0, 360, n_frames, endpoint=False)):
        if i % 10 == 0:
            print_progress(f"  Frame {i+1}/{n_frames}...")

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        sizes = 50 * (0.5 + (energies - vmin) / (vmax - vmin + 1e-10))

        scatter = ax.scatter(
            x, y, z,
            c=energies,
            cmap=cmap,
            s=sizes,
            alpha=0.6,
            vmin=vmin,
            vmax=vmax,
            edgecolors='none',
        )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-0.5, N - 0.5)
        ax.set_ylim(-0.5, N - 0.5)
        ax.set_zlim(-0.5, N - 0.5)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'3D Slice at w={w_value}')

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

        plt.close(fig)

    print_progress(f"Saving animation to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print_progress(f"Animation saved to {output_path}")

    return output_path
