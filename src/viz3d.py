"""3D visualization module using Plotly and PyVista.

Provides high-quality 3D visualizations for 4D lattice dislocations:
- Animated 4D→3D projections (slicing through w dimension)
- Dislocation core structure visualization
- Relaxation process animations
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import json

from .lattice import Lattice4D


def get_3d_slice(lattice: Lattice4D, w_value: int,
                 include_energy: bool = True) -> Dict:
    """Extract a 3D slice at fixed w coordinate.

    Args:
        lattice: The 4D lattice
        w_value: Fixed w coordinate for the slice
        include_energy: Whether to compute site energies

    Returns:
        Dictionary with positions, displacements, and optionally energies
    """
    N = lattice.N
    positions = []
    displacements = []
    energies = []

    for x in range(N):
        for y in range(N):
            for z in range(N):
                pos = (x, y, z, w_value)
                if pos in lattice.displacements:
                    disp = lattice.displacements[pos]
                    positions.append([x, y, z])
                    displacements.append(disp[:3])  # Take x, y, z components
                    if include_energy:
                        energies.append(lattice.compute_site_energy(pos))

    return {
        'positions': np.array(positions),
        'displacements': np.array(displacements),
        'energies': np.array(energies) if include_energy else None,
        'w_value': w_value
    }


def get_3d_projection(lattice: Lattice4D, method: str = 'sum',
                      include_energy: bool = True) -> Dict:
    """Project 4D lattice to 3D by aggregating over w.

    Args:
        lattice: The 4D lattice
        method: 'sum', 'max', or 'mean' for aggregating w dimension
        include_energy: Whether to compute site energies

    Returns:
        Dictionary with aggregated 3D data
    """
    N = lattice.N

    # Initialize aggregation arrays
    positions = []
    disp_agg = {}
    energy_agg = {}

    for x in range(N):
        for y in range(N):
            for z in range(N):
                key = (x, y, z)
                disp_agg[key] = []
                energy_agg[key] = []

                for w in range(N):
                    pos = (x, y, z, w)
                    if pos in lattice.displacements:
                        disp = lattice.displacements[pos]
                        disp_agg[key].append(disp[:3])
                        if include_energy:
                            energy_agg[key].append(lattice.compute_site_energy(pos))

    # Aggregate
    positions = []
    displacements = []
    energies = []

    for key in disp_agg:
        positions.append(list(key))
        disps = np.array(disp_agg[key])

        if method == 'sum':
            displacements.append(np.sum(disps, axis=0))
            if include_energy:
                energies.append(np.sum(energy_agg[key]))
        elif method == 'max':
            # Take displacement with maximum magnitude
            mags = np.linalg.norm(disps, axis=1)
            idx = np.argmax(mags)
            displacements.append(disps[idx])
            if include_energy:
                energies.append(np.max(energy_agg[key]))
        else:  # mean
            displacements.append(np.mean(disps, axis=0))
            if include_energy:
                energies.append(np.mean(energy_agg[key]))

    return {
        'positions': np.array(positions),
        'displacements': np.array(displacements),
        'energies': np.array(energies) if include_energy else None,
        'method': method
    }


def create_lattice_3d_figure(
    data: Dict,
    title: str = "3D Lattice Visualization",
    color_by: str = 'energy',
    show_displacements: bool = True,
    displacement_scale: float = 1.0,
    point_size: float = 8,
    colorscale: str = 'Viridis',
    show_bonds: bool = False,
    opacity: float = 0.8,
) -> go.Figure:
    """Create a Plotly 3D figure for lattice visualization.

    Args:
        data: Dictionary from get_3d_slice or get_3d_projection
        title: Figure title
        color_by: 'energy', 'displacement_mag', or 'x'/'y'/'z' displacement
        show_displacements: Whether to show displaced positions
        displacement_scale: Scale factor for displacements
        point_size: Size of lattice points
        colorscale: Plotly colorscale name
        show_bonds: Whether to show bonds between neighbors
        opacity: Point opacity

    Returns:
        Plotly Figure object
    """
    positions = data['positions']
    displacements = data['displacements']
    energies = data['energies']

    # Calculate display positions
    if show_displacements:
        display_pos = positions + displacements * displacement_scale
    else:
        display_pos = positions

    # Determine colors
    if color_by == 'energy' and energies is not None:
        colors = energies
        colorbar_title = 'Strain Energy'
    elif color_by == 'displacement_mag':
        colors = np.linalg.norm(displacements, axis=1)
        colorbar_title = 'Displacement Magnitude'
    elif color_by in ['x', 'y', 'z']:
        idx = {'x': 0, 'y': 1, 'z': 2}[color_by]
        colors = displacements[:, idx]
        colorbar_title = f'{color_by.upper()} Displacement'
    else:
        colors = energies if energies is not None else np.zeros(len(positions))
        colorbar_title = 'Value'

    # Create figure
    fig = go.Figure()

    # Add lattice points
    fig.add_trace(go.Scatter3d(
        x=display_pos[:, 0],
        y=display_pos[:, 1],
        z=display_pos[:, 2],
        mode='markers',
        marker=dict(
            size=point_size,
            color=colors,
            colorscale=colorscale,
            opacity=opacity,
            colorbar=dict(title=colorbar_title),
            line=dict(width=0.5, color='black')
        ),
        text=[f'pos: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})<br>'
              f'disp: ({d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f})<br>'
              f'energy: {e:.4f}' if energies is not None else
              f'pos: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})<br>'
              f'disp: ({d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f})'
              for p, d, e in zip(positions, displacements,
                                 energies if energies is not None else [0]*len(positions))],
        hoverinfo='text',
        name='Lattice Sites'
    ))

    # Add displacement arrows if showing displacements
    if show_displacements and np.max(np.abs(displacements)) > 0.01:
        # Create cone plot for arrows
        # Filter to significant displacements
        mag = np.linalg.norm(displacements, axis=1)
        mask = mag > 0.01

        if np.any(mask):
            fig.add_trace(go.Cone(
                x=positions[mask, 0],
                y=positions[mask, 1],
                z=positions[mask, 2],
                u=displacements[mask, 0],
                v=displacements[mask, 1],
                w=displacements[mask, 2],
                sizemode='absolute',
                sizeref=0.3,
                colorscale='Blues',
                showscale=False,
                opacity=0.6,
                name='Displacements'
            ))

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True
    )

    return fig


def create_4d_slice_animation(
    lattice: Lattice4D,
    title: str = "4D→3D Projection (w slices)",
    colorscale: str = 'Viridis',
    point_size: float = 6,
) -> go.Figure:
    """Create an animated figure showing slices through the w dimension.

    Args:
        lattice: The 4D lattice
        title: Figure title
        colorscale: Plotly colorscale
        point_size: Size of points

    Returns:
        Plotly Figure with animation frames
    """
    N = lattice.N

    # Get all slices
    slices = [get_3d_slice(lattice, w) for w in range(N)]

    # Find global energy range for consistent coloring
    all_energies = np.concatenate([s['energies'] for s in slices])
    e_min, e_max = np.min(all_energies), np.max(all_energies)
    if e_max - e_min < 1e-10:
        e_max = e_min + 1

    # Create frames
    frames = []
    for w, data in enumerate(slices):
        pos = data['positions'] + data['displacements']

        frame = go.Frame(
            data=[go.Scatter3d(
                x=pos[:, 0],
                y=pos[:, 1],
                z=pos[:, 2],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=data['energies'],
                    colorscale=colorscale,
                    cmin=e_min,
                    cmax=e_max,
                    opacity=0.8,
                ),
            )],
            name=str(w),
            layout=go.Layout(title=f"{title} - w={w}")
        )
        frames.append(frame)

    # Initial frame
    initial = slices[0]
    pos = initial['positions'] + initial['displacements']

    fig = go.Figure(
        data=[go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=initial['energies'],
                colorscale=colorscale,
                cmin=e_min,
                cmax=e_max,
                opacity=0.8,
                colorbar=dict(title='Strain Energy')
            ),
        )],
        frames=frames
    )

    # Add animation controls
    fig.update_layout(
        title=dict(text=f"{title} - w=0", x=0.5),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=0,
                x=0.1,
                xanchor='right',
                yanchor='top',
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=500, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=200)
                        )]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    )
                ]
            )
        ],
        sliders=[dict(
            active=0,
            yanchor='top',
            xanchor='left',
            currentvalue=dict(
                font=dict(size=16),
                prefix='w = ',
                visible=True,
                xanchor='right'
            ),
            transition=dict(duration=200),
            pad=dict(b=10, t=50),
            len=0.9,
            x=0.1,
            y=0,
            steps=[dict(
                args=[[str(w)], dict(
                    frame=dict(duration=200, redraw=True),
                    mode='immediate',
                    transition=dict(duration=200)
                )],
                label=str(w),
                method='animate'
            ) for w in range(N)]
        )]
    )

    return fig


def create_dislocation_core_visualization(
    lattice: Lattice4D,
    center_x: Optional[int] = None,
    radius: int = 3,
    w_value: int = 0,
    title: str = "Dislocation Core Structure",
) -> go.Figure:
    """Create detailed visualization of the dislocation core region.

    Args:
        lattice: The 4D lattice with dislocation
        center_x: X coordinate of dislocation core (default: N//2)
        radius: Radius around core to visualize
        w_value: W slice to visualize
        title: Figure title

    Returns:
        Plotly Figure showing core structure
    """
    N = lattice.N
    if center_x is None:
        center_x = N // 2

    # Get positions near the core
    positions = []
    displacements = []
    energies = []

    for x in range(N):
        for y in range(N):
            for z in range(N):
                # Check if within radius of core (in x direction)
                dx = min(abs(x - center_x), N - abs(x - center_x))
                if dx <= radius:
                    pos = (x, y, z, w_value)
                    if pos in lattice.displacements:
                        disp = lattice.displacements[pos]
                        positions.append([x, y, z])
                        displacements.append(disp[:3])
                        energies.append(lattice.compute_site_energy(pos))

    positions = np.array(positions)
    displacements = np.array(displacements)
    energies = np.array(energies)

    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=['Ideal Positions (colored by energy)',
                       'Displaced Positions (with arrows)']
    )

    # Ideal positions colored by energy
    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=energies,
                colorscale='Hot',
                opacity=0.9,
                colorbar=dict(title='Energy', x=0.45)
            ),
            name='Ideal'
        ),
        row=1, col=1
    )

    # Add cut plane indicator
    cut_y = np.linspace(0, N-1, 10)
    cut_z = np.linspace(0, N-1, 10)
    cut_Y, cut_Z = np.meshgrid(cut_y, cut_z)

    fig.add_trace(
        go.Surface(
            x=np.full_like(cut_Y, center_x + 0.5),
            y=cut_Y,
            z=cut_Z,
            opacity=0.3,
            colorscale=[[0, 'green'], [1, 'green']],
            showscale=False,
            name='Cut Plane'
        ),
        row=1, col=1
    )

    # Displaced positions
    displaced = positions + displacements
    fig.add_trace(
        go.Scatter3d(
            x=displaced[:, 0],
            y=displaced[:, 1],
            z=displaced[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=energies,
                colorscale='Hot',
                opacity=0.9,
                colorbar=dict(title='Energy', x=1.0)
            ),
            name='Displaced'
        ),
        row=1, col=2
    )

    # Displacement arrows
    mag = np.linalg.norm(displacements, axis=1)
    mask = mag > 0.01
    if np.any(mask):
        fig.add_trace(
            go.Cone(
                x=positions[mask, 0],
                y=positions[mask, 1],
                z=positions[mask, 2],
                u=displacements[mask, 0],
                v=displacements[mask, 1],
                w=displacements[mask, 2],
                sizemode='absolute',
                sizeref=0.4,
                colorscale='Blues',
                showscale=False,
                opacity=0.7,
                name='Displacement Vectors'
            ),
            row=1, col=2
        )

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5),
        showlegend=False,
        scene=dict(
            aspectmode='cube',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        scene2=dict(
            aspectmode='cube',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        height=600,
        width=1200
    )

    return fig


def create_relaxation_animation(
    checkpoints: List[Tuple[str, Lattice4D]],
    w_value: int = 0,
    title: str = "Relaxation Process",
) -> go.Figure:
    """Create animation showing the relaxation process.

    Args:
        checkpoints: List of (label, lattice) tuples for each relaxation stage
        w_value: W slice to visualize
        title: Figure title

    Returns:
        Plotly Figure with animation
    """
    if not checkpoints:
        raise ValueError("Need at least one checkpoint")

    # Get all slices and find global ranges
    slices = [(label, get_3d_slice(lat, w_value)) for label, lat in checkpoints]

    all_energies = np.concatenate([s['energies'] for _, s in slices])
    e_min, e_max = np.min(all_energies), np.max(all_energies)
    if e_max - e_min < 1e-10:
        e_max = e_min + 1

    # Create frames
    frames = []
    for i, (label, data) in enumerate(slices):
        pos = data['positions'] + data['displacements']
        total_energy = np.sum(data['energies'])

        frame = go.Frame(
            data=[go.Scatter3d(
                x=pos[:, 0],
                y=pos[:, 1],
                z=pos[:, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    color=data['energies'],
                    colorscale='Hot',
                    cmin=e_min,
                    cmax=e_max,
                    opacity=0.8,
                ),
            )],
            name=str(i),
            layout=go.Layout(
                title=f"{title}<br>{label} (E={total_energy:.2f})"
            )
        )
        frames.append(frame)

    # Initial state
    label0, data0 = slices[0]
    pos0 = data0['positions'] + data0['displacements']
    e0 = np.sum(data0['energies'])

    fig = go.Figure(
        data=[go.Scatter3d(
            x=pos0[:, 0],
            y=pos0[:, 1],
            z=pos0[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=data0['energies'],
                colorscale='Hot',
                cmin=e_min,
                cmax=e_max,
                opacity=0.8,
                colorbar=dict(title='Strain Energy')
            ),
        )],
        frames=frames
    )

    # Animation controls
    fig.update_layout(
        title=dict(text=f"{title}<br>{label0} (E={e0:.2f})", x=0.5),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=0,
                x=0.1,
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=1000, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=500)
                        )]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate'
                        )]
                    )
                ]
            )
        ],
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix='Stage: ', visible=True),
            steps=[dict(
                args=[[str(i)], dict(
                    frame=dict(duration=500, redraw=True),
                    mode='immediate'
                )],
                label=label,
                method='animate'
            ) for i, (label, _) in enumerate(slices)]
        )]
    )

    return fig


def save_figure(fig: go.Figure, path: str,
                format: str = 'html',
                width: int = 1200,
                height: int = 800) -> None:
    """Save a Plotly figure to file.

    Args:
        fig: Plotly Figure
        path: Output path (without extension)
        format: 'html', 'png', 'svg', or 'json'
        width: Image width (for png/svg)
        height: Image height (for png/svg)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'html':
        fig.write_html(str(path.with_suffix('.html')), include_plotlyjs=True)
    elif format == 'png':
        fig.write_image(str(path.with_suffix('.png')), width=width, height=height)
    elif format == 'svg':
        fig.write_image(str(path.with_suffix('.svg')), width=width, height=height)
    elif format == 'json':
        fig.write_json(str(path.with_suffix('.json')))
    else:
        raise ValueError(f"Unknown format: {format}")


def generate_all_visualizations(
    lattice: Lattice4D,
    output_dir: str = 'output/viz3d',
    burgers_label: str = 'bx',
    relaxed_lattice: Optional[Lattice4D] = None,
) -> Dict[str, str]:
    """Generate all 3D visualizations for a lattice.

    Args:
        lattice: The unrelaxed 4D lattice with dislocation
        output_dir: Output directory
        burgers_label: Label for Burgers vector (e.g., 'bx', 'by')
        relaxed_lattice: Optional relaxed version of the lattice

    Returns:
        Dictionary mapping visualization names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # 1. 4D slice animation
    print(f"Creating 4D slice animation for {burgers_label}...")
    fig = create_4d_slice_animation(
        lattice,
        title=f"4D→3D Projection ({burgers_label})"
    )
    path = output_dir / f"4d_slice_animation_{burgers_label}"
    save_figure(fig, str(path), format='html')
    outputs['4d_slice_animation'] = str(path.with_suffix('.html'))

    # 2. Dislocation core
    print(f"Creating dislocation core visualization for {burgers_label}...")
    fig = create_dislocation_core_visualization(
        lattice,
        title=f"Dislocation Core ({burgers_label})"
    )
    path = output_dir / f"dislocation_core_{burgers_label}"
    save_figure(fig, str(path), format='html')
    outputs['dislocation_core'] = str(path.with_suffix('.html'))

    # 3. 3D projection
    print(f"Creating 3D projection for {burgers_label}...")
    data = get_3d_projection(lattice, method='sum')
    fig = create_lattice_3d_figure(
        data,
        title=f"3D Projection (sum over w) - {burgers_label}",
        color_by='energy'
    )
    path = output_dir / f"3d_projection_{burgers_label}"
    save_figure(fig, str(path), format='html')
    outputs['3d_projection'] = str(path.with_suffix('.html'))

    # 4. Relaxation animation (if relaxed lattice provided)
    if relaxed_lattice is not None:
        print(f"Creating relaxation animation for {burgers_label}...")
        checkpoints = [
            ('Unrelaxed', lattice),
            ('Relaxed', relaxed_lattice)
        ]
        fig = create_relaxation_animation(
            checkpoints,
            title=f"Relaxation Process ({burgers_label})"
        )
        path = output_dir / f"relaxation_animation_{burgers_label}"
        save_figure(fig, str(path), format='html')
        outputs['relaxation_animation'] = str(path.with_suffix('.html'))

    print(f"Generated {len(outputs)} visualizations in {output_dir}")
    return outputs
