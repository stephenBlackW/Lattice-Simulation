"""Visualization for 3D lattice showing actual atom positions.

Creates clear visualizations where atoms are shown at their actual
displaced positions, with optional arrows showing displacement vectors.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple, Optional
from pathlib import Path

from .lattice3d import Lattice3D


def create_atom_figure(
    lattice: Lattice3D,
    title: str = "3D Crystal Lattice",
    show_displacement_arrows: bool = True,
    show_bonds: bool = True,
    show_cut_plane: bool = False,
    cut_plane_z: float = None,
    atom_size: float = 15,
    arrow_scale: float = 1.0,
    color_by: str = 'z',  # 'z', 'displacement', or 'fixed'
) -> go.Figure:
    """Create a 3D figure showing atoms at their actual positions.

    Args:
        lattice: The 3D lattice to visualize
        title: Figure title
        show_displacement_arrows: Show arrows from ideal to actual position
        show_bonds: Show bonds between neighboring atoms
        show_cut_plane: Show the shear cut plane
        cut_plane_z: Z coordinate of cut plane (for visualization)
        atom_size: Size of atom markers
        arrow_scale: Scale for displacement arrows
        color_by: How to color atoms ('z', 'displacement', or 'fixed')

    Returns:
        Plotly Figure
    """
    positions = lattice.get_positions()
    ideal_positions = lattice.get_ideal_positions()
    displacements = lattice.get_displacements()

    # Determine colors
    if color_by == 'z':
        colors = positions[:, 2]
        colorbar_title = 'Z Position'
    elif color_by == 'displacement':
        colors = np.linalg.norm(displacements, axis=1)
        colorbar_title = 'Displacement'
    else:
        colors = np.ones(len(positions))
        colorbar_title = None

    fig = go.Figure()

    # Add atoms as spheres at their ACTUAL positions
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='markers',
        marker=dict(
            size=atom_size,
            color=colors,
            colorscale='Viridis',
            opacity=0.9,
            line=dict(width=1, color='black'),
            colorbar=dict(title=colorbar_title) if colorbar_title else None,
        ),
        name='Atoms',
        hovertemplate=(
            'Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>'
            'Displacement: %{customdata:.3f}<extra></extra>'
        ),
        customdata=np.linalg.norm(displacements, axis=1),
    ))

    # Add displacement arrows (from ideal to actual position)
    if show_displacement_arrows:
        disp_mag = np.linalg.norm(displacements, axis=1)
        mask = disp_mag > 0.01  # Only show significant displacements

        if np.any(mask):
            # Draw lines from ideal to actual position
            for i in np.where(mask)[0]:
                ideal = ideal_positions[i]
                actual = positions[i]

                fig.add_trace(go.Scatter3d(
                    x=[ideal[0], actual[0]],
                    y=[ideal[1], actual[1]],
                    z=[ideal[2], actual[2]],
                    mode='lines',
                    line=dict(color='red', width=3),
                    showlegend=False,
                    hoverinfo='skip',
                ))

            # Add arrowheads using cones
            fig.add_trace(go.Cone(
                x=positions[mask, 0],
                y=positions[mask, 1],
                z=positions[mask, 2],
                u=-displacements[mask, 0] * 0.3,
                v=-displacements[mask, 1] * 0.3,
                w=-displacements[mask, 2] * 0.3,
                sizemode='absolute',
                sizeref=0.15,
                colorscale=[[0, 'red'], [1, 'red']],
                showscale=False,
                opacity=0.8,
                name='Displacement direction',
            ))

    # Add bonds between neighbors
    if show_bonds:
        bond_x, bond_y, bond_z = [], [], []

        for i, j in lattice.bonds:
            p1 = positions[i]
            p2 = positions[j]
            bond_x.extend([p1[0], p2[0], None])
            bond_y.extend([p1[1], p2[1], None])
            bond_z.extend([p1[2], p2[2], None])

        fig.add_trace(go.Scatter3d(
            x=bond_x,
            y=bond_y,
            z=bond_z,
            mode='lines',
            line=dict(color='gray', width=2),
            opacity=0.3,
            name='Bonds',
            hoverinfo='skip',
        ))

    # Add cut plane
    if show_cut_plane and cut_plane_z is not None:
        # Create a semi-transparent plane
        plane_x = np.array([[0, lattice.nx * lattice.a],
                           [0, lattice.nx * lattice.a]])
        plane_y = np.array([[0, 0],
                           [lattice.ny * lattice.a, lattice.ny * lattice.a]])
        plane_z = np.array([[cut_plane_z, cut_plane_z],
                           [cut_plane_z, cut_plane_z]])

        fig.add_trace(go.Surface(
            x=plane_x,
            y=plane_y,
            z=plane_z,
            colorscale=[[0, 'rgba(255,0,0,0.3)'], [1, 'rgba(255,0,0,0.3)']],
            showscale=False,
            name='Cut Plane',
            hoverinfo='skip',
        ))

    # Update layout
    max_coord = max(lattice.nx, lattice.ny, lattice.nz) * lattice.a
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        scene=dict(
            xaxis=dict(title='X', range=[-0.5, max_coord + 0.5]),
            yaxis=dict(title='Y', range=[-0.5, max_coord + 0.5]),
            zaxis=dict(title='Z', range=[-0.5, max_coord + 0.5]),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0),
                up=dict(x=0, y=0, z=1)
            ),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
    )

    return fig


def create_shear_animation(
    states: List[Lattice3D],
    cut_plane_z: float,
    title: str = "Shear Dislocation Process",
    atom_size: float = 12,
) -> go.Figure:
    """Create an animation of the shearing process.

    Args:
        states: List of lattice states from create_shear_sequence()
        cut_plane_z: Z coordinate of cut plane
        title: Figure title
        atom_size: Size of atom markers

    Returns:
        Plotly Figure with animation
    """
    n_states = len(states)

    # Create frames
    frames = []
    for i, lattice in enumerate(states):
        positions = lattice.get_positions()
        fraction = i / (n_states - 1)

        frame = go.Frame(
            data=[go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers',
                marker=dict(
                    size=atom_size,
                    color=positions[:, 2],
                    colorscale='Viridis',
                    opacity=0.9,
                    line=dict(width=1, color='black'),
                ),
            )],
            name=str(i),
            layout=go.Layout(
                title=f"{title}<br>Shear: {fraction*100:.0f}%"
            )
        )
        frames.append(frame)

    # Initial state
    init_pos = states[0].get_positions()
    max_coord = max(states[0].nx, states[0].ny, states[0].nz) * states[0].a

    fig = go.Figure(
        data=[go.Scatter3d(
            x=init_pos[:, 0],
            y=init_pos[:, 1],
            z=init_pos[:, 2],
            mode='markers',
            marker=dict(
                size=atom_size,
                color=init_pos[:, 2],
                colorscale='Viridis',
                opacity=0.9,
                line=dict(width=1, color='black'),
                colorbar=dict(title='Z Position'),
            ),
        )],
        frames=frames
    )

    # Add cut plane (static)
    plane_x = np.array([[0, max_coord], [0, max_coord]])
    plane_y = np.array([[0, 0], [max_coord, max_coord]])
    plane_z = np.array([[cut_plane_z, cut_plane_z], [cut_plane_z, cut_plane_z]])

    fig.add_trace(go.Surface(
        x=plane_x, y=plane_y, z=plane_z,
        colorscale=[[0, 'rgba(255,0,0,0.2)'], [1, 'rgba(255,0,0,0.2)']],
        showscale=False,
        hoverinfo='skip',
    ))

    # Animation controls
    fig.update_layout(
        title=dict(text=f"{title}<br>Shear: 0%", x=0.5),
        scene=dict(
            xaxis=dict(title='X', range=[-0.5, max_coord + 1.5]),
            yaxis=dict(title='Y', range=[-0.5, max_coord + 0.5]),
            zaxis=dict(title='Z', range=[-0.5, max_coord + 0.5]),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.0)),
        ),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            y=0,
            x=0.1,
            buttons=[
                dict(label='▶ Play',
                     method='animate',
                     args=[None, dict(frame=dict(duration=200, redraw=True),
                                     fromcurrent=True)]),
                dict(label='⏸ Pause',
                     method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                       mode='immediate')]),
            ]
        )],
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix='Step: ', visible=True),
            steps=[dict(
                args=[[str(i)], dict(frame=dict(duration=100, redraw=True),
                                    mode='immediate')],
                label=str(i),
                method='animate'
            ) for i in range(n_states)]
        )]
    )

    return fig


def create_relaxation_animation(
    states: List[Tuple[Lattice3D, float]],
    title: str = "Relaxation Process",
    atom_size: float = 12,
) -> go.Figure:
    """Create an animation of the relaxation process.

    Args:
        states: List of (lattice, energy) tuples from create_relaxation_sequence()
        title: Figure title
        atom_size: Size of atom markers

    Returns:
        Plotly Figure with animation
    """
    n_states = len(states)

    # Create frames
    frames = []
    for i, (lattice, energy) in enumerate(states):
        positions = lattice.get_positions()
        displacements = lattice.get_displacements()
        disp_mag = np.linalg.norm(displacements, axis=1)

        frame = go.Frame(
            data=[go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers',
                marker=dict(
                    size=atom_size,
                    color=disp_mag,
                    colorscale='Hot',
                    cmin=0,
                    cmax=1,
                    opacity=0.9,
                    line=dict(width=1, color='black'),
                ),
            )],
            name=str(i),
            layout=go.Layout(
                title=f"{title}<br>Step {i}, Energy: {energy:.3f}"
            )
        )
        frames.append(frame)

    # Initial state
    init_lattice, init_energy = states[0]
    init_pos = init_lattice.get_positions()
    init_disp = np.linalg.norm(init_lattice.get_displacements(), axis=1)
    max_coord = max(init_lattice.nx, init_lattice.ny, init_lattice.nz) * init_lattice.a

    fig = go.Figure(
        data=[go.Scatter3d(
            x=init_pos[:, 0],
            y=init_pos[:, 1],
            z=init_pos[:, 2],
            mode='markers',
            marker=dict(
                size=atom_size,
                color=init_disp,
                colorscale='Hot',
                cmin=0,
                cmax=1,
                opacity=0.9,
                line=dict(width=1, color='black'),
                colorbar=dict(title='Displacement'),
            ),
        )],
        frames=frames
    )

    # Animation controls
    fig.update_layout(
        title=dict(text=f"{title}<br>Step 0, Energy: {init_energy:.3f}", x=0.5),
        scene=dict(
            xaxis=dict(title='X', range=[-0.5, max_coord + 1.5]),
            yaxis=dict(title='Y', range=[-0.5, max_coord + 0.5]),
            zaxis=dict(title='Z', range=[-0.5, max_coord + 0.5]),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.0)),
        ),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            y=0,
            x=0.1,
            buttons=[
                dict(label='▶ Play',
                     method='animate',
                     args=[None, dict(frame=dict(duration=300, redraw=True),
                                     fromcurrent=True)]),
                dict(label='⏸ Pause',
                     method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                       mode='immediate')]),
            ]
        )],
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix='Step: ', visible=True),
            steps=[dict(
                args=[[str(i)], dict(frame=dict(duration=100, redraw=True),
                                    mode='immediate')],
                label=str(i),
                method='animate'
            ) for i in range(n_states)]
        )]
    )

    return fig


def create_full_animation(
    shear_states: List[Lattice3D],
    relax_states: List[Tuple[Lattice3D, float]],
    cut_plane_z: float,
    title: str = "Shear Dislocation: Full Process",
    atom_size: float = 12,
) -> go.Figure:
    """Create a combined animation showing shearing then relaxation.

    Args:
        shear_states: States from create_shear_sequence()
        relax_states: States from create_relaxation_sequence()
        cut_plane_z: Z coordinate of cut plane
        title: Figure title
        atom_size: Size of atom markers

    Returns:
        Plotly Figure with combined animation
    """
    all_frames = []
    frame_labels = []

    # Shearing frames
    for i, lattice in enumerate(shear_states):
        positions = lattice.get_positions()
        displacements = lattice.get_displacements()
        disp_mag = np.linalg.norm(displacements, axis=1)
        fraction = i / (len(shear_states) - 1)

        frame = go.Frame(
            data=[go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers',
                marker=dict(
                    size=atom_size,
                    color=disp_mag,
                    colorscale='Hot',
                    cmin=0,
                    cmax=1,
                    opacity=0.9,
                    line=dict(width=1, color='black'),
                ),
            )],
            name=str(len(all_frames)),
            layout=go.Layout(
                title=f"{title}<br>SHEARING: {fraction*100:.0f}%"
            )
        )
        all_frames.append(frame)
        frame_labels.append(f"Shear {fraction*100:.0f}%")

    # Relaxation frames
    for i, (lattice, energy) in enumerate(relax_states):
        positions = lattice.get_positions()
        displacements = lattice.get_displacements()
        disp_mag = np.linalg.norm(displacements, axis=1)

        frame = go.Frame(
            data=[go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers',
                marker=dict(
                    size=atom_size,
                    color=disp_mag,
                    colorscale='Hot',
                    cmin=0,
                    cmax=1,
                    opacity=0.9,
                    line=dict(width=1, color='black'),
                ),
            )],
            name=str(len(all_frames)),
            layout=go.Layout(
                title=f"{title}<br>RELAXING: E={energy:.2f}"
            )
        )
        all_frames.append(frame)
        frame_labels.append(f"Relax E={energy:.2f}")

    # Initial state
    init_pos = shear_states[0].get_positions()
    init_disp = np.linalg.norm(shear_states[0].get_displacements(), axis=1)
    max_coord = max(shear_states[0].nx, shear_states[0].ny, shear_states[0].nz) * shear_states[0].a

    fig = go.Figure(
        data=[go.Scatter3d(
            x=init_pos[:, 0],
            y=init_pos[:, 1],
            z=init_pos[:, 2],
            mode='markers',
            marker=dict(
                size=atom_size,
                color=init_disp,
                colorscale='Hot',
                cmin=0,
                cmax=1,
                opacity=0.9,
                line=dict(width=1, color='black'),
                colorbar=dict(title='Displacement'),
            ),
        )],
        frames=all_frames
    )

    # Add cut plane
    plane_x = np.array([[0, max_coord], [0, max_coord]])
    plane_y = np.array([[0, 0], [max_coord, max_coord]])
    plane_z = np.array([[cut_plane_z, cut_plane_z], [cut_plane_z, cut_plane_z]])

    fig.add_trace(go.Surface(
        x=plane_x, y=plane_y, z=plane_z,
        colorscale=[[0, 'rgba(255,0,0,0.15)'], [1, 'rgba(255,0,0,0.15)']],
        showscale=False,
        hoverinfo='skip',
    ))

    # Animation controls
    fig.update_layout(
        title=dict(text=f"{title}<br>SHEARING: 0%", x=0.5),
        scene=dict(
            xaxis=dict(title='X', range=[-0.5, max_coord + 1.5]),
            yaxis=dict(title='Y', range=[-0.5, max_coord + 0.5]),
            zaxis=dict(title='Z', range=[-0.5, max_coord + 0.5]),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.0)),
        ),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            y=0,
            x=0.1,
            buttons=[
                dict(label='▶ Play',
                     method='animate',
                     args=[None, dict(frame=dict(duration=200, redraw=True),
                                     fromcurrent=True)]),
                dict(label='⏸ Pause',
                     method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                       mode='immediate')]),
            ]
        )],
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix='', visible=True),
            steps=[dict(
                args=[[str(i)], dict(frame=dict(duration=100, redraw=True),
                                    mode='immediate')],
                label=frame_labels[i],
                method='animate'
            ) for i in range(len(all_frames))]
        )]
    )

    return fig


def save_figure(fig: go.Figure, path: str) -> None:
    """Save figure to HTML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=True)
    print(f"Saved: {path}")
