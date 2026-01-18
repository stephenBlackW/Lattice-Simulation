"""4D lattice dislocation simulation package."""

from .lattice import Lattice4D
from .dislocation import (
    introduce_dislocation,
    introduce_simple_dislocation,
    get_burgers_vector,
    verify_discontinuity,
    compute_initial_strain_energy,
)
from .visualization import (
    plot_2d_slice,
    plot_displacement_field,
    plot_strain_energy_field,
    plot_comparison,
    plot_convergence,
    plot_relaxed_comparison,
)
from .relaxation import (
    RelaxationResult,
    gradient_descent,
    fire,
    relax,
    estimate_progress,
)
from .projection import (
    get_3d_slice,
    get_3d_shadow,
    plot_3d_scatter,
    plot_3d_slice,
    plot_3d_shadow,
    create_w_slice_animation,
    create_w_slice_frames,
    create_rotating_view,
)
