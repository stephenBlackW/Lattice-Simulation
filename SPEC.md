# 4D Lattice Dislocation Projection Model

## Overview

Build a 4D cubic lattice, introduce dislocations with various Burgers vector orientations, compute strain energy, and visualize the results projected into 3D.

The goal is to explore how higher-dimensional topological defects appear when projected into lower dimensions, as a potential model for understanding quantum mechanical phenomena.

## Development Principles

- Prioritize working code over optimization — we can speed it up later
- Start with small lattice sizes (N=8) for development, scale up once working
- Write tests for geometry operations — 4D is easy to get wrong
- Save checkpoints/artifacts after expensive computations (especially relaxation)
- Output sanity checks at each phase before moving to the next

## Phase 1: Lattice Construction

**4D Cubic Lattice**
- Points at integer coordinates (x, y, z, w)
- Lattice size: N × N × N × N where N is configurable (start with N=8)
- Periodic boundary conditions in all dimensions
- Nearest neighbors: 8 per site (±1 along each of four axes)
- Store as dictionary mapping (x, y, z, w) tuples to displacement vectors [dx, dy, dz, dw]
- Initial displacements all zero

**Sanity Checks**
- Verify neighbor count is 8 for all sites
- Verify periodic boundary distances work correctly (test edge cases)
- Output basic lattice statistics

**Checkpoint**: Save lattice structure to file

## Phase 2: Dislocation Introduction

**Edge Dislocation in 4D**

An edge dislocation requires:
- A cut hyperplane: 3D surface where we make the cut
- A termination surface: 2D surface where the cut ends (the dislocation "core")
- A Burgers vector: 4D vector defining the shift

**Simple geometry to start:**
- Cut hyperplane: all points where w = 0 and x > 0
- Dislocation core: the 2D surface at w = 0, x = 0 (a plane in yz directions)
- Burgers vector: configurable, four cases of interest:
  - b = (1, 0, 0, 0) — Burgers in x
  - b = (0, 1, 0, 0) — Burgers in y
  - b = (0, 0, 1, 0) — Burgers in z
  - b = (0, 0, 0, 1) — Burgers in w (the most interesting case)

**Implementation:**
- For sites on positive side of cut (x > 0, w ≥ 0), add displacement +b/2
- For sites on negative side (x ≤ 0, w ≥ 0), add displacement -b/2
- This creates the discontinuity

**Sanity Checks**
- Visualize unrelaxed dislocation (before relaxation)
- Verify discontinuity exists across cut plane
- Compare initial strain energy for different Burgers vectors

**Checkpoint**: Save lattice with dislocation (pre-relaxation)

## Phase 3: Lattice Relaxation

**Energy Function**

Total elastic energy:

E_total = (k/2) Σᵢ Σⱼ (|rᵢⱼ| - a₀)²

Where:
- k = spring constant (set to 1)
- a₀ = ideal bond length (set to 1)
- rᵢⱼ = vector from site i to neighbor j (respecting periodic boundaries)
- Sum over all nearest-neighbor pairs

**Relaxation Method**
- Use gradient descent or FIRE algorithm
- Iterate until max force < tolerance (suggest 1e-4)
- Respect periodic boundary conditions when computing distances
- Log energy at each iteration for convergence monitoring

**Sanity Checks**
- Verify energy decreases monotonically
- Plot convergence curve
- Compare relaxed vs unrelaxed strain distribution

**Checkpoint**: Save relaxed lattice positions and final energies

## Phase 4: Strain Computation

**Local Strain Energy**

For each site i:

E_local(i) = (1/2) Σⱼ (|rᵢⱼ| - a₀)²

Sum over neighbors j of site i. Factor of 1/2 avoids double-counting.

**Optional: Strain Tensor (implement later if needed)**

Compute local deformation gradient tensor Fᵢ as best-fit linear transformation mapping ideal neighbor vectors to actual neighbor vectors.

Strain tensor: εᵢ = (1/2)(Fᵢᵀ Fᵢ - I)

**Checkpoint**: Save strain field data

## Phase 5: Projection and Visualization

**Projection Methods**

Method A — Slice:
- Select all sites where w = w₀ (configurable, default 0)
- Plot their (x, y, z) positions colored by local strain energy

Method B — Shadow:
- For each (x, y, z), aggregate strain over all w values
- Options: sum, max, or average
- Plot as 3D scalar field

Method C — Animated slices:
- Sweep w from -N/2 to N/2
- Generate frame for each w value
- Output as animation (gif or mp4)

**Visualization Details**

Use matplotlib for 3D scatter plots:
- Position: (x + dx, y + dy, z + dz) projected
- Color: local strain energy (colormap: viridis or hot)
- Size: constant or scaled with strain
- Alpha: adjustable for visibility

Also generate 2D slices:
- Fix z = 0 and w = 0, plot (x, y) colored by strain
- Useful for seeing dislocation core structure clearly

**Output**: Save figures as PNG, animations as GIF

## Phase 6: Analysis and Comparison

**Compare Burgers Vector Orientations**
- Run identical setup with b along x, y, z, and w
- Generate side-by-side visualizations
- Compute and compare:
  - Total strain energy
  - Spatial distribution of strain
  - Decay of strain with distance from core

**Key Question**: How does the b = (0,0,0,1) case differ from the others when projected to 3D?

**Optional: Dislocation Interactions**
- Place two dislocations with separation d
- Compare: same Burgers vector vs opposite Burgers vector
- Look for attraction/repulsion in relaxed configuration

## File Structure
```
4d_dislocation/
├── src/
│   ├── lattice.py        # Lattice class, neighbor finding, periodic boundaries
│   ├── dislocation.py    # Dislocation creation, cut-and-shift operations
│   ├── relaxation.py     # Energy minimization
│   ├── strain.py         # Strain computation
│   ├── projection.py     # 3D projection methods
│   └── visualization.py  # Plotting functions
├── tests/
│   ├── test_lattice.py   # Test neighbor finding, periodicity
│   ├── test_dislocation.py
│   └── test_strain.py
├── checkpoints/          # Saved lattice states
├── output/               # Figures and animations
├── main.py               # Run experiments
├── config.py             # Parameters
├── SPEC.md               # This file
└── README.md
```

## Configuration Parameters (config.py)
```python
LATTICE_SIZE = 8          # N for N×N×N×N lattice
SPRING_CONSTANT = 1.0
IDEAL_BOND_LENGTH = 1.0
RELAXATION_TOLERANCE = 1e-4
MAX_RELAXATION_STEPS = 10000
```

## Output Artifacts

1. `checkpoints/lattice_initial.npz` — Initial lattice
2. `checkpoints/lattice_dislocation_{burgers}.npz` — Post-dislocation, pre-relaxation
3. `checkpoints/lattice_relaxed_{burgers}.npz` — Post-relaxation
4. `output/strain_3d_{burgers}.png` — 3D strain visualization
5. `output/strain_slice_{burgers}.png` — 2D slice through core
6. `output/strain_animation_{burgers}.gif` — Animated w-slices
7. `output/comparison.png` — Side-by-side Burgers vector comparison
8. `output/convergence_{burgers}.png` — Relaxation convergence curves