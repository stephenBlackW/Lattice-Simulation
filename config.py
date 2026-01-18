"""Configuration parameters for 4D lattice dislocation simulation."""

# Lattice parameters
LATTICE_SIZE = 8          # N for N x N x N x N lattice
SPRING_CONSTANT = 1.0
IDEAL_BOND_LENGTH = 1.0

# Relaxation parameters
RELAXATION_TOLERANCE = 1e-4
MAX_RELAXATION_STEPS = 10000

# Burgers vector presets
BURGERS_VECTORS = {
    'x': (1, 0, 0, 0),
    'y': (0, 1, 0, 0),
    'z': (0, 0, 1, 0),
    'w': (0, 0, 0, 1),
}
