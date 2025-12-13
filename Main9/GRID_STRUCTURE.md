# Grid Structure Documentation

## Overview

This simulation uses a **staggered grid** approach commonly found in computational fluid dynamics (CFD):

- **Main Grid**: Stores scalar fields (density) at cell centers
- **Offset Grid**: Stores vector fields (flow, curl) at cell corners

This separation allows for better numerical stability and conservation properties.

---

## Grid Definitions

### Main Grid (Density Grid)
- **Resolution**: `(NX, NY, NZ)` cells
- **Physical extent**: `[-LX*NX/2, LX*NX/2] × [-LY*NY/2, LY*NY/2] × [-LZ*NZ/2, LZ*NZ/2]`
- **Cell size**: `(LX, LY, LZ)`
- **Field storage**: Scalar density field `densityfield[NZ, NY, NX]`

**Cell centers are at:**
```
x_i = -LX*NX/2 + (i + 0.5) * LX    for i in [0, NX)
y_j = -LY*NY/2 + (j + 0.5) * LY    for j in [0, NY)
z_k = -LZ*NZ/2 + (k + 0.5) * LZ    for k in [0, NZ)
```

### Offset Grid (Vector Fields Grid)
- **Resolution**: Same as main grid `(NX, NY, NZ)` (or sometimes `(NX+1, NY+1, NZ+1)` for true corner storage)
- **Field storage**: Vector fields `flowfield[NZ, NY, NX, 3]` and `curlfield[NZ, NY, NX, 3]`
- **Conceptually**: Values at the corners of cells in the main grid

**Corner positions are at:**
```
x_i = -LX*NX/2 + i * LX           for i in [0, NX]
y_j = -LY*NY/2 + j * LY           for j in [0, NY]
z_k = -LZ*NZ/2 + k * LZ           for k in [0, NZ]
```

*Note: Current implementation uses same size as main grid; could be extended to `(NX+1, NY+1, NZ+1)` for true corner coverage.*

---

## Field Layout

| Field | Grid | Shape | Data Type | Purpose |
|-------|------|-------|-----------|---------|
| `densityfield` | Main | `(NZ, NY, NX)` | `float32` | Scalar density at cell centers |
| `flowfield` | Offset | `(NZ, NY, NX, 3)` | `float32` | Vector field (velocity/flow) at corners |
| `curlfield` | Offset | `(NZ, NY, NX, 3)` | `float32` | Curl/vorticity vector at corners |
| `particles` | N/A | `(NX*NY*NZ, 3)` | `float32` | Particle positions (initially on main grid) |

---

## Indexing Convention

All arrays use **NumPy/CuPy indexing**: `[z, y, x]` or `[z, y, x, component]`

### Accessing density at cell center (i, j, k):
```python
value = densityfield[k, j, i]
```

### Accessing flow at corner (i, j, k):
```python
vector = flowfield[k, j, i, :]      # [fx, fy, fz]
fx = flowfield[k, j, i, 0]
fy = flowfield[k, j, i, 1]
fz = flowfield[k, j, i, 2]
```

### Particle initialization (on main grid):
```python
# Particles start at cell centers
x_coord = -LX*NX/2 + i*LX + LX/2
y_coord = -LY*NY/2 + j*LY + LY/2
z_coord = -LZ*NZ/2 + k*LZ + LZ/2
```

---

## Operations & Transformations

### Interpolation from Offset → Main Grid
When you need to evaluate a vector field at a cell center:
```
flow_at_center = 0.125 * sum(flow at 8 corners of the cell)
```

### Interpolation from Main → Offset Grid
When you need to evaluate density at a corner:
```
density_at_corner = 0.125 * sum(density of 8 adjacent cells)
```

### Gradient Computation
Gradients of density fields naturally sit on the offset (corner) grid:
```
∂ρ/∂x at corner ≈ (ρ_right - ρ_left) / (2*LX)
```

### Curl Computation
The curl of a vector field involves cross products and should preserve grid compatibility.

---

## Current Implementation Status

### ✅ Initialized
- `densityfield`: zeros (initialized in `__init__`)
- `flowfield`: zeros (initialized in `__init__`)
- `curlfield`: zeros (initialized in `__init__`)
- `particles`: on main grid corners (from `generate_initial_particles`)

### ⚠️ TODO / Needs Work
1. **Gradient kernels** (`calculate_gradientfield_kernal`) — should map Main → Offset properly
2. **Curl kernels** (`calculate_curlfield_kernal`) — should map Offset → Offset with proper staggering
3. **Field diffusion** — ensure kernel operations respect grid alignment
4. **Particle advection** — interpolate velocity fields correctly at particle positions
5. **Boundary conditions** — decide wrap vs. clamp for grid edges

---

## Visual Representation (2D slice)

```
Main Grid (density at cell centers):
┌─────┬─────┬─────┐
│  ●  │  ●  │  ●  │
├─────┼─────┼─────┤
│  ●  │  ●  │  ●  │
├─────┼─────┼─────┤
│  ●  │  ●  │  ●  │
└─────┴─────┴─────┘

Offset Grid (vectors at corners):
●─────●─────●─────●
│     │     │     │
●─────●─────●─────●
│     │     │     │
●─────●─────●─────●
│     │     │     │
●─────●─────●─────●

Legend:
● = grid point
─ = cell edge

Note: In 3D, offset grid has 8 corners per cell in main grid.
```

---

## Particle-Field Interaction

### Current Flow
1. **Compute gradients** of density field → gradient values on offset grid
2. **Interpolate** gradient at each particle position (bilinear/trilinear)
3. **Apply impulse** to particles based on interpolated gradient

### What Needs Improvement
- `compute_gradient_contributions()` currently assumes gradient is available; should call `calculate_gradientfield()` first
- Interpolation should use proper staggered-grid methods

---

## Helper Functions

### To add or refine:

```python
def interpolate_offset_to_main(offset_field):
    """Interpolate offset grid values to main grid."""
    # Average 8 corner values → cell center
    pass

def interpolate_main_to_offset(main_field):
    """Interpolate main grid values to offset grid."""
    # Average 8 cell values → corner
    pass

def get_world_position(grid_type, i, j, k):
    """Return physical (x, y, z) coordinate of a grid point."""
    if grid_type == "main":
        return (-LX*NX/2 + (i+0.5)*LX, -LY*NY/2 + (j+0.5)*LY, -LZ*NZ/2 + (k+0.5)*LZ)
    elif grid_type == "offset":
        return (-LX*NX/2 + i*LX, -LY*NY/2 + j*LY, -LZ*NZ/2 + k*LZ)
```

---

## References

- **Staggered grids in CFD**: Common in MAC (Marker-And-Cell) schemes
- **Collocated vs. Staggered**: Staggered grids reduce checker-board pressure oscillations
- **Interpolation**: Essential for cross-grid operations; typically use linear (2D) or trilinear (3D) interpolation

---

## Future Extensions

1. **Variable-size offset grid**: Use `(NX+1, NY+1, NZ+1)` for true corner coverage
2. **Boundary layer handling**: Proper Neumann/Dirichlet BC on staggered grids
3. **Higher-order interpolation**: Cubic or spectral methods for smoother advection
4. **Multigrid solvers**: Leverage staggered structure for pressure-projection problems
