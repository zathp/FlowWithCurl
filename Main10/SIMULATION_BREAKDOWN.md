# Main10 Simulation Breakdown

## Overview
Main10 is a **GPU-accelerated 3D fluid dynamics simulation** using CuPy (CUDA Python) that simulates vortex-based flow with particle visualization.

---

## Core Components

### 1. WorldStep.py - The Simulation Engine (745 lines)

This is the heart of the simulation. Here's what happens step-by-step:

#### Initialization (`__init__`)
- Creates a **3D staggered grid** (100×100×100 by default)
- Initializes 5 convolution kernels (k1-k5) with different sizes for various field operations
- Generates two sets of particles that interact with the fields
- Sets up three main field arrays:
  - `densityfield` - scalar field for density (cell centers)
  - `flowfield` - velocity vectors (cell corners)  
  - `curlfield` - vorticity/curl vectors (cell corners)

#### Main Simulation Step (`step()` method)
Each frame executes these substeps:

1. **`step_densityfield()`** - Advects and diffuses density field #1
   - Diffuses density using weighted kernel
   - Advects based on flow divergence
   - Can add vorticity-driven divergence
   
2. **`step_densityfield2()`** - Manages second density field
   - Pure diffusion with exponential decay
   - No flow coupling - used for particle trails

3. **`calculate_gradientfield_kernal()`** - Computes pressure gradient from density

4. **`step_flowfield()`** - Updates velocity field
   - Adds pressure gradient forces
   - Adds eddy/curl contributions (vorticity confinement)
   - Applies diffusion smoothing
   - Applies damping

5. **`step_curlfield()`** - Updates vorticity field
   - Calculates curl of flow field
   - Applies density-dependent diffusion (higher density = less diffusion)
   
6. **`step_particles()`** - Advects two particle sets
   - Samples flow field at particle positions (trilinear interpolation)
   - Applies flow forces scaled by particle mass
   - Applies curl forces (particles rotate around vortices)
   - Uses **Rodrigues rotation formula** for curl-driven rotation
   - Wraps particles at boundaries (toroidal/periodic)

7. **`inject_particles_to_density1/2()`** - Particle→field coupling
   - Deposits particle density back into grid
   - Two particle sets inject opposite signs (creates dynamics)

---

### 2. Main_3d.py - Primary Entry Point

Initializes simulation with these settings:
- Grid: 100×100×100 cells
- Cell size: 0.1×0.1×0.1 units
- Two particle masses (1000 vs 0.1) - creates different behaviors
- Kernel sizes: k1=3, k2=2, k3=2, k4=2, k5=2
- Launches `viz_points_3d.py` 3D viewer

---

### 3. viz_points_3d.py - 3D OpenGL Visualizer (597 lines)

Interactive real-time viewer:

**Features:**
- Renders particles as colored points (size based on velocity)
- Two particle sets: velocity-colored + cyan
- Bounding box wireframe
- Orbital camera with mouse controls
- Multiple render modes (points/vectors/density)

**Controls (from KEYBINDS.md):**
- **SPACE**: Pause/play
- **N**: Single step
- **M**: Cycle render modes
- **B**: Toggle flow/curl display
- **Mouse drag**: Rotate camera
- **+/-**: Zoom
- **IJKL**: Move camera
- **R**: Reset camera
- **D**: Dump state to .npz file

---

### 4. Supporting Files

#### Main.py - Alternative entry point
Simpler configuration for 2D-like simulations (NZ=10)

#### plot_vectors.py - Static vector field plotter
Creates matplotlib 3D quiver plots of flow/curl fields

#### viz_2d_snapshot.py - 2D screenshot utility
Saves particle positions to PNG images

#### GRID_STRUCTURE.md - Documentation
Explains the staggered grid layout:
- Main grid: density at cell centers
- Offset grid: vectors at cell corners

#### KEYBINDS.md - User guide
Complete keyboard/mouse control reference

---

## Physics Pipeline (Every Frame)

```
1. Diffuse & advect density fields
      ↓
2. Calculate density gradient → pressure forces
      ↓
3. Update flow field (pressure + curl + diffusion + damping)
      ↓
4. Calculate curl of flow → vorticity field
      ↓
5. Update curl field (density-modulated diffusion)
      ↓
6. Advect particles (flow forces + curl rotation)
      ↓
7. Inject particles back to density → feedback loop
```

---

## Key Algorithms

1. **Kernel Convolutions**: All field operations use weighted kernels with exponential falloff
2. **Trilinear Interpolation**: 8-corner sampling for particle-field interactions
3. **Rodrigues Rotation**: Particles rotate around local curl vectors
4. **Vorticity Confinement**: Curl changes drive flow (creates turbulent eddies)
5. **Periodic Boundaries**: Toroidal wrapping (particles reappear on opposite side)

---

## GPU Acceleration

- All arrays are CuPy (CUDA) arrays
- Field operations parallelized on GPU
- Particle updates vectorized
- Only visualization data copied to CPU

---

## Kernel Details

The simulation uses 5 different convolution kernels:

| Kernel | Size | Purpose | Grid Type |
|--------|------|---------|-----------|
| k1 | 3×3×3 | Diffusion and curl calculation | In-grid |
| k2 | 4×4×4 | Divergence and gradient calculation | Out-of-grid |
| k3 | 2×2×2 | Density field diffusion | In-grid |
| k4 | 2×2×2 | Double gradient calculation | In-grid |
| k5 | 2×2×2 | Local mean calculation | In-grid |

### Kernel Weight Functions

- **k1 (diffusion/curl)**: `weight = exp(dispersion * r)`
- **k2 (gradient/divergence)**: `weight = r / (1 + r²)`
- **k5 (local mean)**: `weight = 1 / (1 + r²)`

---

## Particle Dynamics

### Two Particle Sets

1. **Particles (Set 1)**
   - Mass: `particle_mass1` (default: 1000)
   - Color: Velocity-based RGB (direction → color)
   - Inject positive density to `densityfield`

2. **Particles2 (Set 2)**
   - Mass: `particle_mass2` (default: 0.1)
   - Color: Fixed cyan (R:0.2, G:0.9, B:0.9)
   - Inject negative density to `densityfield`

### Particle Forces

Particles experience:
1. **Flow forces**: Linear acceleration from flow field
2. **Curl forces**: Rotation around vorticity vectors using Rodrigues formula
3. **Density gradient**: Optional attraction to density field #2
4. **Velocity clamping**: Max speed limit (`particle_velocity_max`)

---

## Simulation Parameters

### Configurable in Main_3d.py

```python
nx, ny, nz = 100, 100, 100     # Grid resolution
lx, ly, lz = 0.1, 0.1, 0.1     # Cell spacing
seed = 22                       # Random seed
dispersion = 0.1                # Kernel falloff rate
damping = 0.01                  # Flow damping coefficient
particle_mass1 = 1000           # Heavy particles
particle_mass2 = 0.1            # Light particles
particle_dispersion = 5         # Particle grid spacing
k1_size = 3                     # Diffusion kernel size
k2_size = 2                     # Gradient kernel size
k3_size = 2                     # Density diffusion kernel size
k4_size = 2                     # Double gradient kernel size
k5_size = 2                     # Local mean kernel size
```

### Density Injection Strengths

- `density1_injection_strength_pos`: Particles → densityfield1 (positive)
- `density1_injection_strength_neg`: Particles2 → densityfield1 (negative)
- `density2_injection_strength`: Both particle sets → densityfield2

---

## Workflow Example

### Running the Simulation

```bash
python Main_3d.py
```

### Taking Snapshots

The simulation automatically saves:
- `my_particles.png` - Initial state
- `my_particles_after_viewer.png` - State after viewer closes

### Dumping State

Press **D** in the viewer to save current state to:
- Temporary `.npz` file with all field data
- Can be loaded with `plot_vectors.py --load-dump`

### Visualizing Vector Fields

```bash
# Plot flow field
python plot_vectors.py --load-dump --field flow --stride 2

# Plot curl field
python plot_vectors.py --load-dump --field curl --stride 2
```

---

## Performance Notes

### Timing Breakdown (Typical)

The `step()` method prints timing for each substep:
- `densityfield`: ~1-5ms
- `densityfield2`: ~1-5ms
- `gradient`: ~5-10ms
- `flowfield`: ~5-10ms
- `curlfield`: ~10-20ms (most expensive)
- `particles`: ~2-5ms
- `inject_density1`: ~1-2ms
- `inject_density2`: ~1-2ms

**Total**: ~30-60ms per frame (~16-33 FPS)

### Optimization Tips

1. Reduce grid resolution (nx, ny, nz)
2. Decrease kernel sizes (k1-k5)
3. Reduce particle count (increase `particle_dispersion`)
4. Disable particles (`enable_particles=False`)
5. Reduce diffusion iterations

---

## Technical Details

### Staggered Grid Approach

This is a classic **Marker-and-Cell (MAC)** style grid:
- **Scalars** (density) stored at cell centers
- **Vectors** (flow, curl) stored at cell corners
- Better numerical stability
- Improved conservation properties

### Periodic Boundary Conditions

The domain wraps around in all dimensions:
```python
x_wrapped = mod(x + half_width, full_width) - half_width
```

This creates a toroidal topology (like Pac-Man in 3D).

### Field Coupling

The simulation has bidirectional coupling:
1. **Fields → Particles**: Flow and curl advect particles
2. **Particles → Fields**: Particle density injects into density fields
3. **Density → Flow**: Density gradients drive flow
4. **Flow → Curl**: Flow curl creates vorticity
5. **Curl → Flow**: Curl changes create eddies (vorticity confinement)

This creates rich, turbulent dynamics!

---

## Summary

Main10 is a sophisticated **curl-noise based fluid simulator** with:
- Two-way particle-field coupling
- Vorticity confinement for turbulent flow
- GPU acceleration via CuPy
- Interactive 3D visualization
- Multiple density fields for complex interactions

Perfect for visual effects, studying vortex dynamics, or exploring computational fluid dynamics concepts!
