"""
Simple 2D visualization script for taking snapshots of particle positions.
Creates a matplotlib image showing particle positions and colors.
"""
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection


def save_2d_snapshot(sim, filename='particles_2d.png', dpi=150, figsize=None):
    """
    Save a 2D snapshot of particles to an image file.
    
    Args:
        sim: WorldStep simulation object
        filename: Output filename (default: 'particles_2d.png')
        dpi: Image resolution (default: 150)
        figsize: Figure size in inches (width, height). If None, auto-calculated
    """
    # Get particle vertices
    verts_cp = sim.build_point_vertices()  # (N, 8): [x, y, z, r, g, b, size, alpha]
    verts = cp.asnumpy(verts_cp)
    
    # Handle empty particle case
    if verts.shape[0] == 0:
        print(f"Warning: No particles to visualize. Skipping snapshot '{filename}'")
        return
    
    positions = verts[:, :3]  # x, y, z
    colors = verts[:, 3:6]    # r, g, b
    sizes = verts[:, 6]       # point size
    alphas = verts[:, 7]      # alpha
    
    # Determine which plane to use
    if sim.NZ <= 2:
        # XY plane (ignore Z)
        x, y = positions[:, 0], positions[:, 1]
        xlabel, ylabel = 'X', 'Y'
    elif sim.NY <= 2:
        # XZ plane (ignore Y)
        x, y = positions[:, 0], positions[:, 2]
        xlabel, ylabel = 'X', 'Z'
    elif sim.NX <= 2:
        # YZ plane (ignore X)
        x, y = positions[:, 1], positions[:, 2]
        xlabel, ylabel = 'Y', 'Z'
    else:
        # 3D - project onto XY plane
        x, y = positions[:, 0], positions[:, 1]
        xlabel, ylabel = 'X', 'Y'
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        aspect = x_range / (y_range + 1e-9)
        if aspect > 1:
            figsize = (12, 12 / aspect)
        else:
            figsize = (12 * aspect, 12)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Create RGBA colors
    rgba_colors = np.column_stack([colors, alphas])
    
    # Scatter plot with actual colors
    scatter = ax.scatter(x, y, c=rgba_colors, s=sizes*2, edgecolors='none')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal')
    ax.set_title(f'Particle Distribution ({sim.num_particles} particles)')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Set background color
    ax.set_facecolor('#1a1a1e')
    fig.patch.set_facecolor('#1a1a1e')
    
    # Adjust tick colors for dark background
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, facecolor='#1a1a1e', edgecolor='none')
    print(f"Saved 2D snapshot to {filename}")
    plt.close()


def show_2d_live(sim):
    """
    Display 2D particles interactively with matplotlib.
    
    Args:
        sim: WorldStep simulation object
    """
    # Get particle vertices
    verts_cp = sim.build_point_vertices()
    verts = cp.asnumpy(verts_cp)
    
    # Handle empty particle case
    if verts.shape[0] == 0:
        print("Warning: No particles to visualize.")
        return
    
    positions = verts[:, :3]
    colors = verts[:, 3:6]
    sizes = verts[:, 6]
    alphas = verts[:, 7]
    
    # Determine plane
    if sim.NZ <= 2:
        x, y = positions[:, 0], positions[:, 1]
        xlabel, ylabel = 'X', 'Y'
    elif sim.NY <= 2:
        x, y = positions[:, 0], positions[:, 2]
        xlabel, ylabel = 'X', 'Z'
    elif sim.NX <= 2:
        x, y = positions[:, 1], positions[:, 2]
        xlabel, ylabel = 'Y', 'Z'
    else:
        x, y = positions[:, 0], positions[:, 1]
        xlabel, ylabel = 'X', 'Y'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    rgba_colors = np.column_stack([colors, alphas])
    scatter = ax.scatter(x, y, c=rgba_colors, s=sizes*2, edgecolors='none')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal')
    ax.set_title(f'Particle Distribution ({sim.num_particles} particles)')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    ax.set_facecolor('#1a1a1e')
    fig.patch.set_facecolor('#1a1a1e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Example usage
    from WorldStep import WorldStep
    
    print("Creating 2D simulation...")
    sim = WorldStep(
        nx=1000,
        ny=1000,
        nz=1,
        lx=0.1,
        ly=0.1,
        lz=0.1,
        seed=1,
        dispersion=0.1,
        particle_mass1=1,
        particle_mass2=0.1,
        particle_dispersion=20,
        k1_size=3,
        k2_size=1,
        k3_size=1,
        k4_size=1,
        k5_size=2
    )
    
    print("Generating 2D snapshot...")
    save_2d_snapshot(sim, 'particles_initial.png', dpi=200)
    
    # Run a few steps
    print("Running simulation for 10 steps...")
    for i in range(10):
        sim.step(0.05, print_timings=False)
    
    print("Generating snapshot after simulation...")
    save_2d_snapshot(sim, 'particles_after_10steps.png', dpi=200)
    
    print("Done! Check particles_initial.png and particles_after_10steps.png")
