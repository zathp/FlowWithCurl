# main.py
from WorldStep import WorldStep
from Vizualizer_haze_with_topdown import run_viewer

# k1 in grid kernel for diffusion and curl calculation
# k2 in grid kernel for divergence and gradient calculation
# k3 in grid kernel for density field diffusion
# k4 in grid kernel for double gradient calculation


def main():
    # Create simulation (you can tweak grid size here)
    sim = WorldStep(
        nx=100,
        ny=100,
        nz=10,
        lx=1.0,
        ly=1.0,
        lz=1.0,
        seed=0,
        dispersion=0.1,
        particle_mass=20.0,
        k1_size=9,
        k2_size=4,
        k3_size=5,
        k4_size=3,
    )

    # Launch viewer
    run_viewer(sim)


if __name__ == "__main__":
    main()
