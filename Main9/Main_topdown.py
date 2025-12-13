"""Launcher for the top-down particle visualizer."""
from WorldStep import WorldStep
from Vizualizer_particles_topdown import run_viewer


def main():
    sim = WorldStep(
        nx=100,
        ny=100,
        nz=1,
        lx=1.0,
        ly=1.0,
        lz=1.0,
        seed=0,
        dispersion=0.1,
        particle_mass=20.0,
        k1_size=5,
    )

    run_viewer(sim)


if __name__ == '__main__':
    main()
