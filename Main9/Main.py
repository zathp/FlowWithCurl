# main.py
from WorldStep import WorldStep
from Vizualizer_haze_with_topdown import run_viewer


def main():
    # Create simulation (you can tweak grid size here)
    sim = WorldStep(
        nx=10,
        ny=10,
        nz=10,
        lx=1.0,
        ly=1.0,
        lz=1.0,
        seed=0,
        dispersion=0.1,
        particle_mass=20.0
    )

    # Launch viewer
    run_viewer(sim)


if __name__ == "__main__":
    main()
