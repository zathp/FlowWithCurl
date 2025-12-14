from WorldStep import WorldStep
from viz_points_3d import run_viewer


def main():
    sim = WorldStep(
        nx=20,
        ny=20,
        nz=20,
        lx=0.1,
        ly=0.1,
        lz=0.1,
        seed=1,
        dispersion=0.1,
        particle_mass=0.1,
        k1_size=5,
        k2_size=3,
        k3_size=3,
        k4_size=2,
        k5_size=3
    )
    run_viewer(sim)


if __name__ == '__main__':
    main()
