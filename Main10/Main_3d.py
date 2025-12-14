from WorldStep import WorldStep
from viz_points_3d import run_viewer
import traceback


def main():
    try:
        sim = WorldStep(
            nx=50,
            ny=50,
            nz=50,
            lx=0.1,
            ly=0.1,
            lz=0.1,
            seed=1,
            dispersion=0.1,
            particle_mass1=1,
            particle_mass2=1,
            particle_dispersion=3,
            k1_size=3,
            k2_size=1,
            k3_size=1,
            k4_size=1,
            k5_size=2
        )
        print("Simulation initialized successfully")
        print("Starting viewer...")
        run_viewer(sim, width=1920, height=1080)
        print("Viewer closed")
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")


if __name__ == '__main__':
    main()
