from WorldStep import WorldStep
from viz_points_3d import run_viewer
import traceback
from viz_2d_snapshot import save_2d_snapshot

def main():
    try:
        sim = WorldStep(
            nx=100,
            ny=100,
            nz=100,
            lx=0.1,
            ly=0.1,
            lz=0.1,
            seed=22,
            dispersion=0.1,
            damping=0.01,
            particle_mass1=1000,
            particle_mass2=0.1,
            particle_dispersion=5,
            k1_size=3,
            k2_size=2,
            k3_size=2,
            k4_size=2,
            k5_size=2,
            enable_particles=True,
            density1_injection_strength_neg=1,
            density1_injection_strength_pos=1
        )
        
        save_2d_snapshot(sim, 'my_particles.png', dpi=200)

        print("Simulation initialized successfully")
        print("Starting viewer...")
        run_viewer(sim, width=1920, height=1080)
        save_2d_snapshot(sim, 'my_particles_after_viewer.png', dpi=200)
        print("Viewer closed")
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")


if __name__ == '__main__':
    main()
