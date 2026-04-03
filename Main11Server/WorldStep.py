import os
import tempfile
import time
import math

import cupy as cp
import numpy as np

class WorldStep:



    # k1 in grid kernel for diffusion and curl calculation
    # k2 out of grid kernel for divergence and gradient calculation
    # k3 in grid kernel for density field diffusion
    # k4 in grid kernel for double gradient calculation
    # k5 in grid kernel for local mean calculation

    def __init__(
            self,
            base_eddy=0.7,
            damping=0.02,
            dispersion=-0.5,
            particle_mass1=1.0,
            particle_mass2=1.0,
            particle_dispersion=1,
            enable_particles=True,
            particle_velocity_max=5.0,
            density1_injection_strength_pos=0.3,
            density1_injection_strength_neg=0.3,
            density2_injection_strength=0.5,
            density2_follow_strength=0.0,
            nx=50, ny=50, nz=50,
            lx=4.0, ly=4.0, lz=4.0,
            k1_size=3, k2_size=2, k3_size=3, k4_size=2, k5_size=2,
            seed=0, device_id=None, global_nz_start=0, global_nz_total=None,
            partition_index=0, partition_count=1, halo_exchange_dir=None, halo_timeout_s=10.0):
        
        self.device_id = device_id
        if device_id is not None:
            cp.cuda.Device(int(device_id)).use()

        self.NX = nx
        self.NY = ny
        self.NZ = nz
        self.LX = lx
        self.LY = ly
        self.LZ = lz
        self.seed = seed
        self.global_nz_start = int(global_nz_start)
        self.global_nz_total = int(global_nz_total) if global_nz_total is not None else int(nz)

        # base magnitude for flow vectors
        self.base_eddy = float(base_eddy)
        self.dispersion = dispersion
        self.particle_mass1 = particle_mass1
        self.particle_mass2 = particle_mass2
        self.damping = damping
        self.enable_particles = False
        
        # Particle behavior parameters
        self.particle_velocity_max = particle_velocity_max
        self.density2_follow_strength = density2_follow_strength
        
        # Density injection strengths
        self.density1_injection_strength_pos = density1_injection_strength_pos
        self.density1_injection_strength_neg = density1_injection_strength_neg
        self.density2_injection_strength = density2_injection_strength

        #initialize kernel average weights for normalization
        self.k1_size = k1_size
        self.k2_size = k2_size
        self.k3_size = k3_size
        self.k4_size = k4_size
        self.k5_size = k5_size

        self.partition_index = max(0, int(partition_index))
        self.partition_count = max(1, int(partition_count))
        self.halo_width = max((self.k1_size - 1) // 2, self.k2_size, self.k5_size)
        self.halo_exchange_dir = halo_exchange_dir or os.path.join(tempfile.gettempdir(), "main11server_halo")
        self.halo_timeout_s = float(halo_timeout_s)
        self.halo_exchange_enabled = self.partition_count > 1
        self.halo_session_id = f"nz{self.global_nz_total}_parts{self.partition_count}"
        self._halo_exchange_seq = 0
        if self.halo_exchange_enabled:
            os.makedirs(self.halo_exchange_dir, exist_ok=True)

        self.init_kernels()
        # Particle functionality is disabled in the server build to simplify NZ splitting.
        self.particles = cp.zeros((0, 3), dtype=cp.float32)
        self.particles_vel = cp.zeros((0, 3), dtype=cp.float32)
        self.particles2 = cp.zeros((0, 3), dtype=cp.float32)
        self.particles2_vel = cp.zeros((0, 3), dtype=cp.float32)
        self.num_particles = 0
        # initialize fields to zeros to avoid uninitialized memory
        self.curlfield = cp.zeros((self.NZ, self.NY, self.NX, 3), dtype=cp.float32)
        self.curlfield_prev = cp.zeros((self.NZ, self.NY, self.NX, 3), dtype=cp.float32)
        self.flowfield = cp.zeros((self.NZ, self.NY, self.NX, 3), dtype=cp.float32)
        self.flowfield_prev = cp.zeros((self.NZ, self.NY, self.NX, 3), dtype=cp.float32)
        self.densityfield = cp.zeros((self.NZ, self.NY, self.NX), dtype=cp.float32)
        self.densityfield2 = cp.zeros((self.NZ, self.NY, self.NX), dtype=cp.float32)

        self.num_particles = self.particles.shape[0]

        self.init_densityfield()
        self.init_flowfield(seed=seed, magnitude=2.0)
        
        # Particle trajectory tracking
        self.tracking_enabled = False
        self.tracked_particle_indices = []
        self.trajectory_data = []
        self.step_count = 0
        pass

    def generate_initial_particles(self, nx, ny, nz, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0), step=1):
        ox, oy, oz = origin
        dx, dy, dz = spacing

        x = ox + dx * cp.arange(nx, step=step)
        y = oy + dy * cp.arange(ny, step=step)
        z = oz + dz * cp.arange(nz, step=step)

        X, Y, Z = cp.meshgrid(x, y, z, indexing="ij")
        field = cp.stack((X, Y, Z), axis=-1)   # (nx, ny, nz, 3)
        num_points = field.shape[0] * field.shape[1] * field.shape[2]
        return field.reshape((num_points, 3)).astype(cp.float32)

    def init_kernels(self):
        # Precompute diffusion kernel weights (k1_size)
        self.diffuse_weights = []
        self.diffuse_shifts = []
        self.ahat1_weight = 0.0
        for i in range(-round((self.k1_size-1)/2), round((self.k1_size-1)/2)+1):
            for j in range(-round((self.k1_size-1)/2), round((self.k1_size-1)/2)+1):
                for k in range(-round((self.k1_size-1)/2), round((self.k1_size-1)/2)+1):
                    r = math.sqrt(i*i + j*j + k*k)
                    weight = math.exp(self.dispersion * r)
                    self.ahat1_weight += weight
                    self.diffuse_weights.append(weight)
                    self.diffuse_shifts.append((i, j, k))
        # Normalize weights
        self.diffuse_weights = [w / self.ahat1_weight for w in self.diffuse_weights]
        
        # Precompute curl kernel data (reuses k1_size, same as diffuse)
        self.curl_weights = []
        self.curl_shifts = []
        self.curl_rvecs = []
        for i in range(-round((self.k1_size-1)/2), round((self.k1_size-1)/2) + 1):
            for j in range(-round((self.k1_size-1)/2), round((self.k1_size-1)/2) + 1):
                for k in range(-round((self.k1_size-1)/2), round((self.k1_size-1)/2) + 1):
                    r = math.sqrt(i*i + j*j + k*k)
                    weight = math.exp(self.dispersion * r) / self.ahat1_weight
                    self.curl_weights.append(weight)
                    self.curl_shifts.append((i, j, k))
                    self.curl_rvecs.append(cp.array([i, j, k], dtype=cp.float32))

        # Precompute gradient kernel data (k2_size)
        self.gradient_weights = []
        self.gradient_shifts = []
        self.gradient_offsets = []
        self.ahat2_weight = 0.0
        for i in range(-self.k2_size, self.k2_size):
            for j in range(-self.k2_size, self.k2_size):
                for k in range(-self.k2_size, self.k2_size):
                    r = math.sqrt((i+0.5)*(i+0.5) + (j+0.5)*(j+0.5) + (k+0.5)*(k+0.5))
                    weight = r/(1 + r*r)
                    self.ahat2_weight += weight
                    self.gradient_weights.append(weight)
                    self.gradient_shifts.append((i, j, k))
                    self.gradient_offsets.append((i+0.5, j+0.5, k+0.5))
        # Normalize gradient weights
        self.gradient_weights = [w / self.ahat2_weight for w in self.gradient_weights]
        
        # Precompute divergence kernel data (k2_size with different range)
        self.divergence_weights = []
        self.divergence_shifts = []
        self.divergence_offsets = []
        for i in range(-self.k2_size + 1, self.k2_size + 1):
            for j in range(-self.k2_size + 1, self.k2_size + 1):
                for k in range(-self.k2_size + 1, self.k2_size + 1):
                    r = math.sqrt((i - 0.5)*(i - 0.5) + (j - 0.5)*(j - 0.5) + (k - 0.5)*(k - 0.5))
                    weight = r/(1 + r*r) / self.ahat2_weight
                    self.divergence_weights.append(weight)
                    self.divergence_shifts.append((i, j, k))
                    self.divergence_offsets.append((i - 0.5, j - 0.5, k - 0.5))

        self.ahat3_weight = 0.0

        self.ahat4_weight = 0.0

        # Precompute fieldmean kernel weights (k5_size)
        self.mean_weights = []
        self.mean_shifts = []
        self.ahat5_weight = 0.0
        for i in range(-self.k5_size, self.k5_size + 1):
            for j in range(-self.k5_size, self.k5_size + 1):
                for k in range(-self.k5_size, self.k5_size + 1):
                    r = math.sqrt(i*i + j*j + k*k)
                    weight = 1/(1 + r*r)
                    self.ahat5_weight += weight
                    self.mean_weights.append(weight)
                    self.mean_shifts.append((i, j, k))
        # Normalize weights
        self.mean_weights = [w / self.ahat5_weight for w in self.mean_weights]

    def enable_particle_tracking(self, num_particles_to_track=5):
        """Particle tracking is disabled in the server build."""
        print("Particle tracking has been removed from Main11Server.")
    
    def _record_tracked_particles(self):
        return
    
    def export_particle_trajectories(self, csv_path='particle_trajectories.csv'):
        """Particle trajectory export is disabled in the server build."""
        print("Particle trajectory export has been removed from Main11Server.")

    def init_densityfield(self):
        # Properly initialize densityfield with random values in [-1,1]
        self.densityfield = cp.random.uniform(low=-1.0, high=1.0, size=(self.NZ, self.NY, self.NX)).astype(cp.float32)
        # Initialize densityfield2 the same way
        self.densityfield2 = cp.random.uniform(low=-1.0, high=1.0, size=(self.NZ, self.NY, self.NX)).astype(cp.float32)
        return self.densityfield

    def init_flowfield(self, seed=None, magnitude=None):
        """Initialize `self.flowfield` as random directions with uniform magnitude.

        - `seed`: optional RNG seed (defaults to self.seed)
        - `magnitude`: if provided overrides `self.base_eddy`
        """
        if seed is None:
            seed = int(getattr(self, "seed", 0))
        mag = float(magnitude) if magnitude is not None else float(self.base_eddy)

        rng = cp.random.RandomState(seed + self.partition_index)
        shape = (self.NZ, self.NY, self.NX, 3)
        # Draw normal components, normalize to unit vectors, then scale
        vecs = rng.normal(loc=0.0, scale=1.0, size=shape).astype(cp.float32)
        norms = cp.linalg.norm(vecs, axis=-1, keepdims=True)
        norms = cp.where(norms == 0, 1e-9, norms)
        dirs = vecs / norms
        self.flowfield = dirs * mag
        return self.flowfield

    def _next_halo_exchange_tag(self, tag_base):
        tag = f"session_{self.halo_session_id}_step{int(self.step_count):06d}_seq{int(self._halo_exchange_seq):03d}_{tag_base}"
        self._halo_exchange_seq += 1
        return tag

    def _halo_file_path(self, tag, partition_index, side):
        return os.path.join(self.halo_exchange_dir, f"{tag}_p{int(partition_index)}_{side}.npy")

    def _write_halo_file(self, file_path, arr):
        tmp_path = f"{file_path}.tmp.npy"
        np.save(tmp_path, arr, allow_pickle=False)
        os.replace(tmp_path, file_path)

    def _wait_for_halo_file(self, file_path):
        deadline = time.perf_counter() + self.halo_timeout_s
        while time.perf_counter() < deadline:
            if os.path.exists(file_path):
                try:
                    age_s = time.time() - os.path.getmtime(file_path)
                    if age_s > max(5.0, self.halo_timeout_s * 4.0):
                        try:
                            os.remove(file_path)
                        except OSError:
                            pass
                        time.sleep(0.002)
                        continue
                    arr = np.load(file_path, allow_pickle=False)
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass
                    return arr
                except Exception:
                    time.sleep(0.002)
                    continue
            time.sleep(0.002)
        raise TimeoutError(f"Timed out waiting for halo file: {file_path}")

    def _exchange_halos_z(self, field, halo_width, tag_base):
        if not self.halo_exchange_enabled or halo_width <= 0:
            return None, None, 0

        actual_width = min(int(halo_width), int(field.shape[0]))
        tag = self._next_halo_exchange_tag(tag_base)
        left_neighbor = (self.partition_index - 1) % self.partition_count
        right_neighbor = (self.partition_index + 1) % self.partition_count

        self._write_halo_file(self._halo_file_path(tag, self.partition_index, "left"), cp.asnumpy(field[:actual_width]))
        self._write_halo_file(self._halo_file_path(tag, self.partition_index, "right"), cp.asnumpy(field[-actual_width:]))

        left_np = self._wait_for_halo_file(self._halo_file_path(tag, left_neighbor, "right"))
        right_np = self._wait_for_halo_file(self._halo_file_path(tag, right_neighbor, "left"))
        return cp.asarray(left_np), cp.asarray(right_np), actual_width

    def _get_field_with_halo(self, field, halo_width, tag_base):
        if not self.halo_exchange_enabled or halo_width <= 0:
            return field, 0
        left_halo, right_halo, actual_width = self._exchange_halos_z(field, halo_width, tag_base)
        return cp.concatenate((left_halo, field, right_halo), axis=0), actual_width

    def _crop_local_z(self, field, halo_width):
        if halo_width <= 0:
            return field
        return field[halo_width:halo_width + self.NZ, ...]

    def _roll_z(self, field, shift, tag_base="zroll"):
        shift = int(shift)
        if not self.halo_exchange_enabled or shift == 0:
            return cp.roll(field, shift=shift, axis=0)
        padded_field, halo = self._get_field_with_halo(field, abs(shift), tag_base=tag_base)
        shifted = cp.roll(padded_field, shift=shift, axis=0)
        return self._crop_local_z(shifted, halo)
    
    def calculate_gradientfield_kernal(self, field):
        padded_field, halo = self._get_field_with_halo(field, self.k2_size, tag_base="gradient")
        gradientfield = cp.zeros_like(field)
        gradientfield = cp.stack((gradientfield, gradientfield, gradientfield), axis=-1)
        for weight, shift, offset in zip(self.gradient_weights, self.gradient_shifts, self.gradient_offsets):
            shifted = cp.roll(padded_field, shift=shift, axis=(0, 1, 2))
            shifted = self._crop_local_z(shifted, halo)
            gradientfield[..., 0] += weight * shifted * offset[0]
            gradientfield[..., 1] += weight * shifted * offset[1]
            gradientfield[..., 2] += weight * shifted * offset[2]
        return gradientfield
    
    def calculate_divergence_from_flow_kernal(self, field):
        padded_field, halo = self._get_field_with_halo(field, self.k2_size, tag_base="divergence")
        divergencefield = cp.zeros((self.NZ, self.NY, self.NX), dtype=cp.float32)
        for weight, shift, offset in zip(self.divergence_weights, self.divergence_shifts, self.divergence_offsets):
            shifted = cp.roll(padded_field, shift=shift, axis=(0, 1, 2))
            shifted = self._crop_local_z(shifted, halo)
            divergencefield += weight * (shifted[...,0] * offset[0] + shifted[...,1] * offset[1] + shifted[...,2] * offset[2])
        return divergencefield
    
    def calculate_curlfield_kernal(self, field):
        curl_halo = max(0, (self.k1_size - 1) // 2)
        padded_field, halo = self._get_field_with_halo(field, curl_halo, tag_base="curl")
        curlfield = cp.zeros((self.NZ, self.NY, self.NX, 3), dtype=cp.float32)
        for weight, shift, r_vec in zip(self.curl_weights, self.curl_shifts, self.curl_rvecs):
            shifted = cp.roll(padded_field, shift=shift, axis=(0, 1, 2))
            shifted = self._crop_local_z(shifted, halo)
            curlfield += weight * cp.cross(r_vec, shifted)
        return curlfield
    

    def diffuse_field_kernal(self, field):
        diffuse_halo = max(0, (self.k1_size - 1) // 2)
        padded_field, halo = self._get_field_with_halo(field, diffuse_halo, tag_base="diffuse")
        diffused_field = cp.zeros_like(field)
        for weight, shift in zip(self.diffuse_weights, self.diffuse_shifts):
            shifted = cp.roll(padded_field, shift=shift, axis=(0, 1, 2))
            shifted = self._crop_local_z(shifted, halo)
            diffused_field += weight * shifted
        return diffused_field
    
    def fieldmean(self, field):
        padded_field, halo = self._get_field_with_halo(field, self.k5_size, tag_base="fieldmean")
        mean_fields = cp.zeros_like(field)
        for weight, shift in zip(self.mean_weights, self.mean_shifts):
            shifted = cp.roll(padded_field, shift=shift, axis=(0, 1, 2))
            shifted = self._crop_local_z(shifted, halo)
            mean_fields += weight * shifted
        return mean_fields

    # -------------------------
    # Simulation step functions

    def step(self, dt=0.1, print_timings=True):
        timings = {}
        self._halo_exchange_seq = 0
        
        t0 = time.perf_counter()
        self.step_densityfield(dt)
        timings['densityfield'] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        self.step_densityfield2(dt)
        timings['densityfield2'] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        self.step_flowfield(dt)
        timings['flowfield'] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        self.step_curlfield(dt, print_timings=print_timings)
        timings['curlfield'] = time.perf_counter() - t0
        
        # Particle advection/injection removed in the server build.

        if print_timings:
            total = sum(timings.values())
            print(f"Step timings (ms): total={total*1000:.2f}")
            for name, t in sorted(timings.items(), key=lambda x: -x[1]):
                print(f"  {name:20s}: {t*1000:6.2f} ms ({t/total*100:5.1f}%)")
        
        self.step_count += 1
    
    def step_densityfield(self, dt=0.1, diffusion_rate=0.1, curl_divergence_strength=0.0):
        """Advect and diffuse density field.
        
        curl_divergence_strength: how strongly curl magnitude drives density divergence (0 to disable).
        """
        # Diffuse density
        density_diffused = self.diffuse_field_kernal(self.densityfield)
        self.densityfield = (1 - diffusion_rate) * self.densityfield + diffusion_rate * density_diffused
        
        # Advect density using flow field divergence
        self.densityfield += self.calculate_divergence_from_flow_kernal(self.flowfield) * dt
        
        # Add divergence based on curl magnitude: high vorticity pushes density outward
        if curl_divergence_strength > 0:
            curl_mag = cp.linalg.norm(self.curlfield, axis=-1)  # shape (NZ, NY, NX)
            # normalize curl magnitude to [0, 1] range for stable effect
            curl_mag_normalized = curl_mag / (cp.max(curl_mag) + 1e-9)
            # divergence contribution: positive curl magnitude = outward spreading
            self.densityfield += curl_mag_normalized * curl_divergence_strength * dt
        
        # Clamp to reasonable range
        self.densityfield = cp.clip(self.densityfield, -1.0, 1.0)

    def step_densityfield2(self, dt=0.1, diffusion_rate=0.1, decay_rate=0.998, injection_strength=0.5):
        """Diffuse second density field with exponential decay (no flow, no curl, pure diffusion).
        
        decay_rate: multiplicative decay per frame (0.98 = 2% loss per step)
        """
        # Diffuse density
        density_diffused = self.diffuse_field_kernal(self.densityfield2)
        self.densityfield2 = (1 - diffusion_rate) * self.densityfield2 + diffusion_rate * density_diffused
        
        # Inject from energy of field 1 (optional, can be disabled by setting strength to 0)
        if injection_strength > 0:
            flow0 = self._roll_z(self.flowfield, shift=1, tag_base="density2_energy")
            flow1 = cp.roll(self.flowfield, shift=1, axis=1)
            flow2 = cp.roll(self.flowfield, shift=1, axis=2)
            flow01 = cp.roll(flow0, shift=1, axis=1)
            flow02 = cp.roll(flow0, shift=1, axis=2)
            flow12 = cp.roll(flow1, shift=1, axis=2)
            flow012 = cp.roll(flow01, shift=1, axis=2)
            flowavg = self.flowfield + flow0 + flow1 + flow2 + flow01 + flow02 + flow12 + flow012
            flowavg /= 8 # account for average in gradient kernel
            energy = self.densityfield * self.densityfield + cp.linalg.norm(flowavg, axis=-1)**2
            self.densityfield2 += energy * injection_strength * dt

        # Apply exponential decay
        self.densityfield2 *= decay_rate
        
        # Clamp to reasonable range
        self.densityfield2 = cp.clip(self.densityfield2, -1.0, 1.0)

    def step_flowfield(self, dt=0.1, flow_diffusion_rate=0.05):
        """Update flow field using density gradients and eddy effects."""
        # Pressure gradient from density
        functionfield0 = self._roll_z(self.densityfield2, shift=1, tag_base="flowfield_pressure")
        functionfield1 = cp.roll(self.densityfield2, shift=1, axis=1)
        functionfield2 = cp.roll(self.densityfield2, shift=1, axis=2)
        functionfield01 = cp.roll(functionfield0, shift=1, axis=1)
        functionfield02 = cp.roll(functionfield0, shift=1, axis=2)
        functionfield12 = cp.roll(functionfield1, shift=1, axis=2)
        functionfield012 = cp.roll(functionfield01, shift=1, axis=2)
        functionfield = self.densityfield2 + functionfield0 + functionfield1 + functionfield2 + functionfield01 + functionfield02 + functionfield12 + functionfield012
        functionfield /= 8.0  # average of center and neighbors
        functionfield = cp.stack([functionfield, functionfield, functionfield], axis=-1)  # make 3-channel for gradient calculation
        self.flowfield += self.calculate_gradientfield_kernal(self.densityfield) / (functionfield*functionfield + 1) * dt
        
        # Eddy/curl contribution from vorticity
        curl_change = self.curlfield - self.curlfield_prev
        eddyflowfield = self.calculate_curlfield_kernal(curl_change)
        self.flowfield += eddyflowfield * dt * -0.5  # Scale down eddy effect
        
        # Apply dispersion (diffusion) to smooth flow field
        flow_diffused = self.diffuse_field_kernal(self.flowfield)
        self.flowfield = (1 - flow_diffusion_rate) * self.flowfield + flow_diffusion_rate * flow_diffused
        
        # Damping
        self.flowfield *= (1.0 - self.damping)

    def step_curlfield(self, dt=0.1, curl_diffusion_rate=0.1, print_timings=False):
        """Update curl field from flow field with diffusion."""
        if print_timings:
            timings = {}
            t0 = time.perf_counter()
        
        self.curlfield_prev = cp.copy(self.curlfield)
        
        if print_timings:
            timings['copy'] = time.perf_counter() - t0
            t0 = time.perf_counter()
        
        # Compute curl of the current flow field
        self.curlfield = self.calculate_curlfield_kernal(self.flowfield)
        
        if print_timings:
            timings['calculate_curl'] = time.perf_counter() - t0
            t0 = time.perf_counter()
        
        #calculate diffusion rate based on weighted average of the local density field density
        density_avg = self.fieldmean(self.densityfield)
        
        if print_timings:
            timings['fieldmean'] = time.perf_counter() - t0
            t0 = time.perf_counter()
        
        curl_diffusion_rate = curl_diffusion_rate * (2 / (1 + density_avg * density_avg) - 1) # diffusion rate lowers and becomes negative in high density areas
        curl_diffusion_rate = cp.stack((curl_diffusion_rate, curl_diffusion_rate, curl_diffusion_rate), axis=-1)
        
        if print_timings:
            timings['diffusion_rate_calc'] = time.perf_counter() - t0
            t0 = time.perf_counter()
        
        # Apply diffusion to smooth out vorticity
        curl_diffused = self.diffuse_field_kernal(self.curlfield)
        
        if print_timings:
            timings['diffuse'] = time.perf_counter() - t0
            t0 = time.perf_counter()
        
        self.curlfield = (1 - curl_diffusion_rate) * self.curlfield + curl_diffusion_rate * curl_diffused
        
        if print_timings:
            timings['blend'] = time.perf_counter() - t0
            # Print breakdown
            total = sum(timings.values())
            print(f"  Curlfield breakdown (total={total*1000:.2f}ms):")
            for name, t in sorted(timings.items(), key=lambda x: -x[1]):
                print(f"    {name:20s}: {t*1000:6.2f} ms ({t/total*100:5.1f}%)")

    def step_particles(self, dt=0.1, density2_follow_strength=None):
        """Particle stepping is disabled in the server build."""
        return

    def inject_particles_to_density2(self, strength=0.1):
        """Particle-to-density injection is disabled in the server build."""
        return

    def inject_particles_to_density1(self, strength_pos=-.10, strength_neg=-.10):
        """Particle-to-density injection is disabled in the server build."""
        return


    def clamp_magnitude_gpu(points, max_len):
        mag = cp.linalg.norm(points, axis=1, keepdims=True)
        scale = cp.minimum(1.0, max_len / (mag + 1e-9))
        return points * scale

    def compute_gradient_contributions(self, Points, GradientField):
        # Get the voxel indices for each point

        ceil_X = cp.mod(cp.ceil((Points[:,0] / self.LX) + self.NX / 2).astype(cp.int32), self.NX)
        ceil_Y = cp.mod(cp.ceil((Points[:,1] / self.LY) + self.NY / 2).astype(cp.int32), self.NY)
        ceil_Z = cp.mod(cp.ceil((Points[:,2] / self.LZ) + self.NZ / 2).astype(cp.int32), self.NZ)
        
        floor_X = cp.mod(cp.floor((Points[:,0] / self.LX) + self.NX / 2).astype(cp.int32), self.NX)
        floor_Y = cp.mod(cp.floor((Points[:,1] / self.LY) + self.NY / 2).astype(cp.int32), self.NY)
        floor_Z = cp.mod(cp.floor((Points[:,2] / self.LZ) + self.NZ / 2).astype(cp.int32), self.NZ)

        # Compute contributions from the 8 surrounding voxels

        impulseContributions = cp.zeros_like(Points)

        SelPoints = cp.stack(((ceil_X - self.NX / 2) * self.LX, (ceil_Y - self.NY / 2) * self.LY, (ceil_Z - self.NZ / 2) * self.LZ), axis=-1)
        SelGradients = GradientField[ceil_Z, ceil_Y, ceil_X]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))
        
        SelPoints = cp.stack(((ceil_X - self.NX / 2) * self.LX, (ceil_Y - self.NY / 2) * self.LY, (floor_Z - self.NZ / 2) * self.LZ), axis=-1)
        SelGradients = GradientField[floor_Z, ceil_Y, ceil_X]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))
        
        SelPoints = cp.stack(((ceil_X - self.NX / 2) * self.LX, (floor_Y - self.NY / 2) * self.LY, (ceil_Z - self.NZ / 2) * self.LZ), axis=-1)
        SelGradients = GradientField[ceil_Z, floor_Y, ceil_X]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))
        
        SelPoints = cp.stack(((ceil_X - self.NX / 2) * self.LX, (floor_Y - self.NY / 2) * self.LY, (floor_Z - self.NZ / 2) * self.LZ), axis=-1)
        SelGradients = GradientField[floor_Z, floor_Y, ceil_X]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))
        
        SelPoints = cp.stack(((floor_X - self.NX / 2) * self.LX, (ceil_Y - self.NY / 2) * self.LY, (ceil_Z - self.NZ / 2) * self.LZ), axis=-1)
        SelGradients = GradientField[ceil_Z, ceil_Y, floor_X]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))
        
        SelPoints = cp.stack(((floor_X - self.NX / 2) * self.LX, (ceil_Y - self.NY / 2) * self.LY, (floor_Z - self.NZ / 2) * self.LZ), axis=-1)
        SelGradients = GradientField[floor_Z, ceil_Y, floor_X]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))
        
        SelPoints = cp.stack(((floor_X - self.NX / 2) * self.LX, (floor_Y - self.NY / 2) * self.LY, (ceil_Z - self.NZ / 2) * self.LZ), axis=-1)
        SelGradients = GradientField[ceil_Z, floor_Y, floor_X]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))
        
        SelPoints = cp.stack(((floor_X - self.NX / 2) * self.LX, (floor_Y - self.NY / 2) * self.LY, (floor_Z - self.NZ / 2) * self.LZ), axis=-1)
        SelGradients = GradientField[floor_Z, floor_Y, floor_X]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))

            
        return impulseContributions
    
    def compute_curl_contributions(self, Points, CurlField):
        # Get the voxel indices for each point

        ceil_X = cp.mod(cp.ceil((Points[:,0] / self.LX) + self.NX / 2).astype(cp.int32), self.NX)
        ceil_Y = cp.mod(cp.ceil((Points[:,1] / self.LY) + self.NY / 2).astype(cp.int32), self.NY)
        ceil_Z = cp.mod(cp.ceil((Points[:,2] / self.LZ) + self.NZ / 2).astype(cp.int32), self.NZ)
        
        floor_X = cp.mod(cp.floor((Points[:,0] / self.LX) + self.NX / 2).astype(cp.int32), self.NX)
        floor_Y = cp.mod(cp.floor((Points[:,1] / self.LY) + self.NY / 2).astype(cp.int32), self.NY)
        floor_Z = cp.mod(cp.floor((Points[:,2] / self.LZ) + self.NZ / 2).astype(cp.int32), self.NZ)

        # Compute contributions from the 8 surrounding voxels

        impulseContributions = cp.zeros_like(Points)

        SelPoints = cp.stack(((ceil_X - self.NX / 2) * self.LX, (ceil_Y - self.NY / 2) * self.LY, (ceil_Z - self.NZ / 2) * self.LZ), axis=-1)
        SelCurls = CurlField[ceil_Z, ceil_Y, ceil_X]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelCurls, cp.stack((R_factor, R_factor, R_factor), 1))
        
        SelPoints = cp.stack(((ceil_X - self.NX / 2) * self.LX, (ceil_Y - self.NY / 2) * self.LY, (floor_Z - self.NZ / 2) * self.LZ), axis=-1)
        SelCurls = CurlField[floor_Z, ceil_Y, ceil_X]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelCurls, cp.stack((R_factor, R_factor, R_factor), 1))
        
        SelPoints = cp.stack(((ceil_X - self.NX / 2) * self.LX, (floor_Y - self.NY / 2) * self.LY, (ceil_Z - self.NZ / 2) * self.LZ), axis=-1)
        SelCurls = CurlField[ceil_Z, floor_Y, ceil_X]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelCurls, cp.stack((R_factor, R_factor, R_factor), 1))
        
        SelPoints = cp.stack(((ceil_X - self.NX / 2) * self.LX, (floor_Y - self.NY / 2) * self.LY, (floor_Z - self.NZ / 2) * self.LZ), axis=-1)
        SelCurls = CurlField[floor_Z, floor_Y, ceil_X]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelCurls, cp.stack((R_factor, R_factor, R_factor), 1))
        
        SelPoints = cp.stack(((floor_X - self.NX / 2) * self.LX, (ceil_Y - self.NY / 2) * self.LY, (ceil_Z - self.NZ / 2) * self.LZ), axis=-1)
        SelCurls = CurlField[ceil_Z, ceil_Y, floor_X]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelCurls, cp.stack((R_factor, R_factor, R_factor), 1))
        
        SelPoints = cp.stack(((floor_X - self.NX / 2) * self.LX, (ceil_Y - self.NY / 2) * self.LY, (floor_Z - self.NZ / 2) * self.LZ), axis=-1)
        SelCurls = CurlField[floor_Z, ceil_Y, floor_X]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelCurls, cp.stack((R_factor, R_factor, R_factor), 1))
        
        SelPoints = cp.stack(((floor_X - self.NX / 2) * self.LX, (floor_Y - self.NY / 2) * self.LY, (ceil_Z - self.NZ / 2) * self.LZ), axis=-1)
        SelCurls = CurlField[ceil_Z, floor_Y, floor_X]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelCurls, cp.stack((R_factor, R_factor, R_factor), 1))
        
        SelPoints = cp.stack(((floor_X - self.NX / 2) * self.LX, (floor_Y - self.NY / 2) * self.LY, (floor_Z - self.NZ / 2) * self.LZ), axis=-1)
        SelCurls = CurlField[floor_Z, floor_Y, floor_X]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelCurls, cp.stack((R_factor, R_factor, R_factor), 1))

            
        return impulseContributions


    # -------------------------
    # Vertex generation for rendering
    # -------------------------
    def build_point_vertices(self, min_size=3.0, max_size=18.0):
        """Particle vertices are disabled in the server build."""
        return cp.empty((0, 8), dtype=cp.float32)

    def build_point_vertices_region(self, region=None, min_size=3.0, max_size=18.0, stride=1):
        """Particle vertices are disabled in the server build."""
        return cp.empty((0, 8), dtype=cp.float32)

    def _get_region_slices(self, region=None, stride=1, max_cells=None):
        stride = max(1, int(stride))

        def _axis_bounds(axis_name, axis_size, spacing, global_start=0, global_total=None):
            if not region or axis_name not in region or region[axis_name] is None:
                return 0, axis_size

            lo, hi = region[axis_name]
            lo = float(lo)
            hi = float(hi)
            lo, hi = min(lo, hi), max(lo, hi)

            if global_total is None:
                start = int(math.floor(lo / spacing + axis_size / 2))
                stop = int(math.ceil(hi / spacing + axis_size / 2))
            else:
                start = int(math.floor(lo / spacing + global_total / 2)) - global_start
                stop = int(math.ceil(hi / spacing + global_total / 2)) - global_start

            start = max(0, min(axis_size - 1, start))
            stop = max(start + 1, min(axis_size, stop))
            return start, stop

        z0, z1 = _axis_bounds("z", self.NZ, self.LZ, global_start=self.global_nz_start, global_total=self.global_nz_total)
        y0, y1 = _axis_bounds("y", self.NY, self.LY)
        x0, x1 = _axis_bounds("x", self.NX, self.LX)

        if max_cells:
            while (math.ceil((z1 - z0) / stride) * math.ceil((y1 - y0) / stride) * math.ceil((x1 - x0) / stride)) > max_cells:
                stride += 1

        index_bounds = {
            "z": (int(z0), int(z1)),
            "y": (int(y0), int(y1)),
            "x": (int(x0), int(x1)),
        }
        world_bounds = {
            "x": (float((x0 - self.NX / 2) * self.LX), float((max(x0, x1 - 1) - self.NX / 2) * self.LX)),
            "y": (float((y0 - self.NY / 2) * self.LY), float((max(y0, y1 - 1) - self.NY / 2) * self.LY)),
            "z": (
                float(((self.global_nz_start + z0) - self.global_nz_total / 2) * self.LZ),
                float(((self.global_nz_start + max(z0, z1 - 1)) - self.global_nz_total / 2) * self.LZ),
            ),
        }

        return slice(z0, z1, stride), slice(y0, y1, stride), slice(x0, x1, stride), index_bounds, world_bounds, stride

    def _axis_world_coord(self, axis_name, index):
        if axis_name == "x":
            return float((index - self.NX / 2) * self.LX)
        if axis_name == "y":
            return float((index - self.NY / 2) * self.LY)
        if axis_name == "z":
            return float(((self.global_nz_start + index) - self.global_nz_total / 2) * self.LZ)
        raise ValueError(f"Unknown axis_name: {axis_name}")

    def _resolve_slice_index(self, axis_name, requested_index, start, stop):
        if requested_index is None:
            return max(start, min(stop - 1, (start + stop - 1) // 2))

        idx = int(requested_index)
        if axis_name == "z":
            idx -= self.global_nz_start

        return max(start, min(stop - 1, idx))

    def extract_field_block(self, field_name="density", region=None, stride=1, max_cells=None,
                            transfer_mode="cube", slice_axis="z", slice_index=None):
        field_map = {
            "density": self.densityfield,
            "density2": self.densityfield2,
            "flow": self.flowfield,
            "curl": self.curlfield,
        }
        if field_name not in field_map:
            raise ValueError(f"Unknown field_name: {field_name}")

        z_slice, y_slice, x_slice, index_bounds, world_bounds, stride = self._get_region_slices(
            region=region,
            stride=stride,
            max_cells=max_cells,
        )

        data = field_map[field_name][z_slice, y_slice, x_slice]
        transfer_mode = str(transfer_mode).lower()
        slice_axis = str(slice_axis).lower()

        if transfer_mode == "slice":
            z0, z1 = index_bounds["z"]
            y0, y1 = index_bounds["y"]
            x0, x1 = index_bounds["x"]

            if slice_axis == "z":
                idx = self._resolve_slice_index("z", slice_index, z0, z1)
                data = field_map[field_name][idx:idx+1, y_slice, x_slice]
                index_bounds["z"] = (int(idx), int(idx + 1))
                z_world = self._axis_world_coord("z", idx)
                world_bounds["z"] = (z_world, z_world)
            elif slice_axis == "y":
                idx = self._resolve_slice_index("y", slice_index, y0, y1)
                data = field_map[field_name][z_slice, idx:idx+1, x_slice]
                index_bounds["y"] = (int(idx), int(idx + 1))
                y_world = self._axis_world_coord("y", idx)
                world_bounds["y"] = (y_world, y_world)
            elif slice_axis == "x":
                idx = self._resolve_slice_index("x", slice_index, x0, x1)
                data = field_map[field_name][z_slice, y_slice, idx:idx+1]
                index_bounds["x"] = (int(idx), int(idx + 1))
                x_world = self._axis_world_coord("x", idx)
                world_bounds["x"] = (x_world, x_world)
            else:
                raise ValueError(f"Unknown slice_axis: {slice_axis}")

        spatial_shape = data.shape[:3] if data.ndim >= 3 else data.shape

        return {
            "data": data,
            "spatial_shape": tuple(int(v) for v in spatial_shape),
            "index_bounds": index_bounds,
            "world_bounds": world_bounds,
            "stride": int(stride),
            "transfer_mode": transfer_mode,
            "slice_axis": slice_axis,
            "slice_index": slice_index,
        }

    # -------------------------
    # Diagnostics / helpers
    # -------------------------
    def get_field_stats(self, field: str):
        """Return simple stats (min,max,mean) for a named field.

        field: 'density' | 'flow' | 'curl'
        For 'flow' and 'curl' we report statistics on magnitude.
        """
        if field == "density":
            arr = self.densityfield
            vmin = float(cp.min(arr))
            vmax = float(cp.max(arr))
            vmean = float(cp.mean(arr))
            return {"min": vmin, "max": vmax, "mean": vmean}
        elif field == "flow":
            mag = cp.linalg.norm(self.flowfield, axis=-1)
            vmin = float(cp.min(mag))
            vmax = float(cp.max(mag))
            vmean = float(cp.mean(mag))
            return {"min": vmin, "max": vmax, "mean": vmean}
        elif field == "curl":
            mag = cp.linalg.norm(self.curlfield, axis=-1)
            vmin = float(cp.min(mag))
            vmax = float(cp.max(mag))
            vmean = float(cp.mean(mag))
            return {"min": vmin, "max": vmax, "mean": vmean}
        else:
            raise ValueError("Unknown field: %s" % field)

    def print_field_stats(self):
        """Print diagnostics for density, flow, and curl to console (CuPy -> host floats)."""
        try:
            d = self.get_field_stats("density")
            f = self.get_field_stats("flow")
            c = self.get_field_stats("curl")
            print("Field stats:")
            print(f"  density: min={d['min']:.6g} max={d['max']:.6g} mean={d['mean']:.6g}")
            print(f"  flow mag: min={f['min']:.6g} max={f['max']:.6g} mean={f['mean']:.6g}")
            print(f"  curl mag: min={c['min']:.6g} max={c['max']:.6g} mean={c['mean']:.6g}")
        except Exception as e:
            print("Error computing field stats:", e)

    def export_particle_force_diagnostics(self, csv_path='particle_diagnostics.csv', dt=0.1, num_samples=500):
        """Particle diagnostics are disabled in the server build."""
        print("Particle force diagnostics have been removed from Main11Server.")
    
    def _sample_scalar_field_at_points(self, points, field):
        """Sample scalar field values at particle positions using trilinear interpolation."""
        # Convert positions to grid indices
        ix = ((points[:, 0] / self.LX) + self.NX / 2).astype(cp.int32)
        iy = ((points[:, 1] / self.LY) + self.NY / 2).astype(cp.int32)
        iz = ((points[:, 2] / self.LZ) + self.NZ / 2).astype(cp.int32)
        
        # Wrap indices
        ix = cp.mod(ix, self.NX)
        iy = cp.mod(iy, self.NY)
        iz = cp.mod(iz, self.NZ)
        
        # Sample field at nearest grid point (simple nearest-neighbor for diagnostics)
        return field[iz, iy, ix]


