import cupy as cp
import math
import time

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
            particle_mass=1.0,
            particle_c=1.0,
            nx=50, ny=50, nz=50,
            lx=4.0, ly=4.0, lz=4.0,
            k1_size=3, k2_size=2, k3_size=3, k4_size=2, k5_size=2,
            seed=0):
        
        self.NX = nx
        self.NY = ny
        self.NZ = nz
        self.LX = lx
        self.LY = ly
        self.LZ = lz
        self.seed = seed

        # base magnitude for flow vectors
        self.base_eddy = float(base_eddy)
        self.dispersion = dispersion
        self.particle_mass = particle_mass
        self.damping = damping

        #initialize kernel average weights for normalization
        self.k1_size = k1_size
        self.k2_size = k2_size
        self.k3_size = k3_size
        self.k4_size = k4_size
        self.k5_size = k5_size

        self.init_kernels()
        # initialize fields to zeros (curl/density); flow is seeded below
        self.particles = self.generate_initial_particles(nx, ny, nz,
            origin=(-lx * (nx - 1) / 2, -ly * (ny - 1) / 2, -lz * (nz - 1) / 2), spacing=(lx, ly, lz))
        self.particles_prev = cp.copy(self.particles)
        # second particle set (initialized with slight offset)
        self.particles2 = self.generate_initial_particles(nx, ny, nz,
            origin=(-lx * (nx - 1) / 2 + lx*0.5, -ly * (ny - 1) / 2 + ly*0.5, -lz * (nz - 1) / 2), spacing=(lx, ly, lz))
        self.particles2_prev = cp.copy(self.particles2)
        # initialize fields to zeros to avoid uninitialized memory
        self.curlfield = cp.zeros((self.NZ, self.NY, self.NX, 3), dtype=cp.float32)
        self.curlfield_prev = cp.zeros((self.NZ, self.NY, self.NX, 3), dtype=cp.float32)
        self.flowfield = cp.zeros((self.NZ, self.NY, self.NX, 3), dtype=cp.float32)
        self.flowfield_prev = cp.zeros((self.NZ, self.NY, self.NX, 3), dtype=cp.float32)
        self.densityfield = cp.zeros((self.NZ, self.NY, self.NX), dtype=cp.float32)
        self.densityfield2 = cp.zeros((self.NZ, self.NY, self.NX), dtype=cp.float32)

        self.init_densityfield()
        self.init_flowfield(seed=seed, magnitude=2.0)
        pass

    def generate_initial_particles(self, nx, ny, nz, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)):
        ox, oy, oz = origin
        dx, dy, dz = spacing

        x = ox + dx * cp.arange(nx)
        y = oy + dy * cp.arange(ny)
        z = oz + dz * cp.arange(nz)

        X, Y, Z = cp.meshgrid(x, y, z, indexing="ij")
        field = cp.stack((X, Y, Z), axis=-1)   # (nx, ny, nz, 3)
        return field.reshape((nx * ny * nz, 3))

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

        rng = cp.random.RandomState(seed)
        shape = (self.NZ, self.NY, self.NX, 3)
        # Draw normal components, normalize to unit vectors, then scale
        vecs = rng.normal(loc=0.0, scale=1.0, size=shape).astype(cp.float32)
        norms = cp.linalg.norm(vecs, axis=-1, keepdims=True)
        norms = cp.where(norms == 0, 1e-9, norms)
        dirs = vecs / norms
        self.flowfield = dirs * mag
        return self.flowfield
    
    def calculate_gradientfield_kernal(self, field):
        gradientfield = cp.zeros_like(field)
        gradientfield = cp.stack((gradientfield, gradientfield, gradientfield), axis=-1)
        for weight, shift, offset in zip(self.gradient_weights, self.gradient_shifts, self.gradient_offsets):
            shifted = cp.roll(field, shift=shift, axis=(0, 1, 2))
            gradientfield[..., 0] += weight * shifted * offset[0]
            gradientfield[..., 1] += weight * shifted * offset[1]
            gradientfield[..., 2] += weight * shifted * offset[2]
        return gradientfield
    
    def calculate_divergence_from_flow_kernal(self, field):
        divergencefield = cp.zeros((self.NZ, self.NY, self.NX), dtype=cp.float32)
        for weight, shift, offset in zip(self.divergence_weights, self.divergence_shifts, self.divergence_offsets):
            shifted = cp.roll(field, shift=shift, axis=(0, 1, 2))
            divergencefield += weight * (shifted[...,0] * offset[0] + shifted[...,1] * offset[1] + shifted[...,2] * offset[2])
        return divergencefield
    
    def calculate_curlfield_kernal(self, field):
        curlfield = cp.zeros((self.NZ, self.NY, self.NX, 3), dtype=cp.float32)
        for weight, shift, r_vec in zip(self.curl_weights, self.curl_shifts, self.curl_rvecs):
            shifted = cp.roll(field, shift=shift, axis=(0, 1, 2))
            curlfield += weight * cp.cross(r_vec, shifted)
        return curlfield
    

    def diffuse_field_kernal(self, field):
        diffused_field = cp.zeros_like(field)
        for weight, shift in zip(self.diffuse_weights, self.diffuse_shifts):
            shifted = cp.roll(field, shift=shift, axis=(0, 1, 2))
            diffused_field += weight * shifted
        return diffused_field
    
    def fieldmean(self, field):
        mean_fields = cp.zeros_like(field)
        for weight, shift in zip(self.mean_weights, self.mean_shifts):
            shifted = cp.roll(field, shift=shift, axis=(0, 1, 2))
            mean_fields += weight * shifted
        return mean_fields

    # -------------------------
    # Simulation step functions

    def step(self, dt=0.1, print_timings=True):
        timings = {}
        
        t0 = time.perf_counter()
        self.step_densityfield(dt)
        timings['densityfield'] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        self.step_densityfield2(dt)
        timings['densityfield2'] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        gradientfield = self.calculate_gradientfield_kernal(self.densityfield)
        timings['gradient'] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        self.step_flowfield(dt)
        timings['flowfield'] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        self.step_curlfield(dt)
        timings['curlfield'] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        self.step_particles(dt)
        timings['particles'] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        self.inject_particles_to_density1(strength_pos=0.3, strength_neg=0.3)
        timings['inject_density1'] = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        self.inject_particles_to_density2(strength=0.5)
        timings['inject_density2'] = time.perf_counter() - t0
        
        if print_timings:
            total = sum(timings.values())
            print(f"Step timings (ms): total={total*1000:.2f}")
            for name, t in sorted(timings.items(), key=lambda x: -x[1]):
                print(f"  {name:20s}: {t*1000:6.2f} ms ({t/total*100:5.1f}%)")
    
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

    def step_densityfield2(self, dt=0.1, diffusion_rate=0.1, decay_rate=0.998):
        """Diffuse second density field with exponential decay (no flow, no curl, pure diffusion).
        
        decay_rate: multiplicative decay per frame (0.98 = 2% loss per step)
        """
        # Diffuse density
        density_diffused = self.diffuse_field_kernal(self.densityfield2)
        self.densityfield2 = (1 - diffusion_rate) * self.densityfield2 + diffusion_rate * density_diffused
        
        # Apply exponential decay
        self.densityfield2 *= decay_rate
        
        # Clamp to reasonable range
        self.densityfield2 = cp.clip(self.densityfield2, -1.0, 1.0)

    def step_flowfield(self, dt=0.1, flow_diffusion_rate=0.05):
        """Update flow field using density gradients and eddy effects."""
        # Pressure gradient from density
        self.flowfield += self.calculate_gradientfield_kernal(self.densityfield) * dt
        
        # Eddy/curl contribution from vorticity
        curl_change = self.curlfield - self.curlfield_prev
        eddyflowfield = self.calculate_curlfield_kernal(curl_change)
        self.flowfield += eddyflowfield * dt * 0.5  # Scale down eddy effect
        
        # Apply dispersion (diffusion) to smooth flow field
        flow_diffused = self.diffuse_field_kernal(self.flowfield)
        self.flowfield = (1 - flow_diffusion_rate) * self.flowfield + flow_diffusion_rate * flow_diffused
        
        # Damping
        self.flowfield *= (1.0 - self.damping)

    def step_curlfield(self, dt=0.1, curl_diffusion_rate=0.1):
        """Update curl field from flow field with diffusion."""
        self.curlfield_prev = cp.copy(self.curlfield)
        # Compute curl of the current flow field
        self.curlfield = self.calculate_curlfield_kernal(self.flowfield)
        
        #calculate diffusion rate based on weighted average of the local density field density
        density_avg = self.fieldmean(self.densityfield)
        curl_diffusion_rate = curl_diffusion_rate * (2 / (1 + density_avg * density_avg) - 1) # diffusion rate lowers and becomes negative in high density areas
        curl_diffusion_rate = cp.stack((curl_diffusion_rate, curl_diffusion_rate, curl_diffusion_rate), axis=-1)
        # Apply diffusion to smooth out vorticity
        curl_diffused = self.diffuse_field_kernal(self.curlfield)
        self.curlfield = (1 - curl_diffusion_rate) * self.curlfield + curl_diffusion_rate * curl_diffused

    def step_particles(self, dt=0.1, density2_follow_strength=0.3):
        """Advect both particle sets using flow field and densityfield2 gradient.
        
        density2_follow_strength: how strongly particles follow densityfield2 gradient (0 to disable)
        """
        # Track previous positions for both sets
        self.particles_prev = cp.copy(self.particles)
        self.particles2_prev = cp.copy(self.particles2)
        
        # Get impulse from gradient field (flow field influences particle motion)
        flow_contrib = self.compute_gradient_contributions(self.particles, self.flowfield)
        flow_contrib2 = self.compute_gradient_contributions(self.particles2, self.flowfield)
        
        # Get gradient contribution from densityfield2 (particles follow density uphill)
        if density2_follow_strength > 0:
            density2_grad = self.calculate_gradientfield_kernal(self.densityfield2)
            density2_contrib = self.compute_gradient_contributions(self.particles, density2_grad)
            density2_contrib2 = self.compute_gradient_contributions(self.particles2, density2_grad)
            flow_contrib += density2_contrib * density2_follow_strength
            flow_contrib2 += density2_contrib2 * density2_follow_strength
        
        # Update velocity and position for both sets
        self.particles += flow_contrib * dt
        self.particles2 += flow_contrib2 * dt
        
        # Apply toroidal wrapping/boundary conditions
        # Domain bounds: [-L*NX/2, L*NX/2] in each dimension
        half_lx = self.LX * self.NX / 2
        half_ly = self.LY * self.NY / 2
        half_lz = self.LZ * self.NZ / 2
        
        # Wrap particles that go out of bounds (periodic boundary conditions)
        self.particles[..., 0] = cp.mod(self.particles[..., 0] + half_lx, self.LX * self.NX) - half_lx
        self.particles[..., 1] = cp.mod(self.particles[..., 1] + half_ly, self.LY * self.NY) - half_ly
        self.particles[..., 2] = cp.mod(self.particles[..., 2] + half_lz, self.LZ * self.NZ) - half_lz
        
        self.particles2[..., 0] = cp.mod(self.particles2[..., 0] + half_lx, self.LX * self.NX) - half_lx
        self.particles2[..., 1] = cp.mod(self.particles2[..., 1] + half_ly, self.LY * self.NY) - half_ly
        self.particles2[..., 2] = cp.mod(self.particles2[..., 2] + half_lz, self.LZ * self.NZ) - half_lz

    def inject_particles_to_density2(self, strength=0.5):
        """Inject particle density into densityfield2 at particle locations.
        
        strength: how much density each particle contributes (0 to 1 range)
        """
        # Map particle positions to grid indices
        half_lx = self.LX * self.NX / 2
        half_ly = self.LY * self.NY / 2
        half_lz = self.LZ * self.NZ / 2
        
        # Convert positions to normalized indices [0, N)
        ix = cp.mod(cp.floor((self.particles[:, 0] + half_lx) / self.LX).astype(cp.int32), self.NX)
        iy = cp.mod(cp.floor((self.particles[:, 1] + half_ly) / self.LY).astype(cp.int32), self.NY)
        iz = cp.mod(cp.floor((self.particles[:, 2] + half_lz) / self.LZ).astype(cp.int32), self.NZ)
        
        # Add density contribution from particles
        cp.add.at(self.densityfield2, (iz, iy, ix), strength)
        
        # Also inject particles2 (opposite set)
        ix2 = cp.mod(cp.floor((self.particles2[:, 0] + half_lx) / self.LX).astype(cp.int32), self.NX)
        iy2 = cp.mod(cp.floor((self.particles2[:, 1] + half_ly) / self.LY).astype(cp.int32), self.NY)
        iz2 = cp.mod(cp.floor((self.particles2[:, 2] + half_lz) / self.LZ).astype(cp.int32), self.NZ)
        
        cp.add.at(self.densityfield2, (iz2, iy2, ix2), strength)

    def inject_particles_to_density1(self, strength_pos=0.3, strength_neg=0.3):
        """Inject particles into densityfield with opposite signs by type.

        particles  (set1) add +strength_pos; particles2 add -strength_neg.
        """
        half_lx = self.LX * self.NX / 2
        half_ly = self.LY * self.NY / 2
        half_lz = self.LZ * self.NZ / 2

        # set1 indices
        ix = cp.mod(cp.floor((self.particles[:, 0] + half_lx) / self.LX).astype(cp.int32), self.NX)
        iy = cp.mod(cp.floor((self.particles[:, 1] + half_ly) / self.LY).astype(cp.int32), self.NY)
        iz = cp.mod(cp.floor((self.particles[:, 2] + half_lz) / self.LZ).astype(cp.int32), self.NZ)
        cp.add.at(self.densityfield, (iz, iy, ix), strength_pos)

        # set2 indices
        ix2 = cp.mod(cp.floor((self.particles2[:, 0] + half_lx) / self.LX).astype(cp.int32), self.NX)
        iy2 = cp.mod(cp.floor((self.particles2[:, 1] + half_ly) / self.LY).astype(cp.int32), self.NY)
        iz2 = cp.mod(cp.floor((self.particles2[:, 2] + half_lz) / self.LZ).astype(cp.int32), self.NZ)
        cp.add.at(self.densityfield, (iz2, iy2, ix2), -strength_neg)


    def clamp_magnitude_gpu(points, max_len):
        mag = cp.linalg.norm(points, axis=1, keepdims=True)
        scale = cp.minimum(1.0, max_len / (mag + 1e-9))
        return points * scale

    def compute_gradient_contributions(self, Points, GradientField):

        # Max and Min values for wrapping
        Max_X = self.LX * (self.NX + 1) / 2
        Max_Y = self.LY * (self.NY + 1) / 2
        Max_Z = self.LZ * (self.NZ + 1) / 2

        Min_X = -self.LX * (self.NX + 1) / 2
        Min_Y = -self.LY * (self.NY + 1) / 2
        Min_Z = -self.LZ * (self.NZ + 1) / 2
        # 
        Points_Copy = cp.copy(Points)
        Points_Copy[...,0] = cp.minimum(Points_Copy[...,0], Min_X)
        Points_Copy[...,0] = cp.maximum(Points_Copy[...,0], Max_X)
        Points_Copy[...,1] = cp.minimum(Points_Copy[...,1], Min_Y)
        Points_Copy[...,1] = cp.maximum(Points_Copy[...,1], Max_Y)
        Points_Copy[...,2] = cp.minimum(Points_Copy[...,2], Min_Z)
        Points_Copy[...,2] = cp.maximum(Points_Copy[...,2], Max_Z)


        # Get the voxel indices for each point

        ceil_X = cp.mod(cp.ceil((Points_Copy[:,0] / self.LX) + (self.NX - 2) / 2).astype(cp.int32), self.NX)
        ceil_Y = cp.mod(cp.ceil((Points_Copy[:,1] / self.LY) + (self.NY - 2) / 2).astype(cp.int32), self.NY)
        ceil_Z = cp.mod(cp.ceil((Points_Copy[:,2] / self.LZ) + (self.NZ - 2) / 2).astype(cp.int32), self.NZ)
        
        floor_X = cp.mod(cp.floor((Points_Copy[:,0] / self.LX) + (self.NX - 2) / 2).astype(cp.int32), self.NX)
        floor_Y = cp.mod(cp.floor((Points_Copy[:,1] / self.LY) + (self.NY - 2) / 2).astype(cp.int32), self.NY)
        floor_Z = cp.mod(cp.floor((Points_Copy[:,2] / self.LZ) + (self.NZ - 2) / 2).astype(cp.int32), self.NZ)

        # Compute contributions from the 8 surrounding voxels

        impulseContributions = cp.zeros_like(Points)

        SelPoints = cp.stack((ceil_X, ceil_Y, ceil_Z), axis=-1)
        SelGradients = GradientField[ceil_X, ceil_Y, ceil_Z]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))
        
        SelPoints = cp.stack((ceil_X, ceil_Y, floor_Z), axis=-1)
        SelGradients = GradientField[ceil_X, ceil_Y, floor_Z]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))
        
        SelPoints = cp.stack((ceil_X, floor_Y, ceil_Z), axis=-1)
        SelGradients = GradientField[ceil_X, floor_Y, ceil_Z]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))
        
        SelPoints = cp.stack((ceil_X, floor_Y, floor_Z), axis=-1)
        SelGradients = GradientField[ceil_X, floor_Y, floor_Z]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))
        
        SelPoints = cp.stack((floor_X, ceil_Y, ceil_Z), axis=-1)
        SelGradients = GradientField[floor_X, ceil_Y, ceil_Z]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))
        
        SelPoints = cp.stack((floor_X, ceil_Y, floor_Z), axis=-1)
        SelGradients = GradientField[floor_X, ceil_Y, floor_Z]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))
        
        SelPoints = cp.stack((floor_X, floor_Y, ceil_Z), axis=-1)
        SelGradients = GradientField[floor_X, floor_Y, ceil_Z]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))
        
        SelPoints = cp.stack((floor_X, floor_Y, floor_Z), axis=-1)
        SelGradients = GradientField[floor_X, floor_Y, floor_Z]
        R_vec = Points - SelPoints
        R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
        impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))

            
        return impulseContributions
    
    def compute_curl_contributions(self, Points, Points_Prev, CurlField):
        pass


    # -------------------------
    # Vertex generation for rendering
    # -------------------------
    def build_point_vertices(self, min_size=3.0, max_size=18.0):
        """
        Returns a CuPy array of shape (2*NZ*NY*NX, 7):
          [x, y, z, r, g, b, size]
        for point cloud visualization with two particle sets.
        First 50% colored by velocity, second 50% colored cyan.
        """
        n_particles = self.NZ * self.NY * self.NX
        verts = cp.empty((2 * n_particles, 7), dtype=cp.float32)

        # First particle set: positions and velocity-based colors
        verts[0:n_particles, 0:3] = self.particles
        A = self.particles - self.particles_prev
        mag_A = cp.sqrt(A[..., 0]**2 + A[..., 1]**2 + A[..., 2]**2) + 1e-6
        normA = A / mag_A[:, cp.newaxis]
        verts[0:n_particles, 3] = normA[..., 0] * 0.5 + 0.5
        verts[0:n_particles, 4] = normA[..., 1] * 0.5 + 0.5
        verts[0:n_particles, 5] = normA[..., 2] * 0.5 + 0.5
        # size scaled by speed magnitude
        speed = mag_A
        speed_norm = speed / (cp.max(speed) + 1e-6)
        size = min_size + (max_size - min_size) * speed_norm
        verts[0:n_particles, 6] = size

        # Second particle set: positions and fixed cyan color
        verts[n_particles:2*n_particles, 0:3] = self.particles2
        verts[n_particles:2*n_particles, 3] = 0.2  # cyan: low red
        verts[n_particles:2*n_particles, 4] = 0.9  # cyan: high green
        verts[n_particles:2*n_particles, 5] = 0.9  # cyan: high blue
        # give second set a slightly larger base size
        verts[n_particles:2*n_particles, 6] = min_size * 1.2

        return verts

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


