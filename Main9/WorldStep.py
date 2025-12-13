import cupy as cp
import math

class WorldStep:




    def __init__(
            self,
            base_eddy=0.7,
            damping=0.02,
            dispersion=-0.5,
            particle_mass=1.0,
            particle_c=1.0,
            nx=50, ny=50, nz=50,
            lx=4.0, ly=4.0, lz=4.0,
            k1_size=3, k2_size=2, k3_size=3, k4_size=2,
            seed=0):
        
        self.NX = nx
        self.NY = ny
        self.NZ = nz
        self.LX = lx
        self.LY = ly
        self.LZ = lz
        self.seed = seed

        self.dispersion = dispersion
        self.particle_mass = particle_mass

        self.k1_size = k1_size
        self.k2_size = k2_size
        self.init_kernels()

        # Initialize fields

        self.particles = self.generate_initial_particles(nx, ny, nz,
            origin=(-lx * (nx - 1) / 2, -ly * (ny - 1) / 2, -lz * (nz - 1) / 2), spacing=(lx, ly, lz))
        self.particles_prev = cp.copy(self.particles)
        # initialize fields to zeros to avoid uninitialized memory
        self.curlfield = cp.zeros((self.NZ, self.NY, self.NX, 3), dtype=cp.float32)
        self.flowfield = cp.zeros((self.NZ, self.NY, self.NX, 3), dtype=cp.float32)
        self.densityfield = cp.zeros((self.NZ, self.NY, self.NX), dtype=cp.float32)

        self.init_densityfield()
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
        self.ahat1_weight = 0.0
        for i in range(-round((self.k1_size-1)/2), round((self.k1_size-1)/2)+1):
            for j in range(-round((self.k1_size-1)/2), round((self.k1_size-1)/2)+1):
                for k in range(-round((self.k1_size-1)/2), round((self.k1_size-1)/2)+1):
                    r = math.sqrt(i*i + j*j + k*k)
                    self.ahat1_weight += math.exp(self.dispersion * r)

        self.ahat2_weight = 0.0
        for i in range(-self.k2_size, self.k2_size+1):
            for j in range(-self.k2_size, self.k2_size+1):
                for k in range(-self.k2_size, self.k2_size+1):
                    r = math.sqrt(i*i + j*j + k*k)
                    self.ahat2_weight += r/(1 + r*r)

    def init_densityfield(self):
        # Properly initialize densityfield with random values in [-1,1]
        self.densityfield = cp.random.uniform(low=-1.0, high=1.0, size=(self.NZ, self.NY, self.NX)).astype(cp.float32)
        return self.densityfield
    
    def calculate_gradientfield_kernal(self, field):
        gradientfield = cp.zeros_like(field)
        gradientfield = cp.stack((gradientfield, gradientfield, gradientfield), axis=-1)
        for i in range(-self.k2_size, self.k2_size):
            for j in range(-self.k2_size, self.k2_size):
                for k in range(-self.k2_size, self.k2_size):
                    r = math.sqrt((i+0.5)*(i+0.5) + (j+0.5)*(j+0.5) + (k+0.5)*(k+0.5))
                    weight = r/(1 + r*r) / self.ahat2_weight
                    shifted = cp.roll(field, shift=(i, j, k), axis=(0, 1, 2))
                    gradientfield[..., 0] += weight * shifted * (i+0.5)
                    gradientfield[..., 1] += weight * shifted * (j+0.5)
                    gradientfield[..., 2] += weight * shifted * (k+0.5)
        return gradientfield
    
    def calculate_divergence_from_flow_kernal(self, field):
        divergencefield = cp.zeros((self.NZ, self.NY, self.NX), dtype=cp.float32)
        for i in range(-self.k2_size, self.k2_size):
            for j in range(-self.k2_size, self.k2_size):
                for k in range(-self.k2_size, self.k2_size):
                    r = math.sqrt((i - 0.5)*(i - 0.5) + (j - 0.5)*(j - 0.5) + (k - 0.5)*(k - 0.5))
                    weight = r/(1 + r*r) / self.ahat2_weight
                    shifted = cp.roll(field, shift=(i, j, k), axis=(0, 1, 2))
                    divergencefield += weight * (shifted[...,0] * (i - 0.5) + shifted[...,1] * (j - 0.5) + shifted[...,2] * (k - 0.5))
        return divergencefield
    
    def calculate_curlfield_kernal(self, field):
        curlfield = cp.empty((self.NZ, self.NY, self.NX, 3), dtype=cp.float32)
        for i in range(-(self.k1_size-1)/2, (self.k1_size-1)/2 + 1):
            for j in range(-(self.k1_size-1)/2, (self.k1_size-1)/2 + 1):
                for k in range(-(self.k1_size-1)/2, (self.k1_size-1)/2 + 1):
                    r = math.sqrt(i*i + j*j + k*k)
                    weight = math.exp(self.dispersion * r) / self.ahat1_weight
                    shifted = cp.roll(field, shift=(i, j, k), axis=(0, 1, 2))
                    # Compute contributions to curl here
                    r_vec = cp.array([i, j, k], dtype=cp.float32)
                    curlfield += weight * cp.cross(r_vec, shifted)
        return curlfield
    

    def diffuse_field_kernal(self, field):
        diffused_field = cp.zeros_like(field)
        for i in range(-round((self.k1_size-1)/2), round((self.k1_size-1)/2)+1):
            for j in range(-round((self.k1_size-1)/2), round((self.k1_size-1)/2)+1):
                for k in range(-round((self.k1_size-1)/2), round((self.k1_size-1)/2)+1):
                    r = math.sqrt(i*i + j*j + k*k)
                    weight = math.exp(self.dispersion * r) / self.ahat1_weight
                    shifted = cp.roll(field, shift=(i, j, k), axis=(0, 1, 2))
                    diffused_field += weight * shifted
        return diffused_field

    # -------------------------
    # Simulation step functions

    def step(self, dt=0.1):
        self.step_densityfield(dt)
        gradientfield = self.calculate_gradientfield()
        self.step_flowfield(dt, gradientfield)
        self.step_curlfield(dt)
        self.step_particles(dt)
        pass
    
    def step_densityfield(self, dt=0.1, diffusion_rate=0.1):
        # diffusion
        # diffusion
        density_diffused = self.diffuse_field_kernal(self.densityfield)
        self.densityfield = (1 - diffusion_rate) * self.densityfield + diffusion_rate * density_diffused

        self.densityfield += cp.sum(self.calculate_gradientfield_kernal(self.flowfield), (3,4)) * dt
        pass

    def step_flowfield(self, gradientfield, dt=0.1):
        self.flowfield += self.calculate_gradientfield_kernal(self.densityfield) * dt
        eddyflowfield += self.calculate_curlfield_kernal(self.curlfield)
        pass

    def step_curlfield(self, dt=0.1):
        pass

    def step_particles(self, dt=0.1):
        particleMotion = self.particles - self.particles_prev

        impulse = self.compute_gradient_contributions(self.particles, self.calculate_gradientfield(), self.curlfield)
        print(impulse)

        particleMotion += impulse * (dt / self.particle_mass)

        self.particles_prev = cp.copy(self.particles)

        self.particles += particleMotion * dt
        self.particles = cp.mod(self.particles + cp.array([[(self.LX * self.NX / 2), (self.LY * self.NY) / 2, (self.LZ * self.NZ) / 2]]), cp.array([[self.LX * self.NX, self.LY * self.NY, self.LZ * self.NZ]])) - cp.array([self.LX * self.NX / 2, self.LY * self.NY / 2, self.LZ * self.NZ / 2])


        pass


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

        for i in range(8):
            if(i==0):
                SelPoints = cp.stack((ceil_X, ceil_Y, ceil_Z), axis=-1)
                SelGradients = GradientField[ceil_X, ceil_Y, ceil_Z]
                R_vec = Points - SelPoints
                R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
                #print(SelGradients.shape)
                #print(SelPoints.shape)
                #print(GradientField.shape)
                #print(R_factor.shape)
                #print(impulseContributions.shape)
                #print(cp.stack((R_factor, R_factor, R_factor), 1).shape)
                impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))
            elif(i==1):
                SelPoints = cp.stack((ceil_X, ceil_Y, floor_Z), axis=-1)
                SelGradients = GradientField[ceil_X, ceil_Y, floor_Z]
                R_vec = Points - SelPoints
                R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
                impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))
            elif(i==2):
                SelPoints = cp.stack((ceil_X, floor_Y, ceil_Z), axis=-1)
                SelGradients = GradientField[ceil_X, floor_Y, ceil_Z]
                R_vec = Points - SelPoints
                R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
                impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))
            elif(i==3):
                SelPoints = cp.stack((ceil_X, floor_Y, floor_Z), axis=-1)
                SelGradients = GradientField[ceil_X, floor_Y, floor_Z]
                R_vec = Points - SelPoints
                R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
                impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))
            elif(i==4):
                SelPoints = cp.stack((floor_X, ceil_Y, ceil_Z), axis=-1)
                SelGradients = GradientField[floor_X, ceil_Y, ceil_Z]
                R_vec = Points - SelPoints
                R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
                impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))
            elif(i==5):
                SelPoints = cp.stack((floor_X, ceil_Y, floor_Z), axis=-1)
                SelGradients = GradientField[floor_X, ceil_Y, floor_Z]
                R_vec = Points - SelPoints
                R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
                impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))
            elif(i==6):
                SelPoints = cp.stack((floor_X, floor_Y, ceil_Z), axis=-1)
                SelGradients = GradientField[floor_X, floor_Y, ceil_Z]
                R_vec = Points - SelPoints
                R_factor = 1 + (R_vec[...,0] * R_vec[...,0] + R_vec[...,1] * R_vec[...,1] + R_vec[...,2] * R_vec[...,2])
                impulseContributions += cp.divide(SelGradients, cp.stack((R_factor, R_factor, R_factor), 1))
            elif(i==7):
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
    def build_point_vertices(self):
        """
        Returns a CuPy array of shape (NZ, NY, NX, 6):
          [x, y, z, r, g, b]
        for point cloud visualization.
        """
        verts = cp.empty((self.NZ * self.NY * self.NX, 6), dtype=cp.float32)

        # Positions
        verts[..., 0:3] = self.particles

        # Colors
        A = self.particles - self.particles_prev
        mag_A = cp.sqrt(A[..., 0]**2 + A[..., 1]**2 + A[..., 2]**2) + 1e-6
        normA = A / mag_A[:, cp.newaxis]

        verts[..., 3] = normA[..., 0] * 0.5 + 0.5
        verts[..., 4] = normA[..., 1] * 0.5 + 0.5
        verts[..., 5] = normA[..., 2] * 0.5 + 0.5

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


