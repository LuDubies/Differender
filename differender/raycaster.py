import torch
from torch.cuda.amp import autocast, custom_fwd, custom_bwd
import taichi as ti
import taichi.math as tm
import matplotlib.pyplot as plt
import numpy as np
from enum import IntEnum
from typing import Union, Tuple, Optional
from torchvtk.rendering import plot_tfs


#################
### UTILITY   ###
#################
@ti.func
def isnan(x):
    ''' Checks if input is NaN

    Args:
        x (ti.field): Any taichi field

    Returns:
        ti.field: Boolean Taichi field indicating whether value is NaN.
    '''
    return not (x >= 0 or x <= 0)

@ti.func
def low_high_frac(x: float):
    ''' Returns the integer value below and above, as well as the frac

    Args:
        x (float): Floating point number

    Returns:
        int, int, float: floor, ceil, frac of `x`
    '''
    x = ti.max(x, 0.0)
    low = ti.floor(x)
    high = low + 1
    frac = x - float(low)
    return int(low), int(high), frac

@ti.func
def premultiply_alpha(rgba):
    ''' Premultiplies alpha value to RGB component

    Args:
        rgba (tm.vec4): Vec4 with RGBA

    Returns:
        tm.vec4: Vec4 with (RGB * A, A)
    '''
    rgba.xyz *= rgba.w
    return rgba

@ti.func
def get_entry_exit_points(look_from, view_dir, bl, tr):
    ''' Computes the entry and exit points of a given ray

    Args:
        look_from (tm.vec3): Camera Position as vec3
        view_dir (tm.vec): View direction as vec3, normalized
        bl (tm.vec3): Bottom left of the bounding box
        tr (tm.vec3): Top right of the bounding box

    Returns:
        float, float, bool: Distance to entry, distance to exit, bool whether box is hit
    '''
    dirfrac = 1.0 / view_dir
    t1 = (bl.x - look_from.x) * dirfrac.x
    t2 = (tr.x - look_from.x) * dirfrac.x
    t3 = (bl.y - look_from.y) * dirfrac.y
    t4 = (tr.y - look_from.y) * dirfrac.y
    t5 = (bl.z - look_from.z) * dirfrac.z
    t6 = (tr.z - look_from.z) * dirfrac.z

    tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6))
    tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6))
    hit = True
    if tmax < 0.0 or tmin > tmax:
        hit = False
    else:
        hit = True
    return tmin, tmax, hit


@ti.data_oriented
class VolumeRaycaster:
    def __init__(self, volume_resolution, render_resolution, tf_resolution=128, samples_per_step=128,
                       fov=30.0, nearfar=(0.1, 100.0), background_color=0.0, image_tiling=8, volume_tiling=4):
        ''' Initializes Volume Raycaster. Make sure to .set_volume(), .set_tf_tex(), .set_cam_pos().
        
        Args:
            volume_resolution (3-tuple of int): Resolution of the volume data (W, H, D)
            render_resolution (2-tuple of int): Resolution of rendering (W, H)
            tf_resolution (int): Resolution of the Transfer Function texture
            samples_per_step (int, optional): Number of samples along the ray that are computed at once. After this many samples an intermediate result is saved for backwards computation
            fov (float, optional): Field of view of the camera in degrees.
            nearfar (2-tuple of float, optional): Near and far plane distance for perspective projection
            background_color (tm.vec4 or float): Background color
            image_tiling (int): Size of the tiles for hierarchical image data structure
            volume_tiling (int): Size of the tiles/bricks for hierarchical volume data structure
        '''
        self.resolution = render_resolution
        self.aspect = render_resolution[0] / render_resolution[1]
        self.fov_deg, self.fov_rad = fov, np.radians(fov)
        self.near, self.far = nearfar
        self.image_tiling, self.volume_tiling = image_tiling, volume_tiling
        self.tiled_resolution = tuple(map(lambda d: d // image_tiling, render_resolution))
        self.samples_per_step = samples_per_step

        # Taichi Fields
        ## Camera
        self.entry, self.exit = ti.field(ti.f32), ti.field(ti.f32)
        self.ray_dirs = ti.Vector.field(3, ti.f32, needs_grad=True)
        self.n_samples = ti.field(ti.i32)
        self.cam_pos = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)

        ## Lighting
        self.light_pos = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.ambient, self.diffuse, self.specular = 0.4, 0.8, 0.3
        self.shininess = 32.0
        self.light_color = tm.vec3(1.0)

        ## Buffer
        self.buffer = ti.Vector.field(4, dtype=ti.f32, needs_grad=True)
        self.positions = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.output = ti.Vector.field(4, dtype=ti.f32, needs_grad=True)
        self.gt_depth = ti.field(ti.f32)
        self.max_samples = ti.field(ti.i32)
        # Memory Layout
        ti.root.dense(ti.ij, self.tiled_resolution
              ).dense(ti.ij, (image_tiling, image_tiling)
              ).place(self.entry, self.exit, self.ray_dirs, self.n_samples)

        ti.root.dense(ti.ijk, (*self.tiled_resolution, samples_per_step)
              ).dense(ti.ijk, (image_tiling, image_tiling, 1)
              ).place(self.buffer, self.positions)

        ti.root.place(self.cam_pos, self.cam_pos.grad)
        ti.root.place(self.light_pos, self.light_pos.grad)
        ti.root.place(self.gt_depth)
        ti.root.place(self.output)
        ti.root.place(self.max_samples)
        self.volume_snode, self.tf_snode, self.layer_snode = None, None, None

### Setters
    def set_volume(self, volume: torch.Tensor):
        ''' Sets the volume data in Taichi 
        
        Args:
            volume (torch.Tensor): Tensor of shape (W, H, D)
        '''
        assert volume.ndim == 3
        if self.volume_snode is not None:
            if tuple(self.volume.shape) != tuple(volume.shape):
                self.volume_snode.destroy() # Clear memory if it cannot be re-used
            else: # Memory already allocated and of correct shape, just write data
                self.volume.from_torch(volume.float())
                return
        # Allocate new memory for the volume
        fb = ti.FieldsBuilder()
        T = self.volume_tiling
        tiles = tuple(map(lambda d: d // T, volume.shape))

        self.volume = ti.field(ti.f32, needs_grad=volume.requires_grad)
        fb.dense(ti.ijk, tiles).dense(ti.ijk, (T,T,T)).place(self.volume)
        self.volume_snode = fb.finalize()
        self.volume.from_torch(volume.float())

    def set_tf_tex(self, tf_tex: torch.Tensor):
        ''' Sets the TF data in Taichi 
        
        Args:
            tf_tex (torch.Tensor): Tensor of shape (W, 4)
        '''
        assert tf_tex.ndim == 2 and tf_tex.size(1) == 4
        if self.tf_snode is not None:
            if tuple(self.tf_tex.shape) != tuple(tf_tex.shape):
                self.tf_snode.destory()
            else:
                self.tf_tex.from_torch(tf_tex.float())
                return
        fb = ti.FieldsBuilder()
        self.tf_tex = ti.Vector.field(4, ti.f32, needs_grad=tf_tex.requires_grad)
        fb.dense(ti.i, tf_tex.size(0)).place(self.tf_tex)
        self.tf_snode = fb.finalize()
        self.tf_tex.from_torch(tf_tex.float())

    def set_cam_pos(self, cam_pos: torch.Tensor):
        ''' Sets the camera look from in Taichi 
        
        Args:
            cam_pos (torch.Tensor): Tensor of shape (3,)
        '''
        assert cam_pos.ndim == 1 and cam_pos.size(0) == 3
        self.cam_pos.from_torch(cam_pos.float())

    def set_light_pos(self, light_pos: torch.Tensor):
        ''' Sets the light position in Taichi 
        
        Args:
            light_pos (torch.Tensor): Tensor of shape (3,)
        '''
        assert light_pos.ndim == 1 and light_pos.size(0) == 3
        self.light_pos.from_torch(light_pos.float())

    def set_gt_depth(self, depth: torch.Tensor):
        assert tuple(depth.shape) == self.resolution
        self.gt_depth.from_torch(depth.float())

    def allocate_checkpoints(self, num_steps: int):
        if self.layer_snode is not None:
            if self.layers.shape[0] != num_steps:
                self.layer_snode.destroy()
            else:
                self.layers.fill(0.0)
                return
        # Rebuild checkpoint layers from scratch
        fb = ti.FieldsBuilder()
        self.layers = ti.Vector.field(4, dtype=ti.f32, needs_grad=True)
        fb.pointer(ti.i, num_steps).bitmasked(ti.ij, self.resolution).place(self.layers)
        self.layer_snode = fb.finalize()

### Samplers
    @ti.func
    def sample_volume_intensity(self, pos):
        ''' Samples volume data at `pos` and trilinearly interpolates the value

        Args:
            pos (tm.vec3): Position to sample the volume in [-1, 1]^3

        Returns:
            float: Sampled interpolated intensity
        '''
        pos = tm.clamp(
            ((0.5 * pos) + 0.5), 0.0,
            1.0) * (tm.vec3(*self.volume.shape) - 1.0 - 1e-4)
        x_low, x_high, x_frac = low_high_frac(pos.x)
        y_low, y_high, y_frac = low_high_frac(pos.y)
        z_low, z_high, z_frac = low_high_frac(pos.z)

        x_high = min(x_high, ti.static(self.volume.shape[0] - 1))
        y_high = min(y_high, ti.static(self.volume.shape[1] - 1))
        z_high = min(z_high, ti.static(self.volume.shape[2] - 1))
        # on z_low
        v000 = self.volume[x_low, y_low, z_low]
        v100 = self.volume[x_high, y_low, z_low]
        x_val_y_low = tm.mix(v000, v100, x_frac)
        v010 = self.volume[x_low, y_high, z_low]
        v110 = self.volume[x_high, y_high, z_low]
        x_val_y_high = tm.mix(v010, v110, x_frac)
        xy_val_z_low = tm.mix(x_val_y_low, x_val_y_high, y_frac)
        # on z_high
        v001 = self.volume[x_low, y_low, z_high]
        v101 = self.volume[x_high, y_low, z_high]
        x_val_y_low = tm.mix(v001, v101, x_frac)
        v011 = self.volume[x_low, y_high, z_high]
        v111 = self.volume[x_high, y_high, z_high]
        x_val_y_high = tm.mix(v011, v111, x_frac)
        xy_val_z_high = tm.mix(x_val_y_low, x_val_y_high, y_frac)
        return tm.mix(xy_val_z_low, xy_val_z_high, z_frac)

    @ti.func
    def sample_volume_normal(self, pos):
        ''' Computes the volume normal at `pos` using central differences of trilinearly interpolated intensities

        Args:
            pos (tm.vec3): Position to sample the volume in [-1, 1]^3

        Returns:
            float: Sampled interpolated normal
        '''
        delta = 1e-3
        x_delta = tm.vec3(delta, 0.0, 0.0)
        y_delta = tm.vec3(0.0, delta, 0.0)
        z_delta = tm.vec3(0.0, 0.0, delta)
        dx = self.sample_volume_intensity(
            pos + x_delta) - self.sample_volume_intensity(pos - x_delta)
        dy = self.sample_volume_intensity(
            pos + y_delta) - self.sample_volume_intensity(pos - y_delta)
        dz = self.sample_volume_intensity(
            pos + z_delta) - self.sample_volume_intensity(pos - z_delta)
        return tm.vec3(dx, dy, dz).normalized()

    @ti.func
    def sample_transfer_function(self, intensity: float):
        ''' Samples the 1D transfer function at a given intensity value

        Args:
            intensity (float): Intensity in [0,1]

        Returns:
            tm.vec4: Color and opacity for given `intensity`
        '''
        length = ti.static(float(self.tf_tex.shape[0] - 1))
        low, high, frac = low_high_frac(intensity * length)
        return tm.mix(
            self.tf_tex[low],
            self.tf_tex[min(high, ti.static(self.tf_tex.shape[0] - 1))], frac)

### Camera
    @ti.kernel
    def compute_rays(self):
        ''' Compute Ray directions based on camera parameters. Sets the `self.ray_dirs` field storing a normalized viewing direction for each pixel. '''
        for i,j in self.ray_dirs:
            max_x = ti.static(float(self.ray_dirs.shape[0]))
            max_y = ti.static(float(self.ray_dirs.shape[1]))
            up = tm.vec3(0.0, 1.0, 0.0)
            # Shift to 0 center
            u = (float(i) + 0.5) / max_x - 0.5
            v = (float(j) + 0.5) / max_y - 0.5
            # Compute up & right, as well as near plane extents
            view_dir = (-self.cam_pos[None]).normalized()
            right = tm.cross(view_dir, up).normalized()
            up = tm.cross(right, view_dir).normalized()
            near_h = 2.0 * ti.tan(self.fov_rad) * self.near
            near_w = near_h * self.aspect
            near_m = self.cam_pos[None] + self.near * view_dir
            near_pos = near_m + u * near_w * right + v * near_h * up

            self.ray_dirs[i,j] = (near_pos - self.cam_pos[None]).normalized()

    @ti.kernel
    def compute_num_samples(self, sampling_rate: float):
        ''' Computes the Entry/Exit points for each pixel with the maximum number of samples. '''
        for i,j in self.entry:
            vol_diag = (tm.vec3(*self.volume.shape) - tm.vec3(1.0)).norm()
            bb_bl = tm.vec3(-1.0)
            bb_tr = tm.vec3( 1.0)
            tmin, tmax, hit = get_entry_exit_points(self.cam_pos[None], self.ray_dirs[i,j], bb_bl, bb_tr)

            if hit:
                self.n_samples[i, j] = ti.cast(ti.floor(sampling_rate * vol_diag * (tmax - tmin)), ti.int32) + 1
                self.max_samples[None] = max(self.max_samples[None], self.n_samples[i, j])
            else:
                self.n_samples[i, j] = 0
            self.entry[i,j] = tmin
            self.exit[i,j] = tmax

### Raycasting
    @ti.func
    def compute_current_pos(self, i, j, sample_idx):
        ''' Computes the current position along the ray, together with the current depth. Returns
        
        Args:
            i (int): Index to the image buffer (x dimension)
            j (int): Index to the image buffer (y dimension)
            sample_idx (int): Index along the ray
            
        Returns:
            Tuple[tm.vec3, float]: World position of the `sample_idx`-th step along the ray (`i`, `j`) 
        '''
        ray_len = self.exit[i, j] - self.entry[i, j]
        tmin = self.entry[i, j] + 0.5 * ray_len / self.n_samples[i, j]
        dist = tm.mix(tmin, self.exit[i, j], float(sample_idx) / float(self.n_samples[i, j] - 1))
        return self.cam_pos[None] + dist * self.ray_dirs[i, j]

    @ti.func
    def compute_shading(self, pos, rgba, view_direction):
        ''' Computes Phong shading at a given `pos` with a given `rgba` sample.
        
        Args:
            pos (tm.vec3): Position in the scene. (Used to sample normal)
            rgba (tm.vec4): RGB color and Alpha at the given `pos`
            view_direction (tm.vec3): View direction of the given ray.

        Returns:
            tm.vec4: Shaded and alpha-premultiplied RGBA value at `pos`
        '''
        light_dir = (pos - self.light_pos[None]).normalized()  # Direction to light source
        normal = self.sample_volume_normal(pos)
        n_dot_l = max(normal.dot(light_dir), 0.0)
        diffuse = self.diffuse * n_dot_l
        r = tm.reflect(light_dir, normal)  # Direction of reflected light
        r_dot_v = max(r.dot(-view_direction), 0.0)
        specular = self.specular * pow(r_dot_v, self.shininess)

        return tm.vec4(ti.min(1.0, diffuse + specular + self.ambient)
                        * rgba.xyz * rgba.w * self.light_color, rgba.w)

    @ti.kernel
    def raycast_step(self, step: int, sampling_rate: float):
        for i, j in self.n_samples:
            for sample_idx in ti.static(range(1, self.samples_per_step)):
                ray_idx = step * self.samples_per_step + sample_idx
                prev_idx = sample_idx - 1
                pos = self.compute_current_pos(i, j, ray_idx)
                intensity = self.sample_volume_intensity(pos)
                local_rgba = self.sample_transfer_function(intensity)
                local_rgba.w = 1.0 - ti.pow(1.0 - local_rgba.w, 1.0 / sampling_rate)
                self.positions[i, j, sample_idx] = pos
                if local_rgba.w < 1e-3 or self.buffer[i, j, prev_idx].w > 0.99: # Empty Space Skipping
                    self.buffer[i, j, sample_idx] = self.buffer[i, j, prev_idx]
                else:
                    #shaded_color = self.compute_shading(pos, local_rgba, self.ray_dirs[i, j])
                    shaded_color = local_rgba
                    self.buffer[i, j, sample_idx] = (1.0 - self.buffer[i, j, prev_idx].w) * shaded_color + self.buffer[i, j, prev_idx]

    def raycast(self, sampling_rate: float, ert_threshold: float = 0.99):
        self.compute_rays()
        self.compute_num_samples(sampling_rate)
        max_samples = self.max_samples.to_torch().item()
        steps = int(-1 * (-max_samples // self.samples_per_step)) # Ceil int division
        print(f'{steps} steps running now')
        self.allocate_checkpoints(steps)
        self.initialize_checkpoint_mask()
        print('go')
        for i in range(steps):
            print(f'Raycast {i}')
            self.raycast_step(i, sampling_rate=sampling_rate)
            print(f'Save Checkpoint {i}')
            self.save_checkpoint(i)
            if i < steps-1:
                print(f'Update Checkpoint masks {i}')
                self.update_checkpoint_mask(i+1, ert_threshold)
        print('Getting output')
        self.get_output()
        ## in backward:
        # for i in range(0, steps, -1):
        # Initialize buffer with i-th checkpoint
        # self.raycast_step(i)
        # self.raycast_step.grad(i)

    @ti.kernel
    def initialize_checkpoint_mask(self):
        for i, j in self.n_samples:
            if self.n_samples[i, j] > 0:
                ti.activate(self.layers[0].parent(), [i, j])
            else:
                ti.deactivate(self.layers[0].parent(), [i, j])

    @ti.kernel
    def update_checkpoint_mask(self, step: ti.i32, ert_threshold: float):
        for i, j in self.n_samples:
            if self.layers[step-1, i, j].w < ert_threshold:
                ti.activate(self.layers.parent(), [i, j])
            else:
                ti.deactivate(self.layers.parent(), [i, j])

    @ti.kernel
    def save_checkpoint(self, step: ti.i32):
        for i, j in self.n_samples:
            self.layers[step, i, j] = self.buffer[i, j, -1]
            self.buffer[i, j, 0] = self.buffer[i, j, -1]

    @ti.kernel
    def get_output(self, num_steps: ti.i32):
        ''' Retrieves the final image from the `buffer` to `output`. '''
        for i, j in self.output:
            for step in range(0, num_steps, -1):
                if ti.is_active(self.layers.parent(), [i, j]):
                    self.output[i, j] = self.layers[step, i, j]
                    break

    def clear_framebuffer(self):
        ''' Clears the framebuffer `output_rgba` and the `render_tape`'''
        self.max_valid_sample_step_count.fill(0)
        self.render_tape.fill(tm.vec4(0.0))
        self.valid_sample_step_count.fill(1)
        self.output_rgba.fill(tm.vec4(0.0))
        self.depth.fill(self.no_hit_depth)

    def clear_grad(self):
        ti.sync()
        self.cam_pos.grad.fill(tm.vec3(0.0))
        self.volume.grad.fill(0.0)
        self.tf_tex.grad.fill(tm.vec4(0.0))
        self.render_tape.grad.fill(tm.vec4(0.0))
        self.pos_tape.grad.fill(tm.vec3(0.0))
        self.output_rgba.grad.fill(tm.vec4(0.0))
        self.entry.grad.fill(0.0)
        self.exit.grad.fill(0.0)
        self.rays.grad.fill(tm.vec3(0.0))
        self.depth.grad.fill(0.0)

    @ti.kernel
    def grad_nan_to_num(self):
        self.cam_pos.grad[None] = tm.vec3(0.0)
        for i, j in self.rays.grad:
            if any(isnan(self.rays.grad[i, j])):
                self.rays.grad[i, j] = tm.vec3(0.0)
            if isnan(self.entry.grad[i, j]):
                self.entry.grad[i, j] = 0.0
            if isnan(self.exit.grad[i, j]):
                self.exit.grad[i, j] = 0.0
