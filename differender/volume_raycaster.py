from numpy.core.fromnumeric import shape
from taichi_glsl.sampling import D
import torch
from torch.cuda.amp import autocast, custom_fwd, custom_bwd
import taichi as ti
import taichi_glsl as tl
import matplotlib.pyplot as plt
import numpy as np
from enum import IntEnum
from typing import Union, Tuple

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
    rgba.xyz *= rgba.w
    return rgba

@ti.func
def get_entry_exit_points(look_from, view_dir, bl, tr):
    ''' Computes the entry and exit points of a given ray

    Args:
        look_from (tl.vec3): Camera Position as vec3
        view_dir (tl.vec): View direction as vec3, normalized
        bl (tl.vec3): Bottom left of the bounding box
        tr (tl.vec3): Top right of the bounding box

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
class VolumeRaycaster():
    def __init__(self,
                 volume_resolution,
                 render_resolution,
                 max_samples=512,
                 tf_resolution=128,
                 fov=30.0,
                 nearfar=(0.1, 100.0),
                 background_color=0.0,):
        ''' Initializes Volume Raycaster. Make sure to .set_volume() and .set_tf_tex() after initialization

        Args:
            volume_resolution (3-tuple of int): Resolution of the volume data (w,h,d)
            render_resolution (2-tuple of int): Resolution of the rendering (w,h)
            tf_resolution (int): Resolution of the transfer function texture
            fov (float, optional): Field of view of the camera in degrees. Defaults to 60.0.
            nearfar (2-tuple of float, optional): Near and far plane distance used for perspective projection. Defaults to (0.1, 100.0).
        '''
        self.resolution = render_resolution
        self.aspect = render_resolution[0] / render_resolution[1]
        self.fov_deg = fov
        self.fov_rad = np.radians(fov)
        self.near, self.far = nearfar
        # Taichi Fields
        self.volume = ti.field(ti.f32, needs_grad=True)
        self.tf_tex = ti.Vector.field(4, dtype=ti.f32, needs_grad=True)
        self.render_tape = ti.Vector.field(4, dtype=ti.f32, needs_grad=True)
        self.output_rgba = ti.Vector.field(4, dtype=ti.f32, needs_grad=True)
        self.pos_tape = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.valid_sample_step_count = ti.field(ti.i32)
        self.sample_step_nums = ti.field(ti.i32)
        self.entry = ti.field(ti.f32)
        self.exit = ti.field(ti.f32)
        self.loss = ti.field(ti.f32, shape=(), needs_grad=True)
        self.rays = ti.Vector.field(3, dtype=ti.f32)
        self.max_valid_sample_step_count = ti.field(ti.i32, ())
        self.max_samples = max_samples
        self.background_color = background_color
        self.ambient = 0.4
        self.diffuse = 0.8
        self.specular = 0.3
        self.shininess = 32.0
        self.light_color = tl.vec3(1.0)
        self.cam_pos = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.cam_pos_field = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        
        self.ground_truth_depth = ti.field(shape=self.resolution, dtype=ti.f32)    

        # add depth tape and depth_out
        self.depth = ti.field(ti.f32, needs_grad=True)
        self.depth_tape = ti.field(ti.f32, needs_grad=True)

        volume_resolution = tuple(map(lambda d: d // 4, volume_resolution))
        render_resolution = tuple(map(lambda d: d // 8, render_resolution))
        ti.root.dense(ti.ijk, volume_resolution).dense(ti.ijk, (4, 4, 4)).place(self.volume, self.volume.grad)
        ti.root.dense(ti.ijk, (*render_resolution, max_samples)).dense(
            ti.ijk, (8, 8, 1)).place(self.render_tape, self.render_tape.grad, self.pos_tape, self.pos_tape.grad)

        ti.root.dense(ti.ij, render_resolution).dense(ti.ij, (8, 8)).place(self.valid_sample_step_count, self.sample_step_nums)
        ti.root.dense(ti.ij, render_resolution).dense(ti.ij, (8, 8)).place(self.output_rgba, self.output_rgba.grad)
        ti.root.dense(ti.ij, render_resolution).dense(ti.ij, (8, 8)).place(self.entry, self.entry.grad, self.exit, self.exit.grad)
        ti.root.dense(ti.ij, render_resolution).dense(ti.ij, (8, 8)).place(self.rays, self.rays.grad)
        ti.root.dense(ti.i, (tf_resolution)).place(self.tf_tex, self.tf_tex.grad)
        ti.root.place(self.cam_pos, self.cam_pos.grad)
        ti.root.dense(ti.ij, render_resolution).dense(ti.ij, (8, 8)).place(self.cam_pos_field, self.cam_pos_field.grad)
        ti.root.dense(ti.ij, render_resolution).dense(ti.ij, (8, 8)).place(self.depth, self.depth.grad)
        ti.root.dense(ti.ijk, (*render_resolution, max_samples)).dense(
            ti.ijk, (8, 8, 1)).place(self.depth_tape, self.depth_tape.grad)

    def set_volume(self, volume):
        self.volume.from_torch(volume.float())

    def set_tf_tex(self, tf_tex):
        self.tf_tex.from_torch(tf_tex.float())

    def set_cam_pos(self, cam_pos):
        self.cam_pos.from_torch(cam_pos.float())

    def set_gtd(self, gtd):
        self.ground_truth_depth.from_numpy(gtd)

    @ti.func # TODO: remove
    def get_ray_direction(self, orig, view_dir, x: float, y: float):
        ''' Compute ray direction for perspecive camera.

        Args:
            orig (tl.vec3): Camera position
            view_dir (tl.vec3): View direction, normalized
            x (float): Image coordinate in [0,1] along width
            y (float): Image coordinate in [0,1] along height

        Returns:
            tl.vec3: Ray direction from camera origin to pixel specified through `x` and `y`
        '''
        u = x - 0.5
        v = y - 0.5

        up = ti.static(tl.vec3(0.0, 1.0, 0.0))
        right = tl.cross(view_dir, up).normalized()
        up = tl.cross(right, view_dir).normalized()
        near_h = 2.0 * ti.tan(self.fov_rad) * self.near
        near_w = near_h * self.aspect
        near_m = orig + self.near * view_dir
        near_pos = near_m + u * near_w * right + v * near_h * up

        return (near_pos - orig).normalized()

    @ti.func
    def sample_volume_trilinear(self, pos):
        ''' Samples volume data at `pos` and trilinearly interpolates the value

        Args:
            pos (tl.vec3): Position to sample the volume in [-1, 1]^3

        Returns:
            float: Sampled interpolated intensity
        '''
        pos = tl.clamp(
            ((0.5 * pos) + 0.5), 0.0,
            1.0) * ti.static(tl.vec3(*self.volume.shape) - 1.0 - 1e-4)
        x_low, x_high, x_frac = low_high_frac(pos.x)
        y_low, y_high, y_frac = low_high_frac(pos.y)
        z_low, z_high, z_frac = low_high_frac(pos.z)

        x_high = min(x_high, ti.static(self.volume.shape[0] - 1))
        y_high = min(y_high, ti.static(self.volume.shape[1] - 1))
        z_high = min(z_high, ti.static(self.volume.shape[2] - 1))
        # on z_low
        v000 = self.volume[x_low, y_low, z_low]
        v100 = self.volume[x_high, y_low, z_low]
        x_val_y_low = tl.mix(v000, v100, x_frac)
        v010 = self.volume[x_low, y_high, z_low]
        v110 = self.volume[x_high, y_high, z_low]
        x_val_y_high = tl.mix(v010, v110, x_frac)
        xy_val_z_low = tl.mix(x_val_y_low, x_val_y_high, y_frac)
        # on z_high
        v001 = self.volume[x_low, y_low, z_high]
        v101 = self.volume[x_high, y_low, z_high]
        x_val_y_low = tl.mix(v001, v101, x_frac)
        v011 = self.volume[x_low, y_high, z_high]
        v111 = self.volume[x_high, y_high, z_high]
        x_val_y_high = tl.mix(v011, v111, x_frac)
        xy_val_z_high = tl.mix(x_val_y_low, x_val_y_high, y_frac)
        return tl.mix(xy_val_z_low, xy_val_z_high, z_frac)

    @ti.func
    def get_volume_normal(self, pos):
        delta = 1e-3
        x_delta = tl.vec3(delta, 0.0, 0.0)
        y_delta = tl.vec3(0.0, delta, 0.0)
        z_delta = tl.vec3(0.0, 0.0, delta)
        dx = self.sample_volume_trilinear(
            pos + x_delta) - self.sample_volume_trilinear(pos - x_delta)
        dy = self.sample_volume_trilinear(
            pos + y_delta) - self.sample_volume_trilinear(pos - y_delta)
        dz = self.sample_volume_trilinear(
            pos + z_delta) - self.sample_volume_trilinear(pos - z_delta)
        return tl.vec3(dx, dy, dz).normalized()

    @ti.func
    def apply_transfer_function(self, intensity: float):
        ''' Applies a 1D transfer function to a given intensity value

        Args:
            intensity (float): Intensity in [0,1]

        Returns:
            tl.vec4: Color and opacity for given `intensity`
        '''
        length = ti.static(float(self.tf_tex.shape[0] - 1))
        low, high, frac = low_high_frac(intensity * length)
        return tl.mix(
            self.tf_tex[low],
            self.tf_tex[min(high, ti.static(self.tf_tex.shape[0] - 1))], frac)

    @ti.kernel
    def compute_rays(self):
        for i,j in self.rays:
            max_x = ti.static(float(self.rays.shape[0]))
            max_y = ti.static(float(self.rays.shape[1]))
            up = ti.static(tl.vec3(0.0, 1.0, 0.0))
            # Shift to 0 center
            u = (float(i) + 0.5) / max_x - 0.5
            v = (float(j) + 0.5) / max_y - 0.5
            # Compute up & right, as well as near plane extents
            view_dir = (-self.cam_pos[None]).normalized()
            right = tl.cross(view_dir, up).normalized()
            up = tl.cross(right, view_dir).normalized()
            near_h = 2.0 * ti.tan(self.fov_rad) * self.near
            near_w = near_h * self.aspect
            near_m = self.cam_pos[None] + self.near * view_dir
            near_pos = near_m + u * near_w * right + v * near_h * up

            self.rays[i,j] = (near_pos - self.cam_pos[None]).normalized()

    @ti.kernel
    def compute_rays_backward(self):
        for i,j in self.rays:
            if tl.length(self.rays[i,j]) > 1e-8:
                updated_ray = (self.rays[i,j] - self.rays.grad[i,j]).normalized()
                pix_pos = self.cam_pos[None] + self.rays[i, j] * self.entry[i, j]
                new_campos = pix_pos - updated_ray * (self.entry[i,j] - self.entry.grad[i,j])
                self.cam_pos.grad[None] += (self.cam_pos[None] - new_campos) #/ ti.static(self.rays.shape[0] * self.rays.shape[1])
                self.cam_pos_field[i,j] = new_campos

    @ti.kernel
    def compute_intersections(self, sampling_rate: float, jitter: int):
        for i,j in self.entry:
            vol_diag = ti.static((tl.vec3(*self.volume.shape) - tl.vec3(1.0)).norm())
            bb_bl = ti.static(tl.vec3(-1.0))
            bb_tr = ti.static(tl.vec3( 1.0))
            tmin, tmax, hit = get_entry_exit_points(self.cam_pos[None], self.rays[i,j], bb_bl, bb_tr)

            n_samples = 1
            if hit:
                n_samples = ti.cast(ti.floor(sampling_rate * vol_diag * (tmax - tmin)), ti.int32) + 1
            self.entry[i,j] = tmin
            self.exit[i,j] = tmax
            self.sample_step_nums[i,j] = min(self.max_samples, n_samples)

    @ti.kernel
    def compute_intersections_nondiff(self, sampling_rate: float, jitter: int):
        for i,j in self.entry:
            vol_diag = ti.static((tl.vec3(*self.volume.shape) - tl.vec3(1.0)).norm())
            bb_bl = ti.static(tl.vec3(-1.0))
            bb_tr = ti.static(tl.vec3( 1.0))
            tmin, tmax, hit = get_entry_exit_points(self.cam_pos[None], self.rays[i,j], bb_bl, bb_tr)

            n_samples = 1
            if hit:
                n_samples = ti.cast(ti.floor(sampling_rate * vol_diag * (tmax - tmin)), ti.int32) + 1
            self.entry[i,j] = tmin
            self.exit[i,j] = tmax
            self.sample_step_nums[i,j] = n_samples

    @ti.kernel
    def raycast(self, sampling_rate: float):
        ''' Produce a rendering. Run compute_entry_exit first! '''
        for i, j in self.valid_sample_step_count:  # For all pixels
            for sample_idx in range(1, self.sample_step_nums[i, j]):
                look_from = self.cam_pos[None]
                if self.render_tape[i, j, sample_idx -1].w < 0.99 and sample_idx < ti.static(self.max_samples):
                    tmax = self.exit[i, j]
                    n_samples = self.sample_step_nums[i, j]
                    ray_len = (tmax - self.entry[i, j])
                    tmin = self.entry[
                        i,
                        j] + 0.5 * ray_len / n_samples  # Offset tmin as t_start
                    vd = self.rays[i, j]
                    self.pos_tape[i,j, sample_idx] = look_from + tl.mix(
                        tmin, tmax, float(sample_idx) / float(n_samples - 1)) * vd  # Current Pos
                    light_pos = look_from + tl.vec3(0.0, 1.0, 0.0)
                    intensity = self.sample_volume_trilinear(self.pos_tape[i,j, sample_idx])
                    sample_color = self.apply_transfer_function(intensity)
                    opacity = 1.0 - ti.pow(1.0 - sample_color.w,
                                           1.0 / sampling_rate)
                    
                    if sample_color.w > 1e-3 and self.depth_tape[i, j, sample_idx - 1] == 0.0:
                        self.depth_tape[i, j, sample_idx] = tl.mix(tmin, tmax, float(sample_idx) / float(n_samples - 1))     
                    else:
                        self.depth_tape[i, j, sample_idx] = self.depth_tape[i, j, sample_idx-1]


                    normal = self.get_volume_normal(self.pos_tape[i,j, sample_idx])
                    light_dir = (
                        self.pos_tape[i,j, sample_idx] -
                        light_pos).normalized()  # Direction to light source
                    n_dot_l = max(normal.dot(light_dir), 0.0)
                    diffuse = self.diffuse * n_dot_l
                    r = tl.reflect(light_dir,
                                   normal)  # Direction of reflected light
                    r_dot_v = max(r.dot(-vd), 0.0)
                    specular = self.specular * pow(r_dot_v, self.shininess)
                    shaded_color = tl.vec4(
                        ti.min(1.0, diffuse + specular + self.ambient) *
                        sample_color.xyz * opacity * self.light_color, opacity)
                    self.render_tape[i, j, sample_idx] = (
                        1.0 - self.render_tape[i, j, sample_idx - 1].w
                    ) * shaded_color + self.render_tape[i, j, sample_idx - 1]
                    self.valid_sample_step_count[i, j] += 1
                else:
                    self.render_tape[i, j, sample_idx] = self.render_tape[
                        i, j, sample_idx - 1]

    @ti.kernel
    def raycast_nondiff(self, sampling_rate: float):
        ''' Raycasts in a non-differentiable (but faster and cleaner) way. Use `get_final_image_nondiff` with this.

        Args:
            sampling_rate (float): Sampling rate (multiplier with Nyquist frequence)
        '''
        for i, j in self.valid_sample_step_count:  # For all pixels
            for cnt in range(self.sample_step_nums[i, j]):
                look_from = self.cam_pos[None]
                if self.render_tape[i, j, 0].w < 0.99:
                    tmax = self.exit[i, j]
                    n_samples = self.sample_step_nums[i, j]
                    ray_len = (tmax - self.entry[i, j])
                    tmin = self.entry[
                        i,
                        j] + 0.5 * ray_len / n_samples  # Offset tmin as t_start
                    vd = self.rays[i, j]
                    pos = look_from + tl.mix(
                        tmin, tmax,
                        float(cnt) / float(n_samples - 1)) * vd  # Current Pos
                    light_pos = look_from + tl.vec3(0.0, 1.0, 0.0)
                    intensity = self.sample_volume_trilinear(pos)
                    sample_color = self.apply_transfer_function(intensity)
                    opacity = 1.0 - ti.pow(1.0 - sample_color.w,
                                           1.0 / sampling_rate)
                    # if sample_color.w > 1e-3:
                    normal = self.get_volume_normal(pos)
                    light_dir = (pos - light_pos).normalized(
                    )  # Direction to light source
                    n_dot_l = max(normal.dot(light_dir), 0.0)
                    diffuse = self.diffuse * n_dot_l
                    r = tl.reflect(light_dir,
                                   normal)  # Direction of reflected light
                    r_dot_v = max(r.dot(-vd), 0.0)
                    specular = self.specular * pow(r_dot_v, self.shininess)
                    shaded_color = tl.vec4(
                        (diffuse + specular + self.ambient) *
                        sample_color.xyz * opacity * self.light_color,
                        opacity)
                    self.render_tape[
                        i, j,
                        0] = (1.0 - self.render_tape[i, j, 0].w
                              ) * shaded_color + self.render_tape[i, j, 0]

    @ti.kernel
    def get_final_image_nondiff(self):
        ''' Retrieves the final image from the tape if the `raycast_nondiff` method was used. '''
        for i, j in self.valid_sample_step_count:
            valid_sample_step_count = self.valid_sample_step_count[i, j] - 1
            rgba = self.render_tape[i, j, 0]
            self.output_rgba[i, j] += rgba
            if valid_sample_step_count > self.max_valid_sample_step_count[None]:
                self.max_valid_sample_step_count[
                    None] = valid_sample_step_count

    @ti.kernel
    def get_final_image(self):
        ''' Retrieves the final image from the `render_tape` to `output_rgba`. '''
        for i, j in self.valid_sample_step_count:
            valid_sample_step_count = self.valid_sample_step_count[i, j] - 1
            ns = tl.clamp(self.sample_step_nums[i, j], 1, self.max_samples-1)
            rgba = self.render_tape[i, j, ns - 1]
            self.output_rgba[i, j] += tl.vec4(tl.mix(rgba.xyz, tl.vec3(self.background_color), 1.0 - rgba.w), rgba.w)
            if valid_sample_step_count > self.max_valid_sample_step_count[None]:
                self.max_valid_sample_step_count[
                    None] = valid_sample_step_count

    @ti.kernel
    def get_depth_image(self):
        for i, j in self.valid_sample_step_count:
            ns = tl.clamp(self.sample_step_nums[i, j], 1, self.max_samples-1)
            self.depth[i, j] += self.depth_tape[i, j, ns - 1]

    def clear_framebuffer(self):
        ''' Clears the framebuffer `output_rgba` and the `render_tape`'''
        self.max_valid_sample_step_count.fill(0)
        self.render_tape.fill(tl.vec4(0.0))
        self.valid_sample_step_count.fill(1)
        self.output_rgba.fill(tl.vec4(0.0))
        self.depth.fill(0.0)
        self.depth_tape.fill(0.0)

    def clear_grad(self):
        ti.sync()
        self.cam_pos.grad.fill(tl.vec3(0.0))
        self.volume.grad.fill(0.0)
        self.tf_tex.grad.fill(tl.vec4(0.0))
        self.render_tape.grad.fill(tl.vec4(0.0))
        self.pos_tape.grad.fill(tl.vec3(0.0))
        self.output_rgba.grad.fill(tl.vec4(0.0))
        self.entry.grad.fill(0.0)
        self.exit.grad.fill(0.0)
        self.rays.grad.fill(tl.vec3(0.0))
        self.depth.grad.fill(0.0)
        self.depth_tape.grad.fill(0.0)

    @ti.kernel
    def grad_nan_to_num(self):
        self.cam_pos.grad[None] = tl.vec3(0.0)
        for i,j in self.rays.grad:
            if any(tl.isnan(self.rays.grad[i,j])):
                self.rays.grad[i,j] = tl.vec3(0.0)
            if tl.isnan(self.entry.grad[i,j]):
                self.entry.grad[i,j] = 0.0
            if tl.isnan(self.exit.grad[i,j]):
                self.exit.grad[i,j] = 0.0

    @ti.kernel
    def compute_loss(self):
        ''' exclude loss for rays missing the volume '''
        for i, j in self.valid_sample_step_count:
            if self.output_rgba[i, j].w < 1E-8:
                self.ground_truth_depth[i, j] = 1

        for i,j in self.valid_sample_step_count:
            # calculate the sample where we expect to have max opacity based on gt_depth
            # could use better calculation to go from depth -> sample but like this we cant miss the calculated ra part
            if self.ground_truth_depth[i, j] < 1:
                expected_sample = ti.cast(self.ground_truth_depth[i, j] * self.sample_step_nums[i, j], ti.i32)      # ceil resulting in autodiff error, typecasting does not??
                actual_opacity = self.render_tape[i, j, expected_sample].w
                self.loss[None] += (1 - actual_opacity)**2 / (self.resolution[0] * self.resolution[1])

    @ti.kernel
    def loss_grad(self):
        ''' manually calculate loss from distance to ground truth depth'''
        for i,j in self.valid_sample_step_count:
            if self.ground_truth_depth[i, j] < 1:
                expected_sample = ti.cast(self.ground_truth_depth[i, j] * self.sample_step_nums[i, j], ti.i32)      # ceil resulting in autodiff error, typecasting does not??
                actual_opacity = self.render_tape[i, j, expected_sample].w
                opacity_grad_at_ijs = -2 * (1 - actual_opacity) / (self.resolution[0] * self.resolution[1])
                self.render_tape.grad[i, j, expected_sample] += tl.vec4(0.0, 0.0, 0.0, opacity_grad_at_ijs)

    @ti.kernel
    def apply_tf_grad(self):
        for i in range(self.tf_tex.shape[0]):
            self.tf_tex[i] -= self.tf_tex.grad[i]

    @ti.kernel
    def clear_loss(self):
        self.loss[None] = 0.0

    def visualize_ray(self, rgba: Union[str, None] = None, i: int = None, j: int = None, filename: str = None):
        r = rgba is None or 'r' in rgba
        g = rgba is None or 'g' in rgba
        b = rgba is None or 'b' in rgba
        a = rgba is None or 'a' in rgba

        def trim_to_volume(data: np.array) -> Tuple[int, int]:
            not_zero = data > 0.0001
            first_idx = not_zero.argmax()
            last_idx = not_zero.size - not_zero[::-1].argmax() -1

            return first_idx, last_idx

        def get_deriv(data: np.array) -> np.array:
            return np.clip(np.gradient(data), 0, None)

        def plot_data(data: np.array, axes: Tuple[plt.axes, plt.axes], color_fmt: str):
            deriv = get_deriv(data)
            f, l = trim_to_volume(data)
            fd, ld = trim_to_volume(deriv)
            f = min(f, fd)
            l = max(l, ld)
            axes[0].plot(range(f, l+1), data[f:l+1], color_fmt)
            axes[1].plot(range(f, l+1), deriv[f:l+1], color_fmt)

        if i is None:
            i = self.resolution[0] // 2
        if j is None:
            j = self.resolution[1] // 2
        np_tape = self.render_tape.to_numpy()[i, j, :, :]
        fig, (ax, dx) = plt.subplots(2, 1)
        if r:
            plot_data(np_tape[:, 0], (ax, dx), 'r-')
        if g:
            plot_data(np_tape[:, 1], (ax, dx), 'g-')
        if b:
            plot_data(np_tape[:, 2], (ax, dx), 'b-')
        if a:
            plot_data(np_tape[:, 3], (ax, dx), 'k-')
        if filename is None:
            fig.savefig('demo.png', bbox_inches='tight')
        else:
            fig.savefig(filename, bbox_inches='tight')

    def visualize_tf(self, filename=None):
        fig, ax = plt.subplots()
        tf_np = self.tf_tex.to_numpy()
        ax.plot(tf_np[:, 0], 'r-')
        ax.plot(tf_np[:, 1], 'g-')
        ax.plot(tf_np[:, 2], 'b-')
        ax.plot(tf_np[:, 3], 'k-')
        if filename is None:
            fig.savefig('tf.png', bbox_inches='tight')
        else:
            fig.savefig(filename, bbox_inches='tight')


class RaycastFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, vr, volume, tf, look_from, sampling_rate, batched, jitter=True):
        ''' Performs Volume Raycasting with the given `volume` and `tf`

        Args:
            ctx (obj): Context used for torch.autograd.Function
            vr (VolumeRaycaster): VolumeRaycaster taichi class
            volume (Tensor): PyTorch Tensor representing the volume of shape ([BS,] W, H, D)
            tf (Tensor): PyTorch Tensor representing a transfer fucntion texture of shape ([BS,] W, C)
            look_from (Tensor): Look From for Raycaster camera. Shape ([BS,] 3)
            sampling_rate (float): Sampling rate as multiplier to the Nyquist frequency
            batched (4-bool): Whether the input is batched (i.e. has an extra dimension or is a list) and a bool for each volume, tf and look_from
            jitter (bool, optional): Turn on ray jitter (random shift of ray starting points). Defaults to True.

        Returns:
            Tensor: Resulting rendered image of shape (C, H, W)
        '''
        ctx.vr = vr # Save Volume Raycaster for backward
        ctx.sampling_rate = sampling_rate
        ctx.batched, ctx.bs = batched
        ctx.jitter = jitter
        if ctx.batched: # Batched Input
            ctx.save_for_backward(volume, tf, look_from) # unwrap tensor if it's a list
            result = torch.zeros(ctx.bs, *vr.resolution, 4, dtype=torch.float32, device=volume.device)
            for i, vol, tf_, lf in zip(range(ctx.bs), volume, tf, look_from):
                vr.set_cam_pos(lf)
                vr.set_volume(vol)
                vr.set_tf_tex(tf_)
                vr.clear_framebuffer()
                vr.compute_rays()
                vr.compute_intersections(sampling_rate , jitter)
                # vr.compute_entry_exit(sampling_rate, jitter)
                vr.raycast(sampling_rate)
                vr.get_depth_image()
                result[i] = vr.depth.to_torch(device=volume.device)
            return result
        else: # Non-batched, single item
            # No saving via ctx.save_for_backward needed for single example, as it's saved inside vr
            # TODO: is this a problem when using the Raycast multiple times, before calling backward()?
            vr.set_cam_pos(look_from)
            vr.set_volume(volume)
            vr.set_tf_tex(tf)
            vr.clear_framebuffer()
            vr.compute_rays()
            vr.compute_intersections(sampling_rate , jitter)
            vr.raycast(sampling_rate)
            vr.get_depth_image()
            return vr.depth.to_torch(device=volume.device)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        dev = grad_output.device
        if ctx.batched: # Batched Gradient
            vols, tfs, lfs = ctx.saved_tensors
            # Volume Grad Shape (BS, W, H, D)
            volume_grad = torch.zeros(ctx.bs, *ctx.vr.volume.shape, dtype=torch.float32, device=dev)
            # TF Grad Shape (BS, W, C)
            tf_grad = torch.zeros(ctx.bs, *ctx.vr.tf_tex.shape, ctx.vr.tf_tex.n, dtype=torch.float32, device=dev)
            # Look From Grad Shape (BS, 3)
            lf_grad = torch.zeros(ctx.bs, 3, dtype=torch.float32, device=dev)
            for i, vol, tf, lf in zip(range(ctx.bs), vols, tfs, lfs):
                # Clear & Setup
                ctx.vr.set_cam_pos(lf)
                ctx.vr.set_volume(vol)
                ctx.vr.set_tf_tex(tf)
                ctx.vr.clear_grad()
                ctx.vr.clear_framebuffer()
                # Forward
                ctx.vr.compute_rays()
                ctx.vr.compute_intersections(ctx.sampling_rate , ctx.jitter)
                ctx.vr.raycast(ctx.sampling_rate)
                ctx.vr.get_final_image()
                # Backward
                ctx.vr.output_rgba.grad.from_torch(grad_output[i])
                # ctx.vr.depth.grad.from_torch(grad_output[i])
                ctx.vr.get_final_image.grad()
                ctx.vr.raycast.grad(ctx.sampling_rate)
                # print('Output RGBA', torch.nan_to_num(ctx.vr.output_rgba.grad.to_torch(device=dev)).abs().max())
                # print('Render Tape', torch.nan_to_num(ctx.vr.render_tape.grad.to_torch(device=dev)).abs().max())
                # print('TF', torch.nan_to_num(ctx.vr.volume.grad.to_torch(device=dev)).abs().max())
                # print('Volume', torch.nan_to_num(ctx.vr.volume.grad.to_torch(device=dev)).abs().max())
                # print('Entry', torch.nan_to_num(ctx.vr.entry.grad.to_torch(device=dev)).abs().max())
                # print('Exit', torch.nan_to_num(ctx.vr.entry.grad.to_torch(device=dev)).abs().max())
                ctx.vr.grad_nan_to_num()
                ctx.vr.compute_rays.grad()
                # print('AUTODIFF CamPos Field', ctx.vr.cam_pos_field)
                # print('AUTODIFF Camera Pos / Grad', ctx.vr.cam_pos, ctx.vr.cam_pos.grad)
                # ctx.vr.grad_nan_to_num()
                # ctx.vr.compute_rays_backward()
                # ti.sync()
                # print('MANUAL CamPos Field', ctx.vr.cam_pos_field)
                # print('MANUAL Camera Pos', ctx.vr.cam_pos.grad)
                ctx.vr.compute_intersections.grad(ctx.sampling_rate , ctx.jitter)
                # print('Entry', ctx.vr.entry.grad)
                # print('Camera Pos', ctx.vr.cam_pos.grad)
                # ctx.vr.compute_entry_exit.grad(ctx.sampling_rate, ctx.jitter)

                volume_grad[i] = torch.nan_to_num(ctx.vr.volume.grad.to_torch(device=dev))
                tf_grad[i] = torch.nan_to_num(ctx.vr.tf_tex.grad.to_torch(device=dev))
                lf_grad[i] = torch.nan_to_num(ctx.vr.cam_pos.grad.to_torch(device=dev))
            return None, volume_grad, tf_grad, lf_grad, None, None, None

        else: # Non-batched, single item
            ctx.vr.clear_grad()
            ctx.vr.output_rgba.grad.from_torch(grad_output)
            # ctx.vr.depth.grad.from_torch(grad_output)
            ctx.vr.get_final_image.grad()
            ctx.vr.raycast.grad(ctx.sampling_rate)

            return None, \
                torch.nan_to_num(ctx.vr.volume.grad.to_torch(device=dev)), \
                torch.nan_to_num(ctx.vr.tf_tex.grad.to_torch(device=dev)), \
                torch.nan_to_num(ctx.vr.cam_pos.grad.to_torch(device=dev)), \
                    None, None, None
    
    
class Mode(IntEnum):
    FirstHitDepth = 1
    MaxOpacity = 2
    MaxGradient = 3
    WYSIWYP = 4


@ti.data_oriented
class DepthRaycaster(VolumeRaycaster):
        def __init__(self,
                 volume_resolution,
                 render_resolution,
                 max_samples=512,
                 tf_resolution=128,
                 fov=30.0,
                 nearfar=(0.1, 100.0),
                 background_color=0.0):
            ''' Initializes Depth Raycaster. Make sure to .set_volume() and .set_tf_tex() after initialization '''

            ''' Extends the VolumeRaycaster with multiple Depth compositing modes'''
    
            super().__init__(volume_resolution, render_resolution, max_samples, tf_resolution, fov, nearfar)

            render_tiles = tuple(map(lambda x: x // 8, render_resolution))
            self.depth = ti.field(ti.f32)
            self.depth_tape = ti.field(ti.f32)  # tape to record depth so far for every sample, need to be able to check prev measured depth
            ti.root.dense(ti.ij, render_tiles).dense(ti.ij, (8, 8)).place(self.depth)
            ti.root.dense(ti.ijk, (*render_tiles, max_samples)).dense(ti.ijk, (8, 8, 1)).place(self.depth_tape)

        @ti.func
        def get_depth_from_sx(self, sample_index: int, i: int, j: int) -> float:
            tmax = self.exit[i, j]
            n_samples = self.sample_step_nums[i, j]
            ray_len = (tmax - self.entry[i, j])
            tmin = self.entry[i, j] + 0.5 * ray_len / n_samples
            dist = tl.mix(tmin, tmax, float(sample_index) / float(n_samples - 1))
            return dist / self.far

        @ti.kernel
        def raycast_nondiff(self, sampling_rate: float, mode: int):
            ''' Raycasts in a non-differentiable (but faster and cleaner) way. Use `get_final_image_nondiff` with this.

            Args:
                sampling_rate (float): Sampling rate (multiplier with Nyquist frequence)
                mode (Mode): Rendering mode (Standard or different depth modes)
            '''
            for i, j in self.valid_sample_step_count:  # For all pixels
                maximum = 0.0

                # WYSIWYP fields
                biggest_jump = 0.0
                interval_start = 0
                interval_start_acc_opac = 0.0
                current_d = 0.0
                last_d = 0.0
                current_dd = 0.0
                last_dd = 0.0

                for cnt in range(self.sample_step_nums[i, j]):
                    look_from = self.cam_pos[None]
                    opacity, old_agg_opacity = 0.0, 0.0
                    if self.render_tape[i, j, 0].w < 0.99:
                        tmax = self.exit[i, j]      # letztes sample am austrittsort?
                        n_samples = self.sample_step_nums[i, j]
                        ray_len = (tmax - self.entry[i, j])
                        tmin = self.entry[i, j] + 0.5 * ray_len / n_samples  # Offset tmin as t_start
                        vd = self.rays[i, j]

                        dist = tl.mix(tmin, tmax, float(cnt) / float(n_samples - 1))
                        pos = look_from + dist * vd  # Current Pos
                        depth = dist / self.far

                        light_pos = look_from + tl.vec3(0.0, 1.0, 0.0)
                        intensity = self.sample_volume_trilinear(pos)
                        sample_color = self.apply_transfer_function(intensity)
                        opacity = 1.0 - ti.pow(1.0 - sample_color.w, 1.0 / sampling_rate)
                        if sample_color.w > 1e-3:
                            normal = self.get_volume_normal(pos)
                            light_dir = (pos - light_pos).normalized()  # Direction to light source
                            n_dot_l = max(normal.dot(light_dir), 0.0)
                            diffuse = self.diffuse * n_dot_l
                            r = tl.reflect(light_dir, normal)  # Direction of reflected light
                            r_dot_v = max(r.dot(-vd), 0.0)
                            specular = self.specular * pow(r_dot_v, self.shininess)

                            # fill render tape according to selected mode
                            render_output = tl.vec4((diffuse + specular + self.ambient) * sample_color.xyz * opacity * self.light_color, opacity)
                            old_agg_opacity = self.render_tape[i, j, 0].w
                            new_agg_sample = (1.0 - self.render_tape[i, j, 0].w) * render_output + self.render_tape[i, j, 0]
                            if mode == Mode.FirstHitDepth:
                                if sample_color.w > 1e-3 and self.depth[i, j] == 0.0:
                                    self.depth[i, j] = depth
                            elif mode == Mode.MaxOpacity:
                                if sample_color.w > maximum:
                                    self.depth[i, j] = depth
                                    maximum = sample_color.w
                            elif mode == Mode.MaxGradient:
                                grad = new_agg_sample.w - old_agg_opacity
                                if grad > maximum:
                                    self.depth[i, j] = depth
                                    maximum = grad
                            elif mode == Mode.WYSIWYP and cnt > 0:
                                # calculate current derivative and dd (and think about better notation)
                                current_d = new_agg_sample.w - old_agg_opacity
                                current_dd = current_d - last_d

                                # check for interval end (dd up from negative or ray end or ray finished)
                                if last_dd < 0.0 <= current_dd or cnt == self.sample_step_nums[i, j] - 1 or\
                                        new_agg_sample.w >= 0.99:
                                    if new_agg_sample.w - interval_start_acc_opac > biggest_jump:
                                        biggest_jump = new_agg_sample.w - interval_start_acc_opac
                                        # take start of interval (could also take depth from cnt - (cnt-last_interval_start) / 2)
                                        self.depth[i, j] = self.get_depth_from_sx(interval_start, i, j)

                                # check for interval start (dd from 0 or neg to positive)
                                if last_dd <= 0.0 < current_dd:
                                    interval_start = cnt

                                # save current values in last_fields
                                last_d = current_d
                                last_dd = current_dd

                            self.render_tape[i, j, 0] = new_agg_sample


class Raycaster(torch.nn.Module):
    def __init__(self, volume_shape, output_shape, tf_shape, sampling_rate=1.0, jitter=True, max_samples=512,
                 fov=30.0, near=0.1, far=100.0, ti_kwargs=None, background_color=0.0, mode=None):
        super().__init__()
        if ti_kwargs is None:
            ti_kwargs = {}
        self.volume_shape = (volume_shape[2], volume_shape[0], volume_shape[1])
        self.output_shape = output_shape
        self.tf_shape = tf_shape
        self.sampling_rate = sampling_rate
        self.jitter = jitter
        self.mode = mode
        ti.init(arch=ti.cuda, default_fp=ti.f32, **ti_kwargs)

        self.vr = DepthRaycaster(self.volume_shape, output_shape, max_samples=max_samples, tf_resolution=self.tf_shape,
         fov=fov, nearfar=(near, far), background_color=background_color)

    def raycast_nondiff(self, volume, tf, look_from, sampling_rate=None, mode: Union[None, Mode] = None):
        if mode is None:
            if self.mode is not None:
                mode = self.mode
            else:
                mode = Mode.FirstHitDepth

        with torch.no_grad() as _, autocast(False) as _:
            batched, bs, vol_in, tf_in, lf_in = self._determine_batch(volume, tf, look_from)
            sr = sampling_rate if sampling_rate is not None else 4.0 * self.sampling_rate
            if batched:  # Batched Input
                result = torch.zeros(bs,
                                    *self.vr.resolution,
                                    5,
                                    dtype=torch.float32,
                                    device=volume.device)
                # Volume: remove intensity dim, reorder to (BS, W, H, D)
                # TF: Reorder to (BS, W, 4)
                for i, vol, tf_, lf in zip(range(bs), vol_in, tf_in, lf_in):
                    with autocast(False):
                        self.vr.set_cam_pos(lf)
                    self.vr.set_volume(vol)
                    self.vr.set_tf_tex(tf_)
                    self.vr.clear_framebuffer()
                    self.vr.compute_rays()
                    self.vr.compute_intersections_nondiff(sr, False)
                    self.vr.raycast_nondiff(sr, mode)
                    self.vr.get_final_image_nondiff()
                    result[i,...,:4] = self.vr.output_rgba.to_torch(device=volume.device)
                    result[i,...,4]  = self.vr.depth.to_torch(device=volume.device)
                # First reorder render to (BS, C, H, W), then flip Y to correct orientation
                return torch.flip(result, (2,)).permute(0, 3, 2, 1).contiguous()
            else:
                self.vr.set_cam_pos(lf_in)
                self.vr.set_volume(vol_in)
                self.vr.set_tf_tex(tf_in)
                self.vr.clear_framebuffer()
                self.vr.compute_rays()
                self.vr.compute_intersections_nondiff(sr, False)
                self.vr.raycast_nondiff(sr, mode)
                self.vr.get_final_image_nondiff()
                # First reorder to (C, H, W), then flip Y to correct orientation
                rgbad = torch.cat([self.vr.output_rgba.to_torch(device=volume.device), self.vr.depth.to_torch(device=volume.device)], dim=-1)
                return torch.flip(rgbad, (1, )).permute(2, 1, 0).contiguous()

    def raycast_notorch(self, volume, tf, look_from, sampling_rate=None):
        ''' seeks to mimic raycast_nondiff, but using the raycast method of the Volume raycaster
            depth renderings can then be extracted from the render tape instead of during the rendering process'''
        pass

    def forward(self, volume, tf, look_from):
        ''' Raycasts through `volume` using the transfer function `tf` from given camera position (volume is in [-1,1]^3, centered around 0)

        Args:
            volume (Tensor): Volume Tensor of shape ([BS,] 1, D, H, W)
            tf (Tensor): Transfer Function Texture of shape ([BS,] 4, W)
            look_from (Tensor): Camera position of shape ([BS,] 3)

        Returns:
            Tensor: Rendered image of shape ([BS,] 4, H, W)
        '''
        batched, bs, vol_in, tf_in, lf_in = self._determine_batch(volume, tf, look_from)
        if batched: # Anything batched Batched
            return torch.flip(
                RaycastFunction.apply(self.vr, vol_in, tf_in, lf_in, self.sampling_rate, (batched, bs), self.jitter),
                (2,) # First reorder render to (BS, C, H, W), then flip Y to correct orientation
            ).permute(0, 3, 2, 1).contiguous()
        else:
            return torch.flip(
                RaycastFunction.apply(self.vr, vol_in, tf_in, lf_in,
                                      self.sampling_rate, (batched, bs),
                                      self.jitter),
                (1,) # First reorder to (C, H, W), then flip Y to correct orientation
            ).permute(2, 1, 0).contiguous()


    def _determine_batch(self, volume, tf, look_from):
        ''' Determines whether there's a batched input and returns lists of non-batched inputs.

        Args:
            volume (Tensor): Volume input, either 4D or 5D (batched)
            tf (Tensor): Transfer Function input, either 2D or 3D (batched)
            look_from (Tensor): Camera Look From input, either 1D or 2D (batched)

        Returns:
            ([bool], Tensor, Tensor, Tensor): (is anything batched?, batched input or list of non-batched inputs (for all inputs))
        '''
        batched = torch.tensor([volume.ndim == 5, tf.ndim == 3, look_from.ndim == 2])

        if batched.any():
            bs = [volume, tf, look_from][batched.long().argmax().item()].size(0)
            vol_out = volume.squeeze(1).permute(0,3,1,2).contiguous() if batched[0].item() else volume.squeeze(0).permute(2, 0, 1).expand(bs, -1,-1,-1).clone()
            tf_out  = tf.permute(0, 2, 1).contiguous()                if batched[1].item() else tf.permute(1,0).expand(bs, -1,-1).clone()
            lf_out  = look_from if batched[2].item() else look_from.expand(bs, -1).clone()
            return True, bs, vol_out, tf_out, lf_out
        else:
            return False, 0, volume.squeeze(0).permute(2, 0, 1).contiguous(), tf.permute(1,0).contiguous(), look_from

    def extra_repr(self):
        return f'Volume ({self.volume_shape}), Output Render ({self.output_shape}), TF ({self.tf_shape}), Max Samples = {self.vr.max_samples}'
