from turtle import color
from numpy.core.fromnumeric import shape
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

class Mode(IntEnum):
    FirstHitDepth = 1
    MaxOpacity = 2
    MaxGradient = 3
    WYSIWYP = 4


#####################
### Taichi Class  ###
#####################
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
        self.loss = ti.field(ti.f32, shape=(), needs_grad=False)
        self.rays = ti.Vector.field(3, dtype=ti.f32)
        self.max_valid_sample_step_count = ti.field(ti.i32, ())
        self.max_samples = max_samples
        self.background_color = background_color
        self.ambient = 0.4
        self.diffuse = 0.8
        self.specular = 0.3
        self.shininess = 32.0
        self.light_color = tm.vec3(1.0)
        self.cam_pos = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.cam_pos_field = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)



        # add depth field
        self.no_hit_depth = 1.0
        self.depth = ti.field(ti.f32, needs_grad=True)
        self.ground_truth_depth = ti.field(shape=self.resolution, dtype=ti.f32)

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

#################
### SET DATA  ###
#################

    def set_volume(self, volume):
        self.volume.from_torch(volume.float())

    def set_tf_tex(self, tf_tex):
        self.tf_tex.from_torch(tf_tex.float())

    def set_cam_pos(self, cam_pos):
        self.cam_pos.from_torch(cam_pos.float())

    def set_gtd(self, gtd):
        self.ground_truth_depth.from_torch(gtd.float())

#######################
### BASIC SAMPLERS  ###
#######################

    @ti.func
    def sample_volume_trilinear(self, pos):
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
    def get_volume_normal(self, pos):
        delta = 1e-3
        x_delta = tm.vec3(delta, 0.0, 0.0)
        y_delta = tm.vec3(0.0, delta, 0.0)
        z_delta = tm.vec3(0.0, 0.0, delta)
        dx = self.sample_volume_trilinear(
            pos + x_delta) - self.sample_volume_trilinear(pos - x_delta)
        dy = self.sample_volume_trilinear(
            pos + y_delta) - self.sample_volume_trilinear(pos - y_delta)
        dz = self.sample_volume_trilinear(
            pos + z_delta) - self.sample_volume_trilinear(pos - z_delta)
        return tm.vec3(dx, dy, dz).normalized()

    @ti.func
    def apply_transfer_function(self, intensity: float):
        ''' Applies a 1D transfer function to a given intensity value

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

####################
### SETUP CAMERA ###
####################

    @ti.kernel
    def compute_rays(self):
        for i,j in self.rays:
            max_x = ti.static(float(self.rays.shape[0]))
            max_y = ti.static(float(self.rays.shape[1]))
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

            self.rays[i,j] = (near_pos - self.cam_pos[None]).normalized()

    @ti.kernel
    def compute_rays_backward(self):
        for i,j in self.rays:
            if tm.length(self.rays[i,j]) > 1e-8:
                updated_ray = (self.rays[i,j] - self.rays.grad[i,j]).normalized()
                pix_pos = self.cam_pos[None] + self.rays[i, j] * self.entry[i, j]
                new_campos = pix_pos - updated_ray * (self.entry[i,j] - self.entry.grad[i,j])
                self.cam_pos.grad[None] += (self.cam_pos[None] - new_campos) #/ ti.static(self.rays.shape[0] * self.rays.shape[1])
                self.cam_pos_field[i,j] = new_campos

    @ti.kernel
    def compute_intersections(self, sampling_rate: float, jitter: int):
        for i,j in self.entry:
            vol_diag = (tm.vec3(*self.volume.shape) - tm.vec3(1.0)).norm()
            bb_bl = tm.vec3(-1.0)
            bb_tr = tm.vec3( 1.0)
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
            vol_diag = (tm.vec3(*self.volume.shape) - tm.vec3(1.0)).norm()
            bb_bl = tm.vec3(-1.0)
            bb_tr = tm.vec3( 1.0)
            tmin, tmax, hit = get_entry_exit_points(self.cam_pos[None], self.rays[i,j], bb_bl, bb_tr)

            n_samples = 1
            if hit:
                n_samples = ti.cast(ti.floor(sampling_rate * vol_diag * (tmax - tmin)), ti.int32) + 1
            self.entry[i,j] = tmin
            self.exit[i,j] = tmax
            self.sample_step_nums[i,j] = n_samples

    @ti.func
    def get_pos_parameters(self, i, j, idx, lf):
        tmax = self.exit[i, j]
        n_samples = self.sample_step_nums[i, j]
        ray_len = (tmax - self.entry[i, j])
        tmin = self.entry[i, j] + 0.5 * ray_len / n_samples
        dist = tm.mix(tmin, tmax, float(idx) / float(n_samples - 1))
        depth = dist / self.far
        vd = self.rays[i, j]
        pos = lf + dist * vd

        return depth, vd, pos

    @ti.func
    def sample_for_color_and_opacity(self, sr, pos):
        intensity = self.sample_volume_trilinear(pos)
        sample_color = self.apply_transfer_function(intensity)
        opacity = 1.0 - ti.pow(1.0 - sample_color.w, 1.0 / sr)
        return sample_color, opacity

###################
### RAYCASTING  ###
###################
    @ti.func
    def get_shading(self, vd, pos, lf):
        light_pos = lf + tm.vec3(0.0, 1.0, 0.0)
        normal = self.get_volume_normal(pos)
        light_dir = (pos - light_pos).normalized()  # Direction to light source
        n_dot_l = max(normal.dot(light_dir), 0.0)
        diffuse = self.diffuse * n_dot_l
        r = tm.reflect(light_dir, normal)  # Direction of reflected light
        r_dot_v = max(r.dot(-vd), 0.0)
        specular = self.specular * pow(r_dot_v, self.shininess)
        return diffuse, specular

    @ti.kernel
    def raycast(self, sampling_rate: float):
        ''' Produce a rendering. Run compute_entry_exit first! '''
        for i, j in self.valid_sample_step_count:  # For all pixels
            # variables for calculating different depths
            opacity = 0.0
            new_agg_sample = tm.vec4(0.0)

            for sample_idx in range(1, self.sample_step_nums[i, j]):
                look_from = self.cam_pos[None]
                if self.render_tape[i, j, sample_idx -1].w < 0.99 and sample_idx < ti.static(self.max_samples):
                    depth, vd, pos = self.get_pos_parameters(i, j, sample_idx, look_from)
                    self.pos_tape[i,j, sample_idx] = pos  # Current Pos

                    sample_color, opacity = self.sample_for_color_and_opacity(sampling_rate, pos)

                    diffuse, specular = self.get_shading(vd, pos, look_from)

                    # render rgba image
                    render_output = tm.vec4(ti.min(1.0, diffuse + specular + self.ambient) * sample_color.xyz * opacity * self.light_color, opacity)
                    old_agg_opacity = self.render_tape[i, j, sample_idx - 1].w
                    new_agg_sample = (1.0 - self.render_tape[i, j, sample_idx - 1].w) * render_output + self.render_tape[i, j, sample_idx - 1]
                    self.valid_sample_step_count[i, j] += 1
                    self.render_tape[i, j, sample_idx] = new_agg_sample
                else:
                    self.render_tape[i, j, sample_idx] = self.render_tape[i, j, sample_idx - 1]

    @ti.kernel
    def raycast_nondiff(self, sampling_rate: float, mode: int):
        ''' Raycasts in a non-differentiable (but faster and cleaner) way. Use `get_final_image_nondiff` with this.

        Args:
            sampling_rate (float): Sampling rate (multiplier with Nyquist frequence)
            mode (Mode): Rendering mode (Standard or different depth modes)
        '''
        for i, j in self.valid_sample_step_count:  # For all pixels
            # variables for calculating different depths
            maximum = 0.0
            opacity, old_agg_opacity = 0.0, 0.0
            new_agg_sample = tm.vec4(0.0)

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
                if self.render_tape[i, j, 0].w < 0.99:

                    depth, vd, pos = self.get_pos_parameters(i, j, cnt, look_from)
                    sample_color, opacity = self.sample_for_color_and_opacity(sampling_rate, pos)

                    if sample_color.w > 1e-3:
                        diffuse, specular = self.get_shading(vd, pos, look_from)

                        # render rgba image
                        render_output = tm.vec4((diffuse + specular + self.ambient) * sample_color.xyz * opacity * self.light_color, opacity)
                        old_agg_opacity = self.render_tape[i, j, 0].w
                        new_agg_sample = (1.0 - self.render_tape[i, j, 0].w) * render_output + self.render_tape[i, j, 0]
                    else:
                        old_agg_opacity = self.render_tape[i, j, 0].w
                        new_agg_sample = self.render_tape[i, j, 0]

                    # calculate depth information according to mode
                    if mode == Mode.FirstHitDepth:
                        if sample_color.w > 1e-3 and self.depth[i, j] == self.no_hit_depth:
                            self.depth[i, j] = depth
                    elif mode == Mode.MaxOpacity:
                        if sample_color.w > maximum and sample_color.w > 1e-3:
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

                        # check for interval end (2nd derivative changes from negative to zero or positive or ray end or ray finished)
                        if (last_dd < 0.0 and current_dd >= 0.0) or cnt == self.sample_step_nums[i, j] - 1 or\
                                new_agg_sample.w >= 0.99:
                            if new_agg_sample.w - interval_start_acc_opac > biggest_jump:
                                biggest_jump = new_agg_sample.w - interval_start_acc_opac
                                # take start of interval (could also take depth from cnt - (cnt-last_interval_start) / 2)
                                self.depth[i, j] = self.get_depth_from_sx(interval_start, i, j)

                        # check for interval start (2nd derivative becomes positive)
                        if last_dd <= 0.0 < current_dd:
                            interval_start = cnt
                            interval_start_acc_opac = new_agg_sample.w

                        # save current values in last_fields
                        last_d = current_d
                        last_dd = current_dd

                    self.render_tape[i, j, 0] = new_agg_sample

###################
### DEPTH STUFF ###
###################

    @ti.kernel
    def calculate_depth(self, mode: int):
        # run after raycast

        for i, j in self.valid_sample_step_count:  # For all pixels
            maximum = 0.0  # for the max modes

            # WYSIWYP fields
            biggest_jump = 0.0
            interval_start = 0
            interval_start_acc_opac = 0.0
            current_d = 0.0
            last_d = 0.0
            current_dd = 0.0
            last_dd = 0.0

            for sample_idx in range(1, self.sample_step_nums[i, j]):
                depth, _, _ = self.get_pos_parameters(i, j, sample_idx,
                                                      self.cam_pos[None])

                if mode == Mode.FirstHitDepth:
                    if self.render_tape[i, j,
                                        sample_idx].w > 1e-3 and self.depth[
                                            i, j] == self.no_hit_depth:
                        self.depth[i, j] = depth
                if mode == Mode.MaxOpacity:
                    last_acc = self.render_tape[i, j, sample_idx - 1]
                    current_sample = (self.render_tape[i, j, sample_idx] -
                                      last_acc) / (1.0 - last_acc.w)
                    if current_sample.w > maximum and current_sample.w > 1e-3:
                        self.depth[i, j] = depth
                        maximum = current_sample.w
                if mode == Mode.MaxGradient:
                    grad = self.render_tape[i, j,
                                            sample_idx].w - self.render_tape[
                                                i, j, sample_idx - 1].w
                    if grad > maximum:
                        self.depth[i, j] = depth
                        maximum = grad
                if mode == Mode.WYSIWYP:
                    # calculate current derivative and dd (and think about better notation)
                    current_d = self.render_tape[
                        i, j,
                        sample_idx].w - self.render_tape[i, j,
                                                         sample_idx - 1].w
                    current_dd = current_d - last_d

                    # check for interval end (2nd derivative changes from negative to zero or positive or ray end or ray finished)
                    if (last_dd < 0.0 and current_dd >= 0.0) or sample_idx == self.sample_step_nums[i, j] - 1 or\
                            self.render_tape[i, j, sample_idx].w >= 0.99:
                        if self.render_tape[
                                i, j,
                                sample_idx].w - interval_start_acc_opac > biggest_jump:
                            biggest_jump = self.render_tape[
                                i, j, sample_idx].w - interval_start_acc_opac
                            # take start of interval (could also take depth from cnt - (cnt-last_interval_start) / 2)
                            self.depth[i, j] = self.get_depth_from_sx(
                                interval_start, i, j)

                    # check for interval start (2nd derivative becomes positive)
                    if last_dd <= 0.0 < current_dd:
                        interval_start = sample_idx
                        interval_start_acc_opac = self.render_tape[
                            i, j, sample_idx].w

                    # save current values in last_fields
                    last_d = current_d
                    last_dd = current_dd

    @ti.kernel
    def calculate_depth_backward(self):
        ''' Place gradients for samples that have depth-defying opacity in the render tape. '''
        for i, j in self.valid_sample_step_count:
            gt_sx = self.get_sx_from_depth(self.ground_truth_depth[i, j], i,
                                           j)  # ground truth depth index
            cd_sx = self.get_sx_from_depth(self.depth[i, j], i,
                                           j)  # actual depth index

            if cd_sx > gt_sx:
                #    [gt_sx: cd_sx) need to raise opacity at gtd
                self.render_tape.grad[i, j, gt_sx] += tm.vec4(
                    0.0, 0.0, 0.0, -self.depth.grad[i, j])

            if cd_sx < gt_sx:
                #    [cd_sx: gt_sx) need to decrease opacity at cd_sx
                self.render_tape.grad[i, j, cd_sx] += tm.vec4(
                    0.0, 0.0, 0.0, -self.depth.grad[i, j])

    @ti.func
    def get_depth_from_sx(self, sample_index: int, i: int, j: int) -> float:
        """ Calculate a depth value for a given sample index on ray i, j."""
        tmax = self.exit[i, j]
        n_samples = self.sample_step_nums[i, j]
        ray_len = (tmax - self.entry[i, j])
        tmin = self.entry[i, j] + 0.5 * ray_len / n_samples
        dist = tm.mix(tmin, tmax, float(sample_index) / float(n_samples - 1))
        return dist / self.far

    @ti.func
    def get_sx_from_depth(self, depth: float, i: int, j: int) -> int:
        """ Calculate the nearest sample index for a given depth value on ray i, j. """
        n_samples = self.sample_step_nums[i, j]
        # if depth is 1 (backplane is hit) we return n_samples
      
        if depth == 1.0:
            sx = n_samples
        else:
            tmax = self.exit[i, j]
            ray_len = (tmax - self.entry[i, j])
            tmin = self.entry[i, j] + 0.5 * ray_len / n_samples
            dist = depth * self.far

            sx = ti.cast(
                ti.floor(((dist - tmin) / (tmax - tmin)) * (n_samples - 1)),
                ti.i32)
            sx = tm.clamp(sx, 0, n_samples - 1)
        return sx


###################
### DEPTH STUFF ###
###################

    @ti.kernel
    def calculate_depth(self, mode: int):
        # run after raycast

        for i, j in self.valid_sample_step_count:  # For all pixels
            maximum = 0.0  # for the max modes

            # WYSIWYP fields
            biggest_jump = 0.0
            interval_start = 0
            interval_start_acc_opac = 0.0
            current_d = 0.0
            last_d = 0.0
            current_dd = 0.0
            last_dd = 0.0

            for sample_idx in range(1, self.sample_step_nums[i, j]):
                depth, _, _ = self.get_pos_parameters(i, j, sample_idx, self.cam_pos[None])

                if mode == Mode.FirstHitDepth:
                    if self.render_tape[i, j, sample_idx].w > 1e-3 and self.depth[i, j] == self.no_hit_depth:
                        self.depth[i, j] = depth
                if mode == Mode.MaxOpacity:
                    last_acc = self.render_tape[i, j, sample_idx - 1]
                    current_sample = (self.render_tape[i, j, sample_idx] - last_acc) / (1.0 - last_acc.w)
                    if current_sample.w > maximum and current_sample.w > 1e-3:
                        self.depth[i, j] = depth
                        maximum = current_sample.w
                if mode == Mode.MaxGradient:
                    grad = self.render_tape[i, j, sample_idx].w - self.render_tape[i, j, sample_idx - 1].w
                    if grad > maximum:
                        self.depth[i, j] = depth
                        maximum = grad
                if mode == Mode.WYSIWYP:
                    # calculate current derivative and dd (and think about better notation)
                    current_d = self.render_tape[i, j, sample_idx].w - self.render_tape[i, j, sample_idx - 1].w
                    current_dd = current_d - last_d

                    # check for interval end (2nd derivative changes from negative to zero or positive or ray end or ray finished)
                    if (last_dd < 0.0 and current_dd >= 0.0) or sample_idx == self.sample_step_nums[i, j] - 1 or\
                            self.render_tape[i, j, sample_idx].w >= 0.99:
                        if self.render_tape[i, j, sample_idx].w - interval_start_acc_opac > biggest_jump:
                            biggest_jump = self.render_tape[i, j, sample_idx].w - interval_start_acc_opac
                            # take start of interval (could also take depth from cnt - (cnt-last_interval_start) / 2)
                            self.depth[i, j] = self.get_depth_from_sx(interval_start, i, j)

                    # check for interval start (2nd derivative becomes positive)
                    if last_dd <= 0.0 < current_dd:
                        interval_start = sample_idx
                        interval_start_acc_opac = self.render_tape[i, j, sample_idx].w

                    # save current values in last_fields
                    last_d = current_d
                    last_dd = current_dd

    @ti.kernel
    def calculate_depth_backward(self):
        ''' Place gradients for samples that have depth-defying opacity in the render tape. '''
        for i, j in self.valid_sample_step_count:
            gt_sx = self.get_sx_from_depth(self.ground_truth_depth[i, j], i,
                                           j)  # ground truth depth index
            cd_sx = self.get_sx_from_depth(self.depth[i, j], i,
                                           j)  # actual depth index

            if cd_sx > gt_sx:
                #    [gt_sx: cd_sx) need to raise opacity at gtd
                self.render_tape.grad[i, j, gt_sx] += tm.vec4(
                    0.0, 0.0, 0.0, -self.depth.grad[i, j])

            if cd_sx < gt_sx:
                #    [cd_sx: gt_sx) need to decrease opacity at cd_sx
                self.render_tape.grad[i, j, cd_sx] += tm.vec4(
                    0.0, 0.0, 0.0, -self.depth.grad[i, j])

    @ti.func
    def get_depth_from_sx(self, sample_index: int, i: int, j: int) -> float:
        """ Calculate a depth value for a given sample index on ray i, j."""
        tmax = self.exit[i, j]
        n_samples = self.sample_step_nums[i, j]
        ray_len = (tmax - self.entry[i, j])
        tmin = self.entry[i, j] + 0.5 * ray_len / n_samples
        dist = tm.mix(tmin, tmax, float(sample_index) / float(n_samples - 1))
        return dist / self.far

    @ti.func
    def get_sx_from_depth(self, depth: float, i: int, j: int) -> int:
        """ Calculate the nearest sample index for a given depth value on ray i, j. """
        n_samples = self.sample_step_nums[i, j]
        # if depth is 1 (backplane is hit) we return n_samples
        sx = -1
        if depth == 1.0:
            sx = n_samples
        else:
            tmax = self.exit[i, j]
            ray_len = (tmax - self.entry[i, j])
            tmin = self.entry[i, j] + 0.5 * ray_len / n_samples
            dist = depth * self.far

            sx = ti.cast(
                ti.floor(((dist - tmin) / (tmax - tmin)) * (n_samples - 1)),
                ti.i32)
            sx = tm.clamp(sx, 0, n_samples - 1)
        return sx


##################################
### GET RESULT / CLEAR BUFFERS ###
##################################

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
            ns = tm.clamp(self.sample_step_nums[i, j], 1, self.max_samples-1)
            rgba = self.render_tape[i, j, ns - 1]
            self.output_rgba[i, j] += tm.vec4(tm.mix(rgba.xyz, tm.vec3(self.background_color), 1.0 - rgba.w), rgba.w)
            if valid_sample_step_count > self.max_valid_sample_step_count[None]:
                self.max_valid_sample_step_count[
                    None] = valid_sample_step_count

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
        for i,j in self.rays.grad:
            if any(isnan(self.rays.grad[i,j])):
                self.rays.grad[i,j] = tm.vec3(0.0)
            if isnan(self.entry.grad[i,j]):
                self.entry.grad[i,j] = 0.0
            if isnan(self.exit.grad[i,j]):
                self.exit.grad[i,j] = 0.0

##################################
### PyTorch autograd.Function  ###
##################################


class RaycastFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, vr, volume, tf, look_from, sampling_rate, jitter=True, mode=Mode.FirstHitDepth, depth_gt=None):
        ''' Performs Volume Raycasting with the given `volume` and `tf`

        Args:
            ctx (obj): Context used for torch.autograd.Function
            vr (VolumeRaycaster): VolumeRaycaster taichi class
            volume (Tensor): PyTorch Tensor representing the volume of shape ([BS,] W, H, D)
            tf (Tensor): PyTorch Tensor representing a transfer fucntion texture of shape ([BS,] W, C)
            look_from (Tensor): Look From for Raycaster camera. Shape ([BS,] 3)
            sampling_rate (float): Sampling rate as multiplier to the Nyquist frequency
            jitter (bool, optional): Turn on ray jitter (random shift of ray starting points). Defaults to True.
            mode (Mode, optional): Depth Mode. Defaults to FirstHitDepth.
            depth_gt (Tensor): Depth ground truth tensor to be used in backward later ([BS,], 1, H, W)

        Returns:
            Tensor: Resulting rendered image of shape (C, H, W)
        '''
        ctx.vr = vr # Save Volume Raycaster for backward
        ctx.sampling_rate = sampling_rate
        ctx.bs = volume.size(0)
        ctx.jitter = jitter
        ctx.mode = mode
        if depth_gt is None:
            ctx.save_for_backward(volume, tf, look_from)
            ctx.has_depth_gt = False
        else:
            ctx.has_depth_gt = True
            ctx.save_for_backward(volume, tf, look_from, depth_gt) 

        result = torch.zeros(ctx.bs, *vr.resolution, 5, dtype=torch.float32, device=volume.device)
        for i, vol, tf_, lf in zip(range(ctx.bs), volume, tf, look_from):
            vr.set_cam_pos(lf)
            vr.set_volume(vol)
            vr.set_tf_tex(tf_)
            vr.clear_framebuffer()
            vr.compute_rays()
            vr.compute_intersections(sampling_rate, jitter)
            vr.raycast(sampling_rate)
            vr.calculate_depth(mode)
            vr.get_final_image()
            result[i,...,:4] = vr.output_rgba.to_torch(device=volume.device)
            result[i,...,4]  = vr.depth.to_torch(device=volume.device)
        return result

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        dev = grad_output.device
        if ctx.has_depth_gt:
            vols, tfs, lfs, depth_gts = ctx.saved_tensors
        else:
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
            ctx.vr.compute_intersections(ctx.sampling_rate, ctx.jitter)
            ctx.vr.raycast(ctx.sampling_rate)
            ctx.vr.calculate_depth(ctx.mode)
            ctx.vr.get_final_image()

            # Backward
            ctx.vr.output_rgba.grad.from_torch(grad_output[i, ..., :4])
            ctx.vr.depth.grad.from_torch(grad_output[i,..., 4])
            ctx.vr.get_final_image.grad()
            if ctx.has_depth_gt:
                ctx.vr.set_gtd(depth_gts[i].squeeze().flip(0).permute(1,0))
                ctx.vr.calculate_depth_backward()
                # print(ctx.vr.render_tape.grad.to_torch()[..., -1].abs().max())
            ctx.vr.raycast.grad(ctx.sampling_rate)
            # print('Output RGBA', torch.nan_to_num(ctx.vr.output_rgba.grad.to_torch(device=dev)).abs().max())
            # print('Render Tape', torch.nan_to_num(ctx.vr.render_tape.grad.to_torch(device=dev)).abs().max())
            # print('TF', torch.nan_to_num(ctx.vr.volume.grad.to_torch(device=dev)).abs().max())
            ctx.vr.grad_nan_to_num()
            ctx.vr.compute_rays.grad()
            ctx.vr.compute_intersections.grad(ctx.sampling_rate , ctx.jitter)

            volume_grad[i] = torch.nan_to_num(ctx.vr.volume.grad.to_torch(device=dev))
            tf_grad[i] = torch.nan_to_num(ctx.vr.tf_tex.grad.to_torch(device=dev))
            lf_grad[i] = torch.nan_to_num(ctx.vr.cam_pos.grad.to_torch(device=dev))
        return None, volume_grad, tf_grad, lf_grad, None, None, None, None

#######################
### PyTorch Module  ###
#######################

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

        self.vr = VolumeRaycaster(self.volume_shape, output_shape, max_samples=max_samples, tf_resolution=self.tf_shape,
         fov=fov, nearfar=(near, far), background_color=background_color)

    def raycast_nondiff(self, volume, tf, look_from, sampling_rate=None, mode: Union[None, Mode] = None):
        if mode is None:
            if self.mode is not None:
                mode = self.mode
            else:
                mode = Mode.FirstHitDepth

        with torch.no_grad() as _, autocast(False) as _:
            bs, vol_in, tf_in, lf_in = self._determine_batch(volume, tf, look_from)
            sr = sampling_rate if sampling_rate is not None else 4.0 * self.sampling_rate
            result = torch.zeros(bs, *self.vr.resolution, 5, dtype=torch.float32, device=volume.device)
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

    def forward(self, volume, tf, look_from, mode: Optional[Mode] = None, depth_gt=None):
        ''' Raycasts through `volume` using the transfer function `tf` from given camera position (volume is in [-1,1]^3, centered around 0)

        Args:
            volume (Tensor): Volume Tensor of shape ([BS,] 1, D, H, W)
            tf (Tensor): Transfer Function Texture of shape ([BS,] 4, W)
            look_from (Tensor): Camera position of shape ([BS,] 3)
            mode (Optional[Mode]: Depth Mode
            depth_gt (Tensor): Depth ground truth tensor to be used in backward later ([BS,], 1, H, W)

        Returns:
            Tensor: Rendered image of shape ([BS,] 4, H, W)
        '''
        bs, vol_in, tf_in, lf_in = self._determine_batch(volume, tf, look_from)
        if mode is None:
            if self.mode is not None:
                mode = self.mode
            else:
                mode = Mode.FirstHitDepth

        return torch.flip(
            RaycastFunction.apply(self.vr, vol_in, tf_in, lf_in, self.sampling_rate, self.jitter, mode, depth_gt),
            (2,)  # First reorder render to (BS, C, H, W), then flip Y to correct orientation
        ).permute(0, 3, 2, 1).contiguous()

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
            return bs, vol_out, tf_out, lf_out
        else:
            return 1, volume.squeeze(0).permute(2, 0, 1).unsqueeze(0).contiguous(), tf.permute(1,0).unsqueeze(0).contiguous(), look_from.unsqueeze(0)

    def extra_repr(self):
        return f'Volume ({self.volume_shape}), Output Render ({self.output_shape}), TF ({self.tf_shape}), Max Samples = {self.vr.max_samples}'
