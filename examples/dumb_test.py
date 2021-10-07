import taichi as ti
import taichi_glsl as tl
import numpy as np

ti.init(arch=ti.cuda, default_fp=ti.f32)

imag_depth = ti.field(ti.f32, needs_grad=True)
ti.root.dense(ti.ij, (4, 4)).place(imag_depth)
loss = ti.field(ti.f32, needs_grad=True)
x = ti.field(ti.f32, (), needs_grad=True)
#ti.root.place(x)                                       Why does place() fuck up all grad stuff for me??
ti.root.place(loss)
ti.root.lazy_grad()

@ti.kernel
def calc_some_loss():
    for i in range(4):
        for j in range(4):
            loss[None] += ((imag_depth[i, j] - 1) **2) / 16

@ti.kernel
def super_like_the_example():
    loss[None] = ti.sin(x[None])

np_rand = np.random.rand(4, 4)
imag_depth.from_numpy(np_rand)
x[None] = 2

print(imag_depth)


calc_some_loss()
calc_some_loss.grad()

print(loss)
#print(x.grad[None])

#calc_some_loss.grad()

print(imag_depth.grad)