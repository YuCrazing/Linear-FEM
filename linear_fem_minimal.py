import taichi as ti

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)

dt = 4e-3
substeps = int(1 / 60 // dt)

E = 1e3
nu = 0.3
rho = 1.0
friction_coeff = 0.01
gravity = ti.Vector([0, -9.8, 0])

NEO_HOOKEAN = 0
COROTATED = 1
FIXED_COROTATED = 2

CONSTITUTIVE_MODEL = NEO_HOOKEAN

vn = 4

x = ti.Vector.field(3, dtype=float, shape=vn)
v = ti.Vector.field(3, dtype=float, shape=vn)
f = ti.Vector.field(3, dtype=float, shape=vn)

tet_indices = ti.Vector.field(4, int, shape=1)
tri_indices = ti.field(int, shape=4*3)
vertices = ti.Vector.field(3, dtype=float, shape=vn)


plane_height = 0.0
plane_half_length = 10.0
plane_vertices = ti.Vector.field(3, dtype=float, shape=4)
plane_indices = ti.field(int, shape=2*3)


@ti.kernel
def init():
    plane_vertices[0] = ti.Vector([-plane_half_length, plane_height, -plane_half_length])
    plane_vertices[1] = ti.Vector([-plane_half_length, plane_height, plane_half_length])
    plane_vertices[2] = ti.Vector([plane_half_length, plane_height, plane_half_length])
    plane_vertices[3] = ti.Vector([plane_half_length, plane_height, -plane_half_length])
    plane_indices[0], plane_indices[1], plane_indices[2] = 0, 1, 2
    plane_indices[3], plane_indices[4], plane_indices[5] = 0, 2, 3


    tet_indices[0] = [0, 1, 2, 3]
    tri_indices[0], tri_indices[1], tri_indices[2] = 0, 2, 1
    tri_indices[3], tri_indices[4], tri_indices[5] = 0, 1, 3
    tri_indices[6], tri_indices[7], tri_indices[8] = 0, 3, 2
    tri_indices[9], tri_indices[10], tri_indices[11] = 1, 2, 3

    scale = 1.0
    center = ti.Vector([0.0, 2.0, 0.0])
    vertices[0] = ti.Vector([ti.sqrt(8/9), 0, -1/3]) * scale + center
    vertices[1] = ti.Vector([-ti.sqrt(2/9), ti.sqrt(2/3), -1/3]) * scale + center
    vertices[2] = ti.Vector([-ti.sqrt(2/9), -ti.sqrt(2/3), -1/3]) * scale + center
    vertices[3] = ti.Vector([0, 0, 1]) * scale + center

    for i in range(vn):
        x[i] = vertices[i]
        v[i] = ti.Vector([0.0, 0.0, 0.0])

@ti.func
def tensor_contraction(A: ti.template(), B: ti.template()):
    res = 0.0
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            ti.atomic_add(res, A[i, j] * B[i, j])
    return res


@ti.kernel
def substep():

    for i in range(vn):
        f[i] = ti.Vector([0.0, 0.0, 0.0])
    
    for I in ti.grouped(tet_indices):
        a, b, c, d = tet_indices[I]
        pa, pb, pc, pd = x[a], x[b], x[c], x[d]
        pa0, pb0, pc0, pd0 = vertices[a], vertices[b], vertices[c], vertices[d]
        D = ti.Matrix.cols([pa - pd, pb - pd, pc - pd])
        D0 = ti.Matrix.cols([pa0 - pd0, pb0 - pd0, pc0 - pd0])
        D0_inv = D0.inverse()
        F = D @ D0_inv
        J = F.determinant()
        F_INV_T = F.inverse().transpose()
        mu, la = E / (
            2 * (1 + nu)), E* nu / (
                (1 + nu) * (1 - 2 * nu))  # Lame parameters

        P = ti.Matrix.zero(ti.f32, 3, 3)

        if CONSTITUTIVE_MODEL == NEO_HOOKEAN:
            P = mu * (F - F_INV_T) + la * ti.log(J) * F_INV_T
        elif CONSTITUTIVE_MODEL == COROTATED:
            U, sig, V = ti.svd(F)
            R = U @ V.transpose()
            P = 2 * mu * (F - R) + la * (R.transpose() @ F - ti.Matrix.identity(ti.f32, 3)).trace() * R
        elif CONSTITUTIVE_MODEL == FIXED_COROTATED:
            U, sig, V = ti.svd(F)
            R = U @ V.transpose()
            P = 2 * mu * (F - R) + la * (J - 1) * J * F_INV_T

        # acc = -(\partial U / \partial x) / m
        #             = -V0 * P : (\partial F / \partial x) / (V * rho)
        #             = - (V0/V) * P : (\partial F / \partial x) / rho
        #             = - 1/det(F) * P : (\partial F / \partial x) / rho

        f[a].x += tensor_contraction(
            P,
            ti.Matrix([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]) @ D0_inv) / J
        f[a].y += tensor_contraction(
            P,
            ti.Matrix([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]) @ D0_inv) / J
        f[a].z += tensor_contraction(
            P,
            ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) @ D0_inv) / J

        f[b].x += tensor_contraction(
            P,
            ti.Matrix([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]) @ D0_inv) / J
        f[b].y += tensor_contraction(
            P,
            ti.Matrix([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]) @ D0_inv) / J
        f[b].y += tensor_contraction(
            P,
            ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) @ D0_inv) / J

        f[c].x += tensor_contraction(
            P,
            ti.Matrix([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]) @ D0_inv) / J
        f[c].y += tensor_contraction(
            P,
            ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]) @ D0_inv) / J
        f[c].z += tensor_contraction(
            P,
            ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) @ D0_inv) / J

        f[d].x += tensor_contraction(
            P,
            ti.Matrix([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]) @ D0_inv) / J
        f[d].y += tensor_contraction(
            P,
            ti.Matrix([[0.0, 0.0, 0.0], [-1.0, -1.0, -1.0], [0.0, 0.0, 0.0]]) @ D0_inv) / J
        f[d].z += tensor_contraction(
            P,
            ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]) @ D0_inv) / J

    for i in range(vn):
        v[i] += dt * (-f[i] / rho + gravity)
        x[i] += dt * v[i]

        if x[i].y < plane_height:
            x[i].y = plane_height
            n_outward = ti.Vector([0.0, 1.0, 0.0])
            v_co = ti.Vector([0.0, 0.0, 0.0])
            v_rel = v[i] - v_co
            v_n_l = v_rel.dot(n_outward)
            v_t = v_rel - v_n_l * n_outward
            v_rel -= ti.min(v_n_l, 0.0) * n_outward
            if v_n_l < 0.0:
                v_rel -= ti.min(friction_coeff, 1.0) * v_t
            v[i] = v_rel + v_co


def main():
    window = ti.ui.Window("Linear FEM", (800, 800), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = window.get_scene()
    camera = ti.ui.Camera()

    camera.position(0.0, 1.0, 10)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    init()

    while window.running:

        for i in range(substeps):
            substep()

        camera.track_user_inputs(window, movement_speed=0.3, hold_key=ti.ui.LMB)
        scene.set_camera(camera)

        scene.point_light(pos=(0, 5, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))

        scene.mesh(plane_vertices, indices=plane_indices, two_sided=True, color=(0.5, 0.5, 0.5))
        scene.mesh(x, indices=tri_indices, two_sided=True, color=(0.7, 0.0, 0.0))
        
        canvas.scene(scene)
        window.show()

    # TODO: include self-collision handling


if __name__ == "__main__":
    main()
