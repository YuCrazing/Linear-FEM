import taichi as ti
import tetgen_parser

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch, debug=True)

dt = 1e-4
substeps = int(1 / 60 // dt)

E = 1e3
nu = 0.2
rho = 1.0
friction_coeff = 0.01
gravity = ti.Vector([0, -9.8, 0])
damping = 15.0


NEO_HOOKEAN = 0
COROTATED = 1
FIXED_COROTATED = 2

# Invertible corotated model is equivalent to the fixed corotated model but avoid F^{-T} in the computation of P
# From paper "Energetically Consistent Invertible Elasticity"
INVERTIBLE_COROTATED = 3

# From paper "Stable Neo-Hookean Flesh Simulation"
INVERTIBLE_NEOHOOKEAN = 4

CONSTITUTIVE_MODEL = INVERTIBLE_NEOHOOKEAN

# nodes, faces, elements = tetgen_parser.parse_tetgen_output('./data/minimal.1')
# nodes, faces, elements = tetgen_parser.parse_tetgen_output('./data/cube.1')
nodes, faces, elements = tetgen_parser.parse_tetgen_output('./data/cube.2')
# nodes, faces, elements = tetgen_parser.parse_tetgen_output('./data/cube.3')

vn = nodes.shape[0]
fn = faces.shape[0]
en = elements.shape[0]
faces = faces.reshape((fn*3, ))

x = ti.Vector.field(3, dtype=float, shape=vn)
v = ti.Vector.field(3, dtype=float, shape=vn)
f = ti.Vector.field(3, dtype=float, shape=vn)

tet_indices = ti.Vector.field(4, int, shape=en)
tri_indices = ti.field(int, shape=fn*3)
all_tri_indices = ti.field(int, shape=en*4*3)
vertices = ti.Vector.field(3, dtype=float, shape=vn)

vertices.from_numpy(nodes)
tri_indices.from_numpy(faces)
tet_indices.from_numpy(elements)

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


    for I in ti.grouped(tet_indices):
        all_tri_indices[I*12+0], all_tri_indices[I*12+1], all_tri_indices[I*12+2] = tet_indices[I][0], tet_indices[I][2], tet_indices[I][1]
        all_tri_indices[I*12+3], all_tri_indices[I*12+4], all_tri_indices[I*12+5] = tet_indices[I][0], tet_indices[I][1], tet_indices[I][3]
        all_tri_indices[I*12+6], all_tri_indices[I*12+7], all_tri_indices[I*12+8] = tet_indices[I][0], tet_indices[I][3], tet_indices[I][2]
        all_tri_indices[I*12+9], all_tri_indices[I*12+10], all_tri_indices[I*12+11] = tet_indices[I][1], tet_indices[I][2], tet_indices[I][3]

    offset = ti.Vector([0.0, 2.0, 0.0])
    for i in range(vn):
        vertices[i] += offset

    for i in range(vn):
        x[i] = vertices[i]
        x[i].y = 0.5
        x[i] = ti.Vector([ti.random(), ti.random(), ti.random()])
        v[i] = ti.Vector([0.0, 0.0, 0.0])

@ti.func
def tensor_contraction(A: ti.template(), B: ti.template()):
    res = 0.0
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            ti.atomic_add(res, A[i, j] * B[i, j])
    return res

@ti.func
def col(M: ti.template(), i: ti.template()):
    return ti.Vector([M[0, i], M[1, i], M[2, i]])

@ti.kernel
def substep():

    for i in range(vn):
        f[i] = ti.Vector([0.0, 0.0, 0.0])

    for ei in ti.grouped(tet_indices):
        a, b, c, d = tet_indices[ei]
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
        elif CONSTITUTIVE_MODEL == INVERTIBLE_COROTATED:
            I = ti.Matrix.identity(ti.f32, 3)
            U, sig, V = ti.svd(F)
            # Psi = mu * ((sig[0, 0]-1)**2+(sig[1, 1]-1)**2+(sig[2, 2]-1)**2) + 0.5*la * (J-1)**2
            J_sig_inv = ti.Matrix([[sig[1, 1]*sig[2, 2], 0.0, 0.0], [0.0, sig[2, 2]*sig[0, 0], 0.0], [0, 0, sig[0, 0]*sig[1, 1]]])
            # P_hat = 2 * mu * (sig - I) + la * sig.trace() * I
            P_hat = 2 * mu * (sig - I) + la * (J-1) * J_sig_inv
            P = U @ P_hat @ V.transpose()
        elif CONSTITUTIVE_MODEL == INVERTIBLE_NEOHOOKEAN:
            Ic = (F.transpose() @ F).trace()
            I2 = Ic
            I3 = J
            U, sig, V = ti.svd(F)

            # Paper "Stable Neo-Hookean Flesh Simulation"
            alpha = 1 + mu/la - mu/(4*la)
            f0, f1, f2 = col(F, 0), col(F, 1), col(F, 2)
            # Psi = 0.5*mu*(Ic-3)+0.5*la*(J-alpha)**2 - 0.5*ti.log(Ic+1)
            dJdF = ti.Matrix.cols([f1.cross(f2), f2.cross(f0), f0.cross(f1)])
            J_sig_inv = ti.Matrix([[sig[1, 1]*sig[2, 2], 0.0, 0.0], [0.0, sig[2, 2]*sig[0, 0], 0.0], [0, 0, sig[0, 0]*sig[1, 1]]])
            P = mu * (1 - 1/(Ic+1)) * F + la * (I3-alpha) * dJdF
            # P = mu * F + la * (I3-alpha) * U @ J_sig_inv @ V.transpose() - mu / (Ic+1) * F

            # # Course "Dynamic Deformables: Implementation and Production Practicalities"
            # alpha = 1 + mu/la
            # # Psi = 0.5*mu*(I2-3)+0.5*la*(I3-1)**2 - mu*(I3-1)
            # J_sig_inv = ti.Matrix([[sig[1, 1]*sig[2, 2], 0.0, 0.0], [0.0, sig[2, 2]*sig[0, 0], 0.0], [0, 0, sig[0, 0]*sig[1, 1]]])
            # P = mu * F + la * (I3-alpha) * U @ J_sig_inv @ V.transpose() - mu * U @ J_sig_inv @ V.transpose()
            

        # acc = -(\partial U / \partial x) / m
        #             = -V0 * P : (\partial F / \partial x) / (V * rho)
        #             = - (V0/V) * P : (\partial F / \partial x) / rho
        #             = - 1/det(F) * P : (\partial F / \partial x) / rho

        # A hack to avoid division by zero
        J = 1.0
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
        v[i] *= ti.exp(-dt * damping)
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
    window = ti.ui.Window("Invertible Constitutive Model", (800, 800), vsync=True)
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
        scene.mesh(x, indices=tri_indices, two_sided=True, color=(0.7, 0.0, 0.0), show_wireframe=True)
        # scene.mesh(x, indices=all_tri_indices, two_sided=True, color=(0.7, 0.0, 0.0), show_wireframe=True)
        
        canvas.scene(scene)
        window.show()

    # TODO: include self-collision handling


if __name__ == "__main__":
    main()
