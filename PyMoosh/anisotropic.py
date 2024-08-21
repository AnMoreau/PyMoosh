"""
This file contains all functions for anisotropic material computations
"""
import numpy as np
from numpy import linalg as la_np



def calc_cp(Sx, Sy):
    """
        Compute poynting in the x direction in the x,y plane
    """
    deno = (np.abs(Sx) ** 2 + np.abs(Sy) ** 2)
    if deno == 0:
        return 0
    else:
        return np.abs(Sx) ** 2 / deno


def Halfspace_method(Structure, layer_number, wl, theta_entry): #AV_Added#
        """
        This function calculates the HalfSpace's eigenvectors and eigenvalues analytically for a given layer and returns them sorted.

        Args:
        Structure (class): description of the multi-mayered
        layer_number (int): position in the stack of the studied layer
        wl (float): the working wavelength (in nanometers)
        theta_entry (float): angle of the incident light

        return: p_sorted 4x4 array of eigenvectors whose columns are the same as p_unsorted's but sorted
        and q_sorted 4x4 diagonal matrix of eigenvalues whose columns are the same as q_unsorted's but sorted
        """
        k_0 = 2 * np.pi / wl
        eps_entry = Structure.permittivity_tensor_list(wl)[layer_number]
        n = np.sqrt(eps_entry[0,0])
        Kx = n * np.sin(theta_entry)


        # Getting the angles necessary for the eigen vectors of the halfspac
        sin_phi = Kx / n
        cos_phi = np.sqrt(1 - sin_phi ** 2 + 0j)
        # q_sorted = [n * cos_phi, n * cos_phi, -n * cos_phi, -n * cos_phi]
        p_sorted_mat = np.array([[cos_phi, 0, cos_phi, 0],
                                [n, 0, -n, 0],
                                [0, 1, 0, 1],
                                [0, n * cos_phi, 0, -n * cos_phi]])
        Q = np.array([1,1,1,1], dtype=complex) #ref: PyLlama p14

        return p_sorted_mat, Q


def Berreman_method(Structure, layer_number, wl, theta_entry): #AV_Added#
    """
    This function computes Berreman's matrix D for a given layer "layer_number" in the stack and its associated eigenvalues q and associated eigenvectors for a given layer "layer_number" in the stack .
    Then P (interface matrix) and Q (propagation matirix) are computed layer in the stack

    Args:
    Structure (class): description of the multi-mayered
    layer_number (int): position in the stack of the studied layer
    wl (float): the working wavelength (in nanometers)
    theta_entry (float): angle of the incident light

    return: Delta, P and Q matrices as 4x4 Numpy array

    """
    k_0 = 2 * np.pi / wl
    eps_entry = Structure.permittivity_tensor_list(wl, layer=0)
    n_entry = np.sqrt(eps_entry[0,0])
    Kx = n_entry * np.sin(theta_entry)

    eps = Structure.permittivity_tensor_list(wl, layer=layer_number)

    eps_R = Structure.rotate_permittivity_tensor(eps, Structure.ani_rot_angle[layer_number], Structure.ani_rot_axis[layer_number])

    # Delta matrix  (i.e Berreman matrix)

    eps_xx = eps_R[0, 0]
    eps_xy = eps_R[0, 1]
    eps_yx = eps_R[1, 0]
    eps_yy = eps_R[1, 1]
    eps_xz = eps_R[0, 2]
    eps_zx = eps_R[2, 0]
    eps_zz = eps_R[2, 2]
    eps_zy = eps_R[2, 1]
    eps_yz = eps_R[1, 2]
    Delta = np.array([[- Kx * eps_zx / eps_zz, 1 - Kx ** 2 / eps_zz, - Kx * eps_zy / eps_zz, 0],
                      [eps_xx - (eps_xz * eps_zx) / eps_zz, - Kx * eps_xz / eps_zz, eps_xy - (eps_xz * eps_zy) / eps_zz, 0],
                      [0, 0, 0, 1],
                      [eps_yx - (eps_yz * eps_zx) / eps_zz, - Kx * eps_yz / eps_zz, - Kx ** 2 + eps_yy - (eps_yz * eps_zy) / eps_zz, 0]])

    q,P  = la_np.eig(Delta)
    # q is the row vector containing unsorted Delta's eigenvalues and
    # # P is the array which column P[:,i] is the eigenvector corresponding to the eigenvalues q[i].

    Q = np.array([np.exp(1j*k_0*Structure.thickness[layer_number]*q[0]), np.exp(1j*k_0*Structure.thickness[layer_number]*q[1])
                , np.exp(1j*k_0*Structure.thickness[layer_number]*q[2]), np.exp(1j*k_0*Structure.thickness[layer_number]*q[3])] )
    # Propagation matrix

    # Sorting wavevectors and eigenvalues to construct "forward" and "backward" matrices
    thr = 1e-7 # birefringence detection threshold
    id_refl = []
    id_trans = []
    Ex_list=[]
    Ey_list=[]
    Sx_list=[]
    Sy_list=[]
    Sz_list=[]
    for k in range(0, 4, 1):
        # Computing fields => ref. PyLlama equation (7) p9) :
        Ex, Ey, Hx, Hy = P[0,k], P[2,k], -P[3,k], P[1,k]
        Ez = - (eps_zx / eps_zz) * Ex - (eps_zy / eps_zz) * Ey - (Kx / eps_zz) * Hy
        Hz = Kx * Ey

        # Computing Poynting vector :
        Sx = Ey * Hz - Ez * Hy
        Sy = Ez * Hx - Ex * Hz
        Sz = Ex * Hy - Ey * Hx
        Ex_list.append(Ex)
        Ey_list.append(Ey)
        Sx_list.append(Sx)
        Sy_list.append(Sy)
        Sz_list.append(Sz)
        # poynting = [Sx, Sy, Sz]
        if np.isreal(Sz):
            # We sort first along real Poynting values
            test_variable = np.real(Sz)
        else:
            # and then we sort complex Poynting values wrt their imaginary parts
            test_variable = np.imag(Sz)
        if test_variable > 0:
            # There'll always be one positive and one negative value.
            # The positive is going down (like transmission)
            id_trans.append(k)
        else:
            # The negative is going up (like reflection)
            id_refl.append(k)
    Cp0 = calc_cp(Sx_list[id_trans[0]], Sy_list[id_trans[0]])
    Cp1 = calc_cp(Sx_list[id_trans[1]], Sy_list[id_trans[1]])
    if np.abs(Cp0 - Cp1) > thr:
        # It is birefringent, we sort according to Poynting vectors
        if Cp1 < Cp0:
            id_trans = [id_trans[1], id_trans[0]]
        Cp0 = calc_cp(Sx_list[id_refl[0]], Sy_list[id_refl[0]])
        Cp1 = calc_cp(Sx_list[id_refl[1]], Sy_list[id_refl[1]])
        if Cp1 < Cp0:
            id_refl = [id_refl[1], id_refl[0]]
    else:
        # It is not birefringent, we sort according to Electric fields
        Cp0 = calc_cp(Ex_list[id_trans[0]], Ey_list[id_trans[0]])
        Cp1 = calc_cp(Ex_list[id_trans[1]], Ey_list[id_trans[1]])
        if (Cp1 - Cp0) < thr:
            id_trans = [id_trans[1], id_trans[0]]
        Cp0 = calc_cp(Ex_list[id_refl[0]], Ey_list[id_refl[0]])
        Cp1 = calc_cp(Ex_list[id_refl[1]], Ey_list[id_refl[1]])
        if (Cp1 - Cp0) < thr:
            id_refl = [id_refl[1], id_refl[0]]

    # Sort p and q with the newly computed order
    order = [id_trans[1], id_trans[0], id_refl[1], id_refl[0]]

    Q = np.array([Q[order[0]], Q[order[1]], Q[order[2]], Q[order[3]]])

    P = np.stack((P[:, order[0]].transpose(), P[:, order[1]].transpose(),
                            P[:, order[2]].transpose(), P[:, order[3]].transpose()), axis=1)

    return Delta,P,Q


def build_scattering_matrix_to_next(P_a, Q_a, P_b):
    """
    This function constructs the scattering matrix S_{ab} between two successive layers a and
    b by taking into acount the following phenomena:
    - the propagation through the first layer with the propagation matrix Q_a of layer_a
    - the transition from the first layer (layer_a's matrix P_a) and the second layer (layer_b's matrix P_b)

    Args:
        transition matrix : P_a
        propagation matrix : Q_a
        transition matrix : P_b

    :return: partial scattering matrix from layer a to layer b, a 4x4 Numpy array
    """

    Q_forward = np.diag([Q_a[0], Q_a[1], 1, 1])

    Q_backward = np.diag([1, 1, Q_a[2], Q_a[3]])


    P_out = np.block([[P_a[:,:2], -P_b[:,2:]]])
    # P_out = np.array([
    #     [P_a[0, 0], P_a[0, 1], -P_b[0, 2], -P_b[0, 3]],
    #     [P_a[1, 0], P_a[1, 1], -P_b[1, 2], -P_b[1, 3]],
    #     [P_a[2, 0], P_a[2, 1], -P_b[2, 2], -P_b[2, 3]],
    #     [P_a[3, 0], P_a[3, 1], -P_b[3, 2], -P_b[3, 3]]
    # ])

    P_in = np.block([P_b[:,:2], -P_a[:,2:]])
    # P_in = np.array([
    #     [P_b[0, 0], P_b[0, 1], -P_a[0, 2], -P_a[0, 3]],
    #     [P_b[1, 0], P_b[1, 1], -P_a[1, 2], -P_a[1, 3]],
    #     [P_b[2, 0], P_b[2, 1], -P_a[2, 2], -P_a[2, 3]],
    #     [P_b[3, 0], P_b[3, 1], -P_a[3, 2], -P_a[3, 3]]
    # ])
    Sab = la_np.multi_dot((la_np.inv(Q_backward), la_np.inv(P_in), P_out, Q_forward))
    return Sab


def combine_scattering_matrices(S_ab, S_bc): #AV_Aded#
    """
    This function constructs the scattering matrix between three successive layers a, b and c by combining
    the scattering matrices S_{ab} from layer a to layer b and S_{bc} from layer b to layer c.

    :param ndarray S_ab: the scattering matrix from layer a to layer b, a 4x4 Numpy array
    :param ndarray S_bc: the scattering matrix from layer b to layer c, a 4x4 Numpy array
    :return: partial scattering matrix from layer a to layer c, a 4x4 Numpy array
    """
    S_ab00 = np.array([
        [S_ab[0, 0], S_ab[0, 1]],
        [S_ab[1, 0], S_ab[1, 1]],
    ])
    S_ab01 = np.array([
        [S_ab[0, 2], S_ab[0, 3]],
        [S_ab[1, 2], S_ab[1, 3]],
    ])
    S_ab10 = np.array([
        [S_ab[2, 0], S_ab[2, 1]],
        [S_ab[3, 0], S_ab[3, 1]],
    ])
    S_ab11 = np.array([
        [S_ab[2, 2], S_ab[2, 3]],
        [S_ab[3, 2], S_ab[3, 3]],
    ])
    S_bc00 = np.array([
        [S_bc[0, 0], S_bc[0, 1]],
        [S_bc[1, 0], S_bc[1, 1]],
    ])
    S_bc01 = np.array([
        [S_bc[0, 2], S_bc[0, 3]],
        [S_bc[1, 2], S_bc[1, 3]],
    ])
    S_bc10 = np.array([
        [S_bc[2, 0], S_bc[2, 1]],
        [S_bc[3, 0], S_bc[3, 1]],
    ])
    S_bc11 = np.array([
        [S_bc[2, 2], S_bc[2, 3]],
        [S_bc[3, 2], S_bc[3, 3]],
    ])
    C = la_np.inv(np.identity(2) - np.dot(S_ab01, S_bc10))
    S_ac00 = la_np.multi_dot((S_bc00, C, S_ab00))
    S_ac01 = S_bc01 + la_np.multi_dot((S_bc00, C, S_ab01, S_bc11))
    S_ac10 = S_ab10 + la_np.multi_dot((S_ab11, S_bc10, C, S_ab00))
    S_ac11 = la_np.multi_dot((S_ab11, (np.identity(2) + la_np.multi_dot((S_bc10, C, S_ab01))), S_bc11))

    S_ac = np.array([
        [S_ac00[0, 0], S_ac00[0, 1], S_ac01[0, 0], S_ac01[0, 1]],
        [S_ac00[1, 0], S_ac00[1, 1], S_ac01[1, 0], S_ac01[1, 1]],
        [S_ac10[0, 0], S_ac10[0, 1], S_ac11[0, 0], S_ac11[0, 1]],
        [S_ac10[1, 0], S_ac10[1, 1], S_ac11[1, 0], S_ac11[1, 1]]
    ])
    return S_ac


def coefficients_ani(structure, wl, theta_inc): #AV_Aded# automatic calculation of coefficients r_kj and t_kj using the previously defined functions
    """
    This function returns the four reflection and transmission coefficients of the all structure from its global scattering matrix.
    To get P and Q for each layer, calculation of the eigenvalues and eigenvectors are done analytically for the superstrate and the substrate with Halfspace_method()
    In every other layer we computes these e.val and e.vect with Berreman_method() (containing the sorting algorithm).

    Args:
    Structure (class): description of the multi-mayered
    wl (float): the working wavelength (in nanometers)
    theta_inc (float): angle of the incident light

    return: t_pp, t_ps, t_sp, t_ss, r_pp, r_ps, r_sp and r_ss

    """
    # step 1: create all P, Q matrices (interface and propagation resp.)
    P_list = []
    Q_list = []
    for layer_number in range(len(structure.layer_type)):
        # print(layer_number, structure.layer_type)
        if (layer_number == 0) or (layer_number == len(structure.layer_type) -1):
            P, Q = Halfspace_method(structure, layer_number, wl, theta_inc)
            P_list.append(P)
            Q_list.append(Q)

        else:
            D, P, Q = Berreman_method(structure, layer_number, wl, theta_inc)
            P_list.append(P)
            Q_list.append(Q)

    # step 2 : create S matrices
    S_last_period = np.identity(4)
    for lay in range(len(structure.layer_type)-2, -1, -1):
        Sab = build_scattering_matrix_to_next(P_list[lay], Q_list[lay], P_list[lay +1])
        S_last_period = combine_scattering_matrices(Sab, S_last_period)


    # step 3 : extract r and t coefficients (r_kj <=> reflection coeff. of a k polarized incident wave reflected in a j polarized wave)
                                           #(t_kj <=> transmison coeff. of a k polarized incident wave transmitted in a j polarized wave)

    t_pp = S_last_period[0,0]
    t_ps = S_last_period[0,1]
    t_sp = S_last_period[1,0]
    t_ss = S_last_period[1,1]


    r_pp = S_last_period[2,0]
    r_ps = S_last_period[2,1]
    r_sp = S_last_period[3,0]
    r_ss = S_last_period[3,1]


    return t_pp,t_ps,t_sp,t_ss,r_pp,r_ps,r_sp,r_ss

