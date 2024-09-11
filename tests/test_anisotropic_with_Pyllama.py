import numpy as np
import pyllama as pl
import PyMoosh as PM
import PyMoosh.anisotropic as ani
from numpy import linalg as la_np

wl_nm = 640

rot_angle = 0

rot_axis = 'z'

thickness_nm = 100

theta_in_rad = np.pi/6 # angle d'incidence

perm_1 = 1.2

perm_2 = 2.2


opt_ind = [perm_1, perm_2, perm_2]
mat_1=ani.AniMaterial(opt_ind, specialType="ANI")

mat_2=PM.Material(1.0)

material_list = [mat_1, mat_2]
stack = [1, 0, 0, 1]
thickness = [0, thickness_nm, 25, 0]#LES EPAISSEUR DU MILIEU EXT A 0 nm
ani_rot_angle = [0., 0.5,-0.2, 0.]
ani_rot_axis = ['z', 'x', [0,1,1], 'z']


structure1 = ani.AniStructure(material_list, stack, thickness, ani_rot_angle, ani_rot_axis, verbose=False)
epsilon1 = structure1.rotate_permittivity_tensor(np.diag(opt_ind), ani_rot_angle=ani_rot_angle[1], ani_rot_axis=ani_rot_axis[1])
epsilon2 = structure1.rotate_permittivity_tensor(np.diag(opt_ind), ani_rot_angle=ani_rot_angle[2], ani_rot_axis=ani_rot_axis[2])
n_entry = np.sqrt(1.0) # ATTENTION CHEZ PYMOOSH ON FAIT MATERIAL(PERM)
n_exit = np.sqrt(1.0)
k0 = 2 * np.pi / wl_nm

thetas = np.linspace(0,80,80)*np.pi/180
l_rpp_moosh = []
l_rpp_llama = []
l_rps_moosh = []
l_rps_llama = []
l_rsp_moosh = []
l_rsp_llama = []
l_rss_moosh = []
l_rss_llama = []
l_tpp_moosh = []
l_tpp_llama = []
l_tps_moosh = []
l_tps_llama = []
l_tsp_moosh = []
l_tsp_llama = []
l_tss_moosh = []
l_tss_llama = []


for theta_in_rad in thetas:
    print(theta_in_rad)
    Kx = n_entry * np.sin(theta_in_rad)
    Kz_entry = n_entry * np.cos(theta_in_rad)
    theta_out_rad = np.arcsin((n_entry / n_exit) * np.sin(theta_in_rad))
    Kz_exit = n_exit * np.cos(theta_out_rad)

    epsilon_entry = np.array([[n_entry ** 2, 0, 0],
                                [0, n_entry ** 2, 0],
                                [0, 0, n_entry ** 2]])

    epsilon_exit = np.array([[n_exit ** 2, 0, 0],
                                [0, n_exit ** 2, 0],
                                [0, 0, n_exit ** 2]])

    entry = pl.HalfSpace(epsilon_entry, Kx, Kz_entry, k0)
    exit = pl.HalfSpace(epsilon_exit, Kx, Kz_exit, k0)

    Ky = 0
    Kz = n_entry * np.cos(theta_in_rad)
    my_stack_structure = pl.Structure(entry, exit, Kx, Ky, Kz_entry, Kz_exit, k0)#at this point my_stack_structure represent only two half-spaces


    my_layer = pl.Layer(epsilon1, thickness_nm, Kx, k0)
    my_stack_structure.add_layer(my_layer) # remove_layer and replace_layer() also exist
    my_layer = pl.Layer(epsilon2, 25, Kx, k0)
    my_stack_structure.add_layer(my_layer) 


    my_stack_structure.build_scattering_matrix()  # scattering matrix
    # print("PYLLAMA LAYERS", my_stack_structure.layers[0].D)
    # print("PYLLAMA LAYERS", my_stack_structure.layers[0].Q)

    # And we get refl / trans coefficients with
    J_refl_lin, J_trans_lin = my_stack_structure.get_fresnel()
    l_tpp_llama.append(J_trans_lin[0,0])
    l_tps_llama.append(J_trans_lin[0,1])
    l_tsp_llama.append(J_trans_lin[1,0])
    l_tss_llama.append(J_trans_lin[1,1])
    l_rpp_llama.append(J_refl_lin[0,0])
    l_rps_llama.append(J_refl_lin[0,1])
    l_rsp_llama.append(J_refl_lin[1,0])
    l_rss_llama.append(J_refl_lin[1,1])


    res = ani.coefficients_ani(structure1, wl_nm, theta_in_rad)
    l_tpp_moosh.append(res[0])
    l_tps_moosh.append(res[1])
    l_tsp_moosh.append(res[2])
    l_tss_moosh.append(res[3])
    l_rpp_moosh.append(res[4])
    l_rps_moosh.append(res[5])
    l_rsp_moosh.append(res[6])
    l_rss_moosh.append(res[7])

import matplotlib.pyplot as plt
plt.subplot(2,2,1)
plt.plot(thetas*180/np.pi, l_tpp_llama, label='tpp llama')
plt.plot(thetas*180/np.pi, l_tpp_moosh, label='tpp moosh', linestyle='--')
plt.plot(thetas*180/np.pi, l_tss_llama, label='tss llama')
plt.plot(thetas*180/np.pi, l_tss_moosh, label='tss moosh', linestyle='--')
plt.legend()

plt.subplot(2,2,2)
plt.plot(thetas*180/np.pi, l_rpp_llama, label='rpp llama')
plt.plot(thetas*180/np.pi, l_rpp_moosh, label='rpp moosh', linestyle='--')
plt.plot(thetas*180/np.pi, l_rss_llama, label='rss llama')
plt.plot(thetas*180/np.pi, l_rss_moosh, label='rss moosh', linestyle='--')
plt.legend()

plt.subplot(2,2,3)
plt.plot(thetas*180/np.pi, l_tsp_llama, label='tsp llama')
plt.plot(thetas*180/np.pi, l_tsp_moosh, label='tsp moosh', linestyle='--')
plt.plot(thetas*180/np.pi, l_tps_llama, label='tps llama')
plt.plot(thetas*180/np.pi, l_tps_moosh, label='tps moosh', linestyle='--')
plt.legend()

plt.subplot(2,2,4)
plt.plot(thetas*180/np.pi, l_rsp_llama, label='rsp llama')
plt.plot(thetas*180/np.pi, l_rsp_moosh, label='rsp moosh', linestyle='--')
plt.plot(thetas*180/np.pi, l_rps_llama, label='rps llama')
plt.plot(thetas*180/np.pi, l_rps_moosh, label='rps moosh', linestyle='--')
plt.legend()
plt.show()

"""
REFL PYLLAMA[[pp, ps],[sp, ss]] [[ 0.16188752-0.04516487j -0.05824699-0.0139216j ]
 [ 0.02671249+0.00853741j -0.62690456+0.17740569j]]
TRANS PYLLAMA[[pp, ps],[sp, ss]] [[0.34817659+0.92131957j 0.02982136+0.00448614j]
 [0.02982136+0.00448614j 0.20989743+0.72592355j]]
 t_pp,t_ps,t_sp,t_ss, ((0.3481765881478537+0.9213195674475575j), (0.029821358152069155+0.004486139750694154j), (0.02982135815206921+0.00448613975069413j), (0.20989742722970214+0.7259235488651471j)
 r_pp,r_ps,r_sp,r_ss, (0.1618875201114852-0.04516487426543376j), (-0.05824699196060018-0.013921603350022954j), (0.026712485577587423+0.00853740774923072j), (-0.6269045611049386+0.177405694798719j))
"""