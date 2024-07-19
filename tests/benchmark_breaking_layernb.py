import numpy as np
from context import PM
import matplotlib.pyplot as plt
import itertools as iter

## Bragg mirror with increasing number of layers
layers = np.arange(1, 81, 3)
angles = np.arange(0, 81, 10)
unit = "nm"
wav = 600

mats1 = [1.1, 1.5, "Gold"]
mats2 = [1.2, 2]
eps1 = [1, 100, 200]
eps2 = [2, 200, 400]
for mat1, mat2, ep1, ep2 in iter.product(mats1, mats2, eps1, eps2):
    rs_s_te = np.zeros((len(layers), len(angles)))
    ts_s_te = np.zeros((len(layers), len(angles)))
    rs_s_tm = np.zeros((len(layers), len(angles)))
    ts_s_tm = np.zeros((len(layers), len(angles)))

    rs_t_te = np.zeros((len(layers), len(angles)))
    ts_t_te = np.zeros((len(layers), len(angles)))
    rs_t_tm = np.zeros((len(layers), len(angles)))
    ts_t_tm = np.zeros((len(layers), len(angles)))

    rs_a_te = np.zeros((len(layers), len(angles)))
    ts_a_te = np.zeros((len(layers), len(angles)))
    rs_a_tm = np.zeros((len(layers), len(angles)))
    ts_a_tm = np.zeros((len(layers), len(angles)))

    rs_i_te = np.zeros((len(layers), len(angles)))
    rs_i_tm = np.zeros((len(layers), len(angles)))

    rs_dn_te = np.zeros((len(layers), len(angles)))
    ts_dn_te = np.zeros((len(layers), len(angles)))
    rs_dn_tm = np.zeros((len(layers), len(angles)))
    ts_dn_tm = np.zeros((len(layers), len(angles)))


    if (isinstance(mat1, float)):
        materials = [1, mat1**2, (mat2)**2]
    else:
        materials = [1, mat1, (mat2)**2]

    for ilay, nb_couches in enumerate(layers):
        for iinc, incidence in enumerate(angles):




            ## Case 1: single layer, TE
            #structure = np.random.random(nb_couches*2+1)*w_mean
            structure = np.array([ep1, ep2+100]*nb_couches)

            stack = [0]+[1,2]*nb_couches


            epaisseurs = np.concatenate(([0],structure))
            multi_stack = PM.Structure(materials,stack,epaisseurs, verbose=False, unit=unit, si_units=True)
            r, t, R, T = PM.coefficient_S(multi_stack,wav,incidence,0)
            rs_s_te[ilay, iinc] = R
            ts_s_te[ilay, iinc] = T

            r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(multi_stack,wav,incidence,0)
            rs_a_te[ilay, iinc] = R_ab
            ts_a_te[ilay, iinc] = T_ab

            r_t, t_t, R_t, T_t = PM.coefficient_T(multi_stack,wav,incidence,0)
            rs_t_te[ilay, iinc] = R_t
            ts_t_te[ilay, iinc] = T_t

            r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(multi_stack,wav,incidence,0)
            rs_dn_te[ilay, iinc] = R_dn
            ts_dn_te[ilay, iinc] = T_dn

            r_i, t_i, R_i, T_i = PM.coefficient_I(multi_stack,wav,incidence,0)
            rs_i_te[ilay, iinc] = R_i


            r, t, R, T = PM.coefficient_S(multi_stack,wav,incidence,1)
            rs_s_tm[ilay, iinc] = R
            ts_s_tm[ilay, iinc] = T


            r_ab, t_ab, R_ab, T_ab = PM.coefficient_A(multi_stack,wav,incidence,1)
            rs_a_tm[ilay, iinc] = R_ab
            ts_a_tm[ilay, iinc] = T_ab

            r_t, t_t, R_t, T_t = PM.coefficient_T(multi_stack,wav,incidence,1)
            rs_t_tm[ilay, iinc] = R_t
            ts_t_tm[ilay, iinc] = T_t

            r_dn, t_dn, R_dn, T_dn = PM.coefficient_DN(multi_stack,wav,incidence,1)
            rs_dn_tm[ilay, iinc] = R_dn
            ts_dn_tm[ilay, iinc] = T_dn

            r_i, t_i, R_i, T_i = PM.coefficient_I(multi_stack,wav,incidence,1)
            rs_i_tm[ilay, iinc] = R_i


    # fig, axs = plt.subplots(2, 2, sharex=True, figsize=(10,10))
    # axs[0,0].plot(layers, abs(rs_s_te-rs_a_te)/np.abs(rs_s_te), 'b-v', label="abeles", markersize=4)
    # axs[0,0].plot(layers, abs(rs_s_te-rs_dn_te)/np.abs(rs_s_te), 'r-o', label="D2N", markersize=4)
    # axs[0,0].plot(layers, abs(rs_s_te-rs_t_te)/np.abs(rs_s_te), 'g-^', label="T", markersize=4)
    # axs[0,0].plot(layers, abs(rs_s_te-rs_i_te)/np.abs(rs_s_te), 'c-+', label="Impedance", markersize=4)
    # axs[0,0].set_ylabel("Reflection relative error TE Normal incidence")
    # axs[0,0].set_xlabel("Nb Layers")
    # #axs[0,0].set_ylim([0-.0001,.15])
    # #axs[0,0].set_yscale("log")
    # axs[0,0].legend()


    # axs[0,1].plot(layers, abs(rs_s_tm-rs_a_tm)/np.abs(rs_s_tm), 'b-v', label="abeles", markersize=4)
    # axs[0,1].plot(layers, abs(rs_s_tm-rs_dn_tm)/np.abs(rs_s_tm), 'r-o', label="D2N", markersize=4)
    # axs[0,1].plot(layers, abs(rs_s_tm-rs_t_tm)/np.abs(rs_s_tm), 'g-^', label="T", markersize=4)
    # axs[0,1].plot(layers, abs(rs_s_tm-rs_i_tm)/np.abs(rs_s_tm), 'c-+', label="Impedance", markersize=4)
    # axs[0,1].set_ylabel("Reflection relative error TM Normal incidence")
    # axs[0,1].set_xlabel("Nb Layers")
    # #axs[0,1].set_ylim([-0.001,.15])
    # #axs[0,1].set_yscale("log")
    # axs[0,1].legend()


    # axs[1,0].plot(layers, abs(ts_s_te-ts_a_te)/np.abs(ts_s_te), 'b-v', label="abeles", markersize=4)
    # axs[1,0].plot(layers, abs(ts_s_te-ts_dn_te)/np.abs(ts_s_te), 'r-o', label="D2N", markersize=4)
    # axs[1,0].plot(layers, abs(ts_s_te-ts_t_te)/np.abs(ts_s_te), 'g-^', label="T", markersize=4)
    # axs[1,0].set_ylabel("Transmission relative error TE Normal incidence")
    # axs[1,0].set_xlabel("Nb Layers")
    # axs[1,0].set_ylim([-0.001,.15])
    # #axs[1,0].set_yscale("log")
    # axs[1,0].legend()


    # axs[1,1].plot(layers, abs(ts_s_tm-ts_a_tm)/np.abs(ts_s_tm), 'b-v', label="abeles", markersize=4)
    # axs[1,1].plot(layers, abs(ts_s_tm-ts_dn_tm)/np.abs(ts_s_tm), 'r-o', label="D2N", markersize=4)
    # axs[1,1].plot(layers, abs(ts_s_tm-ts_t_tm)/np.abs(ts_s_tm), 'g-^', label="T", markersize=4)
    # axs[1,1].set_ylabel("Transmission relative error TM Normal incidence")
    # axs[1,1].set_xlabel("Nb Layers")
    # axs[1,1].set_ylim([-0.001,.15])
    # #axs[1,1].set_yscale("log")
    # axs[1,1].legend()
    # plt.tight_layout()
    # plt.show()


    X,Y = np.meshgrid(layers, angles)
    err_ts_t_te = np.abs(ts_t_te - ts_s_te)
    err_ts_t_tm = np.abs(ts_t_tm - ts_s_tm)

    err_ts_a_te = np.abs(ts_a_te - ts_s_te)
    err_ts_a_tm = np.abs(ts_a_tm - ts_s_tm)

    err_ts_dn_te = np.abs(ts_dn_te - ts_s_te)
    err_ts_dn_tm = np.abs(ts_dn_tm - ts_s_tm)

    if np.sum(err_ts_t_te + err_ts_t_tm + err_ts_a_te + err_ts_a_tm + err_ts_dn_te + err_ts_dn_tm)>0:
        fig, ax = plt.subplots(3, 2, figsize=(10,10))
        ax[0,0].pcolormesh(X, Y, err_ts_t_te.T, vmax=1)
        ax[0,0].set_title("Abs error transmission TE T")
        ax[0,0].set_xlabel("Nb Layers")
        ax[0,0].set_ylabel("Inc. Angle")

        ax[1,0].pcolormesh(X, Y, err_ts_a_te.T, vmax=1)
        ax[1,0].set_title("Abs error transmission TE Ab")
        ax[1,0].set_xlabel("Nb Layers")
        ax[1,0].set_ylabel("Inc. Angle")

        ax[2,0].pcolormesh(X, Y, err_ts_dn_te.T, vmax=1)
        ax[2,0].set_title("Abs error transmission TE D2N")
        ax[2,0].set_xlabel("Nb Layers")
        ax[2,0].set_ylabel("Inc. Angle")

        ax[0,1].pcolormesh(X, Y, err_ts_t_tm.T, vmax=1)
        ax[0,1].set_title("Abs error transmission TM T")
        ax[0,1].set_xlabel("Nb Layers")
        ax[0,1].set_ylabel("Inc. Angle")

        ax[1,1].pcolormesh(X, Y, err_ts_a_tm.T, vmax=1)
        ax[1,1].set_title("Abs error transmission TM Ab")
        ax[1,1].set_xlabel("Nb Layers")
        ax[1,1].set_ylabel("Inc. Angle")

        ax[2,1].pcolormesh(X, Y, err_ts_dn_tm.T, vmax=1)
        ax[2,1].set_title("Abs error transmission TM D2N")
        ax[2,1].set_xlabel("Nb Layers")
        ax[2,1].set_ylabel("Inc. Angle")

        plt.tight_layout()
        plt.savefig(f"Tuto_article_figs/Error_Trans_plot_mat1_{mat1}_mat2_{mat2}_ep1_{ep1}_ep2_{ep2}.png")
        plt.close()

    err_rs_t_te = np.abs(rs_t_te - rs_s_te)
    err_rs_t_tm = np.abs(rs_t_tm - rs_s_tm)

    err_rs_a_te = np.abs(rs_a_te - rs_s_te)
    err_rs_a_tm = np.abs(rs_a_tm - rs_s_tm)

    err_rs_dn_te = np.abs(rs_dn_te - rs_s_te)
    err_rs_dn_tm = np.abs(rs_dn_tm - rs_s_tm)

    err_rs_i_te = np.abs(rs_i_te - rs_s_te)
    err_rs_i_tm = np.abs(rs_i_tm - rs_s_tm)
    if np.sum(err_rs_t_te + err_rs_t_tm + err_rs_a_te + err_rs_a_tm + err_rs_dn_te + err_rs_dn_tm)>0:

        fig, ax = plt.subplots(4, 2, figsize=(10,10))
        ax[0,0].pcolormesh(X, Y, err_rs_t_te.T, vmax=1)
        ax[0,0].set_title("Abs error reflection TE T")
        ax[0,0].set_xlabel("Nb Layers")
        ax[0,0].set_ylabel("Inc. Angle")

        ax[1,0].pcolormesh(X, Y, err_rs_a_te.T, vmax=1)
        ax[1,0].set_title("Abs error reflection TE Ab")
        ax[1,0].set_xlabel("Nb Layers")
        ax[1,0].set_ylabel("Inc. Angle")

        ax[2,0].pcolormesh(X, Y, err_rs_dn_te.T, vmax=1)
        ax[2,0].set_title("Abs error reflection TE D2N")
        ax[2,0].set_xlabel("Nb Layers")
        ax[2,0].set_ylabel("Inc. Angle")

        ax[3,0].pcolormesh(X, Y, err_rs_i_te.T, vmax=1)
        ax[3,0].set_title("Abs error reflection TE Imp")
        ax[3,0].set_xlabel("Nb Layers")
        ax[3,0].set_ylabel("Inc. Angle")

        ax[0,1].pcolormesh(X, Y, err_rs_t_tm.T, vmax=1)
        ax[0,1].set_title("Abs error reflection TM T")
        ax[0,1].set_xlabel("Nb Layers")
        ax[0,1].set_ylabel("Inc. Angle")

        ax[1,1].pcolormesh(X, Y, err_rs_a_tm.T, vmax=1)
        ax[1,1].set_title("Abs error reflection TM Ab")
        ax[1,1].set_xlabel("Nb Layers")
        ax[1,1].set_ylabel("Inc. Angle")

        ax[2,1].pcolormesh(X, Y, err_rs_dn_tm.T, vmax=1)
        ax[2,1].set_title("Abs error reflection TM D2N")
        ax[2,1].set_xlabel("Nb Layers")
        ax[2,1].set_ylabel("Inc. Angle")

        ax[3,1].pcolormesh(X, Y, err_rs_i_tm.T, vmax=1)
        ax[3,1].set_title("Abs error reflection TM Imp")
        ax[3,1].set_xlabel("Nb Layers")
        ax[3,1].set_ylabel("Inc. Angle")

        plt.tight_layout()
        plt.savefig(f"Tuto_article_figs/Error_Refl_plot_mat1_{mat1}_mat2_{mat2}_ep1_{ep1}_ep2_{ep2}.png")
        plt.close()
    #plt.show()