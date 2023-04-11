import numpy as np
import PyMoosh as PM
import matplotlib.pyplot as plt


#materials = [1.513**2, 1.455**2, 2.079**2, (1.9+4.8j)**2, 1.0003**2]
materials = [4., 1.5**2, 2.+0.2j]
wav = 20
eps = 1e-10


print("WARNING: the impedance formalism only computes r and R for the moment")

print("Normal incidence:")
incidence = 0
nb_prob = 0
prob = False

## Case 1: single layer, TE
#structure = np.random.random(nb_couches*2+1)*w_mean
structure = np.array([1, 2.1])

# stack = [0]+[1,2]*nb_couches+[1,0]
stack = [0,2, 1,0]


epaisseurs = np.concatenate(([0],structure,[0]))

chose = PM.Structure(materials,stack,epaisseurs, verbose=False)
r, t, R, T = PM.coefficient_S(chose,wav,incidence,0)



chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_ab, t_ab, R_ab, T_ab, A_ab, B_ab = PM.coefficient_with_grad_A(chose1,wav,incidence,0)


chose1 = PM.Structure(materials,stack,epaisseurs, verbose=False)
r_t, t_t, R_t, T_t, A_T, B_T = PM.coefficient_with_grad_T(chose1,wav,incidence,0)

print(f"r_t = {r_t}, r_ab = {r_ab}")

#print(f"matrices, A_ab={A_ab}\n, B_ab={B_ab}\n, A_T={A_T}\n, B_T={B_T}")

#print("pop1")
#for j in range(len(A_ab)):
#    print(B_ab[-j-1] @ A_ab[j])
#
#print("pop2")
#for j in range(len(A_T)):
#    print(A_T[j] @ B_T[-j-1])

for i in range(len(structure)+1):
    r_ab, t_ab, R_ab, T_ab = PM.coefficient_with_grad_A(chose1,wav,incidence,0, mode="grad", saved_mat=[A_ab,B_ab], i_change=i)
    print(f"step {i}: r_ab = {r_ab}")

for i in range(len(structure)+1):
    r_t, t_t, R_t, T_t = PM.coefficient_with_grad_T(chose1,wav,incidence,0, mode="grad", saved_mat=[A_T,B_T], i_change=i)
    print(f"step {i}: r_t = {r_t}")
