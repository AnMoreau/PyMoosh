import numpy as np
import copy

def cascade_nl(T,U):
    """
        Cascading Scattering non local matrices
    """
    n=min(np.shape(T)[0], np.shape(U)[0]) - 1
    m=np.shape(T)[0] - n
    p=np.shape(U)[0] - n

    A=T[0: m, 0: m]
    B=T[0: m, m: m + n]
    C=T[m: m + n, 0: m]
    D=T[m: m + n, m: m + n]

    E=U[0: n, 0: n]
    F=U[0: n, n: n + p]
    G=U[n: n + p, 0: n]
    H=U[n: n + p, n: n + p]

    J=np.linalg.inv(np.eye(n, n) - E @ D)
    K=np.linalg.inv(np.eye(n, n) - D @ E)
    matrix = np.vstack((np.hstack((A + B @ J @ E @ C, B @ J @ F)), np.hstack((G @ K @ C, H + G @ K @ D @ F))))
    return  matrix

def coefficient_non_local(struct, wavelength, incidence, polarization) :
    """
    This function computes the reflection and transmission coefficients
    of the structure, and takes account the possibility to have NonLocal materials

    Args:
        struct (Structure): belongs to the Structure class
        wavelength (float): wavelength of the incidence light (in nm)
        incidence (float): incidence angle in radians
        polarization (float): 0 for TE, 1 (or anything) for TM

    returns:
        r (complex): reflection coefficient, phase origin at first interface
        t (complex): transmission coefficient
        R (float): Reflectance (energy reflection)
        T (float): Transmittance (energie transmission)


    R and T are the energy coefficients (real quantities)

    .. warning: The transmission coefficients have a meaning only if the lower medium
    is lossless, or they have no true meaning.
    Also, you should not use two adjacent nonlocal material layers, it doesn't work for the moment..
    """

    # In order to get a phase that corresponds to the expected reflected coefficient, we make the height of the upper (lossless) medium vanish. It changes only the phase of the reflection coefficient.
    # The medium may be dispersive. The permittivity and permability of each layer has to be computed each time.
    if (struct.unit != "nm") :
        wavelength = conv_to_nm(wavelength, struct.unit)

    Epsilon_mat, Mu_mat = struct.polarizability(wavelength)
    Type = struct.layer_type
    Epsilon = [Epsilon_mat[i] for i in Type]
    Mu = [Mu_mat[i] for i in Type]
    thickness = copy.deepcopy(struct.thickness)
    # In order to ensure that the phase reference is at the beginning
    # of the first layer.
    thickness[0] = 0

    if len(struct.thickness) != len(struct.layer_type) :
        print(f"ArgumentMatchError : layer_type has {len(struct.layer_type)} arguments and thickness has {len(struct.thickness)} arguments")
        return None

    # The boundary conditions will change when the polarization changes. (A demander à Antoine pourquoi)
    if polarization == 0 :
        print("Non local materials should be used with polarization = 1 (TM)")
        return 0
    else :
        f = Epsilon


    k0 = 2 * np.pi / wavelength
    g = len(Type)
    omega_p = [0] * (g - 1)
    chi_b = [0] * (g - 1)
    chi_f = [0] * (g - 1)
    beta2 = [0] * (g - 1)
    for k in range(g):
        if struct.materials[Type[k]].specialType == "NonLocal":
            beta2[k], chi_b[k], chi_f[k], omega_p[k] = struct.materials[Type[k]].get_values_nl(wavelength)

    alpha = np.sqrt(Epsilon[0]) * k0 * np.sin(incidence)
    gamma = np.array(np.sqrt([(1+0j)*Epsilon[i] * k0 ** 2 - alpha ** 2 for i in range(g)]), dtype = complex)
    #print(f"Données \nEpsilon vaut {Epsilon} \nMu vaut {Mu} \nType vaut {Type} \nthickness vaut {thickness} \nalpha vaut {alpha}  \ngamma vaut {gamma} \nbeta vaut {beta}")

    T = []
    thickness[0] = 0
    T.append(np.array([[0, 1], [1, 0]], dtype = complex))
    #print(f"Matrice {0} de couche locale (initialisation) \nt vaut : {1., 1.0j}, \n {T[0]}")

    for k in range(g - 1) :
        # Stability of square root in complex world :)
        if np.imag(gamma[k + 1]) < 0 :
            gamma[k + 1] *= -1

        b1 = gamma[k] / f[k]
        b2 = gamma[k + 1] / f[k + 1]
        #print(f"b1 vaut {b1} \nb2 vaut {b2}")

        # local layer matrix
        if beta2[k] == 0 :
            t = np.exp(1j * gamma[k] * thickness[k])
            T.append(np.array([[0, t],
                               [t, 0]], dtype = complex))

            if beta2[k+1]==0:
                # local local interface
                T.append(np.array([[b1 - b2, 2 * b2],
                                   [2 * b1, b2 - b1]] / (b1 + b2), dtype = complex))

            else:
                # local non-local interface
                Kl = np.sqrt(alpha**2 + (omega_p[k+1]**2 / beta2[k+1]) * (1 / chi_f[k+1] + 1 / (1 + chi_b[k+1]) ))
                omega = (alpha**2 / Kl) * (1 / Epsilon[k+1] - 1 / (1 + chi_b[k+1]))

                T.append(np.array([[b1 - b2 + 1j * omega, 2 * b2, 2],
                                   [2 * b1, b2 - b1 + 1j * omega, 2],
                                   [2 * 1j * omega * b1, 2 * 1j * omega * b2, b1 + b2 + 1j * omega]] / (b1 + b2 - 1j * omega), dtype = complex))
                #print(f"Matrice {2 * k + 1} de couche locale \nt vaut : {t} \n {T[2 * k + 1]}")

        else : # if beta[k] != 0 :
            Kl = np.sqrt(alpha**2 + (omega_p[k]**2 / beta2[k]) * (1 / chi_f[k] + 1 / (1 + chi_b[k]) ))
            omega = (alpha**2 / Kl) * (1 / Epsilon[k] - 1 / (1 + chi_b[k]))
            t = np.exp(1j * gamma[k] * thickness[k])
            l = np.exp(- Kl * thickness[k])
            T.append(np.array([[0, 0, t, 0],
                               [0, 0, 0, l],
                               [t, 0, 0, 0],
                               [0, l, 0, 0]], dtype = complex))
            #print(f"Matrice {2 * k + 1} de couche non-locale \nt vaut : {t} \nl vaut : {l} \n, {T[2 * k + 1]}")

            if k == g-2 or beta2[k+1]==0:
                # non-local local interface
                T.append(np.array([[b1 - b2 + 1j * omega, -2, 2 * b2],
                          [-2 * 1j * omega *  b1, b1 + b2 + 1j * omega, -2  * 1j * omega * b2],
                          [2 * b1, -2, b2 - b1 + 1j * omega]] / (b1 + b2 - 1j * omega), dtype = complex))

            else:
                # non-local non-local interface
                print("We can't use cascadage for non local - non local layers (yet)")

    # Last layer
    t = np.exp(1j * gamma[g - 1] * thickness[g - 1])
    T.append(np.array([[0, t],
                       [t, 0]], dtype = complex))
    #print(f"Matrice {2 * g - 1} de couche locale, t vaut : {t} \n {T[2 * g - 1]}")
    #print(f"kl vaut {Kl} \nomega vaut {omega}")

    # INITIALISATION
    A = T[0] # np.array([[0, 1], [1, 0]], dtype = complex)

    # Cascading scattering matrices
    for p in range(len(T) - 1) : # len(T) - 1 = 2 * g - 1
        A = cascade_nl(A, T[p])
    # Reflection coefficient
    r = A[0, 0]
    # Transmission coefficient
    t = A[1, 0]
    # Energy reflexion coefficient
    R = np.real(abs(r) ** 2)
    # Energy transmission coefficient
    T = np.real(abs(t) ** 2 * gamma[g - 1] * f[Type[0]] / (gamma[0] * f[Type[g - 1]]))

    return r, t, R, T
