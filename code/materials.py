import numpy as np

# A few materials.

# BK7, glass.
def epsbk7(lam):
    """Permittivity of BK7 glass, obtained using an interpolation.

    Args:
        lam (float): wavelength in nm
    """
    wl=np.array([190.75,193.73,196.8,199.98,203.25,206.64,210.14,213.77,217.52,221.4,225.43,229.6,233.93,238.43,243.11,247.97,253.03,258.3,263.8,269.53,275.52,281.78,288.34,295.2,302.4,309.96,317.91,326.28,335.1,344.4,354.24,364.66,375.71,387.45,399.95,413.28,427.54,442.8,459.2,476.87,495.94,516.6,539.0700000000001,563.5700000000001,590.41,619.9299999999999,652.55,688.8099999999999,729.3200000000001,774.91,826.5700000000001,885.61,953.73,1033.21,1127.14,1239.85])
    permittivity=np.array([2.8406742849,2.804261715649,2.770563579001,2.739329528464,2.710376127684,2.683571461921,2.658732435844,2.635706792196,2.614339739664,2.594480126121,2.575996110081,2.558755351321,2.542647106624,2.527572147556,2.513432915161,2.500146004225,2.487638700625,2.475842457361,2.464692764356,2.4541415649,2.444132010384,2.434623467584,2.425575745476,2.416958387649,2.4087350401,2.400878973529,2.393369890704,2.386181504529,2.379296995009,2.372696606736,2.366369966601,2.360300650929,2.354472356041,2.348881151236,2.343513969316,2.3383608889,2.333415112704,2.328666844009,2.324109397009,2.319733086489,2.315534369344,2.311503651769,2.307628351744,2.303905051044,2.300324289124,2.296867553764,2.2935285136,2.290285730161,2.287120856329,2.284009554436,2.2809154729,2.277793303696,2.274585797929,2.271202716601,2.267514953929,2.263330686969])
    epsilon=np.interp(lam,wl,permittivity)

    return epsilon

# Water
def epsH2O(lam):
    """Permittivity of water, obtained using an interpolation.

    Args:
        lam (float): wavelength in nm
    """
    wl=np.array([404.7,435.8,467.8,480,508.5,546.1,577,579.1,589.1,643.8,700,750,800,850,900,950,1000,1050,1100])
    permittivity=np.array([1.8056640625,1.7988442641,1.7932691569,1.7914216336,1.7875957401,1.7833999936,1.7804632356,1.7802764329,1.7794226025,1.7753164081,1.7718005881,1.766241,1.763584,1.760929,1.763584,1.760929,1.758276,1.755625,1.752976])
    epsilon=np.interp(lam,wl,permittivity)

    return epsilon

# Glass
def epsglass(lam):

    epsilon=2.978645+0.008777808/(lam**2*1e-6-0.010609)+84.06224/(lam**2*1e-6-96)
    return epsilon

# Amorphous silicon
def epsSia(lam):
    """Permittivity of amorphous silicon, obtained using an interpolation.

    Args:
        lam (float): wavelength in nm
    """

    i=1j
    wl=np.array([103.3,107.8,112.7,118.1,124,130.5,137.8,145.9,155,165.3,177.1,190.7,206.6,225.4,248,258.3,269.5,281.8,295.2,310,326.3,344.4,354.3,364.7,387.5,413.3,442.8,476.9,496,516.6,563.6,619.9,652.6,688.8,729.3,774.9,826.6,885.6,953.8,1033,1127,1240,1378,1550,1771,2066])
    permittivity=np.array([-0.420147+i*0.474804,-0.5856399999999999+i*0.614922,-0.7412519999999999+i*0.741664,-0.9026710000000001+i*0.87984,-1.088919+i*1.04652,-1.290591+i*1.23256,-1.527651+i*1.4661,-1.804491+i*1.75518,-2.124400000000001+i*2.112,-2.487375+i*2.5578,-2.879876+i*3.14496,-3.380498999999999+i*3.937139999999999,-3.966299999999999+i*5.0616,-4.477599999999999+i*6.777,-4.761499999999999+i*9.328799999999999,-4.6629+i*10.602,-4.300000000000002+i*12.1302,-3.650100000000002+i*13.754,-2.688+i*15.5648,-1.1267+i*17.5644,1.3041+i*19.26,4.3081+i*20.448,6.1288+i*20.8134,8.134399999999999+i*20.748,11.7245+i*19.8492,15.104+i*17.6952,17.2913+i*14.6616,18.5217+i*11.4944,18.7265+i*10.0128,18.952639+i*8.64348,18.5335+i*6.0168,17.68037900000001+i*3.900060000000001,17.257131+i*3.02742,16.654659+i*2.21678,16.040499+i*1.59598,15.426404+i*1.06896,14.89300656+i*0.626864,14.21129199+i*0.302354,13.5424+i*0,13.0321+i*0,12.7449+i*0,12.5316+i*0,12.25+i*0,12.1104+i*0,11.9025+i*0,11.8336+i*0]);
    epsilon=np.interp(lam,wl,permittivity)
    return epsilon

""" For metals, we use
    Optical properties of metallic films for vertical-cavity optoelectronic devices
    Aleksandar D. Rakic, Aleksandra B. Djurisic, Jovan M. Elazar, and Marian L. Majewski
    APPLIED OPTICS Vol. 37, No. 22, P. 5271
    https://cdn.optiwave.com/wp-content/uploads/2015/06/4.pdf
"""

# Silver
def epsAg(lam):
    """ Permittivity of silver.

    Args:
        lam (float): wavelength in nm

    Reference:
    Optical properties of metallic films for vertical-cavity optoelectronic devices
    Aleksandar D. Rakic, Aleksandra B. Djurisic, Jovan M. Elazar, and Marian L. Majewski
    APPLIED OPTICS Vol. 37, No. 22, P. 5271
    https://cdn.optiwave.com/wp-content/uploads/2015/06/4.pdf

    """
    from scipy import special
    w=6.62606957e-25*299792458/1.602176565e-19/lam
    f0=0.821;
    Gamma0=0.049;
    omega_p=9.01;
    f=np.array([0.050,0.133,0.051,0.467,4.000])
    Gamma=np.array([0.189,0.067,0.019,0.117,0.052])
    omega=np.array([2.025,5.185,4.343,9.809,18.56])
    sigma=np.array([1.894,0.665,0.189,1.170,0.516])
    a=np.sqrt(w*(w+1j*Gamma))
    x=(a-omega)/(np.sqrt(2)*sigma)
    y=(a+omega)/(np.sqrt(2)*sigma)
    # Polarizability due to bound electrons
    chi_b=np.sum(1j*np.sqrt(np.pi)*f*omega_p**2/(2*np.sqrt(2)*a*sigma)*(special.wofz(x)+special.wofz(y)))
    # Equivalent polarizability linked to free electrons (Drude model)
    chi_f=-omega_p**2*f0/(w*(w+1j*Gamma0))
    epsilon=1+chi_f+chi_b
    return epsilon

# Aluminium
def epsAl(lam):
    """ Permittivity of aluminium.

    Args:
        lam (float): wavelength in nm

    Reference:
    Optical properties of metallic films for vertical-cavity optoelectronic devices
    Aleksandar D. Rakic, Aleksandra B. Djurisic, Jovan M. Elazar, and Marian L. Majewski
    APPLIED OPTICS Vol. 37, No. 22, P. 5271
    https://cdn.optiwave.com/wp-content/uploads/2015/06/4.pdf

    """

    from scipy import special
    w=6.62606957e-25*299792458/1.602176565e-19/lam
    f0=0.526
    Gamma0=0.047
    omega_p=14.98
    f=np.array([0.213,0.060,0.182,0.014])
    Gamma=np.array([0.312,0.315,1.587,2.145])
    omega=np.array([0.163,1.561,1.827,4.495])
    sigma=np.array([0.013,0.042,0.256,1.735])
    a=np.sqrt(w*(w+1j*Gamma))
    x=(a-omega)/(np.sqrt(2)*sigma)
    y=(a+omega)/(np.sqrt(2)*sigma)
    # Polarizability due to bound electrons
    chi_b=np.sum(1j*np.sqrt(np.pi)*f*omega_p**2/(2*np.sqrt(2)*a*sigma)*(special.wofz(x)+special.wofz(y)))
    # Equivalent polarizability linked to free electrons (Drude model)
    chi_f=-omega_p**2*f0/(w*(w+1j*Gamma0))
    epsilon=1+chi_f+chi_b
    return epsilon

# Gold
def epsAu(lam):
    """ Permittivity of gold.

    Args:
        lam (float): wavelength in nm

    Reference:
    Optical properties of metallic films for vertical-cavity optoelectronic devices
    Aleksandar D. Rakic, Aleksandra B. Djurisic, Jovan M. Elazar, and Marian L. Majewski
    APPLIED OPTICS Vol. 37, No. 22, P. 5271
    https://cdn.optiwave.com/wp-content/uploads/2015/06/4.pdf

    """

    from scipy import special
    w=6.62606957e-25*299792458/1.602176565e-19/lam
    f0=0.770
    Gamma0=0.050
    omega_p=9.03
    f=np.array([0.054,0.050,0.312,0.719,1.648])
    Gamma=np.array([0.074,0.035,0.083,0.125,0.179])
    omega=np.array([0.218,2.885,4.069,6.137,27.97])
    sigma=np.array([0.742,0.349,0.830,1.246,1.795])
    a=np.sqrt(w*(w+1j*Gamma))
    x=(a-omega)/(np.sqrt(2)*sigma)
    y=(a+omega)/(np.sqrt(2)*sigma)
    # Polarizability due to bound electrons
    chi_b=np.sum(1j*np.sqrt(np.pi)*f*omega_p**2/(2*np.sqrt(2)*a*sigma)*(special.wofz(x)+special.wofz(y)))
    # Equivalent polarizability linked to free electrons (Drude model)
    chi_f=-omega_p**2*f0/(w*(w+1j*Gamma0))
    epsilon=1+chi_f+chi_b
    return epsilon

# Nickel
def epsNi(lam):
    """ Permittivity of Nickel.

    Args:
        lam (float): wavelength in nm

    Reference:
    Optical properties of metallic films for vertical-cavity optoelectronic devices
    Aleksandar D. Rakic, Aleksandra B. Djurisic, Jovan M. Elazar, and Marian L. Majewski
    APPLIED OPTICS Vol. 37, No. 22, P. 5271
    https://cdn.optiwave.com/wp-content/uploads/2015/06/4.pdf

    """

    from scipy import special
    w=6.62606957e-25*299792458/1.602176565e-19/lam
    f0=0.083
    Gamma0=0.022
    omega_p=15.92
    f=np.array([0.357,0.039,0.127,0.654])
    Gamma=np.array([2.820,0.120,1.822,6.637])
    omega=np.array([0.317,1.059,4.583,8.825])
    sigma=np.array([0.606,1.454,0.379,0.510])
    a=np.sqrt(w*(w+1j*Gamma))
    x=(a-omega)/(np.sqrt(2)*sigma)
    y=(a+omega)/(np.sqrt(2)*sigma)
    # Polarizability due to bound electrons
    chi_b=np.sum(1j*np.sqrt(np.pi)*f*omega_p**2/(2*np.sqrt(2)*a*sigma)*(special.wofz(x)+special.wofz(y)))
    # Equivalent polarizability linked to free electrons (Drude model)
    chi_f=-omega_p**2*f0/(w*(w+1j*Gamma0))
    epsilon=1+chi_f+chi_b
    return epsilon

# Platinum
def epsPt(lam):
    """ Permittivity of platinum.

    Args:
        lam (float): wavelength in nm

    Reference:
    Optical properties of metallic films for vertical-cavity optoelectronic devices
    Aleksandar D. Rakic, Aleksandra B. Djurisic, Jovan M. Elazar, and Marian L. Majewski
    APPLIED OPTICS Vol. 37, No. 22, P. 5271
    https://cdn.optiwave.com/wp-content/uploads/2015/06/4.pdf

    """

    from scipy import special
    w=6.62606957e-25*299792458/1.602176565e-19/lam
    f0=0.333
    Gamma0=0.080
    omega_p=9.59
    f=np.array([0.186,0.665,0.551,2.214])
    Gamma=np.array([0.498,1.851,2.604,2.891])
    omega=np.array([0.782,1.317,3.189,8.236])
    sigma=np.array([0.031,0.096,0.766,1.146])
    a=np.sqrt(w*(w+1j*Gamma))
    x=(a-omega)/(np.sqrt(2)*sigma)
    y=(a+omega)/(np.sqrt(2)*sigma)
    # Polarizability due to bound electrons
    chi_b=np.sum(1j*np.sqrt(np.pi)*f*omega_p**2/(2*np.sqrt(2)*a*sigma)*(special.wofz(x)+special.wofz(y)))
    # Equivalent polarizability linked to free electrons (Drude model)
    chi_f=-omega_p**2*f0/(w*(w+1j*Gamma0))
    epsilon=1+chi_f+chi_b
    return epsilon

# Cupper
def epsCu(lam):
    """ Permittivity of cupper.

    Args:
        lam (float): wavelength in nm

    Reference:
    Optical properties of metallic films for vertical-cavity optoelectronic devices
    Aleksandar D. Rakic, Aleksandra B. Djurisic, Jovan M. Elazar, and Marian L. Majewski
    APPLIED OPTICS Vol. 37, No. 22, P. 5271
    https://cdn.optiwave.com/wp-content/uploads/2015/06/4.pdf

    """

    from scipy import special
    w=6.62606957e-25*299792458/1.602176565e-19/lam
    f0=0.562
    Gamma0=0.03
    omega_p=10.83
    f=np.array([0.076,0.081,0.324,0.726])
    Gamma=np.array([0.056,0.047,0.113,0.172])
    omega=np.array([0.416,2.849,4.819,8.136])
    sigma=np.array([0.562,0.469,1.131,1.719])
    a=np.sqrt(w*(w+1j*Gamma))
    x=(a-omega)/(np.sqrt(2)*sigma)
    y=(a+omega)/(np.sqrt(2)*sigma)
    # Polarizability due to bound electrons
    chi_b=np.sum(1j*np.sqrt(np.pi)*f*omega_p**2/(2*np.sqrt(2)*a*sigma)*(special.wofz(x)+special.wofz(y)))
    # Equivalent polarizability linked to free electrons (Drude model)
    chi_f=-omega_p**2*f0/(w*(w+1j*Gamma0))
    epsilon=1+chi_f+chi_b
    return epsilon
