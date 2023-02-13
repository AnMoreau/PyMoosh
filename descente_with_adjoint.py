import numpy as np

def descente_rapide(f_cout_adjoint,pas,start,npas,*args):
    """
        Descente de gradient tout ce qu'il y a de plus classique.
            > On calcule une fonction de coût pour l'étape actuelle
            > On décale le point en remontant le gradient, calculé rapidement
            > Si des paramètres dépassent les limites imposées, on les met à la frontière
            > On calcule la nouvelle fonction de coût
            > Si c'est mieux, on continue, sinon on divise la pas par 2.
            > On s'arrête quand le pas est trop petit
    """
    # Gestion des arguments optionnels et des optionnalité des arguments
    limites=False
    #print(len(args))
    if (len(args)==2):
        xmin=args[0]
        xmax=args[1]
        limites=True
#        print("Limites !")
    n=start.size
    if (len(pas)!=n):
        pas=pas*np.ones(n)

    # Initialisation
    convergence=np.zeros(npas)
    x=start
    f, A, B =f_cout_adjoint(x, mode="value")
    # the cost function needs to give access to the tranfer matrices
    direction=np.zeros(n)

    for k in range(0,npas):
        dpas=np.zeros(n)
        for j in range(0,n):
            dpas[j]=pas[j]/100
            direction[j]=(f-f_cout_adjoint(x+dpas, mode="grad", i_change=j//2, saved_mat=[A,B]))/dpas[j]
            # fixed indices: i_change = j
            # variable indices: i_change = j//2
            dpas[j]=0
        # Determination du nouveau x - prenant les limites en compte.
        xn=x+pas*direction
        if (limites):
            super=(xn<xmax)
            infra=(xn>xmin)
            xn=xn*super*infra+xmin*(1-infra)+xmax*(1-super)
        # Calcul de sa fonction de coût
        tmp=f_cout(xn)
        # Est-ce qu'on avance ou est-ce qu'on divise le pas par deux ?
        if (tmp<f):
            f=tmp
            x=xn
        else:
            print('Division du pas')
            pas=pas/2
            # Si le pas est trop petit, on est sans doute sur le minimum.
            # On sort.
            if (max(pas)<1e-10):
                convergence=convergence[0:k]
                break
        convergence[k]=f # La valeur de la fonction de coût est stockée à chaque étape
    return [x,convergence]
