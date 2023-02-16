import numpy as np

def Differential_Evolution(f_cout,budget,X_min,X_max,population = 30):
    """This is Differentiel Evolution in its current to best version.

    Args:
        f_cout (function): cost function taking a numpy vector as argument
        budget (integer): number of times the cost function can be computed
        X_min (numpy array): lower boundaries of the optimization domain,
                             a vector with the same size as the argument of
                             the cost function.
        X_max (numpy array): upper boundaries of the optimization domain.
        population (integer): size of the population (30 by default)

    Returns:
        best (numpy array): best solution found
        convergence (array): lowest value of the cost function for each
                             generation
    """

    # Hyperparameters
    # Cross-over
    cr=0.5;
    # Mutation
    f1=0.9;
    f2=0.8;

    n=X_min.size

    # Population initialization
    omega=np.zeros((population,n))
    cost=np.zeros(population)
    for k in range(0,population):
        omega[k]=X_min+(X_max-X_min)*np.random.random(n)
        cost[k]=f_cout(omega[k])

    # Who's the best ?
    who=np.argmin(cost)
    best=omega[who]

    # initialization of the rest
    evaluation=population
    convergence=[]
    generation=0
    convergence.append(cost[who])

    # Differential Evolution loop.
    while evaluation<budget-population:
        for k in range(0,population):
            crossover=(np.random.random(n)<cr)
            X=(\
            # Current
            omega[k]\
            +f1*(omega[np.random.randint(population)]\
            -omega[np.random.randint(population)])\
            # To best
            +f2*(best-omega[k]))
            # And cross-over
            *(1-crossover)+
            crossover*omega[k]

            if np.prod((X>=X_min)*(X<=X_max)):
                tmp=f_cout(X)
                evaluation=evaluation+1
                if (tmp<cost[k]) :
                    cost[k]=tmp
                    omega[k]=X
        generation=generation+1
        who=np.argmin(cost)
        best=omega[who]
        convergence.append(cost[who])

    convergence=convergence[0:generation+1]

    return [best,convergence]
