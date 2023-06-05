import numpy as np
from scipy.optimize import minimize

def differential_evolution(f_cout, budget, X_min, X_max, population=30):
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
            
            # Choosing which parameters will be taken from the new individual
            crossover=(np.random.random(n)<cr)
            
            # Choosing 2 random individuals
            pop_1 = omega[np.random.randint(population)]
            pop_2 = omega[np.random.randint(population)]
            rand_step = pop_1 - pop_2
            best_step = best-omega[k]

            new_param = omega[k] + f1*rand_step + f2*best_step

            X = new_param*(1-crossover) + omega[k]*crossover


            if np.prod((X >= X_min)*(X <= X_max)):
                # If the individual is in the parameter domain, proceed
                tmp = f_cout(X)
                evaluation = evaluation+1
                if (tmp < cost[k]) :
                    # If the new individual is better than the parent,
                    # we keep it
                    cost[k] = tmp
                    omega[k] = X

        generation = generation+1
        who = np.argmin(cost)
        best = omega[who]
        convergence.append(cost[who])

    convergence = convergence[0:generation+1]

    return [best, convergence]


def bfgs(f_cout, npas, start, *args):
    """This is a wrapper for the L-BFGS-B method encoded in scipy

    Args:
        f_cout (function): cost function taking a numpy vector as argument
        npas (integer): maximum number of iterations of the BFGS algorithm
        start (numpy array): initial guess for the structure
        *args contains nothing or:
            X_min (numpy array): lower boundaries of the optimization domain,
                                 a vector with the same size as the argument of
                                 the cost function.
            X_max (numpy array): upper boundaries of the optimization domain.

    Returns:
        best (numpy array): best solution found
    """
    if (len(args) == 2):
        xmin = args[0]
        xmax = args[1]
        assert len(xmin)==len(xmax)==len(start), f"starting array and boundary arrays should be of same length, but have lengths {len(start)}, {len(xmin)} and {len(xmax)}"
        limites = True
    else:
        limites = False

    x = np.array(start)
    epsilon = 1.e-7

    def jac(x):
        """ Jacobian defined with a finite difference method.
            It uses a global convergence variable to be able to
            retrieve the cost function value evolution.
        """
        n = len(x)
        grad = np.zeros(n)

        val = f_cout(x)
        for i in range(n):
            xp = np.array(x)
            xp[i] = xp[i] + epsilon
            grad[i] = (f_cout(xp) - val) / epsilon
        return grad

    if (limites):
        res = minimize(f_cout, start, method='L-BFGS-B', jac=jac, tol=1e-99,
                 options = {'disp': False, 'maxiter': npas}, bounds=[(xmin[i], xmax[i]) for i in range(len(xmin))])
    else:
        res = minimize(f_cout, start, method='BFGS', jac=jac, tol=1e-99,
                 options = {'disp': False, 'maxiter': npas})

    best = res.x

    return best, f_cout(best)

