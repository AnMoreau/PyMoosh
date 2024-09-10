import numpy as np
from scipy.optimize import minimize

def differential_evolution(f_cout, budget, X_min, X_max, f1=0.9, f2=0.8, cr=0.5, population=30):
    """This is Differentiel Evolution in its current to best version.

    Args:
        f_cout (function): cost function taking a numpy vector as argument
        budget (integer): number of times the cost function can be computed
        X_min (numpy array): lower boundaries of the optimization domain,
                             a vector with the same size as the argument of
                             the cost function.
        X_max (numpy array): upper boundaries of the optimization domain.
    KwArgs:
        cr (float): value for crossover operation
        f1 (float): value for f1 mutation operation
        f2 (float): value for f2 mutation operation
        population (integer): size of the population (30 by default)

    Returns:
        best (numpy array): best solution found
        convergence (array): lowest value of the cost function for each
                             generation
    """

    # Hyperparameters
    # Cross-over : cr,
    # Mutation : f1 and f2
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


def QODE(f_cout, budget, X_min, X_max, population=30, progression=False):
    """This is Quasi Opposite Differential Evolution.

    Args:
        f_cout (function): cost function taking a numpy vector as argument
        budget (integer): number of times the cost function can be computed
        X_min (numpy array): lower boundaries of the optimization domain,
                             a vector with the same size as the argument of
                             the cost function.
        X_max (numpy array): upper boundaries of the optimization domain.
        population (integer): size of the population (30 by default), should be even!

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
    #center of optimization domain
    c = (X_max + X_min)/2
    for k in range(0,population, 2):
        omega[k]=X_min+(X_max-X_min)*np.random.random(n)
        cost[k]=f_cout(omega[k])
        delta = 2*(c - omega[k])*np.random.random(n)
        omega[k+1] = omega[k] + delta
        cost[k+1]=f_cout(omega[k+1])
    # The specifity of QODE (the initialisation) is done, the rest is usual DE

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
                if progression and ((evaluation*progression) % budget == 0) :
                    print(f'Progression : {np.round(evaluation*100/(budget - population), 2)}%. Current cost : {np.round(f_cout(best),6)}')
                if (tmp < cost[k]) :
                    # If the new individual is better than the parent,
                    # we keep it
                    cost[k] = tmp
                    omega[k] = X

        generation = generation+1
        who = np.argmin(cost)
        best = omega[who]
        convergence.append(cost[who])
        if (evaluation % 50 == 0) and progression:
            print(f'Progression : {np.round(evaluation*100/(budget - population), 2)}%. Current cost : {np.round(f_cout(best),6)}')

    convergence = convergence[0:generation+1]

    return [best, convergence]


def QNDE(f_cout, budget, X_min, X_max, budget_bfgs=None, population=30, progression=False):
    """This is Quasi Newton Differential Evolution.

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
    if budget_bfgs is None:
        budget_bfgs = int(0.1*budget)
    cut_budget = budget - budget_bfgs
    first_best, first_convergence = QODE(f_cout, cut_budget, X_min, X_max, population, progression)
    print('Switching to bfgs gradient descent...') if progression else None
    best, last_convergence = bfgs(f_cout, budget_bfgs, first_best, X_min, X_max)
    convergence = np.append(np.asarray(first_convergence), last_convergence)
    return [best, convergence]


## Strategy-adaptative variants


def SADE(f_cout, budget, X_min, X_max, f1=0.9, f2=0.8, cr=0.5, population=30):
    """This is Strategy-Adaptative Differentiel Evolution in its current to best version.

    Args:
        f_cout (function): cost function taking a numpy vector as argument
        budget (integer): number of times the cost function can be computed
        X_min (numpy array): lower boundaries of the optimization domain,
                             a vector with the same size as the argument of
                             the cost function.
        X_max (numpy array): upper boundaries of the optimization domain.
    KwArgs:
        cr (float or list): value(s) for crossover operation. If several are given, a new
                            one is taken at random for each individual
        f1 (float or list): value(s) for f1 mutation operation. If several are given, a new
                            one is taken at random for each individual
        f2 (float or list): value(s) for f2 mutation operation. If several are given, a new
                            one is taken at random for each individual
        population (integer): size of the population (30 by default)

    Returns:
        best (numpy array): best solution found
        convergence (array): lowest value of the cost function for each
                             generation
    """

    # Hyperparameters
    # Cross-over : cr,
    # Mutation : f1 and f2
    # Managing the possibility to give lists of parameters
    if isinstance(cr, list):
        # a list of crossovers is given
        cross_list = True
        cross_nb = len(cr)
    else:
        cross_list = False
        cur_cr = cr
    if isinstance(f1, list):
        # a list of f1 mutations is given
        f1_list = True
        f1_nb = len(f1)
    else:
        f1_list = False
        cur_f1 = f1
    if isinstance(f2, list):
        # a list of f2 mutations is given
        f2_list = True
        f2_nb = len(f2)
    else:
        f2_list = False
        cur_f2 = f2

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
            if cross_list:
                cur_cr = cr[np.random.randint(cross_nb)]
            if f1_list:
                cur_f1 = f1[np.random.randint(f1_nb)]
            if f2_list:
                cur_f2 = f2[np.random.randint(f2_nb)]
            
            # Choosing which parameters will be taken from the new individual
            crossover=(np.random.random(n)<cur_cr)
            
            # Choosing 2 random individuals
            pop_1 = omega[np.random.randint(population)]
            pop_2 = omega[np.random.randint(population)]
            rand_step = pop_1 - pop_2
            best_step = best-omega[k]

            new_param = omega[k] + cur_f1*rand_step + cur_f2*best_step

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



def SAQODE(f_cout, budget, X_min, X_max, f1=0.9, f2=0.8, cr=0.5, population=30, progression=False):
    """This is Quasi Opposite Differential Evolution.

    Args:
        f_cout (function): cost function taking a numpy vector as argument
        budget (integer): number of times the cost function can be computed
        X_min (numpy array): lower boundaries of the optimization domain,
                             a vector with the same size as the argument of
                             the cost function.
        X_max (numpy array): upper boundaries of the optimization domain.
    KwArgs:
        cr (float or list): value(s) for crossover operation. If several are given, a new
                            one is taken at random for each individual
        f1 (float or list): value(s) for f1 mutation operation. If several are given, a new
                            one is taken at random for each individual
        f2 (float or list): value(s) for f2 mutation operation. If several are given, a new
                            one is taken at random for each individual
        population (integer): size of the population (30 by default), should be even!

    Returns:
        best (numpy array): best solution found
        convergence (array): lowest value of the cost function for each
                             generation
    """

    # Hyperparameters
    # Cross-over
    # Managing the possibility to give lists of parameters
    if isinstance(cr, list):
        # a list of crossovers is given
        cross_list = True
        cross_nb = len(cr)
    else:
        cross_list = False
        cur_cr = cr
    if isinstance(f1, list):
        # a list of f1 mutations is given
        f1_list = True
        f1_nb = len(f1)
    else:
        f1_list = False
        cur_f1 = f1
    if isinstance(f2, list):
        # a list of f2 mutations is given
        f2_list = True
        f2_nb = len(f2)
    else:
        f2_list = False
        cur_f2 = f2

    n=X_min.size

    # Population initialization
    omega=np.zeros((population,n))
    cost=np.zeros(population)
    #center of optimization domain
    c = (X_max + X_min)/2
    for k in range(0,population, 2):
        omega[k]=X_min+(X_max-X_min)*np.random.random(n)
        cost[k]=f_cout(omega[k])
        delta = 2*(c - omega[k])*np.random.random(n)
        omega[k+1] = omega[k] + delta
        cost[k+1]=f_cout(omega[k+1])
    # The specifity of QODE (the initialisation) is done, the rest is usual DE

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
            if cross_list:
                cur_cr = cr[np.random.randint(cross_nb)]
            if f1_list:
                cur_f1 = f1[np.random.randint(f1_nb)]
            if f2_list:
                cur_f2 = f2[np.random.randint(f2_nb)]
            
            # Choosing which parameters will be taken from the new individual
            crossover=(np.random.random(n)<cur_cr)
            
            # Choosing 2 random individuals
            pop_1 = omega[np.random.randint(population)]
            pop_2 = omega[np.random.randint(population)]
            rand_step = pop_1 - pop_2
            best_step = best-omega[k]

            new_param = omega[k] + cur_f1*rand_step + cur_f2*best_step

            X = new_param*(1-crossover) + omega[k]*crossover


            if np.prod((X >= X_min)*(X <= X_max)):
                # If the individual is in the parameter domain, proceed
                tmp = f_cout(X)
                evaluation = evaluation+1
                if progression and ((evaluation*progression) % budget == 0) :
                    print(f'Progression : {np.round(evaluation*100/(budget - population), 2)}%. Current cost : {np.round(f_cout(best),6)}')
                if (tmp < cost[k]) :
                    # If the new individual is better than the parent,
                    # we keep it
                    cost[k] = tmp
                    omega[k] = X

        generation = generation+1
        who = np.argmin(cost)
        best = omega[who]
        convergence.append(cost[who])
        if (evaluation % 50 == 0) and progression:
            print(f'Progression : {np.round(evaluation*100/(budget - population), 2)}%. Current cost : {np.round(f_cout(best),6)}')

    convergence = convergence[0:generation+1]

    return [best, convergence]


def SAQNDE(f_cout, budget, X_min, X_max, budget_bfgs=None, f1=0.9, f2=0.8, cr=0.5, population=30, progression=False):
    """This is Strategy Adaptative Quasi Newton Differential Evolution.

    Args:
        f_cout (function): cost function taking a numpy vector as argument
        budget (integer): number of times the cost function can be computed
        X_min (numpy array): lower boundaries of the optimization domain,
                             a vector with the same size as the argument of
                             the cost function.
        X_max (numpy array): upper boundaries of the optimization domain.
    KwArgs
        budget_bfgs (int): how many function evaluations the algorithm keeps for the gradient descent
                            if nothing is given, 10% of total budget is allocated.
        cr (float or list): value(s) for crossover operation. If several are given, a new
                            one is taken at random for each individual
        f1 (float or list): value(s) for f1 mutation operation. If several are given, a new
                            one is taken at random for each individual
        f2 (float or list): value(s) for f2 mutation operation. If several are given, a new
                            one is taken at random for each individual
        population (integer): size of the population (30 by default)

    Returns:
        best (numpy array): best solution found
        convergence (array): lowest value of the cost function for each
                             generation
    """
    if budget_bfgs is None:
        budget_bfgs = int(0.1*budget)
    cut_budget = budget - budget_bfgs
    first_best, first_convergence = SAQODE(f_cout, cut_budget, X_min, X_max, f1, f2, cr, population, progression)
    print('Switching to bfgs gradient descent...') if progression else None
    best, last_convergence = bfgs(f_cout, budget_bfgs, first_best, X_min, X_max)
    convergence = np.append(np.asarray(first_convergence), last_convergence)
    return [best, convergence]



## Other functions


def QODE_distance(f_cout, budget, X_min, X_max, population=30, progression=False):
    """This is Quasi Opposite Differential Evolution + return a list of
       the distances to the best.

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
        distances (array): distances-to-the-best
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
    #center of optimization domain
    c = (X_max + X_min)/2
    for k in range(0,population, 2):
        omega[k]=X_min+(X_max-X_min)*np.random.random(n)
        cost[k]=f_cout(omega[k])
        delta = 2*(c - omega[k])*np.random.random(n)
        omega[k+1] = omega[k] + delta
        cost[k+1]=f_cout(omega[k+1])

    # Who's the best ?
    who=np.argmin(cost)
    best=omega[who]

    # initialization of the rest
    evaluation=population
    distances=[]
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
        distance_generation = []
        for structure in omega:
            d = np.linalg.norm(structure - best, ord=None)
            distance_generation.append(d)
        distances.append(distance_generation)
        if (evaluation % 50 == 0) and progression:
            print(f'Progression : {np.round(evaluation*100/(budget - population), 2)}%. Current cost : {np.round(f_cout(best),6)}')

    convergence = convergence[0:generation+1]

    return [best, convergence, distances]


def QNDE_distance(f_cout, budget, X_min, X_max, population=30, progression=False):
    """This is Quasi Newton Differential Evolution + return a list of
       the distances to the best.

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
        distances (array): distances-to-the-best
    """
    cut_budget = 100
    first_budget = budget - cut_budget #0.95*budget
    first_best, first_convergence, distances = QODE_distance(f_cout, first_budget, X_min, X_max, population, progression)
    print('Switching to bfgs gradient descent...') if progression else None
    best, last_convergence = bfgs(f_cout, cut_budget, first_best, X_min, X_max)
    convergence = np.append(np.asarray(first_convergence), last_convergence)

    return [best, convergence, distances]


def super_QNDE(f_cout, budget, X_min, X_max, population=30, progression=False):
    """This algorithm is QODE + Bfgs for each individuals in population + 
       return a matrix of the distances between each individuals.

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
        result_population: all individuals of the last generation
        distances (array): distances-to-the-best
        matrix (array)   : distances between all solutions
    """

    ## We divide the budget in two part. The first part goes to QODE.
    cut_budget = 100
    first_budget = budget - cut_budget #0.95*budget

    ## First algorithm.
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
    #center of optimization domain
    c = (X_max + X_min)/2
    for k in range(0,population, 2):
        omega[k]=X_min+(X_max-X_min)*np.random.random(n)
        cost[k]=f_cout(omega[k])
        delta = 2*(c - omega[k])*np.random.random(n)
        omega[k+1] = omega[k] + delta
        cost[k+1]=f_cout(omega[k+1])

    # Who's the best ?
    who=np.argmin(cost)
    best=omega[who]

    # initialization of the rest
    evaluation=population
    distances=[]
    convergence=[]
    generation=0
    convergence.append(cost[who])

    # Differential Evolution loop.
    while evaluation<first_budget-population:
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
        distance_generation = []
        for structure in omega:
            d = np.linalg.norm(structure - best, ord=2)
            distance_generation.append(d)
        distances.append(distance_generation)
        if (evaluation % 50 == 0) and progression:
            print(f'Progression : {np.round(evaluation*100/(first_budget - population), 2)}%. Current cost : {np.round(f_cout(best),6)}')

    convergence = convergence[0:generation+1]

    ## Second algorithm for each indviduals.
    result_population = []
    print('Switching to bfgs gradient descent...') if progression else None
    count = 0
    for individual in omega:
        # do the gradient.
        gradient, _ = bfgs(f_cout, cut_budget, individual, X_min, X_max)
        # counts and prints.
        count += 1
        print(f'Gradient descent progression : {count}/{population} completed.') if progression else None
        # Append result.
        result_population.append(gradient)
    
    ## Who is the best ?
    costs = [f_cout(_) for _ in result_population]
    who = np.argmin(costs)
    best = result_population[who]
    
    distance_generation = []
    for structure in result_population:
        d = np.linalg.norm(structure - best, ord=2)
        distance_generation.append(d)
    distances.append(distance_generation)

    ## Then create the matrix of the distance.
    ## This is a symetric matrix ( the distance from i to j is equal to the distance from j to i.)
    ## with a null diagonal ( the distance from i to itself is always zero).

    matrix_ = []
    for individual in result_population:
        d = [np.linalg.norm(x - individual, ord=2) for x in result_population]
        matrix_.append(d)

    matrix = np.array(matrix_)

    return [best, convergence, result_population, distances, matrix]