# %% [markdown]
# ##### ---
# 
# #### Activity 1.        
# 
# Write a function named `load_mdp` that receives, as input, a string corresponding to the name of the file with the MDP information, and a real number $\gamma$ between $0$ and $1$. The loaded file will contain the transition matrix for the two actions, with P[0] corresponding to the transition of the action stop and P[1] for the action go.
# 
# Create a tuple including
# 
# * An array `X` that contains all the states in the MDP represented as strings.
# * An array `A` that contains all the actions in the MDP, represented as strings. In the domain above, for example, each action is represented as a string `"St"`, and `"Go"`.
# * An array `P` containing `len(A)` subarrays, each with dimension `len(X)` &times; `len(X)` and  corresponding to the transition probability matrix for one action.
# * An array `c` with dimension `len(X)` &times; `len(A)` containing the cost function for the MDP. The cost must be 1 in all states except in the states corresponding to the top level (level 9); for top-level states, the cost must be zero.
# 
# Your function should create the MDP as a tuple `(X, A, (Pa, a = 0, ..., len(A)), c, g)`, where `X` is a tuple containing the states in the MDP represented as strings (see above), `A` is a tuple containing the actions in the MDP represented as strings (see above), `P` is a tuple with `len(A)` elements, where `P[a]` is an `np.array` corresponding to the transition probability matrix for action `a`, `c` is an `np.array` corresponding to the cost function for the MDP, and `g` is a float, corresponding to the discount and provided as the argument $\gamma$ of your function. Your function should return the MDP tuple.
# 
# 
# ---

# %%
import numpy as np

def load_mdp(fname, gamma):
    """
    Builds an MDP model from the provided file.

    :param fname: Name of the file containing the MDP information
    :type: str
    :param gamma: Discount
    :type: float
    :returns: tuple (tuple, tuple, tuple, nd.array, float)
    """
    P1 = np.load(fname)

    n_states = P1[0].shape[0]
    X = tuple(str(i) for i in range(n_states))
    A = ("St", "Go")

    c = np.ones((n_states, len(A)))

    final_states = list(range(n_states - 10, n_states)) # states 9(0) - 9(9)
    c[final_states, :] = 0

    P = (P1[0], P1[1])

    return (X, A, P, c, gamma)

# -- End: load_mdp

# %% [markdown]
# ---
# 
# #### Activity 2.
# 
# Write a function `noisy_policy` that builds a noisy policy "around" a provided action. Your function should receive, as input, an MDP described as a tuple like that of **Activity 1**, an integer `a`, corresponding to the _index_ of an action in the MDP, and a real number `eps`. The function should return, as output, a policy for the provided MDP that selects action with index `a` with a probability `1 - eps` and, with probability `eps`, selects another action uniformly at random. The policy should be a `numpy` array with as many rows as states and as many columns as actions, where the element in position `[x, a]` should contain the probability of action `a` in state `x` according to the desired policy.
# 
# **Note:** The examples provided correspond for the MDP in the previous environment. However, your code should be tested with MDPs of different sizes, so **make sure not to hard-code any of the MDP elements into your code**.
# 
# ---

# %%
def noisy_policy(mdp, a, eps):
    """
    Builds a noisy policy around action a for a given MDP.

    :param mdp: MDP description
    :type: tuple
    :param a: main action for the policy
    :type: integer
    :param eps: noise level
    :type: float
    :return: nd.array
    """

    n_states = len(mdp[0])
    n_actions = len(mdp[1])
    pol = np.full((n_states, n_actions), eps / (n_actions - 1))
    pol[:, a] = 1 - eps

    return pol

# -- End: noisy_policy

# %% [markdown]
# ---
# 
# #### Activity 3.
# 
# You will now write a function called `evaluate_pol` that evaluates a given policy. Your function should receive, as an input, an MDP described as a tuple like that of **Activity 1** and a policy described as an array like that of **Activity 2** and return a `numpy` array corresponding to the cost-to-go function associated with the given policy.
# 
# **Note:** The array returned by your function should have as many rows as the number of states in the received MDP, and exactly one column. Note also that, as before, your function should work with **any** MDP that is specified as a tuple with the same structure as the one from **Activity 1**. In your solution, you may find useful the function `np.linalg.inv`, which can be used to invert a matrix.
# 
# ---

# %%
def evaluate_pol(mdp, pol):
    """
    Computes the cost-to-go function for a given policy in a given MDP.

    :param mdp: MDP description
    :type: tuple
    :param pol: Policy to be evaluated
    :type: nd.array
    :returns: nd.array
    """
    X, A, P, c, gamma = mdp
    n_states = len(X)
    P_pi = sum(pol[:, a].reshape(-1, 1) * P[a] for a in range(len(A)))
    c_pi = np.sum(pol * c, axis=1)
    J = np.linalg.inv(np.eye(n_states) - gamma * P_pi) @ c_pi
    return J.reshape(-1, 1)

# -- End: evaluate

# %% [markdown]
# ---
# 
# #### Activity 4
# 
# In this activity you will show that the policy in Activity 3 is _not_ optimal. For that purpose, you will use value iteration to compute the optimal cost-to-go, $J^*$, and show that $J^*\neq J^\pi$.
# 
# Write a function called `value_iteration` that receives as input an MDP represented as a tuple like that of **Activity 1** and returns an `numpy` array corresponding to the optimal cost-to-go function associated with that MDP. Before returning, your function should print:
# 
# * The time it took to run, in the format `Execution time: xxx seconds`, where `xxx` represents the number of seconds rounded up to $3$ decimal places.
# * The number of iterations, in the format `N. iterations: xxx`, where `xxx` represents the number of iterations.
# 
# **Note 1:** Stop the algorithm when the error between iterations is smaller than $10^{-8}$. To compute the error between iterations, you should use the function `norm` from `numpy.linalg`.
# 
# **Note 2:** You may find useful the function ``time()`` from the module ``time``. You may also find useful the code provided in the theoretical lecture.
# 
# **Note 3:** The array returned by your function should have as many rows as the number of states in the received MDP, and exactly one column. As before, your function should work with **any** MDP that is specified as a tuple with the same structure as the one from **Activity 1**.
# 
# 
# ---

# %%
import time
from numpy.linalg import norm

def value_iteration(mdp):
    """
    Computes the optimal cost-to-go function for a given MDP.

    :param mdp: MDP description
    :type: tuple
    :returns: nd.array
    """
    X, A, P, c, gamma = mdp
    n_states = len(X)
    J = np.zeros(n_states)

    threshold = 1e-8
    delta = float('inf')
    k = 0

    start_time = time.time()

    while delta > threshold:
        J_new = np.min([c[:, a] + gamma * P[a] @ J for a in range(len(A))], axis=0)
        delta = norm(J_new - J)
        J = J_new
        k += 1

    end_time = time.time()

    print(f'Execution time: {end_time - start_time:.3f} seconds')
    print(f'N. iterations: {k}')

    return J.reshape(-1, 1)

# -- End: value_iteration

# %% [markdown]
# ---
# 
# #### Activity 5
# 
# You will now compute the optimal policy using policy iteration. Write a function called `policy_iteration` that receives as input an MDP represented as a tuple like that of **Activity 1** and returns an `numpy` array corresponding to the optimal policy associated with that MDP. Consider the initial policy is the uniformly random policy. Before returning, your function should print:
# * The time it took to run, in the format `Execution time: xxx seconds`, where `xxx` represents the number of seconds rounded up to $3$ decimal places.
# * The number of iterations, in the format `N. iterations: xxx`, where `xxx` represents the number of iterations.
# 
# **Note:** If you find that numerical errors affect your computations (especially when comparing two values/arrays) you may use the `numpy` function `isclose` with adequately set absolute and relative tolerance parameters (e.g., $10^{-8}$). You may also find useful the code provided in the theoretical lecture.
# 
# ---

# %%
def policy_iteration(mdp):
    """
    Computes the optimal policy for a given MDP.

    :param mdp: MDP description
    :type: tuple
    :returns: nd.array
    """
    X, A, P, c, gamma = mdp
    pol = np.ones((len(X), len(A))) / len(A)
    quit = False
    k = 0
    
    start_time = time.time()

    while not quit:
        Q = np.zeros((len(X), len(A)))
        J = evaluate_pol(mdp, pol)

        for a in range(len(A)):
            Q[:, a, None] = c[:, a, None] + gamma * P[a].dot(J)

        Qmin = np.min(Q, axis=1, keepdims=True)
        pnew = np.isclose(Q, Qmin, atol=1e-8, rtol=1e-8).astype(int)
        pnew = pnew / pnew.sum(axis=1, keepdims=True)

        quit = (pol == pnew).all()
        pol = pnew
        k += 1

    end_time = time.time()

    print(f'Execution time: {end_time - start_time:.3f} seconds')
    print(f'N. iterations: {k}')
    return np.round(pol, 3)

# -- End: policy_iteration

# %% [markdown]
# ---
# 
# #### Activity 6
# 
# Write a function `simulate` that receives, as inputs
# 
# * An MDP represented as a tuple like that of **Activity 1**;
# * A policy, represented as an `numpy` array like that of **Activity 2**;
# * An integer, `x0`, corresponding to a state index
# * A second integer, `length`
# 
# Your function should return, as an output, a float corresponding to the estimated cost-to-go associated with the provided policy at the provided state. To estimate such cost-to-go, your function should:
# 
# * Generate **`NRUNS`** trajectories of `length` steps each, starting in the provided state and following the provided policy.
# * For each trajectory, compute the accumulated (discounted) cost.
# * Compute the average cost over the 100 trajectories.
# 
# **Note 1:** You may find useful to import the numpy module `numpy.random`.
# 
# **Note 2:** Each simulation may take a bit of time, don't despair ☺️.
# 
# ---

# %%
import numpy.random as rand

NRUNS = 100 # Do not delete this

def simulate(mdp, pol, x0, length=10000):
    """
    Estimates the cost-to-go for a given MDP, policy and state.

    :param mdp: MDP description
    :type: tuple
    :param pol: policy to be simulated
    :type: nd.array
    :param x0: initial state
    :type: int
    :returns: float
    """
    X, A, P, c, gamma = mdp
    total_costs = []

    for _ in range(NRUNS):
        x = x0
        cost = 0
        discount = 1

        for _ in range(length):
            a = rand.choice(len(A), p=pol[x])
            cost += discount * c[x, a]
            discount *= gamma
            x = rand.choice(len(X), p=P[a][x])

        total_costs.append(cost)

    return np.mean(total_costs)

# -- End: simulate
