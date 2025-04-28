# %% [markdown]
# # Learning and Decision Making

# %% [markdown]
# ---
# 
# #### Activity 1.        
# 
# Write a function named `load_pomdp` that receives, as input, a string corresponding to the name of the file with the POMDP information, and a real number $\gamma$ between $0$ and $1$. The loaded file contains 6 arrays:
# 
# * An array `X` that contains all the states in the POMDP, represented as strings. In the police-bandit scenario above, for example, there is a total of 25 states, each describing a stage in the game. The state is represented as x(y) where x is the location of the police and y the location of the bandit
# * An array `A` that contains all the actions in the POMDP, also represented as strings. In the  domain above, for example, each action is represented as one of the actions 'left', 'right', 'catch', or 'observe'.
# * An array `Z` that contains all the observations in the POMDP, also represented as strings. In the domain above, for example, there is a total of 6 observations.
# * An array `P` containing `len(A)` subarrays, each with dimension `len(X)` &times; `len(X)` and  corresponding to the transition probability matrix for one action.
# * An array `O` containing `len(A)` subarrays, each with dimension `len(X)` &times; `len(Z)` and  corresponding to the observation probability matrix for one action.
# * An array `c` with dimension `len(X)` &times; `len(A)` containing the cost function for the POMDP.
# 
# Your function should create the POMDP as a tuple `(X, A, Z, (Pa, a = 0, ..., len(A)), (Oa, a = 0, ..., len(A)), c, g)`, where `X` is a tuple containing the states in the POMDP represented as strings (see above), `A` is a tuple containing the actions in the POMDP represented as strings (see above), `Z` is a tuple containing the observations in the POMDP represented as strings (see above), `P` is a tuple with `len(A)` elements, where `P[a]` is an `np.array` corresponding to the transition probability matrix for action `a`, `O` is a tuple with `len(A)` elements, where `O[a]` is an `np.array` corresponding to the observation probability matrix for action `a`, `c` is an `np.array` corresponding to the cost function for the POMDP, and `g` is a float, corresponding to the discount and provided as the argument $\gamma$ of your function. Your function should return the POMDP tuple.
# 
# ---

# %%
import numpy as np

def load_pomdp(fname, gamma):
    """
    Builds a POMDP model from the provided file.

    :param fname: Name of the file containing the POMDP information
    :type fname: str
    :param gamma: Discount factor
    :type gamma: float
    :returns: tuple (X, A, Z, P, O, c, gamma)
    """
    data = np.load(fname, allow_pickle=True)

    X = tuple(data['X'])  
    A = tuple(data['A'])  
    Z = tuple(data['Z']) 
    P = tuple(data['P'])  
    O = tuple(data['O'])  
    c = data['c']         

    return (X, A, Z, P, O, c, gamma)

# -- End: load_pomdp

# %% [markdown]
# ### 2. Sampling
# 
# You are now going to sample random trajectories of your POMDP and observe the impact it has on the corresponding belief.

# %% [markdown]
# ---
# 
# #### Activity 2.
# 
# Write a function called `gen_trajectory` that generates a random POMDP trajectory using a uniformly random policy. Your function should receive, as input, a POMDP described as a tuple like that from **Activity 1** and two integers, `x0` and `n` and return a tuple with 3 elements, where:
# 
# 1. The first element is a `numpy` array corresponding to a sequence of `n + 1` state indices, $x_0,x_1,\ldots,x_n$, visited by the agent when following a uniform policy (i.e., a policy where actions are selected uniformly at random) from state with index `x0`. In other words, you should select $x_1$ from $x_0$ using a random action; then $x_2$ from $x_1$, etc.
# 2. The second element is a `numpy` array corresponding to the sequence of `n` action indices, $a_0,\ldots,a_{n-1}$, used in the generation of the trajectory in 1.;
# 3. The third element is a `numpy` array corresponding to the sequence of `n` observation indices, $z_1,\ldots,z_n$, experienced by the agent during the trajectory in 1.
# 
# The `numpy` array in 1. should have a shape `(n + 1,)`; the `numpy` arrays from 2. and 3. should have a shape `(n,)`.
# 
# **Note:** Your function should work for **any** POMDP specified as above.
# 
# ---

# %%
import numpy.random as rnd

def gen_trajectory(pomdp, x0, steps):
    """
    Generates a random trajectory for the provided POMDP from the provided initial state and with
    the provided number of steps.

    :param pomdp: POMDP description
    :type pomdp: tuple
    :param x0: Initial state
    :type x0: int
    :param steps: Number of steps
    :type steps: int
    :returns: tuple (nd.array, nd.array, nd.array)
    """
    
    X, A, Z, P, O, c, gamma = pomdp
    n_states = len(X)
    n_actions = len(A)
    n_observations = len(Z)
    
    state_traj = np.zeros(steps + 1, dtype=int)
    action_traj = np.zeros(steps, dtype=int)
    observation_traj = np.zeros(steps, dtype=int)
    
    state_traj[0] = x0
    
    for t in range(steps):
        action = rand.randint(n_actions)
        action_traj[t] = action
        
        next_state = rand.choice(n_states, p=P[action][state_traj[t], :])
        state_traj[t + 1] = next_state
        
        observation = rand.choice(n_observations, p=O[action][next_state, :])
        observation_traj[t] = observation

    return (state_traj, action_traj, observation_traj)

# -- End: gen_trajectory

# %% [markdown]
# You will now write a function that samples a given number of possible belief points for a POMDP. To do that, you will use the function from **Activity 2**.
# 
# ---
# 
# #### Activity 3.
# 
# Write a function called `sample_beliefs` that receives, as input, a POMDP described as a tuple like that from **Activity 1** and an integer `n`, and return a tuple with `n + 1` elements **or less**, each corresponding to a possible belief state (represented as a $1\times|\mathcal{X}|$ vector). To do so, your function should
# 
# * Generate a trajectory with `n` steps from a random initial state, using the function `gen_trajectory` from **Activity 2**.
# * For the generated trajectory, compute the corresponding sequence of beliefs, assuming that the agent does not know its initial state (i.e., the initial belief is the uniform belief, and should also be considered).
# 
# Your function should return a tuple with the resulting beliefs, **ignoring duplicate beliefs or beliefs whose distance is smaller than $10^{-3}$.**
# 
# **Suggestion:** You may want to define an auxiliary function `belief_update` that receives a POMDP, a belief, an action and an observation and returns the updated belief.
# 
# **Note:** Your function should work for **any** POMDP specified as above. To compute the distance between vectors, you may find useful `numpy`'s function `linalg.norm`.
# 
# 
# ---

# %%
def belief_update(pomdp, b, a, z):
    X, A, Z, P, O, c, gamma = pomdp

    b_next = O[a][:, z] * (P[a].T @ b)  # Bayesian update
    b_next /= np.sum(b_next)  # Normalize
    
    return b_next.flatten() 

# -- belief_update

def sample_beliefs(pomdp, n):
    """
    Generates a random sample of belief states for the provided POMDP.

    :param pomdp: POMDP description
    :type: tuple
    :param n: Maximum number of sampled beliefs
    :type: int
    :returns: tuple (n x nd.array)
    """
    X, A, Z, P, O, c, gamma = pomdp
    n_states = len(X)
    
    x0 = rand.randint(n_states)
    states, actions, observations = gen_trajectory(pomdp, x0, n)
    
    belief_list = []
    b = np.ones(n_states) / n_states
    belief_list.append(b) 
    
    for t in range(n):
        a = actions[t]
        z = observations[t]
        b = belief_update(pomdp, b, a, z)
        
        if not any(np.linalg.norm(b - b_old) < 1e-3 for b_old in belief_list):
            belief_list.append(b)
    
    return np.matrix(tuple(belief_list))

# -- sample_belief

# %% [markdown]
# <font color="white">**Question 1**: Assume the initial belief is, as implemented above, the uniform belief. **Q1.1** If we were to consider a different policy than the random policy to sample beliefs, should we expect to get a different set of sampled beliefs? **Q1.2** Is our code above able to retrieve all possible beliefs that can be attained under all policies (assume we are able to sample an infinite number of trajectories of infinite-length)? </font>

# %% [markdown]
# <font color="white">**Insert answer:** **Q1.1** </font>
# 
# ### Q1.1
# Yes, we should expect to get a different set of sampled beliefs if we use a different policy instead of a random policy. The belief update depends on both the chosen actions and the resulting observations, and a different policy may favor more or less certain actions, which will make some belief states more or less prone to be reached.
# 
# ### Q1.2
# No, our current code is not guaranteed to retrieve all possible beliefs, even if we sample an infinite number of trajectories of infinite length. The belief update depends on actions and observations, meaning that some beliefs may never be reached under a specific policy. Even with infinite sampling, the constraint on duplicate beliefs introduces a threshold that may filter out certain close but distinct beliefs.

# %% [markdown]
# ### 3. MDP-based heuristics
# 
# In this section you are going to compare different heuristic approaches for POMDPs discussed in class.

# %% [markdown]
# ---
# 
# #### Activity 4
# 
# Write a function `solve_mdp` that takes as input a POMDP represented as a tuple like that of **Activity 1** and returns a `numpy` array corresponding to the **optimal $Q$-function for the underlying MDP**. Stop the algorithm when the error between iterations is smaller than $10^{-8}$.
# 
# **Note:** Your function should work for **any** POMDP specified as above. Feel free to reuse one of the functions you implemented in Lab 2 (for example, value iteration).
# 
# ---

# %%
from numpy.linalg import norm

def solve_mdp(pomdp):
    """
    Computes the optimal Q-function for the underlying MDP.

    :param pomdp: POMDP description
    :type: tuple
    :returns: nd.array
    """
    X, A, Z, P, O, c, gamma = pomdp
    n_states = len(X)
    n_actions = len(A)

    Q = np.zeros((n_states, n_actions))

    threshold = 1e-8
    delta = float('inf')

    while delta > threshold:
        Q_new = np.zeros((n_states, n_actions))

        for a in range(n_actions):
            Q_new[:, a] = c[:, a] + gamma * P[a] @ np.min(Q, axis=1)

        delta = norm(Q_new - Q)
        Q = Q_new

    return Q

# -- End: solve_mdp

# %% [markdown]
# ---
# 
# #### Activity 5
# 
# You will now test the different MDP heuristics discussed in class. To that purpose, write down a function that, given a belief vector and the solution for the underlying MDP, computes the action prescribed by each of the three MDP heuristics. In particular, you should write down a function named `get_heuristic_action` that receives, as inputs:
# 
# * A belief state represented as a `numpy` array like those of **Activity 3**;
# * The optimal $Q$-function for an MDP (computed, for example, using the function `solve_mdp` from **Activity 4**);
# * A string that can be either `"mls"`, `"av"`, or `"q-mdp"`;
# 
# Your function should return an integer corresponding to the index of the action prescribed by the heuristic indicated by the corresponding string, i.e., the most likely state heuristic for `"mls"`, the action voting heuristic for `"av"`, and the $Q$-MDP heuristic for `"q-mdp"`. *In all heuristics, ties should be broken randomly, i.e., when maximizing/minimizing, you should randomly select between all maximizers/minimizers*.
# 
# ---

# %%
def get_heuristic_action(belief, qfunction, heuristic):
    """
    Computes the action prescribed by the selected MDP heuristic at the provided belief.

    :param belief: belief vector
    :type: nd.array
    :param qfunction: optimal q-function for the underlying MDP
    :type: nd.array
    :param heuristic: selected heuristic
    :type: str
    :returns: int
    """
    if heuristic == "mls":
        most_likely_state = np.argmax(belief)
        action_values = qfunction[most_likely_state, :]
    
    elif heuristic == "av":
        action_values = np.sum(belief[:, None] * qfunction, axis=0)
    
    elif heuristic == "q-mdp":
        action_values = belief @ qfunction

    action_values = np.asarray(action_values).flatten()

    min_value = np.min(action_values)

    best_actions = [i for i in range(len(action_values)) if np.isclose(action_values[i], min_value)]

    return rand.choice(best_actions)

# -- End: get_heuristic_action

# %% [markdown]
# ---
# 
# #### Activity 6
# 
# You will now test the different heuristics you implemented in the previous question. To do this, you will implement function `test_heuristic`, which receives as arguments:
# * A tuple specifying the POMDP;
# * The optimal $Q$-function for an MDP (computed, for example, using the function `solve_mdp` from **Activity 4**);
# * A string specifying the action-selection heuristic that can be either `"mls"`, `"av"`, or `"q-mdp"`;
# * An integer `n` specifying the legnth of each sampled trajectory;
# * An integer `NRUNS` specifying the number of sampled trajectories.
# 
# The function should first randomly sample an initial state (with equal probability of sampling any state) and consider the initial belief is uniform over the state space; then, the function should let the agent iteratively interact with the environment while updating its belief and choosing actions according to the specified heuristic. The function should sample `NRUNS` trajectories, compute the discounted cumulative costs for each trajectory, and return the mean discounted cumulative costs obtained over the `NRUNS` sampled trajectories.

# %%
def test_heuristic(pomdp, initial_state, qfunction, heuristic, n, NRUNS):
    """
    Simulate NRUNS trajectories of length n using a given action-selection heuristic
      to update the belief and select actions in a POMDP.

    :param pomdp: POMDP description
    :type: tuple
    :param initial_state: initial state
    :type: int
    :param qfunction: optimal q-function for the underlying MDP
    :type: nd.array
    :param heuristic: selected heuristic
    :type: str
    :param n: length of trajectories
    :type: int
    :param NRUNS: number of sampled trajectories
    :type: int

    :returns: float
    """

    X, A, Z, P, O, c, gamma = pomdp
    n_states = len(X)
    total_costs = []

    for run in range(NRUNS):
        state = initial_state
        belief = np.ones(n_states) / n_states 
        cumulative_cost = 0
        discount_factor = 1

        for t in range(n):
            action = get_heuristic_action(belief, qfunction, heuristic)
            next_state = rand.choice(n_states, p=P[action][state])
            observation = rand.choice(len(Z), p=O[action][next_state])

            cumulative_cost += discount_factor * c[state, action]
            discount_factor *= gamma

            belief = belief_update(pomdp, belief, action, observation)
            state = next_state

        total_costs.append(cumulative_cost)

    return np.mean(total_costs)


# %% [markdown]
# <font color="white">**Question 2:** **Q2.1** Which heuristic(s) appear(s) to perform the best? **Q2.2** Do we have any guarantees on the optimality of these action-selection procedures?
# Justify. </font>

# %% [markdown]
# <font color='white'>**Insert answer** **Q2.1** </font>
# 
# ### Q2.1: Best Performing Heuristic?
# Q-MDP heuristic generally performs the best since it considers expected future costs over all belief states.  MLS can fail if the most likely state is wrong, and AV struggles when different states favor different actions.
# 
# ### Q2.2: Are These Heuristics Optimal?
# No, none are guaranteed to be optimal. MLS ignores uncertainty, AV doesn't consider future costs, and Q-MDP assumes full observability after one step. Q-MDP is usually the best approximation but not always optimal.

# %% [markdown]
# <font color="white">**Question 2:** **Q2.2** In activity 6 what is the critical point in terms of efficiency?</font>

# %% [markdown]
# <font color='white'>**Insert answer** **Q2.2** </font>
# 
# The critical efficiency bottleneck in Activity 6 is belief state updates, as they involve costly matrix-vector multiplications. The main factors affecting efficiency are: the belief updates, as recalculating beliefs at each step using Bayes' Rule is computationally expensive, the trajectory sampling, as running NRUNS trajectories of n steps scales linearly in cost and action selection ('q-mdp' is the most expensive, as it requires belief-weighted Q-value computations).


