# %% [markdown]
# ---
# 
# #### Activity 1.        
# 
# Write a function named `load_chain` that receives, as input, a string corresponding to the name of the file with a transition matrix to be loaded, and a real number $\gamma$ between $0$ and $1$. Assume that:
# 
# * The transition matrices in the file have been built from a representation of the game with P[0] corresponding to the transition of the action stop and P[1] for the action go.
# 
# * For this first lab we do not consider the process of choosing action so we consider that the action are choosen at random with the action go selected with probability $\gamma$ .
# 
# Your function should build the transition probability matrix for the chain by combining the two actions using the value of $\gamma$. Your function should then return, as output, a two-element tuple corresponding to the Markov chain, where:
# 
# * ... the first element is a tuple containing an enumeration of the state-space (i.e., each element of the tuple corresponds to a state of the chain, represented as a string);
# * ... the second element is a `numpy` array corresponding to the transition probability matrix for the chain.
# 
# ---

# %%
import numpy as np

def load_chain(filename: str, gamma: float):
    
    P = np.load(filename)
    
    P_stop = P[0]  # stop
    P_go = P[1]    # go 

    num_states = P_stop.shape[0]
    
    state_space = tuple(str(i) for i in range(num_states))
    
    P_combined = (1 - gamma) * P_stop + gamma * P_go

    return (state_space, P_combined)



# %% [markdown]
# The following functions might be useful to convert the state representation. For instance the state with the index 10 can also be represented as 1(0) meaning that the player is in the level 1 with safe level 0. State index 22 corresponds to being in level 2 with safe level 2.
# 

# %%
# Auxiliary function to convert state representation to state index
Lm = 10

def observ2state(observ):
    dimsizes = [Lm, Lm]
    return np.ravel_multi_index(observ, dimsizes)

# Auxiliary function to convert state index to state representation
def state2observ(state):
    dimsizes = [Lm, Lm]
    return np.unravel_index(int(state), dimsizes)

# Auxiliary function to print a sequence of states
def printTraj(seq):
    ss = ""
    for st in seq:
        ss += printState(st) + "\n"

    return ss

# Auxiliary function to print a state
def printState(state):
    if type(state) in [list,tuple]:
        l = state[0]
        s = state[1]
    else:
        l,s = state2observ(state)

    return "%d (%d)" % (l, s)

print(10, state2observ('10'))
print(22, state2observ('22'))

# %% [markdown]
# In the next activity, you will use the Markov chain model to evaluate the likelihood of any given path for the player.
# 
# ---
# 
# #### Activity 2.
# 
# Write a function `prob_trajectory` that receives, as inputs,
# 
# * ... a Markov chain in the form of a tuple like the one returned by the function in Activity 1;
# * ... a trajectory, corresponding to a sequence of states (i.e., a tuple or list of strings, each string corresponding to a state).
# 
# Your function should return, as output, a floating point number corresponding to the probability of observing the provided trajectory, taking the first state in the trajectory as initial state.  
# 
# ---

# %%
def prob_trajectory(markov_chain, trajectory):
    _, P = markov_chain
    
    indices = [observ2state(state2observ(int(state))) for state in trajectory]
    
    prob = 1.0
    for i in range(len(indices) - 1):
        prob *= P[indices[i], indices[i + 1]]  
          
    return prob


# %% [markdown]
# ### 2. Stability

# %% [markdown]
# The next activities explore the notion of *stationary distribution* for the chain.
# 
# ---
# 
# #### Activity 3
# 
# Write a function `stationary_dist` that receives, as input, a Markov chain in the form of a tuple like the one returned by the function in Activity 1. Your function should return, as output, a `numpy` array corresponding to a row vector containing the stationary distribution for the chain.
# 
# **Note:** The stationary distribution is a *left* eigenvector of the transition probability matrix associated to the eigenvalue 1. As such, you may find useful the numpy function `numpy.linalg.eig`. Also, recall that the stationary distribution is *a distribution*. You may also find useful the function `numpy.real` which returns the real part of a complex number.
# 
# ---

# %%
def stationary_dist(markov_chain):
    _, P = markov_chain
    
    eigvals, eigvecs = np.linalg.eig(P.T) # we use transpose because the np.linalg.eig() function computes the right eigenvectors and we want the left ones
    
    stationary_vec = eigvecs[:, np.isclose(eigvals, 1)].T  # transpose to get the row vector
    
    stationary_vec = np.real(stationary_vec)
    stationary_vec /= np.sum(stationary_vec) # normalizing
    
    return stationary_vec.reshape(1, -1) 


# %% [markdown]
# To complement Activity 3, you will now empirically establish that the chain is ergodic, i.e., no matter where the player starts, its visitation frequency will eventually converge to the stationary distribution.
# 
# ---
# 
# #### Activity 4.
# 
# Write a function `compute_dist` that receives, as inputs,
# 
# * ... a Markov chain in the form of a tuple like the one returned by the function in Activity 1;
# * ... a row vector (a numpy array) corresponding to the initial distribution for the chain;
# * ... an integer $N$, corresponding to the number of steps that the chain is expected to take.
# 
# Your function should return, as output, a row vector (a `numpy` array) containing the distribution after $N$ steps of the chain. Use your function to justify that the chain is ergodic.
# 
# ---

# %%
def compute_dist(markov_chain, initial_dist, N):
    _, P = markov_chain

    # P^N
    P_expN = np.linalg.matrix_power(P, N)
    
    final_dist = initial_dist @ P_expN
    
    return final_dist

# %% [markdown]
# <font color='blue'>Write your answer here.</font>
# 
# As demonstrated in the examples below, the probability distribution of the Markov chain converges to a unique stationary distribution u*, regardless of the initial state. This indicates that the chain is ergoic.




# %% [markdown]
# Example of application of the function.
# 
# ```python
# import numpy.random as rnd
# 
# rnd.seed(42)
# 
# REPETITIONS = 5
# 
# print('- Mgo - always select go -')
# 
# # Number of states
# nS = len(Mgo[0])
# u_star = stationary_dist(Mgo)
# 
# # Repeat a number of times
# for n in range(REPETITIONS):
# 
#     print('\n- Repetition', n + 1, 'of', REPETITIONS, '-')
# 
#     # Initial random distribution
#     u = rnd.random((1, nS))
#     u = u / np.sum(u)
# 
#     # Distrbution after 10 steps
#     v = compute_dist(Mgo, u, 100)
#     print('Is u * P^100 = u*?', np.all(np.isclose(v, u_star)))
# 
#     # Distrbution after 100 steps
#     v = compute_dist(Mgo, u, 200)
#     print('Is u * P^2000 = u*?', np.all(np.isclose(v, u_star)))
# 
# print('- Mgostop - select go half of the time -')
# 
# # Number of states
# nS = len(Mgostop[0])
# u_star = stationary_dist(Mgostop)
# 
# # Repeat a number of times
# for n in range(REPETITIONS):
# 
#     print('\n- Repetition', n + 1, 'of', REPETITIONS, '-')
# 
#     # Initial random distribution
#     u = rnd.random((1, nS))
#     u = u / np.sum(u)
# 
#     # Distrbution after 100 steps
#     v = compute_dist(Mgostop, u, 100)
#     print('Is u * P^100 = u*?', np.all(np.isclose(v, u_star)))
# 
#     # Distrbution after 2000 steps
#     v = compute_dist(Mgostop, u, 200)
#     print('Is u * P^2000 = u*?', np.all(np.isclose(v, u_star)))
# ```
# 
# Output:
# ````
# - Mgo - always select go -
# 
# - Repetition 1 of 5 -
# Is u * P^100 = u*? True
# Is u * P^2000 = u*? True
# 
# - Repetition 2 of 5 -
# Is u * P^100 = u*? True
# Is u * P^2000 = u*? True
# 
# - Repetition 3 of 5 -
# Is u * P^100 = u*? True
# Is u * P^2000 = u*? True
# 
# - Repetition 4 of 5 -
# Is u * P^100 = u*? True
# Is u * P^2000 = u*? True
# 
# - Repetition 5 of 5 -
# Is u * P^100 = u*? True
# Is u * P^2000 = u*? True
# - Mgostop - select go half of the time -
# 
# - Repetition 1 of 5 -
# Is u * P^100 = u*? True
# Is u * P^2000 = u*? True
# 
# - Repetition 2 of 5 -
# Is u * P^100 = u*? True
# Is u * P^2000 = u*? True
# 
# - Repetition 3 of 5 -
# Is u * P^100 = u*? True
# Is u * P^2000 = u*? True
# 
# - Repetition 4 of 5 -
# Is u * P^100 = u*? True
# Is u * P^2000 = u*? True
# 
# - Repetition 5 of 5 -
# Is u * P^100 = u*? True
# Is u * P^2000 = u*? True
# ```

# %% [markdown]
# ### 3. Simulation
# 
# In this part of the lab, you will *simulate* the actual player, and empirically compute the visitation frequency of each state.

# %% [markdown]
# ---
# 
# #### Activity 5
# 
# Write down a function `simulate` that receives, as inputs,
# 
# * ... a Markov chain in the form of a tuple like the one returned by the function in Activity 1;
# * ... a row vector (a `numpy` array) corresponding to the initial distribution for the chain;
# * ... an integer $N$, corresponding to the number of steps that the chain is expected to take.
# 
# Your function should return, as output, a tuple containing a trajectory with $N$ states, where the initial state is sampled according to the initial distribution provided. Each element in the tuple should be a string corresponding to a state index.
# 
# ---
# 
# **Note:** You may find useful to import the numpy module `numpy.random`.

# %%
def simulate(markov_chain, initial_dist, N):
    state_space, P = markov_chain
    
    current_state = np.random.choice(state_space, p=initial_dist.flatten()) # use the flatten to get a single collumn vector
    
    trajectory = [current_state] # starts at the current state
    
    for _ in range(N - 1):
        current_index = state_space.index(current_state)
        
        next_state = np.random.choice(state_space, p=P[current_index]) # in current state line any random next state
        
        trajectory.append(next_state) # adds a new state
        current_state = next_state
    
    return tuple(trajectory)


# %% [markdown]
# ---
# 
# #### Activity 6
# 
# We will now compare the relative speed of two chains.
# Create two chains, one where we always choose Go and another where we choose Go 3/4 of the time and Stop 1/4 of the time.
# 
# Which one is faster? Verify using one sampling approach, and one analytical approach.
# 
# Is the best way to choose the action the same for the game with 20% rainy days ('StopITSpider02.npy') and the game with 40% rainy days?.
# 
# ---

# %%
def compare_chains(filename):
    """
    Compare the relative speed of two chains using analytical and simulation approaches.
    """
    chain_go = load_chain(filename, 1.0)
    chain_mixed = load_chain(filename, 0.75)

    initial_dist = np.zeros((1, len(chain_go[0])))
    initial_dist[0, 0] = 1

    steps = 1000
    final_dist_go = compute_dist(chain_go, initial_dist, steps)
    final_dist_mixed = compute_dist(chain_mixed, initial_dist, steps)
    
    final_state_indices = list(range(len(chain_go[0]) - 10, len(chain_go[0])))
    prob_reach_go = np.sum(final_dist_go[0, final_state_indices])
    prob_reach_mixed = np.sum(final_dist_mixed[0, final_state_indices])
    
    # Simulate probability of reaching level 9 using large trajectory
    prob_sim_go = test_simulation(filename, chain_go)
    prob_sim_mixed = test_simulation(filename, chain_mixed)
    
    print(f"\nResults for {filename}:")
    print(f"  Stationary probability of final state:")
    print(f"     Always Go (γ=1.0): {prob_reach_go:.4f}")
    print(f"     Mixed Go/Stop (γ=0.75): {prob_reach_mixed:.4f}")
    print(f"  Probability of being at final state after {steps} steps:")
    print(f"     Always Go (γ=1.0): {prob_sim_go:.4f}")
    print(f"     Mixed Go/Stop (γ=0.75): {prob_sim_mixed:.4f}")

def test_simulation(filename, markov_chain):
    state_space, _ = markov_chain
    nS = len(state_space)
    initial_dist = np.ones((1, nS)) / nS  # Uniform initial distribution
    traj = simulate(markov_chain, initial_dist, 10000)
    level_9_states = {str(i) for i in range(90, 100)}
    count_level_9 = sum(1 for state in traj if state in level_9_states)
    return count_level_9 / len(traj)

print("20% Rainy Days")
compare_chains("StopITSpider02.npy")
print("\n40% Rainy Days")
compare_chains("StopITSpider04.npy")

# %% [markdown]
# Which one is faster?
# 
#     When we always choose Go, as shown bellow.
# 
#     20% Rainy Days
# 
#     Results for StopITSpider02.npy:
#     Stationary probability of final state:
#         Always Go (γ=1.0): 0.1249
#         Mixed Go/Stop (γ=0.75): 0.1105
#     Probability of being at final state after 1000 steps:
#         Always Go (γ=1.0): 0.1229
#         Mixed Go/Stop (γ=0.75): 0.1133
# 
#     40% Rainy Days
# 
#     Results for StopITSpider04.npy:
#     Stationary probability of final state:
#         Always Go (γ=1.0): 0.0618
#         Mixed Go/Stop (γ=0.75): 0.0703
#     Probability of being at final state after 1000 steps:
#         Always Go (γ=1.0): 0.0606
#         Mixed Go/Stop (γ=0.75): 0.0693
# 
# Is the best way to choose the action the same for the game with 20% rainy days ('StopITSpider02.npy') and the game with 40% rainy days?
# 
#     No, it isnt the same since the probabilities of reaching the final states and the stationary distribution, are not the same.
# 

# %%



