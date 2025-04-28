# Intelligent Planning Agents for Dynamic Environments

Group project developed for the Planning, Learning and Intelligent Decision Making course at Instituto Superior Técnico in 2024/25.

## Introduction
This project explores the design and implementation of intelligent agents capable of planning and acting effectively in dynamic environments. We tackled deterministic, stochastic, and partially observable settings using established techniques in automated planning and decision-making.

Our work was organized into three development phases:

- **Classical Planning (Homework 1)**
- **Markov Decision Processes (MDPs) (Homework 2)**
- **Partially Observable Markov Decision Processes (POMDPs) (Homework 3)**

Each phase introduced new layers of complexity, transitioning from full knowledge of the environment to handling uncertainty and partial observability.

## Homework 1 – Classical Planning
**Objective:** Model a planning problem using a classical (deterministic) approach.

**Techniques:** Problem modeling in PDDL (Planning Domain Definition Language).

**Challenges Addressed:**
- Defining domain actions, preconditions, and effects.
- Generating plans using standard deterministic planners.

**Key Outcome:**
- Successful automated generation of action sequences leading from initial states to goal states in deterministic settings.

## Homework 2 – Markov Decision Processes (MDPs)
**Objective:** Solve a planning problem with stochastic transitions using MDP techniques.

**Techniques:**
- Value Iteration
- Policy Iteration

**Challenges Addressed:**
- Defining state spaces, transition probabilities, and reward functions.
- Computing optimal policies that maximize expected cumulative rewards.

**Key Outcome:**
- Development of robust agent strategies under uncertainty, highlighting the advantages of probabilistic modeling in decision-making.

## Homework 3 – Partially Observable MDPs (POMDPs)
**Objective:** Plan and act in environments with partial observability.

**Techniques:**
- Belief State representation
- Policy computation over belief spaces

**Challenges Addressed:**
- Handling uncertainty not only in outcomes but also in state information.
- Managing the complexity of belief space planning.

**Key Outcome:**
- Agents were able to make informed decisions based on probabilistic beliefs, improving adaptability in incomplete information settings.

## Team
- Beatriz Paulo - [@beatrizp03](https://github.com/beatrizp03)
- Leonor Fortes - [@leonorf03](https://github.com/leonorf03)
