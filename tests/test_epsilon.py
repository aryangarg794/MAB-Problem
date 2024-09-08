import pytest
import numpy as np
from mabproblem.mab_problem import EpsilonGreedy

def test_initialization():
    epsilon_greedy = EpsilonGreedy()
    assert epsilon_greedy.epsilon == 0.2
    assert epsilon_greedy.experimental_means == {}
    assert epsilon_greedy.machine_turns == {}
    assert epsilon_greedy.best is None

    custom_epsilon_greedy = EpsilonGreedy(epsilon=0.5)
    assert custom_epsilon_greedy.epsilon == 0.5

def test_single_step_simulation():
    epsilon_greedy = EpsilonGreedy()
    
    machine = 0
    reward = 1.0
    epsilon_greedy.simulate_one_step(machine, reward)
    
    assert epsilon_greedy.machine_turns[machine] == 1
    assert epsilon_greedy.experimental_means[machine] == reward
    assert epsilon_greedy.best == machine

    new_reward = 0.5
    epsilon_greedy.simulate_one_step(machine, new_reward)
    
    assert epsilon_greedy.machine_turns[machine] == 2
    assert np.isclose(epsilon_greedy.experimental_means[machine], 0.75)
    assert epsilon_greedy.best == machine


def test_multiple_machines():
    epsilon_greedy = EpsilonGreedy()
    
    epsilon_greedy.simulate_one_step(0, 1.0)
    epsilon_greedy.simulate_one_step(0, 0.5)
    
    epsilon_greedy.simulate_one_step(1, 0.8)

    assert epsilon_greedy.machine_turns[0] == 2
    assert epsilon_greedy.machine_turns[1] == 1
    assert np.isclose(epsilon_greedy.experimental_means[0], 0.75)
    assert np.isclose(epsilon_greedy.experimental_means[1], 0.8)
    
    assert epsilon_greedy.best == 1  


def test_exploration_exploitation():
    epsilon_greedy = EpsilonGreedy(epsilon=0.2)
    
    np.random.seed(42)
    rewards_machine_0 = [1, 0.8, 0.9]  
    rewards_machine_1 = [0.5, 0.7, 0.6]  
    
    for reward in rewards_machine_0:
        epsilon_greedy.simulate_one_step(0, reward)
    
    for reward in rewards_machine_1:
        epsilon_greedy.simulate_one_step(1, reward)
    
    assert epsilon_greedy.best == 0  
    for _ in range(100):
        chosen_machine = epsilon_greedy.simulate_one_step(1, np.random.choice([0, 1]))
        



