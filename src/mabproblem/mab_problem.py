import numpy as np

class EpsilonGreedy:
    def __init__(self, epsilon: float = 0.2) -> None:
        self.epsilon = epsilon
        self.experimental_means: dict = {}
        self.machine_turns: dict = {}
        self.best = None
        
    def _calculate_running_mean(self, machine: int, new_reward: float) -> None:
        self.experimental_means[machine] = ((self.machine_turns[machine]-1)/self.machine_turns[machine]) * self.experimental_means.get(machine, 0) \
            + (1/self.machine_turns[machine])*new_reward

    def simulate_one_step(self, machine: int, reward: float) -> None:
        if machine not in self.machine_turns:
            self.machine_turns[machine] = 1
        self._calculate_running_mean(machine, reward)
        self.best = np.argmax(list(self.experimental_means.values())) # type: ignore


        
            

