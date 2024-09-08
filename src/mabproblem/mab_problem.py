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
        """ Given a machine and its rewards, simulate a step, should be used in conjuction
         with select machine on external distribution """
        if machine not in self.machine_turns:
            self.machine_turns[machine] = 1
        else:
            self.machine_turns[machine] += 1
        self._calculate_running_mean(machine, reward)
        print(self.experimental_means)
        self.best = np.argmax(list(self.experimental_means.values())) # type: ignore

    def select_machine(self) -> int:
        """ Output a machine to pull """

        nr_of_machines = len(self.machine_turns.keys())
        assert nr_of_machines != 0, "Machines are needed in order to select a machine"
        probs = [self.epsilon / float(nr_of_machines-1) for i in range(nr_of_machines)]
        probs[self.best] = 1.0 - self.epsilon # type: ignore
        return np.random.choice(list(self.machine_turns.keys()), p=probs) if nr_of_machines > 1 else list(self.machine_turns.keys())[0]

        
            

