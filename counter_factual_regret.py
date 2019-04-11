#http://modelai.gettysburg.edu/2013/cfr/cfr.pdf

import numpy as np
import random

'''
Algorithm
For each player, initialize all cumulative regrets to 0

For n number of iterations:
    1) Compute a regret-matching strategy profile
        -If all regrets for a player are non-positive, use a uniform random strategy
    2) Add the strategy profile to the strategy profile sum
    3) Select each player action profile according to the strategy profile
    4) Compute player regrets
    5) Add play regrets to player cumulative regrets
'''

class RPSTrainer():
    ROCK = 0
    PAPER = 1
    SCISSORS = 2
    NUM_ACTIONS = 3

    regret_sum = [0.0] * NUM_ACTIONS
    strategy_sum = [0.0] * NUM_ACTIONS
    opp_strategy = [0.4, 0.3, 0.3]

    def get_strategy(self):
        normalizing_sum = 0.0
        strategy = [0.0] * self.NUM_ACTIONS

        for i in range(0, self.NUM_ACTIONS):
            strategy[i] = self.regret_sum[i] if self.regret_sum[i] > 0 else 0.0
            normalizing_sum += strategy[i]

        for i in range(0, self.NUM_ACTIONS):
            if normalizing_sum > 0:
                strategy[i] /= normalizing_sum
            else:
                strategy[i] = 1.0 / self.NUM_ACTIONS
            self.strategy_sum[i] += strategy[i]

        return strategy

    def get_action(self, strategy):
        r = np.random.uniform()
        a = 0
        cumulative_prob = 0
        while a < (self.NUM_ACTIONS - 1):
            cumulative_prob += strategy[a]
            if r < cumulative_prob:
                break
            a += 1
        return a

    def get_regret_matched_mixed_strategy_actions(self):
        strategy = self.get_strategy()
        my_action = self.get_action(strategy)
        other_action = self.get_action(self.opp_strategy)
        return my_action, other_action

    def compute_action_utilities(self, other_action, action_utility):
        """
        [Rock, Paper, Scissors]

        other_action = 0
            action_utility = [0, 1, -1]
        other_action = 1
            action_utility = [-1, 0, 1]
        other_action = 2
            action_utility = [1, -1, 0]

        :param other_action:
        :param action_utility:
        :return:
        """
        action_utility[other_action] = 0

        # Compute Winner Utility
        i = 0 if other_action == self.NUM_ACTIONS - 1 else other_action + 1
        action_utility[i] = 1

        # Compute Loser Utility
        j = self.NUM_ACTIONS - 1 if other_action == 0 else other_action - 1
        action_utility[j] = -1

        return action_utility

    def accumulate_action_regrets(self, my_action, action_utility):
        for i in range(0, self.NUM_ACTIONS):
            self.regret_sum[i] += action_utility[i] - action_utility[my_action]

    def train(self, iterations=1000000):
        action_utility = [0] * self.NUM_ACTIONS
        for i in range(0, iterations):
            my_action, other_action = self.get_regret_matched_mixed_strategy_actions()
            action_utility = self.compute_action_utilities(other_action, action_utility)
            self.accumulate_action_regrets(my_action, action_utility)

    def get_average_strategy(self):
        avg_strategy = [0] * self.NUM_ACTIONS
        normalizing_sum = 0.0
        for i in range(0, self.NUM_ACTIONS):
            normalizing_sum += self.strategy_sum[i]

        for i in range(0, self.NUM_ACTIONS):
            if normalizing_sum > 0:
                avg_strategy[i] = self.strategy_sum[i] / normalizing_sum
            else:
                avg_strategy[i] = 1.0 / self.NUM_ACTIONS

        return avg_strategy


def main():
    trainer = RPSTrainer()
    trainer.train(1000000)
    print(trainer.get_average_strategy())

main()