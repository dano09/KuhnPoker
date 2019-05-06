from random import shuffle
import numpy as np
import random
from multiplayer import kuhnHelper


class GameInfoSet:
    def __init__(self, info_set, strategy):
        self.info_set = info_set
        self.strategy = strategy
        self.plays = 0
        self.utility_sum = 0

    def get_action(self):
        """
        Determines which action to choose based on the probabilities of the strategy profile.
        searchsorted takes a random value and looks where it would place it in the cumulative array

        Example:
            strategy = [0.3, 0.7]
            np.cumsum(strategy) = [0.3, 1]
            np.searchsorted(np.cumsum(strategy), 0.39) = 1
            np.searchsorted(np.cumsum(strategy), 0.29) = 0
        """
        action = np.searchsorted(np.cumsum(self.strategy), random.random())
        return 'p' if action == 0 else 'b'

    def update(self, utility):
        self.plays += 1
        self.utility_sum += utility


class KuhnPoker:

    def __init__(self, node_map):
        self.node_map = node_map

    def _update_node_utilities(self, info_sets, utility):
        for info_set in info_sets:
            player = kuhnHelper.determine_player_from_infoset(info_set)
            node = self.node_map[info_set]
            node.update(utility[player])

    def _play_round(self, cards, info_sets, history=''):
        """
        Recursively play rounds until a terminal state has beeen reached. Send all info nodes visited up the stack
        and update utilities accordingly
        :param cards:                     list [str] - cards played this round
        :param info_sets: list [{str : GameInfoSet}] - all information sets that have been visited
        :param history:                          str - previous actions

        """
        plays = len(history)
        current_player = plays % 3

        if kuhnHelper.is_terminal_state(plays, history):
            utility = kuhnHelper.calculate_terminal_payoff(history, cards)
            return self._update_node_utilities(info_sets, utility)

        info_set = str(cards[current_player]) + history
        info_sets.append(info_set)
        node = self.node_map[info_set]
        action = node.get_action()

        self._play_round(cards, info_sets, history + action)

    @staticmethod
    def _compute_player_utility(player_positions):
        total_plays = total_utility = 0
        for position in player_positions:
            total_plays += player_positions[position].plays
            total_utility += player_positions[position].utility_sum

        avg_utility = total_utility / total_plays
        print('total plays is: {} and total utility is: {} and average utility for player is: {}'.format(total_plays, total_utility, avg_utility))
        return avg_utility

    def _compute_average_utilities(self):
        p1, p2, p3 = kuhnHelper.get_positions_from_strategy_profile(self.node_map)
        p1_utility = self._compute_player_utility(p1)
        p2_utility = self._compute_player_utility(p2)
        p3_utility = self._compute_player_utility(p3)
        return p1_utility, p2_utility, p3_utility

    def play_poker(self, rounds=100):
        cards = [3, 4, 1, 2]
        for _ in range(rounds):
            shuffle(cards)
            self._play_round(cards, [], '')

        return self.node_map