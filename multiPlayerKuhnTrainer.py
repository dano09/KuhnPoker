import random
from random import shuffle
from collections import defaultdict
from pprint import pprint

# Aggressive - B
B = 'BET'
C = 'CALL'

# Non-Aggressive - P
K = 'CHECK'
F = 'FOLD'


class Node:
    # Information set node class definition

    NUM_ACTIONS = 2

    def __init__(self, info_set=''):
        # Kuhn Node Definitions
        self.info_set = info_set
        self.regret_sum = [0] * self.NUM_ACTIONS
        self.strategy_sum = [0] * self.NUM_ACTIONS

    def get_strategy(self, realization_weight):
        """
        Get current information set mixed strategy through regret matching
        :param realization_weight:
        :return:
        """
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

            self.strategy_sum[i] += realization_weight * strategy[i]

        return strategy

    def get_average_strategy(self):
        """
        Get average information set mixed strategy across all training iterations
        :return:
        """
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

    def to_string(self):
        # Get info set string representation
        avg_strat = self.get_average_strategy()
        print('info_set is: {0} and average strategy for BET: {1.4f} PASS: {2:.4f}'.format(self.info_set, avg_strat[0], avg_strat[1]))


class KuhnTrainer:
    # Kuhn Poker Definitions
    PASS = 0
    BEST = 1
    NUM_ACTIONS = 2
    node_map = defaultdict(Node)

    @staticmethod
    def _is_terminal_state(plays, history):
        """
        Determine if our current state is terminal
        :param plays:  int - what state we are in (length of history)
        :param history: str - sequence of plays that describe game state
        :return: Bool
        """
        if plays < 3:
            return False

        if plays == 3 and history == 'ppp' or history[0] == 'b':
            # Only terminal on round 3 if everyone passes or if player 1 bets
            return True

        elif plays == 4 and history[:2] == 'pb':
            # Player 2 bets after player 1 passes. At most 4 rounds can happen
            return True

        elif plays == 5:
            # At most 5 rounds can happen, so terminal
            return True

        return False

    @staticmethod
    def _round_3_payoff(history, cards):
        """ Terminal Utility for Player Z """
        if history.count('b') == 0:
            # PPP - Showdown
            return 2 if cards[2] > max(cards[0], cards[1]) else -1
        elif history[2] != 'b':
            # Player Z does not bet
            return -1
        elif history.count('b') == 2:
            # BPB - Showdown between Z and X
            return 3 if cards[2] > cards[0] else -2
        else:
            # BBB - Showdown
            return 4 if cards[2] > max(cards[0], cards[1]) else -2

    @staticmethod
    def _round_4_payoff(history, cards):
        """ Terminal Utility for Player X """
        if history[3] == 'p':
            # Player X passes both rounds
            return -1
        elif history.count('b') == 2:
            # PBPB - Showdown between X and Y
            return 3 if cards[0] > cards[1] else -2
        else:
            # PBBB - Showdown
            return 4 if cards[0] > max(cards[1], cards[2]) else -2

    @staticmethod
    def _round_5_payoff(history, cards):
        """ Terminal Utility for Player Y """
        if history[4] == 'p':
            # Player Y passes both rounds
            return -1
        elif history.count('b') == 2:
            # PPBPB - Showdown between Y and Z
            return 3 if cards[1] > cards[2] else -2
        else:
            # PPBBB - Showdown
            return 4 if cards[1] > max(cards[0], cards[2]) else -2

    def calculate_terminal_payoff(self, plays, history, cards):
        """
        Terminal payoff depends on the current player. Each round has a different play payoff
        Round 3 - Payoff for Player Z (3rd player)
        Round 4 - Payoff for Player X (1st player)
        Round 5 - Payoff for Player Y (2nd player)

        :param plays:       int - round we are on
        :param history:     str - sequence of actions made by the players
        :param cards: list[str] - player cards to determine utility
        :return:            int - utility
        """
        if plays == 3:
            return self._round_3_payoff(history, cards)
        elif plays == 4:
            return self._round_4_payoff(history, cards)
        elif plays == 5:
            return self._round_5_payoff(history, cards)

    def cfr(self, cards, history, reach_probabilities):
        """
        Counterfactual regret minimization for Kuhn Poker
        :param cards:   list[int] -
        :param history: string    -
        :param reach_probabilities: list [ float ] - probability of action for players 1, 2, 3

        :return:
        """
        plays = len(history)
        current_player = plays % 3
        rp0 = reach_probabilities[0]
        rp1 = reach_probabilities[1]
        rp2 = reach_probabilities[2]
        util = [0.0] * self.NUM_ACTIONS
        node_util = 0.0

        if self._is_terminal_state(plays, history):
            # Terminal Utility is pre-defined for each player based on the current state
            return self.calculate_terminal_payoff(plays, history, cards)

        # Get information set node or create it if has not been visited yet
        info_set = str(cards[current_player]) + history
        node = self.node_map[info_set]

        # Get updated strategy based on cumulative regret
        strategy = node.get_strategy(reach_probabilities[current_player])

        for a in range(0, self.NUM_ACTIONS):
            # For each action, recursively call cfr with additional history and probability
            next_history = history + ('p' if a == 0 else 'b')
            if current_player == 0:
                util[a] = self.cfr(cards, next_history, [rp0 * strategy[a], rp1, rp2])
            elif current_player == 1:
                util[a] = self.cfr(cards, next_history, [rp0, rp1 * strategy[a], rp2])
            else:  # Player 2
                util[a] = self.cfr(cards, next_history, [rp0, rp1, rp2 * strategy[a]])

            node_util += strategy[a] * util[a]

        # For each action, compute and accumulate counterfactual regret
        for i in range(0, self.NUM_ACTIONS):
            regret = util[i] - node_util
            # CFR is multiplied by reach probability (from previous player) of getting to the current state
            if current_player == 0:
                node.regret_sum[i] += rp2 * regret
            elif current_player == 1:
                node.regret_sum[i] += rp0 * regret
            else:
                node.regret_sum[i] += rp1 * regret

        # Non-Terminal Utility should be negated since the perspective changes between players
        return -node_util

    @staticmethod
    def _return_player_strats(strategy_profile):
        info_sets = list(strategy_profile.keys())
        # Player 1, 2, and 3's betting strategies
        x = {i: strategy_profile[i] for i in info_sets if len(i) == 1 or len(i) == 4}
        y = {i: strategy_profile[i] for i in info_sets if len(i) == 2 or len(i) == 5}
        z = {i: strategy_profile[i] for i in info_sets if len(i) == 3}
        return x, y, z

    def train(self, iterations):
        """
        Train Kuhn Poker
        :param iterations:
        :return:
        """
        strategy_profile = {}
        # index 0 will be card for player 1 (X)
        # index 1 will be card for player 2 (Y)
        # index 2 will be card for player 3 (Z)
        cards = [3, 4, 1, 2]
        util = 0
        for _ in range(iterations):
            shuffle(cards)
            util += self.cfr(cards, '', [1, 1, 1])

        print('Average game value: {}'.format(util / iterations))
        for info_set in sorted(self.node_map):
            node = self.node_map[info_set]
            avg_strat = node.get_average_strategy()
            strategy_profile[info_set] = [round(avg_strat[0], 4), round(avg_strat[1], 4)]
            #print('info_set is: {0} and average strategy for Pass: {1:.4f} Bet: {2:.4f}'.format(node.info_set, avg_strat[0], avg_strat[1]))

        return self._return_player_strats(strategy_profile)
1

def main():
    iterations = 10000
    strat_profiles = KuhnTrainer().train(iterations)
    pprint(strat_profiles)
    print('here')




main()





#
# TERMINAL_STATES = ['PPP', 'BPP', 'BPB', 'BBP', 'BBB', 'PBPP', 'PBPB', 'PBBP', 'PBBB' 'PPBPP', 'PPBPB', 'PPBBP', 'PPBBB']
#
#
# node_map = {'1': Node('1'), '1B': Node('1B'), '1P': Node('1P'),
#             '2': Node('2'), '2B': Node('2B'), '2P': Node('2P'),
#             '3': Node('3'), '3B': Node('3B'), '3P': Node('3P'),
#             '4': Node('4'), '4B': Node('4B'), '4P': Node('4P'),
#
#             '1BB': Node('1BB'), '1BP': Node('1BP'), '1PP': Node('1PP'), '1PB': Node('1PB'),
#             '2BB': Node('2BB'), '2BP': Node('2BP'), '2PP': Node('2PP'), '2PB': Node('2PB'),
#             '3BB': Node('3BB'), '3BP': Node('3BP'), '3PP': Node('3PP'), '3PB': Node('3PB'),
#             '4BB': Node('4BB'), '4BP': Node('4BP'), '4PP': Node('4PP'), '4PB': Node('4PB'),
#
#             '1PBP': Node('1PBP'), '1PBB': Node('1PBB'), '1PPB': Node('1PPB'),
#             '2PBP': Node('2PBP'), '2PBB': Node('2PBB'), '2PPB': Node('2PPB'),
#             '3PBP': Node('3PBP'), '3PBB': Node('3PBB'), '3PPB': Node('3PPB'),
#             '4PBP': Node('4PBP'), '4PBB': Node('4PBB'), '4PPB': Node('4PPB'),
#
#             '1PPBP': Node('1PPBP'), '1PPBB': Node('1PPBB'),
#             '2PPBP': Node('2PPBP'), '2PPBB': Node('2PPBB'),
#             '3PPBP': Node('3PPBP'), '3PPBB': Node('3PPBB'),
#             '4PPBP': Node('4PPBP'), '4PPBB': Node('4PPBB')
#             }
#
#
#
#
# node_map = {'1': Node('1'), '1B': Node('1B'), '1K': Node('1K'),
#             '2': Node('2'), '2B': Node('2B'), '2K': Node('2K'),
#             '3': Node('3'), '3B': Node('3B'), '3K': Node('3K'),
#             '4': Node('4'), '4B': Node('4B'), '4K': Node('4K'),
#
#             '1BC': Node('1BC'), '1BF': Node('1BF'), '1KK': Node('1KK'), '1KB': Node('1KB'),
#             '2BC': Node('2BC'), '2BF': Node('2BF'), '2KK': Node('2KK'), '2KB': Node('2KB'),
#             '3BC': Node('3BC'), '3BF': Node('3BF'), '3KK': Node('3KK'), '3KB': Node('3KB'),
#             '4BC': Node('4BC'), '4BF': Node('4BF'), '4KK': Node('4KK'), '4KB': Node('4KB'),
#
#             '1KBF': Node('1KBF'), '1KBC': Node('1KBC'), '1KKB': Node('1KKB'),
#             '2KBF': Node('2KBF'), '2KBC': Node('2KBC'), '2KKB': Node('2KKB'),
#             '3KBF': Node('3KBF'), '3KBC': Node('3KBC'), '3KKB': Node('3KKB'),
#             '4KBF': Node('4KBF'), '4KBC': Node('4KBC'), '4KKB': Node('4KKB'),
#
#             '1KKBF': Node('1KKBF'), '1KKBC': Node('1KKBC'),
#             '2KKBF': Node('2KKBF'), '2KKBC': Node('2KKBC'),
#             '3KKBF': Node('3KKBF'), '3KKBC': Node('3KKBC'),
#             '4KKBF': Node('4KKBF'), '4KKBC': Node('4KKBC')
#             }
