import random
from random import shuffle

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

    @staticmethod
    def _is_terminal_state(plays, history):
        """
        Determine if our current state is terminal
        :param plays:  int - what state we are in (length of history)
        :param history: str - sequence of plays that describe game state
        :return: Bool
        """

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
    def calculate_terminal_payoff(plays, history, cards):
        if plays == 3:
            # No one bets - showdown
            if history == 'ppp':
                return 2 if cards[0] > max(cards[1], cards[2]) else -1
            elif history.count('b') == 1:
                return 2
            elif history.count('b') == 2:
                if history[1] == 'b':
                    # Showdown with player 2
                    return 3 if cards[0] > cards[1] else -2
                else:
                    # Showdown with player 3
                    return 3 if cards[0] > cards[2] else -2
            else:
                return 4 if cards[0] > max(cards[1], cards[2]) else -2

        # If more than 3 cards in history, player 1 must of passed followed by
        # player 2 or player 3 betting, requiring another round
        else:
            # Player 1 doesn't bet
            if history[3] == 'p':
                return -1
            # Player 1 bets 1 other player
            elif history.count('b') == 2:
                if history[1] == 'b':  #PBPB
                    return 3 if cards[0] > cards[1] else -2
                else:  #PPBBP
                    return 3 if cards[0] > cards[2] else -2

            # Player 1 bets both players
            else:
                return 4 if cards[0] > max(cards[1], cards[2]) else -2

    def cfr(self, cards, history, player0, player1):
        """
        Counterfactual regret minimization for Kuhn Poker
        :param cards:   list[int] -
        :param history: string    -
        :param player0: float     - probability of player 0 actions
        :param player1: float     - probability of player 1 actions
        :return:
        """
        plays = len(history)
        player = plays % 3
        #opponent = 1 - player

        if self._is_terminal_state(plays, history):
            return self.calculate_terminal_payoff(plays, history, cards, player, opponents)

        info_set = str(cards[player]) + history

        # Get information set node or create it if nonexistant
        if info_set not in self.node_map:
            node = Node(info_set)
            self.node_map[info_set] = node
        else:
            node = self.node_map[info_set]

        # For each action, recursively call cfr with additional history and probability
        cur_player = player0 if player == 0 else player1
        strategy = node.get_strategy(cur_player)
        util = [0.0] * self.NUM_ACTIONS
        node_util = 0.0
        for i in range(0, self.NUM_ACTIONS):
            next_history = history + ('p' if i == 0 else 'b')
            if player == 0:
                util[i] = - self.cfr(cards, next_history, player0*strategy[i], player1)
            else:
                util[i] = - self.cfr(cards, next_history, player0, player1*strategy[i])

            node_util += strategy[i] * util[i]

        # For each action, compute and accumulate counterfactual regret
        for i in range(0, self.NUM_ACTIONS):
            regret = util[i] - node_util
            # This also updates node_mapping
            node.regret_sum[i] += player1 * regret if player == 0 else player0 * regret

        return node_util



    def train(self, iterations):
        """
        Train Kuhn Poker
        :param iterations:
        :return:
        """

        # index 0 will be card for player 1
        # index 1 will be card for player 2
        # index 2 will be card for player 3
        cards = [1, 2, 3, 4]
        util = 0
        for _ in range(iterations):
            shuffle(cards)
            util += self.cfr(cards, '', 1, 1)

        print('Average game value: {}'.format(util / iterations))
        for info_set in sorted(self.node_map):
            node = self.node_map[info_set]
            avg_strat = node.get_average_strategy()
            print('info_set is: {0} and average strategy for Pass: {1:.4f} Bet: {2:.4f}'.format(node.info_set, avg_strat[0], avg_strat[1]))


def main():
    iterations = 1000000
    KuhnTrainer().train(iterations)

#main()


TERMINAL_STATES = ['PPP', 'BPP', 'BPB', 'BBP', 'BBB', 'PBPP', 'PBPB', 'PBBP', 'PBBB' 'PPBPP', 'PPBPB', 'PPBBP', 'PPBBB']


node_map = {'1': Node('1'), '1B': Node('1B'), '1P': Node('1P'),
            '2': Node('2'), '2B': Node('2B'), '2P': Node('2P'),
            '3': Node('3'), '3B': Node('3B'), '3P': Node('3P'),
            '4': Node('4'), '4B': Node('4B'), '4P': Node('4P'),

            '1BB': Node('1BB'), '1BP': Node('1BP'), '1PP': Node('1PP'), '1PB': Node('1PB'),
            '2BB': Node('2BB'), '2BP': Node('2BP'), '2PP': Node('2PP'), '2PB': Node('2PB'),
            '3BB': Node('3BB'), '3BP': Node('3BP'), '3PP': Node('3PP'), '3PB': Node('3PB'),
            '4BB': Node('4BB'), '4BP': Node('4BP'), '4PP': Node('4PP'), '4PB': Node('4PB'),

            '1PBP': Node('1PBP'), '1PBB': Node('1PBB'), '1PPB': Node('1PPB'),
            '2PBP': Node('2PBP'), '2PBB': Node('2PBB'), '2PPB': Node('2PPB'),
            '3PBP': Node('3PBP'), '3PBB': Node('3PBB'), '3PPB': Node('3PPB'),
            '4PBP': Node('4PBP'), '4PBB': Node('4PBB'), '4PPB': Node('4PPB'),

            '1PPBP': Node('1PPBP'), '1PPBB': Node('1PPBB'),
            '2PPBP': Node('2PPBP'), '2PPBB': Node('2PPBB'),
            '3PPBP': Node('3PPBP'), '3PPBB': Node('3PPBB'),
            '4PPBP': Node('4PPBP'), '4PPBB': Node('4PPBB')
            }




node_map = {'1': Node('1'), '1B': Node('1B'), '1K': Node('1K'),
            '2': Node('2'), '2B': Node('2B'), '2K': Node('2K'),
            '3': Node('3'), '3B': Node('3B'), '3K': Node('3K'),
            '4': Node('4'), '4B': Node('4B'), '4K': Node('4K'),

            '1BC': Node('1BC'), '1BF': Node('1BF'), '1KK': Node('1KK'), '1KB': Node('1KB'),
            '2BC': Node('2BC'), '2BF': Node('2BF'), '2KK': Node('2KK'), '2KB': Node('2KB'),
            '3BC': Node('3BC'), '3BF': Node('3BF'), '3KK': Node('3KK'), '3KB': Node('3KB'),
            '4BC': Node('4BC'), '4BF': Node('4BF'), '4KK': Node('4KK'), '4KB': Node('4KB'),

            '1KBF': Node('1KBF'), '1KBC': Node('1KBC'), '1KKB': Node('1KKB'),
            '2KBF': Node('2KBF'), '2KBC': Node('2KBC'), '2KKB': Node('2KKB'),
            '3KBF': Node('3KBF'), '3KBC': Node('3KBC'), '3KKB': Node('3KKB'),
            '4KBF': Node('4KBF'), '4KBC': Node('4KBC'), '4KKB': Node('4KKB'),

            '1KKBF': Node('1KKBF'), '1KKBC': Node('1KKBC'),
            '2KKBF': Node('2KKBF'), '2KKBC': Node('2KKBC'),
            '3KKBF': Node('3KKBF'), '3KKBC': Node('3KKBC'),
            '4KKBF': Node('4KKBF'), '4KKBC': Node('4KKBC')
            }
