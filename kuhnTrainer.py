import random

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
    node_map = {'1': Node('1'), '1b': Node('1b'), '1p': Node('1p'), '1pb': Node('1pb'),
                '2': Node('2'), '2b': Node('2b'), '2p': Node('2p'), '2pb': Node('2pb'),
                '3': Node('3'), '3b': Node('3b'), '3p': Node('3p'), '3pb': Node('3pb')}

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
        player = plays % 2
        opponent = 1 - player

        # Terminal State Payoff
        if plays > 1:
            # Both players have made an action
            # Two conditions for a terminal state
            terminal_pass = history[plays - 1] == 'p'
            double_bet = history[plays-2: plays] == 'bb'
            is_player_card_higher = cards[player] > cards[opponent]
            if terminal_pass:
                if history == 'pp':  # Double terminal pass
                    return 1 if is_player_card_higher else -1
                else:  # Player betting wins the hand (bp)
                    return 1
            elif double_bet:
                return 2 if is_player_card_higher else -2

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

    def shuffle_cards(self, cards):
        for c1 in range(len(cards) - 1, 0, -1):
            c2 = random.randint(0, c1)
            tmp = cards[c1]
            cards[c1] = cards[c2]
            cards[c2] = tmp
        return cards

    def train(self, iterations):
        """
        Train Kuhn Poker
        :param iterations:
        :return:
        """

        # index 0 will be card for player 1
        # index 1 will be card for player 2
        cards = [1, 2, 3]
        util = 0
        for _ in range(iterations):
            cards = self.shuffle_cards(cards)
            util += self.cfr(cards, '', 1, 1)

        print('Average game value: {}'.format(util / iterations))
        for info_set in sorted(self.node_map):
            node = self.node_map[info_set]
            avg_strat = node.get_average_strategy()
            print('info_set is: {0} and average strategy for Pass: {1:.4f} Bet: {2:.4f}'.format(node.info_set, avg_strat[0], avg_strat[1]))


def main():
    iterations = 1000000
    KuhnTrainer().train(iterations)

main()

'''
Average game value: -0.059024760010225504
info_set is: 1   and average strategy for Pass: 0.8657 Bet: 0.1343
info_set is: 1b  and average strategy for Pass: 1.0000 Bet: 0.0000
info_set is: 1p  and average strategy for Pass: 0.6663 Bet: 0.3337
info_set is: 1pb and average strategy for Pass: 1.0000 Bet: 0.0000
info_set is: 2   and average strategy for Pass: 1.0000 Bet: 0.0000
info_set is: 2b  and average strategy for Pass: 0.6649 Bet: 0.3351
info_set is: 2p  and average strategy for Pass: 1.0000 Bet: 0.0000
info_set is: 2pb and average strategy for Pass: 0.5278 Bet: 0.4722
info_set is: 3   and average strategy for Pass: 0.5881 Bet: 0.4119
info_set is: 3b  and average strategy for Pass: 0.0000 Bet: 1.0000
info_set is: 3p  and average strategy for Pass: 0.0000 Bet: 1.0000
info_set is: 3pb and average strategy for Pass: 0.0000 Bet: 1.0000
'''