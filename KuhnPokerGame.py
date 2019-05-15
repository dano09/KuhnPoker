
from random import shuffle
import numpy as np
import random
import pickle

PLAYER1 = 0
PLAYER2 = 1
PLAYER3 = 2
GRAPHS_DIR = '/graphs/'
STRATS_DIR = '/trained_strategies/'
RESULTS_DIR = '/results/'
TRAINED_MODEL_FILES = ['cfr_strategy.p']
PLAYER_RESULT_FILES = ['p1_results.p', 'p2_results.p', 'p3_results.p', 'cfr_br_df.p']
CARDS = ['1', '2', '3', '4']
HISTORIES = ['', 'p', 'b', 'pp', 'pb', 'bp', 'bb', 'ppb', 'pbp', 'pbb', 'ppbp', 'ppbb']

def is_terminal_state(plays, history):
    """
    Determine if our current state is terminal
    :param plays:   int - what state we are in (length of history)
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

def calculate_terminal_payoff(history, cards):
    """
    Terminal payoff depends on the current player. Each round has a different play payoff
    :param history:     str - sequence of actions made by the players
    :param cards: list[str] - player cards to determine utility
    :return:      list[int] - utility
    """
    util = [-1, -1, -1]

    if history == 'ppp':
        if cards[PLAYER1] > max(cards[PLAYER2], cards[PLAYER3]):
            util[PLAYER1] = 2
        elif cards[PLAYER2] > max(cards[PLAYER1], cards[PLAYER3]):
            util[PLAYER2] = 2
        else:
            util[PLAYER3] = 2

    elif history.count('b') == 1:
        winner = history.index('b') % 3
        util[winner] = 2

    elif history.count('b') == 2:
        first_bet = history.index('b') % 3
        sec_bet = history.rindex('b') % 3
        if cards[first_bet] > cards[sec_bet]:
            util[first_bet] = 3
            util[sec_bet] = -2
        else:
            util[first_bet] = -2
            util[sec_bet] = 3

    elif history.count('b') == 3:
        if cards[PLAYER1] > max(cards[PLAYER2], cards[PLAYER3]):
            util = [4, -2, -2]
        elif cards[PLAYER2] > max(cards[PLAYER1], cards[PLAYER3]):
            util = [-2, 4, -2]
        else:
            util = [-2, -2, 4]

    return util


def get_positions_from_strategy_profile(strategy_profile):
    """
    The Strategy Profile (info sets) contains positions for every player
    This helper function separates each players info sets
    :param strategy_profile:
    :return:
    """
    p1 = {i: strategy_profile[i] for i in strategy_profile if len(i) == 1 or len(i) == 4}
    p2 = {i: strategy_profile[i] for i in strategy_profile if len(i) == 2 or len(i) == 5}
    p3 = {i: strategy_profile[i] for i in strategy_profile if len(i) == 3}
    return p1, p2, p3

def load_trained_models():
    """
    If we want to load from REPL
    BASE_DIR = 'C:/Users/Justin/PycharmProjects/KuhnPoker/multiplayer'
    TS = '2019_05_07_16_10'
    res = load_trained_models(BASE_DIR + TS)

    :param directory: str
    :return:
    """
    return pickle.load(open('cfr_strategy.p', 'rb'))


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


class KuhnPoker:

    def __init__(self, node_map, human_player):
        self.node_map = node_map
        self.human_player = int(human_player)

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

        if is_terminal_state(plays, history):
            utility = calculate_terminal_payoff(history, cards)
            print('Utility is: {}'.format(utility))
            human_utility = utility[self.human_player]
            print('You lose: {}'.format(human_utility)) if human_utility < 0 else print('You win: {}'.format(human_utility))
            return

        if current_player == self.human_player:
            if history == '':
                action = input('Press B to bet/call or P to check/fold: '.format(history)).lower()
            else:
                action = input('Current history is: {}  -- Press B to bet/call or P to check/fold: '.format(history)).lower()
        else:
            info_set = str(cards[current_player]) + history
            info_sets.append(info_set)
            node = self.node_map[info_set]
            action = node.get_action()

        print('Player {} has passed!'.format(current_player)) if action == 'p' else print('Player {} has betted!'.format(current_player))
        self._play_round(cards, info_sets, history + action)

    def play_poker(self, rounds=100):
        cards = [3, 4, 1, 2]
        for _ in range(rounds):
            shuffle(cards)
            print('\nStarting round. Your card is: {}'.format(cards[self.human_player]))
            self._play_round(cards, [], '')

        return self.node_map



def play():

    starting_balance = 10
    print('Welcome to Kuhn Poker!')
    player = input('Type either 1, 2, or 3 to be the corresponding player: ')
    player = int(player) - 1
    cfr_strat = load_trained_models()
    base_strategy = {i_s: GameInfoSet(info_set=i_s, strategy=cfr_strat[i_s]) for i_s in cfr_strat}
    game = KuhnPoker(node_map=base_strategy, human_player=player)
    game.play_poker(rounds=3)

play()