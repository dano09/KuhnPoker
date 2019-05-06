import pickle
import pandas as pd

# Some helper functions for multiplayer Kuhn Poker
# Player 1 (or X)


X = 0
# Player 2 (or Y)
Y = 1
# Player 3 (or Z)
Z = 2

#MODEL_DIR = 'C:/Users/Justin/PycharmProjects/KuhnPoker/saved_strats/'
MODEL_DIR = '../saved_strats/'
TRAINED_MODEL_FILES = ['strat_profile.p', 'p1_br.p', 'p2_br.p', 'p3_br.p']
PLAYER_RESULT_FILES = ['p1_results.p', 'p2_results.p', 'p3_results.p', 'cfr_br_df.p']


def save_trained_models(models):
    for obj, name in zip(models, TRAINED_MODEL_FILES):
        pickle.dump(obj, open(MODEL_DIR + name, 'wb'))


def load_trained_models():
    return [pickle.load(open(MODEL_DIR + obj, 'rb')) for obj in TRAINED_MODEL_FILES]


def save_results(results):
    for obj, name in zip(results, PLAYER_RESULT_FILES):
        pickle.dump(obj, open(MODEL_DIR + name, 'wb'))


def load_results():
    return [pickle.load(open(MODEL_DIR + obj, 'rb')) for obj in PLAYER_RESULT_FILES]


def df_builder(results):
    """
    Cast over to DataFrame to analyze utility
    :param results: dict { 'infoset': GameNode() }
    :return: pd.DataFrame
    """
    res = [{'infoset': r, 'plays': results[r].plays, 'utility':results[r].utility_sum} for r in results]
    return pd.DataFrame(res).set_index('infoset')


def strat_df_builder(strat):
    """
    For Easy viewing of complete strategy profile
    :param strat:
    :return:
    """
    info_sets = strat.keys()
    strat_df = pd.DataFrame(index=[k[1:] for k in info_sets if '1' in k], columns=['1', '2', '3', '4'])
    strat_df.rename(index={'': '-'}, inplace=True)
    for k, v in strat.items():
        if len(k) == 1:
            strat_df.loc['-', k] = v
        else:
            strat_df.loc[k[1:], k[0]] = v
    return strat_df


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
        if cards[X] > max(cards[Y], cards[Z]):
            util[X] = 2
        elif cards[Y] > max(cards[X], cards[Z]):
            util[Y] = 2
        else:
            util[Z] = 2

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
        if cards[X] > max(cards[Y], cards[Z]):
            util = [4, -2, -2]
        elif cards[Y] > max(cards[X], cards[Z]):
            util = [-2, 4, -2]
        else:
            util = [-2, -2, 4]

    return util


def determine_player_from_infoset(info_set):
    if len(info_set) == 1 or len(info_set) == 4:
        return X
    elif len(info_set) == 2 or len(info_set) == 5:
        return Y
    elif len(info_set) == 3:
        return Z
    else:
        raise Exception('Invalid InfoSet')


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


def view_results():
    s = load_trained_models()
    base_strat = s[0]
    br1_strat = s[1]
    br2_strat = s[2]
    br3_strat = s[3]

    sdf = strat_df_builder(base_strat)
    #sdf1 = strat_df_builder(br1_strat)
    #sdf2 = strat_df_builder(br2_strat)
    #sdf3 = strat_df_builder(br3_strat)

    results = load_results()
    #results[0]  #Player 1 Strat
    #results[1]  #Player 2 Strat
    #results[2]  #Player 3 Strat
    results[3]  # CFR
    #print(results)
    # from multiplayer.main import calculate_nash_equilibrium
    #cf_br, e = calculate_nash_equilibrium(results)