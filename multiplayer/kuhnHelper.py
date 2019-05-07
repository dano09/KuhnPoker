import pickle
import pandas as pd
import matplotlib as mlb
import matplotlib.pyplot as plt
import os
mlb.style.use('seaborn')

PLAYER1 = 0
PLAYER2 = 1
PLAYER3 = 2
GRAPHS_DIR = '/graphs/'
STRATS_DIR = '/trained_strategies/'
RESULTS_DIR = '/results/'
TRAINED_MODEL_FILES = ['cfr_strategy.p', 'p1_br_strategy.p', 'p2_br_strategy.p', 'p3_br_strategy.p']
PLAYER_RESULT_FILES = ['p1_results.p', 'p2_results.p', 'p3_results.p', 'cfr_br_df.p']


def save_results(results, file_names, base_dir, file_dir):
    os.makedirs(base_dir+'/'+file_dir, exist_ok=True)

    for obj, name in zip(results, file_names):
        pickle.dump(obj, open(base_dir + file_dir + name, 'wb'))


def load_trained_models(directory):
    """
    If we want to load from REPL
    BASE_DIR = 'C:/Users/Justin/PycharmProjects/KuhnPoker/multiplayer'
    TS = '2019_05_07_16_10'
    res = load_trained_models(BASE_DIR + TS)

    :param directory: str
    :return:
    """
    return [pickle.load(open(directory + STRATS_DIR + obj, 'rb')) for obj in TRAINED_MODEL_FILES]


def load_results(directory):
    """
    Same applies as load_trained_models
    :param directory: str
    :return:
    """
    return [pickle.load(open(directory + RESULTS_DIR + obj, 'rb')) for obj in PLAYER_RESULT_FILES]


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


def determine_player_from_infoset(info_set):
    if len(info_set) == 1 or len(info_set) == 4:
        return PLAYER1
    elif len(info_set) == 2 or len(info_set) == 5:
        return PLAYER2
    elif len(info_set) == 3:
        return PLAYER3
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


def plot_training(node_map, base_dir):
    histories = ['', 'p', 'b', 'pp', 'pb', 'bp', 'bb', 'ppb', 'pbp', 'pbb', 'ppbp', 'ppbb']
    for h in histories:
        plot_strategy(node_map, h, base_dir)


def plot_strategy(node_map, history='', base_dir=None):
    cards = ['1', '2', '3', '4']
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(8, 4))

    for card in cards:
        infoset = card+history
        dfs = pd.DataFrame([{'Bet': i[1], 'Pass': i[0]} for i in node_map[infoset].strategy_list])
        dfr = pd.DataFrame([{'Bet': i[1], 'Pass': i[0]} for i in node_map[infoset].regret_list])

        dfs.plot(ax=axes[0, cards.index(card)], legend=False, title=infoset)
        dfr.plot(ax=axes[1, cards.index(card)], legend=False)

    for ax, row in zip(axes[:, 0], ['Strategy', 'Regret']):
        ax.set_ylabel(row, rotation=90)

    axes[0, 0].legend()
    plt.tight_layout()

    # Create directory and save figure
    os.makedirs(base_dir + '/' + GRAPHS_DIR, exist_ok=True)
    graph_name = '_training_strat.png'
    graph_name = graph_name if history == '' else history + graph_name
    plt.savefig(base_dir + '/' + GRAPHS_DIR + graph_name)


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