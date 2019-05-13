import pickle
import pandas as pd
import matplotlib as mlb
import matplotlib.pyplot as plt
import os

import xlsxwriter

mlb.style.use('seaborn')

PLAYER1 = 0
PLAYER2 = 1
PLAYER3 = 2
GRAPHS_DIR = '/graphs/'
STRATS_DIR = '/trained_strategies/'
RESULTS_DIR = '/results/'
TRAINED_MODEL_FILES = ['cfr_strategy.p', 'p1_br_strategy.p', 'p2_br_strategy.p', 'p3_br_strategy.p']
PLAYER_RESULT_FILES = ['p1_results.p', 'p2_results.p', 'p3_results.p', 'cfr_br_df.p']
CARDS = ['1', '2', '3', '4']
HISTORIES = ['', 'p', 'b', 'pp', 'pb', 'bp', 'bb', 'ppb', 'pbp', 'pbb', 'ppbp', 'ppbb']

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


def _save_training_data(info_set_data, dir):
    for h in HISTORIES:
        for card in CARDS:
            cur_info_set = card + h
            current_infoset = info_set_data[cur_info_set]
            fs = open(dir + '/' + cur_info_set + '_strat.csv', 'w', newline='')
            fs.write(current_infoset.strat_output.getvalue())
            fs.close()
            f2 = open(dir + '/' + cur_info_set + '_regret.csv', 'w', newline='')
            f2.write(current_infoset.regret_output.getvalue())
            f2.close()
            #with open(dir + 'some.csv', 'w', newline='') as f:
                #writer = csv.writer(f)
                #current_infoset.strat_csv_writer.

    #with open(dir + 'some.csv', 'w', newline='') as f:
    #    current_infoset = info_set_data[infoset]
    #    current_infoset.strat_output.seek(0)
    #    current_infoset.regret_output.seek(0)


def plot_training(base_dir):
    for h in HISTORIES:
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(8, 4))
        for c in CARDS:
            infoset = c + h
            s_csv_file = base_dir + '/' + infoset + '_strat.csv'
            r_csv_file = base_dir + '/' + infoset + '_regret.csv'

            dfs = pd.read_csv(s_csv_file, names=['pass', 'bet'])
            dfr = pd.read_csv(r_csv_file, names=['pass', 'bet'])
            dfs.plot(ax=axes[0, CARDS.index(c)], legend=False, title=infoset)
            dfr.plot(ax=axes[1, CARDS.index(c)], legend=False)
            axes[1, CARDS.index(c)].set_ylim(-10, max(dfr['pass'].max(), dfr['bet'].max()))
        for ax, row in zip(axes[:, 0], ['Strategy', 'Regret']):
            ax.set_ylabel(row, rotation=90)

        axes[0, 0].legend()
        plt.tight_layout()

        # Create directory and save figure
        os.makedirs(base_dir + '/' + GRAPHS_DIR, exist_ok=True)
        graph_name = '_training.png'
        graph_name = graph_name if h == '' else h + graph_name
        plt.savefig(base_dir + '/' + GRAPHS_DIR + graph_name)



#def plot_training(node_map, base_dir):
#    for h in HISTORIES:
#        plot_strategy(node_map, h, base_dir)


def plot_strategy(node_map, history='', base_dir=None):
    #cards = ['1', '2', '3', '4']
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(8, 4))

    for card in CARDS:
        infoset = card+history
        current_infoset = node_map[infoset]
        current_infoset.strat_output.seek(0)
        current_infoset.regret_output.seek(0)
        dfs = pd.read_csv(current_infoset.strat_output, names=['pass', 'bet'])
        dfr = pd.read_csv(current_infoset.regret_output, names=['pass', 'bet'])
        #dfs = pd.DataFrame(node_map[infoset].strategy_list)
        #dfr = pd.DataFrame(node_map[infoset].regret_list)
        dfs.plot(ax=axes[0, CARDS.index(card)], legend=False, title=infoset)
        dfr.plot(ax=axes[1, CARDS.index(card)], legend=False)
        axes[1, CARDS.index(card)].set_ylim(-5, 3)
        node_map[infoset].strategy_list = None
        node_map[infoset].regret_list = None

    for ax, row in zip(axes[:, 0], ['Strategy', 'Regret']):
        ax.set_ylabel(row, rotation=90)

    axes[0, 0].legend()
    plt.tight_layout()

    # Create directory and save figure
    os.makedirs(base_dir + '/' + GRAPHS_DIR, exist_ok=True)
    graph_name = '_training_strat.png'
    graph_name = graph_name if history == '' else history + graph_name
    plt.savefig(base_dir + '/' + GRAPHS_DIR + graph_name)

    dfs = dfr = None


def _pivot_data(df, info_set):
    """
    Pass/Bet have been a list in some of the dataframes. Extract that into the index
    TODO: Refactor
    This could probably be cleaned up with pivot/transforms somehow
    :param df:
    :param info_set:
    :return: Dataframes
    """
    cards = ['1', '2', '3', '4']
    res = pd.DataFrame(columns=['bet', 'pass'])
    for c in cards:
        idx = {k: c + k for k in info_set}
        idx['-'] = c
        df_temp = df[[c + 'pass', c + 'bet']]
        cols = {c + 'pass': 'pass', c + 'bet': 'bet'}
        fin_df = df_temp.rename(index=idx, columns=cols)
        res = res.append(fin_df, sort=False)
    return res


def _transform_data(cfr_strategy, br_strategies, player_result):
    """
    TODO: Refactor
    Most of these transformations were done via trial and error until the desired workbook was made
    Some code here may be redundant
    :param cfr_strategy:
    :param br_strategies:
    :param player_result:
    :return:
    """
    cols = ['1pass', '1bet', '2pass', '2bet', '3pass', '3bet', '4pass', '4bet']
    info_sets = ['-', 'pbb', 'pbp', 'ppb', 'b', 'p', 'ppbb', 'ppbp', 'bb', 'bp', 'pb', 'pp']
    info_sets_by_player = [info_sets, info_sets[:4], info_sets[4:8], info_sets[8:]]
    cfr = strat_df_builder(cfr_strategy)

    cfr_bp = pd.DataFrame([[f for e in i for f in e] for i in cfr.values.tolist()], index=cfr.index, columns=cols)

    br1 = strat_df_builder(br_strategies[0])
    br2 = strat_df_builder(br_strategies[1])
    br3 = strat_df_builder(br_strategies[2])

    br1_bp = pd.DataFrame([[f for e in i for f in e] for i in br1.values.tolist()], index=br1.index, columns=cols)
    br2_bp = pd.DataFrame([[f for e in i for f in e] for i in br2.values.tolist()], index=br2.index, columns=cols)
    br3_bp = pd.DataFrame([[f for e in i for f in e] for i in br3.values.tolist()], index=br3.index, columns=cols)

    cfr_br_dfs = [cfr_bp, br1_bp, br2_bp, br3_bp]
    piv_data = [_pivot_data(d, i) for d, i in zip(cfr_br_dfs, info_sets_by_player)]

    player1 = pd.merge(piv_data[0], piv_data[1], left_index=True, right_index=True, suffixes=('_cfr', '_br'))
    player2 = pd.merge(piv_data[0], piv_data[2], left_index=True, right_index=True, suffixes=('_cfr', '_br'))
    player3 = pd.merge(piv_data[0], piv_data[3], left_index=True, right_index=True, suffixes=('_cfr', '_br'))

    p1 = pd.merge(player1, player_result[0], left_index=True, right_index=True)
    p2 = pd.merge(player2, player_result[1], left_index=True, right_index=True)
    p3 = pd.merge(player3, player_result[2], left_index=True, right_index=True)

    p1.index.name = 'Player1'
    p2.index.name = 'Player2'
    p3.index.name = 'Player3'

    br_cols = ['bet_br', 'pass_br', 'plays_br', 'utility_br', 'avg_br']
    cfr_cols = ['bet_cfr', 'pass_cfr', 'plays_cfr', 'utility_cfr', 'avg_cfr']

    res = [p1[cfr_cols], p1[br_cols], p2[cfr_cols],  p2[br_cols], p3[cfr_cols], p3[br_cols]]
    re = [d.round(2) for d in res]
    return re


def make_excel(cfr_strategy, br_strategies, player_results, base_dir):
    """
    Creates Excel Report
    - Very tedious formatting with XlsxWriter
    :param cfr_strategy:
    :param br_strategies:
    :param player_results:
    :param base_dir:
    :return:
    """
    dfs = _transform_data(cfr_strategy, br_strategies, player_results)

    writer = pd.ExcelWriter(base_dir + '/' + 'Kuhn_Poker_Results.xlsx', engine='xlsxwriter')
    center_format = writer.book.add_format()
    center_format.set_align('center_across')

    dfs[0].to_excel(writer, sheet_name='Sheet1', startcol=1, startrow=1)
    dfs[1].to_excel(writer, sheet_name='Sheet1', startcol=1, startrow=19)
    dfs[2].to_excel(writer, sheet_name='Sheet1', startcol=8, startrow=1)
    dfs[3].to_excel(writer, sheet_name='Sheet1', startcol=8, startrow=19)
    dfs[4].to_excel(writer, sheet_name='Sheet1', startcol=15, startrow=1)
    dfs[5].to_excel(writer, sheet_name='Sheet1', startcol=15, startrow=19)
    player_results[3].index.name = 'Results'
    player_results[3].to_excel(writer, sheet_name='Sheet1', startcol=22, startrow=1)
    sheet1 = writer.sheets['Sheet1']
    sheet1.write(6, 22, 'Epsilon')
    sheet1.write(6, 23, player_results[4])

    strat_cells = ['C3:D18', 'J3:K18', 'Q3:R18', 'C21:D36', 'J21:K36', 'Q21:R36']
    indxs_cells = ['B2:B18', 'I2:I18', 'P2:P18', 'W2:W5', 'B20:B36', 'I20:I36', 'P20:P36']
    col_cells = ['B2:G2', 'I2:N2', 'P2:U2', 'W2:Z2', 'B20:G20', 'I20:N20', 'P20:U20', 'W7']
    game_cells = ['E3:G18', 'L3:N18', 'S3:U18', 'X3:Z5', 'E21:G36', 'L21:N36', 'S21:U36', 'X7']
    top_border_cells = ['C37:G37', 'J37:N37', 'Q37:U37', 'Y6:Z6']
    top_bot_border_cells = ['C19:G19', 'J19:N19', 'Q19:U19', 'X6', 'W7:X7']
    left_right_border_cells = ['H3:H18', 'O3:O18', 'H21:H36', 'O21:O36', 'V3:V5', 'V7', 'W7:X7']
    left_border_cells = ['V6', 'V8:V18', 'V21:V36',  'AA3:AA5', 'Y7']

    df_header_format = writer.book.add_format()
    df_header_format.set_bg_color('#FFDAB9')
    game_values_format = writer.book.add_format()
    game_values_format.set_bg_color('#F0F8FF')
    top_bot_border = writer.book.add_format()
    top_bot_border.set_top()
    top_bot_border.set_bottom()
    top_border = writer.book.add_format()
    top_border.set_top()
    left_right_border = writer.book.add_format()
    left_right_border.set_left()
    left_right_border.set_right()
    left_border = writer.book.add_format()
    left_border.set_left()

    _ = [sheet1.conditional_format(s, {'type': '3_color_scale'}) for s in strat_cells]
    _ = [sheet1.conditional_format(i, {'type': 'no_errors', 'format': df_header_format}) for i in indxs_cells]
    _ = [sheet1.conditional_format(i, {'type': 'no_errors', 'format': df_header_format}) for i in col_cells]
    _ = [sheet1.conditional_format(i, {'type': 'no_errors', 'format': game_values_format}) for i in game_cells]
    _ = [sheet1.conditional_format(i, {'type': 'no_errors', 'format': top_border}) for i in top_border_cells]
    _ = [sheet1.conditional_format(i, {'type': 'no_errors', 'format': top_bot_border}) for i in top_bot_border_cells]
    _ = [sheet1.conditional_format(i, {'type': 'no_errors', 'format': left_right_border}) for i in left_right_border_cells]
    _ = [sheet1.conditional_format(i, {'type': 'no_errors', 'format': left_border}) for i in left_border_cells]

    writer.save()



#plot_training('2019_05_12_11_19')