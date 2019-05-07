from pprint import pprint
from multiplayer import kuhnHelper
from multiplayer import multiPlayerKuhnTrainer as mKuhnTrainer
from multiplayer import multiPlayerKuhnPoker as mKuhnPoker
import pandas as pd
from datetime import datetime
import os

GRAPHS_DIR = '/graphs/'
STRATS_DIR = '/trained_strategies/'
RESULTS_DIR = '/results/'
TRAINED_MODEL_FILES = ['cfr_strategy.p', 'p1_br_strategy.p', 'p2_br_strategy.p', 'p3_br_strategy.p']
PLAYER_RESULT_FILES = ['p1_results.p', 'p2_results.p', 'p3_results.p', 'cfr_br_df.p']


def setup_kuhn_poker_game(strategy, best_response=None):
    """
    Prepare the strategies to be used for each info node. If a best_response strategy is provided,
    then replace the base strategy positions with the best response positions
    :param strategy:  dict
    :param best_response: dict
    :return:
    """
    br_strategy = None
    base_strategy = {i_s: mKuhnPoker.GameInfoSet(info_set=i_s, strategy=strategy[i_s]) for i_s in strategy}
    if best_response:
        br_strategy = {i_s: mKuhnPoker.GameInfoSet(info_set=i_s, strategy=best_response[i_s]) for i_s in best_response}

    return {**base_strategy, **br_strategy} if best_response else base_strategy


def play_kuhn_poker(base_strat, best_response_strat, iterations):
    node_map = setup_kuhn_poker_game(base_strat, best_response_strat)
    results = mKuhnPoker.KuhnPoker(node_map).play_poker(iterations)
    if best_response_strat:
        return {k: results[k] for k in best_response_strat}

    return results


def calculate_utility(strat_df, br_player_df):
    # Join the CFR utilities with the best response utilities for one of the players
    df = strat_df.join(br_player_df, how='right', rsuffix='_br')
    # Calculate utility (antes/hand) for each position
    df['avg'] = df.utility / df.plays
    # Repeat for best response
    df['avg_br'] = df.utility_br / df.plays_br

    return df


def calculate_nash_equilibrium(util_results):
    cfr = {'p1': util_results[0].avg.mean(), 'p2': util_results[1].avg.mean(), 'p3': util_results[2].avg.mean()}
    br = {'p1': util_results[0].avg_br.mean(), 'p2': util_results[1].avg_br.mean(), 'p3': util_results[2].avg_br.mean()}
    cfr_results_df = pd.DataFrame(data=[cfr, br], index=['CFR', 'BR'])
    cfr_results_df.loc['diff'] = cfr_results_df.loc['BR'] - cfr_results_df.loc['CFR']
    epsilon = cfr_results_df.loc['diff'].mean(axis=0)
    return cfr_results_df, epsilon


def train(iterations=100, gen_graphs=False, base_dir=None):
    """
    TODO
    :return:
    """
    # 1) Generate a strategy profile using CFR
    print('Training Strategy Profile, this may take some time')
    cfr_trainer = mKuhnTrainer.KuhnTrainer(training_best_response=False, generate_graphs=gen_graphs, base_dir=base_dir)
    cfr_strategy_profiles = cfr_trainer.train(iterations)

    # 2) Compute a best response strategy for each player
    print('Training Best Response for Player 1')
    p1_br = mKuhnTrainer.KuhnTrainer(training_best_response=True,
                                     best_response_player=0,
                                     strategy_profile=cfr_strategy_profiles).train(iterations)

    print('Training Best Response for Player 2')
    p2_br = mKuhnTrainer.KuhnTrainer(training_best_response=True,
                                     best_response_player=1,
                                     strategy_profile=cfr_strategy_profiles).train(iterations)

    print('Training Best Response for Player 3')
    p3_br = mKuhnTrainer.KuhnTrainer(training_best_response=True,
                                     best_response_player=2,
                                     strategy_profile=cfr_strategy_profiles).train(iterations)

    print('Training complete')
    return cfr_strategy_profiles, p1_br, p2_br, p3_br


def main(iterations=100000, run_training=True, training_mod_dir=None, save_models=False, save_results=False, gen_graphs=False):
    """
    Determine if CFR generated strategy profile is epsilon-Nash Equilibrium
    1) Generate a strategy profile using CFR
    2) Compute a best response strategy, using CFR, for each player
    3) Compute utilities for each position of the strategy profile by playing three strategies against each other*
    4) Compute the utilities of the best response in each position by playing one BR strategy against
        two ordinary strategies
    5) Compare the BR strategy utilities in each position to the original strategies utilities to determine how much
        extra the BR strategy wins in each position
    6) If the best response strategy profile does not improve by more than an epsilon AVERAGED over all positions, then
        our original strategy profile is an epsilon-Nash Equilibrium
    :return:
    """
    if save_models or save_results or gen_graphs:
        # Creates a unique directory for models, graphs and results
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
        os.makedirs(timestamp, exist_ok=True)

    # Step 1 & 2 - done in train method
    if run_training:
        strategy_profiles, *best_responses = train(iterations, gen_graphs=gen_graphs, base_dir=timestamp)

        if save_models:
            res = [strategy_profiles, *best_responses]
            kuhnHelper.save_results(results=res, file_names=TRAINED_MODEL_FILES, base_dir=timestamp, file_dir=STRATS_DIR)

    else:
        strategy_profiles, *best_responses = kuhnHelper.load_trained_models(training_mod_dir)

    # 3) Compute utilities for each position of the strategy profile by playing three strategies against each other

    print('Playing Kuhn Poker with base strategy')
    cfr_game_results = play_kuhn_poker(base_strat=strategy_profiles, best_response_strat=None, iterations=iterations)

    # 4) Compute the utilities of the best response in each position by playing one BR strategy
    #    against two ordinary strategies

    print('Playing Kuhn Poker with best response strategies')
    br_game_results = [play_kuhn_poker(base_strat=strategy_profiles, best_response_strat=br, iterations=iterations) for br in best_responses]

    # 5) Compare the BR strategy utilities in each position to the original strategies utilities to determine how much
    #    extra the BR strategy wins in each position

    cfr_game_results_df = kuhnHelper.df_builder(cfr_game_results)
    br_game_results_df = [kuhnHelper.df_builder(r) for r in br_game_results]

    player_results = [calculate_utility(cfr_game_results_df, br_profile) for br_profile in br_game_results_df]

    cfr_br_df, epsilon = calculate_nash_equilibrium(player_results)

    if save_results:
        player_results.extend([cfr_br_df, epsilon])
        kuhnHelper.save_results(results=player_results, file_names=PLAYER_RESULT_FILES, base_dir=timestamp, file_dir=RESULTS_DIR)

    return cfr_br_df, epsilon


cfr_br_df, epsilon = main(iterations=100000, run_training=True, save_models=True, save_results=True, gen_graphs=True)


#cfr_br_df, epsilon = main(iterations=1000,
#                          run_training=False,
#                          training_mod_dir='2019_05_07_16_10',
#                          save_models=True,
#                          save_results=True,
#                          gen_graphs=True)


