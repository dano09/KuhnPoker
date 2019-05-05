from pprint import pprint
from multiplayer import kuhnHelper
from multiplayer import multiPlayerKuhnTrainer as mKuhnTrainer
from multiplayer import multiPlayerKuhnPoker as mKuhnPoker
import pandas as pd


def setUpKuhnPokerGame(strat, best_response=None):
    """
    Prepare the strategies to be used for each info node. If a best_response strategy is provided,
    then replace the base strategy positions with the best response positions
    :param strat:  dict
    :param best_response: dict
    :return:
    """
    br_strategy = None
    base_strategy = {i_s: mKuhnPoker.GameNode(info_set=i_s, strategy=strat[i_s]) for i_s in strat}
    if best_response:
        br_strategy = {i_s: mKuhnPoker.GameNode(info_set=i_s, strategy=best_response[i_s]) for i_s in best_response}

    return {**base_strategy, **br_strategy} if best_response else base_strategy


def play_kuhn_poker(base_strat, best_response_strat, iterations):
    node_map = setUpKuhnPokerGame(base_strat, best_response_strat)
    results = mKuhnPoker.KuhnPoker(node_map).play_poker(iterations)
    if best_response_strat:
        return {k: results[k] for k in best_response_strat}

    return results


def calculate_utility(strat_df, br_player_df):
    # Filters infosets so only positions for one player is used
    df = strat_df.join(br_player_df, how='right', rsuffix='_br')
    # Calculate utility (antes/hand) for each position
    df['avg'] = df.utility / df.plays
    # Repeat for best response
    df['avg_br'] = df.utility_br / df.plays_br
    df['util_diff'] = df.avg_br - df.avg
    # Average utility over all positions is our target utility
    #avg_utility = df.avg.sum() / df.shape[0]
    #br_avg_utility = df.avg_br.sum() / df.shape[0]
    avg_util_diff = df.util_diff.sum() / df.shape[0]

    return df, avg_util_diff


def train(iterations=100, save_training=False):
    """
    TODO
    :return:
    """

    ''' 
        1) Generate a strategy profile using CFR
    '''
    print('Training Strategy Profile, this may take some time')
    strategy_profiles = mKuhnTrainer.KuhnTrainer(training_best_response=False).train(iterations)

    '''
        2) Compute a best response strategy for each player
    '''
    print('Training Best Response for Player 1')
    p1_br = mKuhnTrainer.KuhnTrainer(training_best_response=True, best_response_player=0, strategy_profile=strategy_profiles).train(iterations)
    print('Training Best Response for Player 2')
    p2_br = mKuhnTrainer.KuhnTrainer(training_best_response=True, best_response_player=1, strategy_profile=strategy_profiles).train(iterations)
    print('Training Best Response for Player 3')
    p3_br = mKuhnTrainer.KuhnTrainer(training_best_response=True, best_response_player=2, strategy_profile=strategy_profiles).train(iterations)

    if save_training:
        kuhnHelper.save_trained_models([strategy_profiles, p1_br, p2_br, p3_br])
    print('Training complete')
    return strategy_profiles, p1_br, p2_br, p3_br


def main(iterations=100000, run_training=True, save_models=False, save_results=False):
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

    '''  
        Step 1 & 2 - done in train method
    '''
    if run_training:
        strategy_profiles, *best_responses = train(iterations, save_training=save_models)
    else:
        strategy_profiles, *best_responses = kuhnHelper.load_trained_models()

    '''  
        3) Compute utilities for each position of the strategy profile by playing three strategies against each other
    '''
    print('Playing Kuhn Poker with base strategy')
    results = play_kuhn_poker(base_strat=strategy_profiles, best_response_strat=None, iterations=iterations)

    '''
        4) Compute the utilities of the best response in each position by playing one BR strategy 
           against two ordinary strategies    
    '''
    print('Playing Kuhn Poker with best response strategies')
    br_results = [play_kuhn_poker(base_strat=strategy_profiles, best_response_strat=br, iterations=iterations) for br in best_responses]

    '''
        5) Compare the BR strategy utilities in each position to the original strategies utilities to determine how much
           extra the BR strategy wins in each position
    '''
    strat_profile_df = kuhnHelper.df_builder(results)
    br_strat_profile_df = [kuhnHelper.df_builder(r) for r in br_results]
    player_results = [calculate_utility(strat_profile_df, br_profile) for br_profile in br_strat_profile_df]

    cfr = {'p1': player_results[0][0].avg.sum(), 'p2': player_results[1][0].avg.sum(),
           'p3': player_results[2][0].avg.sum()}

    br = {'p1': player_results[0][0].avg_br.sum(), 'p2': player_results[1][0].avg_br.sum(),
           'p3': player_results[2][0].avg_br.sum()}

    cfr_results_df = pd.DataFrame(data=[cfr, br], index=['CFR', 'BR'])
    cfr_results_df.loc['diff'] = cfr_results_df.loc['BR'] - cfr_results_df.loc['CFR']
    epsilon = cfr_results_df.loc['diff'].mean(axis=0)

    if save_results:
        kuhnHelper.save_results(player_results)


#main(iterations=1000000, run_training=False, save_models=False, save_results=False)
#main(iterations=1000000, run_training=True, save_models=True, save_results=True)


s = kuhnHelper.load_trained_models()
base_strat = s[0]
br1_strat = s[1]
br2_strat = s[2]
br3_strat = s[3]

sdf = kuhnHelper.strat_df_builder(base_strat)
#sdf1 = kuhnHelper.strat_df_builder(br1_strat)
#sdf2 = kuhnHelper.strat_df_builder(br2_strat)
#sdf3 = kuhnHelper.strat_df_builder(br3_strat)

#print('done')
results = kuhnHelper.load_results()
#p1 = results[0]
#p2 = results[1]
#p3 = results[2]

#print(results)