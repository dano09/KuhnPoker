from multiplayer.multiPlayerKuhnTrainer import KuhnTrainer

"""
Round 3 - Player Z Utility for Winning and Losing Hands
"""
R3_Z_WIN = [1, 2, 3]
R3_Z_LOSE = [3, 2, 1]
ROUND3 = [R3_Z_WIN, R3_Z_LOSE]

PPP_RESULTS = [2, -1]
BPP_RESULTS = [-1, -1]
BPB_RESULTS = [3, -2]
BBP_RESULTS = [-1, -1]
BBB_RESULTS = [4, -2]

"""
Round 4 - Player X Utility for Winning and Losing Hands
"""
R4_X_WIN = [3, 1, 2]
R4_X_LOSE = [1, 2, 3]

PBPP_RESULTS = [-1, -1]
PBPB_RESULTS = [3, -2]
PBBP_RESULTS = [-1, -1]
PBBB_RESULTS = [4, -2]
ROUND4 = [R4_X_WIN, R4_X_LOSE]

"""
Round 5 - Player Y Utility for Winning and Losing Hands
"""
R5_Y_WIN = [1, 3, 2]
R5_Y_LOSE = [3, 1, 2]

PPBPP_RESULTS = [-1, -1]
PPBPB_RESULTS = [3, -2]
PPBBP_RESULTS = [-1, -1]
PPBBB_RESULTS = [4, -2]
ROUND5 = [R5_Y_WIN, R5_Y_LOSE]

"""
Test all rounds
"""
t = KuhnTrainer()
for i in range(2):
    # Round 3
    assert t._calculate_terminal_payoff(plays=3, history='ppp', cards=ROUND3[i]) == PPP_RESULTS[i]
    assert t._calculate_terminal_payoff(plays=3, history='bpp', cards=ROUND3[i]) == BPP_RESULTS[i]
    assert t._calculate_terminal_payoff(plays=3, history='bpb', cards=ROUND3[i]) == BPB_RESULTS[i]
    assert t._calculate_terminal_payoff(plays=3, history='bbp', cards=ROUND3[i]) == BBP_RESULTS[i]
    assert t._calculate_terminal_payoff(plays=3, history='bbb', cards=ROUND3[i]) == BBB_RESULTS[i]
    # Round 4
    assert t._calculate_terminal_payoff(plays=4, history='pbpp', cards=ROUND4[i]) == PBPP_RESULTS[i]
    assert t._calculate_terminal_payoff(plays=4, history='pbpb', cards=ROUND4[i]) == PBPB_RESULTS[i]
    assert t._calculate_terminal_payoff(plays=4, history='pbbp', cards=ROUND4[i]) == PBBP_RESULTS[i]
    assert t._calculate_terminal_payoff(plays=4, history='pbbb', cards=ROUND4[i]) == PBBB_RESULTS[i]
    # Round 5
    assert t._calculate_terminal_payoff(plays=5, history='ppbpp', cards=ROUND5[i]) == PPBPP_RESULTS[i]
    assert t._calculate_terminal_payoff(plays=5, history='ppbpb', cards=ROUND5[i]) == PPBPB_RESULTS[i]
    assert t._calculate_terminal_payoff(plays=5, history='ppbbp', cards=ROUND5[i]) == PPBBP_RESULTS[i]
    assert t._calculate_terminal_payoff(plays=5, history='ppbbb', cards=ROUND5[i]) == PPBBB_RESULTS[i]