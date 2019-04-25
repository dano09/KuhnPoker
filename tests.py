from multiPlayerKuhnTrainer import KuhnTrainer

t = KuhnTrainer()

CARDS_P1_HIGH = [3, 2, 1]
CARDS_P1_MID  = [2, 3, 1]
CARDS_P1_LOW  = [1, 3, 2]

CARD_COMBOS = [CARDS_P1_HIGH, CARDS_P1_MID, CARDS_P1_LOW]

PPP_RESULTS = [2, -1, -1]
BPP_RESULTS = [2, 2, 2]
BPB_RESULTS = [3, 3, -2]
BBP_RESULTS = [3, -2, -2]
BBB_RESULTS = [4, -2, -2]
PBPP_RESULTS = [-1, -1, -1]
PBPB_RESULTS = [3, -2, -2]
PBBP_RESULTS = [-1, -1, -1]
PBBB_RESULTS = [4, -2, -2]
PPBPP_RESULTS = [-1, -1, -1]
PPBPB_RESULTS = [-1, -1, -1]

PPBBP_RESULTS = [3, 3, -2]

PPBBB_RESULTS = [4, -2, -2]


for i in range(len(CARD_COMBOS)):
    assert t.calculate_terminal_payoff(plays=3, history='ppp', cards=CARD_COMBOS[i]) == PPP_RESULTS[i]
    assert t.calculate_terminal_payoff(plays=3, history='bpp', cards=CARD_COMBOS[i]) == BPP_RESULTS[i]
    assert t.calculate_terminal_payoff(plays=3, history='bpb', cards=CARD_COMBOS[i]) == BPB_RESULTS[i]
    assert t.calculate_terminal_payoff(plays=3, history='bbp', cards=CARD_COMBOS[i]) == BBP_RESULTS[i]
    assert t.calculate_terminal_payoff(plays=3, history='bbb', cards=CARD_COMBOS[i]) == BBB_RESULTS[i]

    assert t.calculate_terminal_payoff(plays=4, history='pbpp', cards=CARD_COMBOS[i]) == PBPP_RESULTS[i]
    assert t.calculate_terminal_payoff(plays=4, history='pbpb', cards=CARD_COMBOS[i]) == PBPB_RESULTS[i]
    assert t.calculate_terminal_payoff(plays=4, history='pbbp', cards=CARD_COMBOS[i]) == PBBP_RESULTS[i]
    assert t.calculate_terminal_payoff(plays=4, history='pbbb', cards=CARD_COMBOS[i]) == PBBB_RESULTS[i]

    assert t.calculate_terminal_payoff(plays=5, history='ppbpp', cards=CARD_COMBOS[i]) == PPBPP_RESULTS[i]
    assert t.calculate_terminal_payoff(plays=5, history='ppbpb', cards=CARD_COMBOS[i]) == PPBPB_RESULTS[i]
    assert t.calculate_terminal_payoff(plays=5, history='ppbbp', cards=CARD_COMBOS[i]) == PPBBP_RESULTS[i]
    assert t.calculate_terminal_payoff(plays=5, history='ppbbb', cards=CARD_COMBOS[i]) == PPBBB_RESULTS[i]