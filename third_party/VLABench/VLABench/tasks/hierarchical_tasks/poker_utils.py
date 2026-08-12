"""
Utils for texas holdem poker game.
"""
from itertools import combinations
from VLABench.tasks.hierarchical_tasks.primitive.select_poker_series import value2int, CARDS, SUITES, VALUES

RANKING = {
    "royal_flush": 10,
    "straight_flush": 9,
    "four_of_a_kind": 8,
    "full_house": 7,
    "flush": 6,
    "straight": 5,
    "three_of_a_kind": 4,
    "two_pair": 3,
    "one_pair": 2,
    "high_card": 1
}

def is_flush(cards):
    suites = [suite for value, suite in cards]
    return len(set(suites)) == 1
    
def is_straight(cards):
    values = sorted(card[0] for card in cards)
    if values == [2, 3, 4, 5, 14]:  # Handle ace-low straight
        values = [1, 2, 3, 4, 5]
    return all(values[i] + 1 == values[i + 1] for i in range(len(values) - 1))

def classify_by_value(cards):
    value_count = {}
    for card in cards:
        value = card[0]
        if value not in value_count:
            value_count[value] = 0
        value_count[value] += 1
    return value_count

def sorted_by_count_then_value(cards):
    value_count = classify_by_value(cards)
    return sorted(cards, key=lambda x: (value_count[x[0]], x[0]), reverse=True)

def check_texas_handem_cardtype(cards_combine) -> list:
    """
    """
    cards = [(value2int[card[0]], card[-1]) for card in list(cards_combine).copy()]
    largets_card = max(cards, key=lambda x: x[0])
    
    value_count = classify_by_value(cards)
    counts = sorted(value_count.values(), reverse=True)
    
    if is_flush(cards) and is_straight(cards):
        if largets_card[0] == 14:
            return RANKING["royal_flush"], sorted(cards, key=lambda x: x[0])
        return RANKING["straight_flush"], sorted(cards, key=lambda x: x[0])
    elif counts == [4, 1]:
        return RANKING["four_of_a_kind"], sorted_by_count_then_value(cards)[:4]
    elif counts == [3, 2]:
        return RANKING["full_house"], sorted_by_count_then_value(cards)
    elif is_flush(cards):
        return RANKING["flush"], sorted(cards, key=lambda x: x[0])
    elif is_straight(cards):
        return RANKING["straight"], sorted(cards, key=lambda x: x[0])
    elif counts == [3, 1, 1]:
        return RANKING["three_of_a_kind"], sorted_by_count_then_value(cards)[:3]
    elif counts == [2, 2, 1]:
        return RANKING["two_pair"], sorted_by_count_then_value(cards)[:4]
    elif counts == [2, 1, 1, 1]:
        return RANKING["one_pair"], sorted_by_count_then_value(cards)[:2]
    else:
        return RANKING["high_card"], [largets_card]

def get_largest_combination(cards: list):
    max_cardtype = 0
    max_combination = None
    for combination in combinations(cards, 5):
        cardtype, selected_combination = check_texas_handem_cardtype(combination)
        if cardtype > max_cardtype:
            max_cardtype = cardtype
            max_combination = selected_combination
    return max_cardtype, max_combination