import numpy as np
from enum import Enum
from collections import Counter

class Card:
    class Ranks(Enum):
        TWO = 0
        THREE = 1
        FOUR = 2
        FIVE = 3
        SIX = 4
        SEVEN = 5
        EIGHT = 6
        NINE = 7
        TEN = 8
        JACK = 9
        QUEEN = 10
        KING = 11
        ACE = 12

    base_chip_values = {
        Ranks.TWO: 2,
        Ranks.THREE: 3,
        Ranks.FOUR: 4,
        Ranks.FIVE: 5,
        Ranks.SIX: 6,
        Ranks.SEVEN: 7,
        Ranks.EIGHT: 8,
        Ranks.NINE: 9,
        Ranks.TEN: 10,
        Ranks.JACK: 10,
        Ranks.QUEEN: 10,
        Ranks.KING: 10,
        Ranks.ACE: 11,
    }

    class Suits(Enum):
        SPADES = 0
        CLUBS = 1
        HEARTS = 2
        DIAMONDS = 3

    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.played = False

    def chip_value(self):
        return self.base_chip_values[self.rank]
    
    def encode(self):
        return self.rank.value + self.suit.value * len(self.Ranks)
    
    def __str__(self):
        return self.rank.name + " OF " + self.suit.name


class BalatroGame:
    class State(Enum):
        IN_PROGRESS = 0
        WIN = 1
        LOSS = 2

    def __init__(self, deck="yellow", stake="white"):
        self.deck = [Card(rank, suit) for suit in Card.Suits for rank in Card.Ranks]
        self.hand_indexes = []
        self.highlighted_indexes = []

        self.hand_size = 8
        self.hands = 4
        self.discards = 3

        self.ante = 1
        self.blind_index = 0
        self.blinds = [300, 450, 600]

        self.round_score = 0
        self.last_score = 0
        self.last_played_hand_type = ""
        self.round_hands = self.hands
        self.round_discards = self.discards

        self.state = self.State.IN_PROGRESS
        self.round_in_progress = False

        self.remaining_cards = set(range(len(self.deck)))
        self._draw_cards()

    def highlight_card(self, hand_index: int):
        self.highlighted_indexes.append(self.hand_indexes.pop(hand_index))

    def play_hand(self):
        self.round_hands -= 1

        hand = [self.deck[idx] for idx in self.highlighted_indexes]
        self.last_score, self.last_played_hand_type = self._evaluate_hand(hand)
        self.round_score += self.last_score

        if self.round_score >= self.blinds[self.blind_index]:
            self._end_round()
        elif self.round_hands == 0:
            self.state = self.State.LOSS
        else:
            self._draw_cards()

        return self.last_score

    def discard_hand(self):
        self.round_discards -= 1
        self._draw_cards()
    
    def _start_round(self):
        self._draw_cards()

    def _end_round(self):
        for card in self.deck:
            card.played = False

        self.hand_indexes.clear()
        self.highlighted_indexes.clear()
        self.round_hands = self.hands
        self.round_discards = self.discards
        self.round_score = 0
        self.last_score = 0
        self.remaining_cards = set(range(len(self.deck)))

        self.blind_index += 1
        if self.blind_index == 3:
            self.blind_index = 0
            self.ante += 1
            self.state = self.State.WIN
        self._start_round()

    def _draw_cards(self):
        self.highlighted_indexes.clear()
        draw_count = min(self.hand_size - len(self.hand_indexes) -len(self.highlighted_indexes), len(self.remaining_cards))
        new_cards = np.random.choice(list(self.remaining_cards), draw_count, replace=False)
        self.remaining_cards.difference_update(new_cards)
        for card_index in new_cards:
            self.deck[card_index].played = True
            self.hand_indexes.append(card_index)

    @staticmethod
    def _evaluate_hand(hand):
        rank_counts = Counter(card.rank for card in hand)
        sorted_ranks = sorted(rank.value for rank in rank_counts)
        is_flush = len({card.suit for card in hand}) == 1 and len(hand) == 5
        is_straight = (
            len(hand) == 5
            and (sorted_ranks[-1] - sorted_ranks[0] == 4)
            and len(sorted_ranks) == 5
        )

        chips = 0
        mult = 0
        last_played_hand_type = ""

        if is_flush and is_straight:
            chips += 100
            mult += 8
            last_played_hand_type = "Straight Flush"

        elif is_flush: 
            chips += 35
            mult += 4
            last_played_hand_type = "Flush"
        elif is_straight:
            chips += 30
            mult += 4
            last_played_hand_type = "Straight"
        else:
            primary, secondary = rank_counts.most_common(2) + [(None, 0)] * (2 - len(rank_counts))
            if primary[1] == 4:
                chips += 60
                mult += 7
                last_played_hand_type = "Four of a Kind"
            elif primary[1] == 3 and secondary[1] == 2:
                chips += 40
                mult += 4
                last_played_hand_type = "Full House"
            elif primary[1] == 3:
                chips += 30
                mult += 3
                last_played_hand_type = "Three of a Kind"
            elif primary[1] == 2 and secondary[1] == 2:
                chips += 20
                mult += 2
                last_played_hand_type = "Two Pair"
            elif primary[1] == 2:
                chips += 10
                mult += 2
                last_played_hand_type = "Pair"
            else:
                chips += 5
                mult += 1
                last_played_hand_type = "High Card"

        chips += sum(card.chip_value() for card in hand)
        return chips * mult, last_played_hand_type 

    def deck_to_string(self):
        return ", ".join(map(str, self.deck))

    def hand_to_string(self):
        return ", ".join(str(self.deck[idx]) for idx in self.hand_indexes)

    def highlighted_to_string(self):
        return ", ".join(str(self.deck[idx]) for idx in self.highlighted_indexes)