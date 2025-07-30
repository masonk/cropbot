import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import copy
import sys
import argparse
import json
from enum import Enum
from collections import defaultdict

class Plot(Enum):
    P = "P"
    Y = "Y"
    B = "B"

class Pair(Enum):
    YY = "YY"
    YP = "YP"
    YB = "YB"
    BB = "BB"
    BP = "BP"
    PP = "PP"
    YX = "YX"
    BX = "BX"
    PX = "PX"

class Choice(Enum):
    Y_FROM_YY = "Y_FROM_YY"
    Y_FROM_YB = "Y_FROM_YB"
    Y_FROM_YP = "Y_FROM_YP"

    P_FROM_YP = "P_FROM_YP"
    P_FROM_PP = "P_FROM_PP"
    P_FROM_PB = "P_FROM_PB"

    B_FROM_BB = "B_FROM_BB"
    B_FROM_BY = "B_FROM_BY"
    B_FROM_BP = "B_FROM_BP"

    Y_FROM_YX = "Y_FROM_YX"
    B_FROM_BX = "B_FROM_BX"
    P_FROM_PX = "P_FROM_PX"

    @classmethod
    def from_pair(cls, pair: Pair) -> List['Choice']:
        """Return all possible choices for a given pair."""
        choice_map = {
            Pair.YY: [cls.Y_FROM_YY],
            Pair.YP: [cls.Y_FROM_YP, cls.P_FROM_YP],
            Pair.YB: [cls.Y_FROM_YB, cls.B_FROM_BY],
            Pair.BB: [cls.B_FROM_BB],
            Pair.BP: [cls.B_FROM_BP, cls.P_FROM_PB],
            Pair.PP: [cls.P_FROM_PP],
            Pair.YX: [cls.Y_FROM_YX],
            Pair.BX: [cls.B_FROM_BX],
            Pair.PX: [cls.P_FROM_PX]
        }
        return choice_map.get(pair, [])

    def pair(self) -> Pair:
        pair_map = {
            self.Y_FROM_YY: Pair.YY,
            self.Y_FROM_YB: Pair.YB,
            self.Y_FROM_YP: Pair.YP,

            self.P_FROM_YP: Pair.YP,
            self.P_FROM_PP: Pair.PP,
            self.P_FROM_PB: Pair.BP,

            self.B_FROM_BB: Pair.BB,
            self.B_FROM_BY: Pair.YB,
            self.B_FROM_BP: Pair.BP,

            self.Y_FROM_YX: Pair.YX,
            self.P_FROM_PX: Pair.PX,
            self.B_FROM_BX: Pair.BX
        }
        return pair_map[self]

    def upgrade(self) -> Plot:
        upgrade_map = {
            self.Y_FROM_YY: Plot.Y,
            self.Y_FROM_YB: Plot.Y,
            self.Y_FROM_YP: Plot.Y,

            self.P_FROM_YP: Plot.P,
            self.P_FROM_PP: Plot.P,
            self.P_FROM_PB: Plot.P,

            self.B_FROM_BB: Plot.B,
            self.B_FROM_BY: Plot.B,
            self.B_FROM_BP: Plot.B,

            self.Y_FROM_YX: Plot.Y,
            self.P_FROM_PX: Plot.P,
            self.B_FROM_BX: Plot.B
        }
        return upgrade_map[self]
    
    def leftover_choice(self) -> Optional['Choice']:
        leftover_map = {
            self.Y_FROM_YY: self.Y_FROM_YX,
            self.Y_FROM_YB: self.B_FROM_BX,
            self.Y_FROM_YP: self.P_FROM_PX,

            self.P_FROM_YP: self.Y_FROM_YX,
            self.P_FROM_PP: self.P_FROM_PX,
            self.P_FROM_PB: self.B_FROM_BX,

            self.B_FROM_BB: self.B_FROM_BX,
            self.B_FROM_BY: self.Y_FROM_YX,
            self.B_FROM_BP: self.P_FROM_PX,
        }
        return leftover_map.get(self, None)

@dataclass
class PlotState:
    count: int = 0
    upgrades: Dict[Plot, int] = field(default_factory=lambda: defaultdict(int))
    
    def __post_init__(self):
        if not isinstance(self.upgrades, defaultdict):
            self.upgrades = defaultdict(int, self.upgrades)

# Game constants
P = np.array([
    [0.75, 0.25, 0.00, 0.00],
    [0.00, 0.80, 0.20, 0.00],
    [0.00, 0.00, 0.97, 0.03],
    [0.00, 0.00, 0.00, 1.00]
])

s0 = np.array([1., 0., 0., 0.])
seeds_per_plot = 23
tier_yield = np.array([0.0073 * 12.7, 1.85 * 12.7, 47 * 12.7, 234 * 12.7 / 2]) * seeds_per_plot

yellow_price = 1/3950
blue_price = 1/5300
purple_price = 1/5900

# Calculate expected values for each plot type and upgrade level
evs = {
    Plot.Y: [],
    Plot.B: [],
    Plot.P: []
}

s = s0.copy()
for i in range(10):
    lifeforce = np.dot(s, tier_yield)
    evs[Plot.Y].append(lifeforce * yellow_price)
    evs[Plot.B].append(lifeforce * blue_price)
    evs[Plot.P].append(lifeforce * purple_price)
    s = s @ P

print(f"yellow EVs: {evs[Plot.Y]}")
print(f"blue EVs: {evs[Plot.B]}")
print(f"purple EVs: {evs[Plot.P]}")

class GameState:
    def __init__(self):
        self._plots: Dict[Pair, PlotState] = defaultdict(PlotState)
        self._harvested_value: float = 0.0
    
    @classmethod
    def from_starting_pairs(cls, *pairs: Pair) -> 'GameState':
        game = cls()
        for pair in pairs:
            game._plots[pair].count += 1
        return game
    
    def get_available_choices(self):
        """Generator that yields all choices currently available without allocating memory."""
        for pair, plot_state in self._plots.items():
            if plot_state.count > 0:
                for choice in Choice.from_pair(pair):
                    yield choice
    
    def is_terminal(self) -> bool:
        """Check if no more choices are available."""
        for _ in self.get_available_choices():
            return False
        return True
    
    def choose(self, choice: Choice, wither: bool = False):
        """Make a choice, modifying this game state in place.
        
        Args:
            choice: The choice to make
            wither: If True, the partner plot withers (for X pairs, this means no leftover)
        """
        # Get the pair and plot type for this choice
        pair = choice.pair()
        upgrade_plot = choice.upgrade()
        
        # Check if we have this pair available
        if self._plots[pair].count == 0:
            raise ValueError(f"Can't choose {choice} because no {pair} plots available")
        
        # Get current upgrade level for this plot type
        current_upgrades = self._plots[pair].upgrades[upgrade_plot]
        
        # Calculate harvest value
        if current_upgrades < len(evs[upgrade_plot]):
            harvest_value = evs[upgrade_plot][current_upgrades]
        else:
            # If we've exceeded the calculated EVs, use the last one
            harvest_value = evs[upgrade_plot][-1]
        
        self._harvested_value += harvest_value
        
        # Store the upgrade state of the harvested plot before removing it
        harvested_plot_upgrades = dict(self._plots[pair].upgrades)
        
        # Remove one plot of this pair
        self._plots[pair].count -= 1
        if self._plots[pair].count == 0:
            del self._plots[pair]
        
        # Add leftover plot if it doesn't wither
        leftover_pair = None
        if not wither:
            leftover_choice = choice.leftover_choice()
            if leftover_choice:
                leftover_pair = leftover_choice.pair()
                self._plots[leftover_pair].count += 1
        
        # Upgrade all remaining plots of other types
        upgraded_plots = []
        for remaining_pair, plot_state in self._plots.items():
            if plot_state.count > 0:
                for plot_type in [Plot.Y, Plot.B, Plot.P]:
                    if plot_type != upgrade_plot:
                        plot_state.upgrades[plot_type] += 1
                        upgraded_plots.append((remaining_pair, plot_type))
        
        # Return undo information for backtracking
        return {
            'harvest_value': harvest_value,
            'pair': pair,
            'pair_was_deleted': pair not in self._plots,
            'leftover_pair': leftover_pair,
            'upgraded_plots': upgraded_plots,
            'harvested_plot_upgrades': harvested_plot_upgrades,
            'wither': wither
        }
    
    def undo_choice(self, choice: Choice, undo_info: dict, wither: bool = False):
        """Undo a choice to backtrack, using the undo information."""
        # Restore harvest value
        self._harvested_value -= undo_info['harvest_value']
        
        # Undo upgrades
        for remaining_pair, plot_type in undo_info['upgraded_plots']:
            if remaining_pair in self._plots:
                self._plots[remaining_pair].upgrades[plot_type] -= 1
        
        # Remove leftover plot if it was added
        if undo_info['leftover_pair']:
            self._plots[undo_info['leftover_pair']].count -= 1
            if self._plots[undo_info['leftover_pair']].count == 0:
                del self._plots[undo_info['leftover_pair']]
        
        # Restore the original pair
        pair = undo_info['pair']
        if undo_info['pair_was_deleted']:
            # Recreate the plot state if it was deleted, with its original upgrades
            self._plots[pair] = PlotState(count=1)
            self._plots[pair].upgrades = defaultdict(int, undo_info['harvested_plot_upgrades'])
        else:
            # Just increment the count
            self._plots[pair].count += 1
    
    def find_optimal_strategy(self) -> Tuple[List[Choice], float]:
        """Find the optimal sequence of choices considering wither probability."""
        # Store the initial harvested value to restore it later
        initial_harvested_value = self._harvested_value
        
        def calculate_expected_value(current_path: List[Choice], current_state_value: float) -> float:
            """Calculate expected value for a path considering wither probability."""
            # Check if this is a terminal state
            if self.is_terminal():
                return current_state_value
            
            max_ev = current_state_value  # In case no moves improve EV
            best_choice = None
            
            # Try each available choice
            available_choices = list(self.get_available_choices())
            for choice in available_choices:
                pair = choice.pair()
                
                # Check if this is a paired plot (not XX)
                is_paired = pair not in [Pair.YX, Pair.BX, Pair.PX]
                
                if is_paired:
                    # 60% chance no wither, 40% chance wither
                    # Calculate expected value for no wither case
                    undo_info_no_wither = self.choose(choice, wither=False)
                    harvest_value = undo_info_no_wither['harvest_value']
                    ev_no_wither = calculate_expected_value(current_path + [choice], 
                                                           current_state_value + harvest_value)
                    self.undo_choice(choice, undo_info_no_wither, wither=False)
                    
                    # Calculate expected value for wither case
                    undo_info_wither = self.choose(choice, wither=True)
                    ev_wither = calculate_expected_value(current_path + [choice], 
                                                        current_state_value + harvest_value)
                    self.undo_choice(choice, undo_info_wither, wither=True)
                    
                    # Expected value is weighted average
                    choice_ev = 0.6 * ev_no_wither + 0.4 * ev_wither
                else:
                    # Single plot (XX pair) - no wither possibility
                    undo_info = self.choose(choice, wither=False)
                    harvest_value = undo_info['harvest_value']
                    choice_ev = calculate_expected_value(current_path + [choice], 
                                                       current_state_value + harvest_value)
                    self.undo_choice(choice, undo_info, wither=False)
                
                if choice_ev > max_ev:
                    max_ev = choice_ev
                    best_choice = choice
            
            return max_ev
        
        def find_best_first_move() -> Tuple[Choice, float]:
            """Find the best first move and its expected value."""
            best_choice = None
            best_ev = float('-inf')
            
            available_choices = list(self.get_available_choices())
            for choice in available_choices:
                pair = choice.pair()
                is_paired = pair not in [Pair.YX, Pair.BX, Pair.PX]
                
                if is_paired:
                    # Calculate EV for both wither outcomes
                    undo_info_no_wither = self.choose(choice, wither=False)
                    harvest_value = undo_info_no_wither['harvest_value']
                    ev_no_wither = calculate_expected_value([choice], harvest_value)
                    self.undo_choice(choice, undo_info_no_wither, wither=False)
                    
                    undo_info_wither = self.choose(choice, wither=True)
                    ev_wither = calculate_expected_value([choice], harvest_value)
                    self.undo_choice(choice, undo_info_wither, wither=True)
                    
                    choice_ev = 0.6 * ev_no_wither + 0.4 * ev_wither
                else:
                    undo_info = self.choose(choice, wither=False)
                    harvest_value = undo_info['harvest_value']
                    choice_ev = calculate_expected_value([choice], harvest_value)
                    self.undo_choice(choice, undo_info, wither=False)
                
                print(f"Choice {choice.value}: EV = {choice_ev:.6f}")
                
                if choice_ev > best_ev:
                    best_ev = choice_ev
                    best_choice = choice
            
            return best_choice, best_ev
        
        # Find the best strategy
        print("Calculating expected values for each starting choice...")
        best_choice, best_ev = find_best_first_move()
        
        # For now, return just the best first move
        # A full solution would need to build the entire strategy tree
        print(f"\nBest first move: {best_choice.value if best_choice else 'None'}")
        print(f"Expected value: {best_ev:.6f}")
        
        # Restore the initial harvested value
        self._harvested_value = initial_harvested_value
        
        return [best_choice] if best_choice else [], best_ev
    
    def get_total_value(self) -> float:
        """Get the total harvested value so far."""
        return self._harvested_value
    
    def __str__(self) -> str:
        plot_info = []
        for pair, plot_state in self._plots.items():
            if plot_state.count > 0:
                upgrades_str = dict(plot_state.upgrades) if plot_state.upgrades else {}
                plot_info.append(f"{pair.value}:{plot_state.count}(upgrades:{upgrades_str})")
        
        plots_str = ", ".join(plot_info) if plot_info else "No plots"
        return f"GameState(value={self._harvested_value:.6f}, plots={plots_str})"

# Example usage
if __name__ == "__main__":
    # Create initial game state
    game = GameState.from_starting_pairs(Pair.YY, Pair.BP, Pair.YP, Pair.BP, Pair.YB)
    print(f"Initial state: {game}")
    print(f"Initial harvested value: {game.get_total_value()}")
    print(f"Available choices: {list(c.value for c in game.get_available_choices())}")
    
    print(f"\n=== FINDING OPTIMAL STRATEGY (WITH WITHER PROBABILITY) ===")
    print("Note: Paired plots have 40% chance of partner withering")
    print("Calculating expected values...\n")
    
    # Find the optimal strategy
    optimal_path, optimal_expected_value = game.find_optimal_strategy()
    
    print(f"\n=== OPTIMAL STRATEGY FOUND ===")
    print(f"Best expected value: {optimal_expected_value:.6f}")