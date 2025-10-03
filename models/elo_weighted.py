"""
Elo Rating System with Goal Differential Weighting

This model extends the basic Elo system by incorporating the margin of victory (goal differential).
Teams gain more rating points for bigger wins and lose more points for bigger losses.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from .elo import EloRatingSystem


class EloWeighted(EloRatingSystem):
    
    def __init__(
        self, 
        k_factor: float = 32, 
        initial_rating: float = 1500,
        goal_diff_importance: float = 1.0
    ):
        super().__init__(k_factor, initial_rating)
        self.goal_diff_importance = goal_diff_importance
    
    def calculate_goal_diff_multiplier(self, goal_diff: int) -> float:
        """ 
        goal_diff = 0 (draw): multiplier = 1.0
        goal_diff = 1: multiplier = 1.0
        goal_diff = 2: multiplier = 1.5
        goal_diff = 3: multiplier = 1.75
        goal_diff = 5: multiplier = 2.0
        """
        if goal_diff <= 1:
            return 1.0
        
        # Logarithmic scaling with diminishing returns
        # Formula: 1 + log2(goal_diff) * importance
        multiplier = 1.0 + np.log2(goal_diff) * self.goal_diff_importance
        
        # Cap the maximum multiplier to prevent extreme changes
        max_multiplier = 3.0
        return min(multiplier, max_multiplier)
    
    def process_match(
        self, 
        home_team: str, 
        away_team: str, 
        home_score: int, 
        away_score: int,
        date: Optional[pd.Timestamp] = None,
        k_factor: Optional[float] = None
    ) -> Dict[str, float]:
        # Get ratings before update
        old_home_rating = self.get_rating(home_team)
        old_away_rating = self.get_rating(away_team)
        
        # Calculate goal differential
        goal_diff = abs(home_score - away_score)
        
        # Determine match outcome (from home team perspective)
        if home_score > away_score:
            home_actual_score = 1.0  # Win
        elif home_score < away_score:
            home_actual_score = 0.0  # Loss
        else:
            home_actual_score = 0.5  # Draw
        
        # Calculate goal differential multiplier
        multiplier = self.calculate_goal_diff_multiplier(goal_diff)
        
        # Apply weighted K-factor
        effective_k = (k_factor if k_factor is not None else self.k_factor) * multiplier
        
        # Update ratings with weighted K-factor
        new_home_rating, new_away_rating = self.update_ratings(
            home_team, away_team, home_actual_score, k_factor=effective_k
        )
        
        # Add to team history
        self.rating_history[home_team].append({
            'date': date,
            'opponent': away_team,
            'rating': new_home_rating,
            'rating_change': new_home_rating - old_home_rating,
            'goal_diff': home_score - away_score,
            'multiplier': multiplier
        })
        self.rating_history[away_team].append({
            'date': date,
            'opponent': home_team,
            'rating': new_away_rating,
            'rating_change': new_away_rating - old_away_rating,
            'goal_diff': away_score - home_score,
            'multiplier': multiplier
        })
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'goal_diff': goal_diff,
            'multiplier': multiplier,
            'effective_k': effective_k,
            'home_rating_before': old_home_rating,
            'away_rating_before': old_away_rating,
            'home_rating_after': new_home_rating,
            'away_rating_after': new_away_rating,
            'home_rating_change': new_home_rating - old_home_rating,
            'away_rating_change': new_away_rating - old_away_rating,
        }
