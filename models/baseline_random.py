
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from .ranker_model import RankerModel

class RandomBaseline(RankerModel):
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def get_probability_of_win(self, team_a: str, team_b: str) -> float:
        return np.random.uniform(0, 1)
    
    def reset(self):
        pass