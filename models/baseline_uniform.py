

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from .ranker_model import RankerModel

class UniformBaseline(RankerModel):
    def __init__(self):
        pass

    def get_probability_of_win(self, team_a: str, team_b: str) -> float:
        return 0.5
    
    def reset(self):
        pass