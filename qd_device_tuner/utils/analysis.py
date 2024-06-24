from ..device import *

from typing import Callable

import scipy as sp
import pandas as pd

class Analysis:
    def __init__(self) -> None:
        pass

    def fit(self, x: pd.Series, y: pd.Series, func: Callable, guess: tuple) -> tuple:
        
        fit_params, fit_cov = sp.optimize.curve_fit(
            func, 
            x, 
            y,
            guess
        )

        return fit_params, fit_cov
