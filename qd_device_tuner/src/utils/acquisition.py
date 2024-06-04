import qcodes as qc
import pandas as pd

from ..device import *

import datetime, os

from typing import List

import qcodes as qc


class Acquisition:
    def __init__(self, 
                 device: Device, 
                 save_dir: str = None) -> None:
        
        # Create QCoDeS database to store experiments
        assert save_dir is not None, "Please provide a save directory"
        todays_date = datetime.date.today().strftime("%Y-%m-%d")
        self.db_folder = os.path.join(save_dir, f"{device.name}_{todays_date}")
        os.makedirs(self.db_folder, exist_ok=True)
        db_file =  os.path.join(self.db_folder, f"experiments_{device.name}_{todays_date}.db")
        qc.dataset.initialise_or_create_database_at(db_file)

        # Create QCoDeS experiment to store measurements
        self.experiment = qc.dataset.load_or_create_experiment(
            "tuning",
            sample_name=device.name
        )

    def iv_sweep(self,
                 x: List[qc.dataset.LinSweep] | qc.dataset.LinSweep,
                 y: Drain,
                 break_condition: callable
                 ) -> pd.DataFrame:
        
        def check_bounds():
            """
            Check if a value is outside the specified bounds.

            Parameters:
                value (float): The value to check.
                bounds (tuple): A tuple defining the lower and upper bounds.

            Returns:
                bool: True if the value is outside the bounds, False otherwise.
            """
            return y.get_current() < y.bounds[0] or y.get_current() > y.bounds[1]

        if isinstance(x, List):
            x = qc.dataset.TogetherSweep(*x)
        
        result = qc.dataset.dond(
            x,
            y.source,
            break_condition= check_bounds,
            exp=self.experiment,
            show_progress=True
        )

        # Get last dataset recorded, convert to units of current (A)
        dataset = qc.load_last_experiment().last_data_set()
        df = dataset.to_pandas_dataframe().reset_index()
        df_current = df.copy()

        # Calculate the current column based on existing columns
        df_current[f'?'] = (df_current[f'?'] - y.offset) * y.scale

        return 




