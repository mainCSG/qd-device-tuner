import qcodes as qc
import pandas as pd
import numpy as np 

from ..device import *
from ..components.gate import Gate

import datetime, os

from typing import List, Union, Callable, Dict

General1DSweep = Union[qc.dataset.AbstractSweep, qc.dataset.TogetherSweep]

import time
import qcodes as qc


class Acquisition:
    def __init__(self, station) -> None:
        self.station = station

    def create_database(self, database_name: str, save_dir: str):
        todays_date = datetime.date.today().strftime("%Y-%m-%d")
        self.db_folder = os.path.join(save_dir, f"databases_{todays_date}")
        os.makedirs(self.db_folder, exist_ok=True)
        db_file =  os.path.join(self.db_folder, f"{database_name}_{todays_date}.db")
        qc.dataset.initialise_or_create_database_at(db_file)

    def create_experiment(self, experiment_name: str, sample_name: str):
        self.experiment = qc.dataset.load_or_create_experiment(
            experiment_name,
            sample_name=sample_name
        )

    def iv_sweep(self,
                 V: General1DSweep | List[General1DSweep],
                 I: Drain,
                 measurement_name: str,
                 break_condition: Callable[..., None] = None) -> pd.DataFrame:
        
        if isinstance(V, General1DSweep):
            V = [V]

        result = qc.dataset.dond(
            *V,
            I.source,
            break_condition=break_condition,
            exp=self.experiment,
            measurement_name=measurement_name,
            show_progress=True
        )

        # Get last dataset recorded, convert to units of current (A)
        dataset = qc.load_last_experiment().last_data_set()
        df = dataset.to_pandas_dataframe().reset_index()

        return df
    
    def iv_sweep_2d(self,
                    Vstep: General1DSweep,
                    Vsweep: General1DSweep,
                    I: Drain,
                    measurement_name: str,
                    break_condition: Callable[..., None] = None) -> pd.DataFrame:

        Vstep_min = Vstep.get_setpoints()[0]
        Vstep_max = Vstep.get_setpoints()[-1]
        Vstep_delay = Vstep.delay
        Vstep_steps = len(Vstep.get_setpoints())
        Vstep_param = Vstep.param

        Vsweep_min = Vsweep.get_setpoints()[0]
        Vsweep_max = Vsweep.get_setpoints()[-1]
        Vsweep_delay = Vsweep.delay
        Vsweep_steps = len(Vsweep.get_setpoints())
        Vsweep_param = Vsweep.param

        dummyGate = Gate(name="dummy", source=Vstep_param)

        def reset_smoothly():
            self.set_smoothly({dummyGate: Vsweep_max})

        result = qc.dataset.do2d(
            Vstep_param,
            Vstep_max,
            Vstep_min,
            Vstep_steps,
            Vstep_delay,
            Vsweep_param,
            Vsweep_max,
            Vsweep_min,
            Vsweep_steps,
            Vsweep_delay,
            I.source,
            after_inner_actions = [reset_smoothly],
            set_before_sweep=True, 
            show_progress=True, 
            measurement_name=measurement_name,
            exp=self.experiment
        )

        # Get last dataset recorded, convert to units of current (A)
        dataset = qc.load_last_experiment().last_data_set()
        df = dataset.to_pandas_dataframe().reset_index()

        return df


    def calculate_num_of_steps(self, minV: float, maxV: float, dV: float) -> float:
        return round(np.abs(maxV - minV) / dV) + 1

    def set_smoothly(self, voltage_configuration: Dict[Gate | Source, float], timestep: float = 0.05) -> None:
        
        voltage_values = []
        maxsteps = 0
        dV = {}
        for gate, value in voltage_configuration.items():

            dV[gate.name] = value - gate.source()
            stepsize = gate.step
            num_steps = abs(int(np.ceil(dV[gate.name]/stepsize)))
            if num_steps > maxsteps:
                maxsteps = num_steps
        
        for step in range(maxsteps):
            voltage_values.append({})
            for gate, value in voltage_configuration.items():
                voltage_values[-1][gate.name] = value - dV[gate.name] * ((maxsteps-step-1)/maxsteps)
         
        for voltages in voltage_values:
            for gate, value in voltage_configuration.items():
                gate.source(voltages[gate.name])
            time.sleep(timestep)
                
