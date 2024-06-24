from ..utils.acquisition import *
from ..utils.analysis import *
from ..utils.fit_functions import logarithmic, sigmoid
from ..device import *


import pathlib, time

import scipy as sp
import matplotlib.pyplot as plt
from pathlib import Path

class SingleQD(Analysis):
    def __init__(self, device: Device) -> None:
        super().__init__()
        
        self.device = device

    def fit_turn_on(self, data: pd.DataFrame) -> tuple:
        mask = data.iloc[:, -1].abs() > self.device.minimum_current_threshold
    
        if not mask.any():
            print(f"Did not cross minimum threshold of {self.device.minimum_current_threshold} A")
            return None

        # Only keep points above the threshold
        data_masked = data[mask]
        X = data_masked.iloc[:,0]
        Y = data_masked.iloc[:,-1]
        guess = [Y.iloc[0], self.device.polarity, X.iloc[-1] - self.device.polarity, 0]

        try:
            params, cov = self.fit(
                x=X,
                y=Y,
                func=logarithmic,
                guess=guess
            )
        except RuntimeError:
            print("Unable to fit turn-on curve.")
            return None

        a, b, x0, y0 = params

        self.device.turn_on_voltage = np.round(np.exp(-y0/a)/b + x0, 3)
        self.device.saturation_voltage = X.iloc[-1]
        self.device.saturation_current = Y.iloc[-1]
        self.device.turns_on = True

        return params
    
    def fit_pinch_off(self, data: pd.DataFrame) -> tuple:
        mask = data.iloc[:, -1].abs() > self.device.minimum_current_threshold
    
        if not mask.any():
            print(f"Unable to pinch off below {self.device.minimum_current_threshold} A")
            return None

        # Only keep points above the threshold
        data_masked = data[mask]
        X = data_masked.iloc[:,0]
        Y = data_masked.iloc[:,-1]
        guess = [Y.iloc[0], -1 * self.device.polarity * 100, self.device.turn_on_voltage, 0]

        try:
            params, cov = self.fit(
                x=X,
                y=Y,
                func=sigmoid,
                guess=guess
            )
            a, b, x0, y0 = params
        
            gate = getattr(self.device, X.name, None)
            gate.pinch_off_voltage = float(round(min(
                            np.abs(x0 - np.sqrt(8) / b),
                            np.abs(x0 + np.sqrt(8) / b)
                        ),3))
            gate.pinch_off_width = float(abs(round(2 * np.sqrt(8) / b,2)))

            return params
    
        except RuntimeError:
            print("Unable to fit sigmoid pinch-off curve.")
            return None

        

class SingleChargeTransistorTuner:
    def __init__(self, 
                 acquisition: Acquisition,
                 device: Device, 
                 save_dir: Path) -> None:

        self.device = device
        self.acquisition = acquisition
        self.analysis = SingleQD(self.device)

        # Unpack device geometry for automated usage
        self.source_ohmic = next(iter(self.device.sources.values())) # Get first, should only be one for SCT
        self.drain_ohmic = next(iter(self.device.drains.values()))

        barrier_gates = [gate for name, gate in self.device.barriers.items()]
        plunger_gates = [gate for name, gate in self.device.plungers.items()]
        top_gates = [gate for name, gate in self.device.tops.items()]

        assert len(barrier_gates) == 2, "Tuner expects two barrier gates confining the quantum dot."
        assert len(top_gates) == 1, "Tuner expects one plunger gate by the quantum dot."
        assert len(plunger_gates) == 1, "Tuner expects one top gate inducing charge carriers."

        self.barrier1_gate = barrier_gates[0]
        self.barrier2_gate = barrier_gates[1]
        self.plunger_gate = plunger_gates[0]
        self.top_gate = top_gates[0]

        self.all_connections = [self.source_ohmic, self.drain_ohmic, self.barrier1_gate, self.plunger_gate, self.barrier2_gate, self.top_gate]
        self.all_gates = [self.source_ohmic, self.barrier1_gate, self.plunger_gate, self.barrier2_gate, self.top_gate]
        source_full_names = [connection.source.full_name for connection in self.all_connections]
        gate_names = [connection.name for connection in self.all_connections]
        self.source_to_gate_mapping = dict(zip(source_full_names, gate_names))

    def ground_device(self) -> None:
        voltage_configuration = dict(zip(self.all_gates, [0.0] * len(self.all_gates)))
        self.acquisition.set_smoothly(voltage_configuration)

    def apply_bias(self, voltage: float) -> None:
        voltage_configuration = {self.source_ohmic: voltage}
        self.acquisition.set_smoothly(voltage_configuration)

    def turn_on(self,
                minV: float = 0,
                maxV: float = None,
                dV: float = 0.05,
                delay: float = 0.01) -> pd.DataFrame:

        # Check all inputs for proper signs
        assert np.sign(minV) == 0 or np.sign(minV) == self.device.polarity, "Check minimum voltage"

        # Establish gates involved
        gates = [self.barrier1_gate, self.top_gate, self.barrier2_gate]

        # Slowly ramp towards minV before starting sweep
        voltage_configuration = dict(zip(gates, [minV] * len(gates)))
        self.acquisition.set_smoothly(voltage_configuration)

        # Get maximum allowed voltage to sweep to (minimum of the gate maximums)
        gates_bounds = np.array([gate.bounds for gate in gates])
        gate_maximums = gates_bounds[:, 1]
        if maxV is None:
            maxV = gate_maximums[abs(gate_maximums).argmin()]
            abs_maxV = gate_maximums[abs(gate_maximums).argmax()]
        assert np.sign(maxV) == self.device.polarity, "Check maximum voltage"

        # Need to sweep relevant gates together for turn on
        num_steps = self.acquisition.calculate_num_of_steps(minV, maxV, dV)
        sweep_list = []
        for gate in gates:
            sweep_list.append(
                qc.dataset.LinSweep(gate.source, minV, maxV, num_steps, delay)
            )
        sweep = qc.dataset.TogetherSweep(*sweep_list)

        # Measure current as all channel gates are swept together
        data = self.acquisition.iv_sweep(
            sweep,
            self.drain_ohmic,
            measurement_name="Device Turn On",
            break_condition=self.max_current_check
        )

        # Clean up data to make it understandable
        for source_name, gate_name in self.source_to_gate_mapping.items():
            data = data.rename(columns={source_name: gate_name})
        data.iloc[:,-1] = data.iloc[:,-1].subtract(self.drain_ohmic.offset).mul(self.drain_ohmic.scale)

        # Plot raw data
        axes = data.plot.scatter(y=self.drain_ohmic.name, x=self.top_gate.name, marker='o', s=10)
        axes.set_title("Device Turn-On")
        data.plot.line(y=self.drain_ohmic.name, x=self.top_gate.name, ax=axes, linewidth=1)
        axes.set_ylabel(r'$I$ (A)')
        axes.set_xlabel(r'$V_{GATES}$ (V)')
        axes.axhline(y=self.device.minimum_current_threshold, alpha=0.5, c='g', linestyle=':', label=r'$I_{\min}$')
        axes.axhline(y=-self.device.minimum_current_threshold, alpha=0.5, c='g', linestyle=':')
        axes.axhline(y=-self.device.maximum_current_threshold, alpha=0.5, c='g', linestyle='')
        axes.axhline(y=self.device.maximum_current_threshold, alpha=0.5, c='g', linestyle='--', label=r'$I_{\max}$')

        # Fit data to theoretical logarithmic fit and extract parameters
        params = self.analysis.fit_turn_on(data)
        if params is not None:
            axes.plot(data.iloc[:,0], logarithmic(data.iloc[:,0], *params), 'r-')
            axes.axvline(x=self.device.turn_on_voltage, alpha=0.5, linestyle=':',c='b',label=r'$V_{ON}$')
            axes.axvline(x=self.device.saturation_voltage,alpha=0.5, linestyle='--',c='b',label=r'$V_{SAT}$')
        
        axes.legend(loc='best')
        
        return data
    
    def pinch_off(self,
                  gates: Gate | List[Gate] = None,
                  maxV: float = None,
                  minV: float = None,
                  delay: float = 0.01,
                  voltage_configuration: Dict[Gate | Ohmic, float] = {}) -> List[pd.DataFrame]:
        
        if gates is None:
            gates = [self.barrier1_gate, self.barrier2_gate]

        if minV is None: 
            startV = self.device.saturation_voltage

        self.acquisition.set_smoothly(voltage_configuration)

        if not isinstance(gates, list):
            gates = [gates]

        data_list = []
        for gate in gates:
            
            if maxV is None:
                finalV = gate.bounds[0] 

            num_steps = self.acquisition.calculate_num_of_steps(finalV, startV, gate.step)
            sweep = qc.dataset.LinSweep(gate.source, startV, finalV, num_steps, delay)
            
            data = self.acquisition.iv_sweep(
                sweep,
                self.drain_ohmic,
                measurement_name=f"{gate.name} Pinch Off"
            )

            # Clean up data to make it understandable
            for source_name, gate_name in self.source_to_gate_mapping.items():
                data = data.rename(columns={source_name: gate_name})
            data.iloc[:,-1] = data.iloc[:,-1].subtract(self.drain_ohmic.offset).mul(self.drain_ohmic.scale)

            axes = data.plot.scatter(y=self.drain_ohmic.name, x=gate.name, marker='o', s=10)
            axes.set_title(f"{gate.name} Pinch Off")
            data.plot.line(y=self.drain_ohmic.name, x=gate.name, ax=axes, linewidth=1)
            axes.set_ylabel(r'$I$ (A)')
            axes.set_xlabel(r'$V_{{{gate}}}$ (V)'.format(gate=gate.name))
            axes.set_xlim(0, self.device.saturation_voltage)
            axes.axhline(y=self.device.minimum_current_threshold, alpha=0.5, c='g', linestyle=':', label=r'$I_{\min}$')
            axes.axhline(y=-self.device.minimum_current_threshold, alpha=0.5, c='g', linestyle=':')
            axes.axhline(y=-self.device.maximum_current_threshold, alpha=0.5, c='g', linestyle='')
            axes.axhline(y=self.device.maximum_current_threshold, alpha=0.5, c='g', linestyle='--', label=r'$I_{\max}$')

            self.acquisition.set_smoothly({gate: startV})

            if isinstance(gate, Barrier):
                params = self.analysis.fit_pinch_off(data)
                if params is not None:
                    axes.plot(data.iloc[:,0], sigmoid(data.iloc[:,0], *params), 'r-')
                    axes.axvline(x=gate.pinch_off_voltage, alpha=0.5, linestyle=':', c='b', label=r'$V_{\min}$')
                    axes.axvline(x=gate.pinch_off_voltage + self.device.polarity * gate.pinch_off_width, alpha=0.5, linestyle='--', c='b', label=r'$V_{\max}$')

            axes.axvline(x=self.device.turn_on_voltage, alpha=0.5, linestyle=':',c='b',label=r'$V_{ON}$')
            axes.axvline(x=self.device.saturation_voltage,alpha=0.5, linestyle='--',c='b',label=r'$V_{SAT}$')
            axes.legend(loc='best')

            data_list.append(data)

        return data_list

    def sweep_barriers(self, 
                       delay: float = 0.01,
                       voltage_configuration: Dict[Gate | Ohmic, float] = {}) -> pd.DataFrame:
        
        self.acquisition.set_smoothly(voltage_configuration)

        b1_bounds = (self.barrier1_gate.pinch_off_voltage, self.barrier1_gate.pinch_off_voltage + self.device.polarity * self.barrier1_gate.pinch_off_width)
        b2_bounds = (self.barrier2_gate.pinch_off_voltage, self.barrier2_gate.pinch_off_voltage + self.device.polarity * self.barrier2_gate.pinch_off_width)

        b1_num_steps = self.acquisition.calculate_num_of_steps(b1_bounds[1], b1_bounds[0], self.barrier1_gate.step)
        b2_num_steps = self.acquisition.calculate_num_of_steps(b1_bounds[1], b1_bounds[0], self.barrier1_gate.step)

        b1_sweep = qc.dataset.LinSweep(self.barrier1_gate.source, b1_bounds[1], b1_bounds[0], b1_num_steps, delay)
        b2_sweep = qc.dataset.LinSweep(self.barrier2_gate.source, b2_bounds[1], b2_bounds[0], b2_num_steps, delay)

        data = self.acquisition.iv_sweep_2d(b1_sweep, b2_sweep, self.drain_ohmic, measurement_name="Barrier Barrier Sweep")

        # Clean up data to make it understandable
        for source_name, gate_name in self.source_to_gate_mapping.items():
            data = data.rename(columns={source_name: gate_name})
        data.iloc[:,-1] = data.iloc[:,-1].subtract(self.drain_ohmic.offset).mul(self.drain_ohmic.scale)

        data_pivoted = data.pivot_table(values=self.drain_ohmic.name, index=[self.barrier1_gate.name], columns=[self.barrier2_gate.name])
        B1_data, B2_data = data_pivoted.columns, data_pivoted.index
        raw_current_data = data_pivoted.to_numpy()[:,:-1] / 1.0e-9 # convert to nA
        B1_grad = np.gradient(raw_current_data, axis=1)
        B2_grad = np.gradient(raw_current_data, axis=0)

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,10))
        fig.suptitle("Barrier Barrier Sweep")

        im_ratio = 1

        cbar_ax1 = plt.colorbar(ax1.imshow(
            raw_current_data,
            extent=[B1_data[0], B1_data[-1], B2_data[0], B2_data[-1]],
            origin='lower',
            cmap='coolwarm',
            aspect=im_ratio
        ), ax=ax1,fraction=0.046, pad=0.04)

        cbar_ax1.set_label(r'$I_{SD}$ (nA)')
        ax1.set_title(r'$I_{SD}$')
        ax1.set_xlabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=self.barrier2_gate.name))
        ax1.set_ylabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=self.barrier1_gate.name))

        # V grad is actually horizontal
        grad_vector = (1,1) # Takes the gradient along 45 degree axis

        cbar_ax2 = plt.colorbar(ax2.imshow(
            np.sqrt(grad_vector[0] * B1_grad**2 +   grad_vector[1]* B2_grad**2),
            extent=[B1_data[0], B1_data[-1], B2_data[0], B2_data[-1]],
            origin='lower',
            cmap='coolwarm',
            aspect=im_ratio
        ), ax=ax2,fraction=0.046, pad=0.04)

        cbar_ax2.set_label(r'$\nabla_{\theta=45\circ} I_{SD}$ (nA/V)')
        ax2.set_title(r'$\nabla I_{SD}$')
        ax2.set_xlabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=self.barrier2_gate.name))
        ax2.set_ylabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=self.barrier1_gate.name))

        fig.tight_layout()

        return data
    
    def coulomb_diamonds(self,
                        delay: float = 0.01,
                        voltage_configuration: Dict[Gate | Ohmic, float] = {}) -> pd.DataFrame:
        
        self.acquisition.set_smoothly(voltage_configuration)

        p_bounds = self.plunger_gate.bounds
        s_bounds = self.source_ohmic.bounds

        p_num_steps = self.acquisition.calculate_num_of_steps(p_bounds[1], p_bounds[0], self.plunger_gate.step)
        s_num_steps = self.acquisition.calculate_num_of_steps(s_bounds[1], s_bounds[0], self.source_ohmic.step)

        p_sweep = qc.dataset.LinSweep(self.plunger_gate.source, p_bounds[1], p_bounds[0], p_num_steps, delay)
        s_sweep = qc.dataset.LinSweep(self.source_ohmic.source, s_bounds[1], s_bounds[0], s_num_steps, delay)

        data = self.acquisition.iv_sweep_2d(p_sweep, s_sweep, self.drain_ohmic, measurement_name="Coulomb Diamonds")

        # Clean up data to make it understandable
        for source_name, gate_name in self.source_to_gate_mapping.items():
            data = data.rename(columns={source_name: gate_name})
        data.iloc[:,-1] = data.iloc[:,-1].subtract(self.drain_ohmic.offset).mul(self.drain_ohmic.scale)

        data_pivoted = data.pivot_table(values=self.drain_ohmic.name, index=[self.source_ohmic.name], columns=[self.plunger_gate.name])
        s_data, p_data = data_pivoted.columns, data_pivoted.index
        raw_current_data = data_pivoted.to_numpy()[:,:-1] / 1.0e-9 # convert to nA
        s_grad = np.gradient(raw_current_data, axis=1)
        p_grad = np.gradient(raw_current_data, axis=0)

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,10))
        fig.suptitle("Barrier Barrier Sweep")

        im_ratio = 'auto'

        cbar_ax1 = plt.colorbar(ax1.imshow(
            raw_current_data,
            extent=[s_data[0], s_data[-1], p_data[0], p_data[-1]],
            origin='lower',
            cmap='coolwarm',
            aspect=im_ratio
        ), ax=ax1,fraction=0.046, pad=0.04)

        cbar_ax1.set_label(r'$I_{SD}$ (nA)')
        ax1.set_title(r'$I_{SD}$')
        ax1.set_xlabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=self.plunger_gate.name))
        ax1.set_ylabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=self.source_ohmic.name))

        # V grad is actually horizontal
        grad_vector = (1,1) # Takes the gradient along 45 degree axis

        cbar_ax2 = plt.colorbar(ax2.imshow(
            np.sqrt(grad_vector[0] * s_grad**2 +   grad_vector[1]* p_grad**2),
            extent=[s_data[0], s_data[-1], p_data[0], p_data[-1]],
            origin='lower',
            cmap='coolwarm',
            aspect=im_ratio
        ), ax=ax2,fraction=0.046, pad=0.04)

        cbar_ax2.set_label(r'$\nabla_{\theta=45\circ} I_{SD}$ (nA/V)')
        ax2.set_title(r'$\nabla I_{SD}$')
        ax2.set_xlabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=self.plunger_gate.name))
        ax2.set_ylabel(r"$V_{{{gate_name}}}$ (V)".format(gate_name=self.source_ohmic.name))

        fig.tight_layout() 

        return data 

    
    def current_trace(self, 
                      f_sampling: int, 
                      t_capture: int, 
                      plot_psd: bool = False) -> pd.DataFrame:
        """Records current data from multimeter device at a given sampling rate
        for a given amount of time. 

        Args:
            f_sampling (int): Sampling rate (Hz)
            t_capture (int): Capture time (s)
            plot_psd (bool, optional): Plots power spectral density spectrum. Defaults to "False".
        
        Returns:
            df (pd.DataFrame): Return measurement data. 

        **Example**
        >>>QD_FET_Tuner.current_trace(
            f_sampling=1000,
            t_capture=60, 
            plot_psd=True
        )
        """

        time_param = qc.parameters.ElapsedTimeParameter('time')
        meas = qc.dataset.Measurement(exp=self.acquisition.experiment)
        meas.register_parameter(time_param)
        meas.register_parameter(self.drain_ohmic.source, setpoints=[time_param])

        with meas.run() as datasaver:
            time_param.reset_clock()
            elapsed_time = 0
            while elapsed_time < t_capture:
                elapsed_time = time_param.get()
                datasaver.add_result((self.drain_ohmic.source, self.drain_ohmic.source()),
                                (time_param, time_param()))
                time.sleep(1/f_sampling)
      
        data = datasaver.dataset.to_pandas_dataframe().reset_index()
        for source_name, gate_name in self.source_to_gate_mapping.items():
            data = data.rename(columns={source_name: gate_name})
        data.iloc[:,-1] = data.iloc[:,-1].subtract(self.drain_ohmic.offset).mul(self.drain_ohmic.scale)

        # Plot current v.s. time 
        axes = data.plot.scatter(y=self.drain_ohmic.name, x='time', marker= 'o',s=5)
        data.plot.line(y=self.drain_ohmic.name, x=f'time', ax=axes, linewidth=1)
        axes.set_ylabel(r'$I$ (A)')
        axes.set_xlabel(r'$t$ (s)')
        axes.set_title(rf'Current noise, $f_s={f_sampling}$ Hz, $t_\max={t_capture}$ s')
        plt.show()
        
        if plot_psd:
            # Plot noise spectrum
            t = data[f'time']
            I = data[self.drain_ohmic.name]
            f, Pxx = sp.signal.periodogram(I, fs=f_sampling, scaling='density')

            plt.loglog(f, Pxx)
            plt.xlabel(r'$\omega$ (Hz)')
            plt.ylabel(r'$S_I$ (A$^2$/Hz)')
            plt.title(r"Current noise spectrum")
            plt.show()

        return data
    
    def max_current_check(self):
        is_exceeding_max_current = np.abs(self.drain_ohmic.get_current()) > self.device.maximum_current_threshold
        return np.array([is_exceeding_max_current]).any()
