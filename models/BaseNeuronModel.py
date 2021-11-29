from abc import ABC, abstractmethod
from typing import Iterable

from util.plotting import plot_spikes, plot_voltage_trace


class BaseNeuronModel(ABC):
    """
    Base class for all neuron models.
    """

    def __init__(
        self,
        dt: float,
        E_l: float = -65.0,
        v_th: float = -55.0,
        v_reset: float = -75.0,
        tau_m: float = 10.0,
        R: float = 20.0,
    ):
        """
        Args:
            dt (float): time step size
            E_l (float): resting potential
            v_th (float): threshold voltage
            v_reset (float): reset voltage
            tau_m (float): membrane time constant
            R (float): membrane resistance ratio of capacitance and leak
        """

        self.dt = dt
        self.E_l = E_l
        self.v_th = v_th
        self.v_reset = v_reset
        self.tau_m = tau_m
        self.R = R

        self.spike_times = []
        self.voltages = [self.E_l]

    def plot_spikes(self, sim_length: int):
        if not len(self.spike_times):
            raise (ValueError("No spike times recorded"))

        plot_spikes(self.spike_times, self.dt, sim_length)

    def plot(self, sim_length: int):
        if not len(self.voltages):
            raise (ValueError("No voltages recorded"))

        plot_voltage_trace(self.voltages, self.v_th, self.dt, sim_length)

    def insert_spike_train(self, spike_train: Iterable[float]):
        for inp_current in spike_train:
            self.step(inp_current)

    @abstractmethod
    def step(self):
        pass
