from typing import Iterable

from util.plotting import plot_voltage_trace

from models.BaseNeuronModel import BaseNeuronModel


class LIF(BaseNeuronModel):
    def step(self, input_current: float):
        v = self.voltages[-1]

        # check if voltage is above threshold
        if v >= self.v_th:
            self.spike_times += [len(self.voltages) - 1]
            v = self.v_reset

        # calculate membrane voltage change
        dv = (self.dt / self.tau_m) * (self.E_l - v) + (input_current / self.R)

        v += dv

        self.voltages += [v]
