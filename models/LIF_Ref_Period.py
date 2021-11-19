from models.BaseNeuronModel import BaseNeuronModel


class LIF_Ref_Period(BaseNeuronModel):
    def __init__(self, t_ref: float, *args, **kwargs):
        """
        Args:
            t_ref(float): refractory period in ms
        """
        super().__init__(*args, **kwargs)

        self.t_ref = t_ref
        self.ref_count = 0

    def step(self, input_current: float):
        # check if neuron is in refractory period
        if self.ref_count > 0:
            v = self.v_reset
            self.ref_count -= 1

        else:
            v = self.voltages[-1]

            # check if voltage is above threshold
            if v >= self.v_th:
                self.spike_times += [len(self.voltages) - 1]
                v = self.v_reset

                self.ref_count = self.t_ref / self.dt

            # calculate membrane voltage change
            dv = (self.dt / self.tau_m) * (self.E_l - v) + (input_current / self.R)

            v += dv

        self.voltages += [v]
