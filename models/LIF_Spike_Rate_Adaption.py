from BaseNeuronModel import BaseNeuronModel


class LIF_Spike_Rate_Adaption(BaseNeuronModel):
    def __init__(
        self,
        E_k: float = -65.0,
        g_delta: float = 0.1,
        tau_g: float = 10.0,
        *args,
        **kwargs
    ):
        """
        Args:
            E_k(float): potassium reversal potential in mV
            g_delta(float): increase in sra conductance in nS
            tau_g(float): time constant of sra conductance in ms
        """
        super().__init__(*args, **kwargs)

        self.E_k = E_k
        self.g_delta = g_delta / self.tau_m
        self.tau_g = tau_g

        self.sra = [0.0]

    def step(self, input_current: float):
        v = self.voltages[-1]

        # check if voltage is above threshold
        if v >= self.v_th:
            self.spike_times += [len(self.voltages) - 1]
            v = self.v_reset

            # increase sra conductance
            d_sra = self.sra[-1] + (self.g_delta / self.tau_m)

        else:
            # relax sra conductance exponentially to 0
            d_sra = -self.sra[-1] * (self.dt / self.tau_g)

        # calculate membrane voltage change
        dv = (
            (self.dt / self.tau_m) * (self.E_l - v)
            - (self.tau_m * self.sra[-1] * (self.voltages[-1] - self.E_k))
            + (input_current / self.R)
        )

        v += dv

        self.sra += [self.sra[-1] + d_sra]
        self.voltages += [v]
