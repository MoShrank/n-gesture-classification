import numpy as np

def get_GWN(dt, size, mu, sig, myseed=False):
    """
    Function that generates Gaussian white noise input

    Args:
      pars       : parameter dictionary
      mu         : noise baseline (mean)
      sig        : noise amplitute (standard deviation)
      myseed     : random seed. int or boolean
                   the same seed will give the same
                   random number sequence

    Returns:
      I          : Gaussian white noise input
    """

    # Set random seed
    if myseed:
        np.random.seed(seed=myseed)
    else:
        np.random.seed()

    # Generate GWN
    # we divide here by 1000 to convert units to sec.
    I_gwn = mu + sig * np.random.randn(size) / np.sqrt(dt / 1000.0)

    return I_gwn

def get_poisson_spike_train(rate, no_spikes, no_spike_trains, dt, seed=None):
    if seed:
        np.random.seed(seed=myseed)

    # generate uniformly distributed random variables
    u_rand = np.random.rand(no_spikes, no_spike_trains)

    # generate Poisson train
    poisson_train = 1.0 * (u_rand < rate * (dt / 1000.0))

    return poisson_train