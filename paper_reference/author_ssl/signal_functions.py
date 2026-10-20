"""Function-only extraction from pinned Generate_Dataset.py; no module-level file writes.
Source: usmanakua927/FedSSL-AMC_Github @ 7cf7dc290aa208ef525c56b4591a501e547b8abf.
"""
import numpy as np

def generate_16qam_symbols(num_symbols):
    """Generates random 16-QAM symbols.

    Args:
      num_symbols: The number of symbols to generate.

    Returns:
      A NumPy array of complex numbers representing the 16-QAM symbols.
    """

    # Generate random integers between 0 and 15
    symbols_index = np.random.randint(0, 16, num_symbols)

    # Map integers to 16-QAM constellation points
    # The constellation points are arranged in a 4x4 grid
    constellation = 1 / np.sqrt(10) * np.array([
        -3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j,
        -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j,
        1 - 3j, 1 - 1j, 1 + 1j, 1 + 3j,
        3 - 3j, 3 - 1j, 3 + 1j, 3 + 3j
    ])

    qam_symbols = constellation[symbols_index]

    return qam_symbols

def generate_PSK_symbols(num_symbols, K):
    symbols_index = np.random.randint(0, K, num_symbols)

    if K != 2:
        constellation = np.exp(2j * np.pi * np.arange(0, K) / K + 1j * np.pi / K)
    else:
        constellation = np.exp(2j * np.pi * np.arange(0, K) / K)

    PSK_symbols = constellation[symbols_index]

    return PSK_symbols

def Generate_symbols(modulation_mode=1, SNR_mode='random', SNR=0, num_symbols=1024, rayleigh_scale=0.35,
                     sanity_check=True, del_f = 0.01):
    if modulation_mode == 3:
        syms = generate_16qam_symbols(num_symbols)
    elif modulation_mode == 0:
        syms = generate_PSK_symbols(num_symbols, 2)
    elif modulation_mode == 1:
        syms = generate_PSK_symbols(num_symbols, 4)
    elif modulation_mode==2:
        syms = generate_PSK_symbols(num_symbols, 8)

    A = np.random.rayleigh(rayleigh_scale, 1)
    if SNR_mode == 'random':
        SNR = np.random.uniform(-10, 10)

    del_theta = np.random.uniform(0, np.pi / 16)

    time_series_example_noiseless = A * np.exp(
        1j * 2 * np.pi * del_f * np.arange(0, num_symbols) / num_symbols + 1j * del_theta) * syms;
    time_series_power = np.mean(np.abs(time_series_example_noiseless) ** 2)
    w_n = np.sqrt(10 ** (-SNR / 10) * time_series_power / 2) * (
                np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols));
    time_series_example = time_series_example_noiseless + w_n;
    if sanity_check:
        empirical_SNR = 10 * np.log10(np.mean(np.abs(time_series_example_noiseless) ** 2) / np.mean(np.abs(w_n) ** 2))
        print('Aimed SNR: ', SNR, 'dB')
        print('Empirical SNR: ', empirical_SNR, 'dB')

    return time_series_example
