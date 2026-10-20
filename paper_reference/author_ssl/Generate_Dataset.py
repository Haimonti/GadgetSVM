# prompt: generate random 16 QAM symbols using scipy

import numpy as np
import pickle
import os

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


############################ Generation Parameters ######################################
s=0.2; # Ratio of labeled data to Encoder Data
data_dir='./CFO_Het_Mixed_Data/'; # Where data is saved
os.makedirs(data_dir, exist_ok=True) # Create directory if it doesn't exist

client_dists_train = [[1000, 6000, 6000, 1000], [1000, 1000, 6000, 6000], [6000, 1000, 1000, 6000],
                                 [6000, 6000, 1000, 1000]]; # Number of training examples per modulation per client
client_dists_supervised_train = [[1000*s, 6000*s, 6000*s, 1000*s], [1000*s, 1000*s, 6000*s, 6000*s], [6000*s, 1000*s, 1000*s, 6000*s],
                                 [6000*s, 6000*s, 1000*s, 1000*s]] # Number of supervised training examples per modulation per client

client_CFO_list_lower=[0, 0.01, 0.1, 1]
client_CFO_list_upper=[0.01, 0.1, 1, 20]
CFO_dist=[[0.4, 0.4,0.1,0.1], [0.4, 0.1, 0.4, 0.1], [0.1, 0.4, 0.4, 0.1], [0.1, 0.1, 0.4, 0.4]] # CFO bin probabilities per client

client_dists_supervised_test = [[20, 120, 120, 20], [20, 20, 120, 120], [120, 20, 20, 120], [120, 120, 20, 20]] # Number of supervised test examples per modulation per client

num_clients = len(client_dists_train)
num_mods = len(client_dists_train[-1])
###############################################################################################


encoder_train_client = [];
encoder_train_labels_client = [];
train_supervised_client = []
train_supervised_labels_client = []
test_supervised_client = []
test_supervised_labels_client = []

for i in range(num_clients):
    encoder_train = [];
    encoder_train_labels = [];
    train_supervised = []
    train_supervised_labels = []
    test_supervised = []
    test_supervised_labels = []

    n_symb =384
    CFO_bin_probs = np.array(CFO_dist[i])

    for j in range(num_mods):
        for ex in range(int(client_dists_train[i][j])):

            CFO_bin= np.random.choice(np.arange(len(CFO_bin_probs)),1,p=CFO_bin_probs)
            CFO= np.random.uniform(client_CFO_list_lower[CFO_bin[0]], client_CFO_list_upper[CFO_bin[0]])
            
            
            syms = Generate_symbols(modulation_mode=j, SNR_mode='random', num_symbols=n_symb, rayleigh_scale=0.35,
                                    sanity_check=False, del_f=CFO)
            syms = np.concatenate((np.expand_dims(syms.real, 0), np.expand_dims(syms.imag, 0)), 0)
            encoder_train += [syms]
            encoder_train_labels += [j]

        for ex in range(int(client_dists_supervised_train[i][j])):
            CFO_bin= np.random.choice(np.arange(len(CFO_bin_probs)),1,p=CFO_bin_probs)
            CFO= np.random.uniform(client_CFO_list_lower[CFO_bin[0]], client_CFO_list_upper[CFO_bin[0]])

            syms = Generate_symbols(modulation_mode=j, SNR_mode='random', num_symbols=n_symb, rayleigh_scale=0.35,
                                    sanity_check=False, del_f=CFO)
            syms = np.concatenate((np.expand_dims(syms.real, 0), np.expand_dims(syms.imag, 0)), 0)
            train_supervised += [syms]
            train_supervised_labels += [j]

        for ex in range(int(client_dists_supervised_test[i][j])):
            CFO_bin= np.random.choice(np.arange(len(CFO_bin_probs)),1,p=CFO_bin_probs)
            CFO= np.random.uniform(client_CFO_list_lower[CFO_bin[0]], client_CFO_list_upper[CFO_bin[0]])

            syms = Generate_symbols(modulation_mode=j, SNR_mode='random', num_symbols=n_symb, rayleigh_scale=0.35,
                                    sanity_check=False, del_f=CFO)
            syms = np.concatenate((np.expand_dims(syms.real, 0), np.expand_dims(syms.imag, 0)), 0)
            test_supervised += [syms]
            test_supervised_labels += [j]

    encoder_train_client += [np.array(encoder_train)];
    encoder_train_labels_client += [np.array(encoder_train_labels)];
    train_supervised_client += [np.array(train_supervised)];
    train_supervised_labels_client += [np.array(train_supervised_labels)];
    test_supervised_client += [np.array(test_supervised)];
    test_supervised_labels_client += [np.array(test_supervised_labels)];

for i in range(num_clients):
    with open(data_dir+'train_encoder_x_client_' + str(i), 'wb') as fp:
        pickle.dump(encoder_train_client[i], fp)
    with open(data_dir+'train_encoder_y_client_' + str(i), 'wb') as fp:
        pickle.dump(encoder_train_labels_client[i], fp)

    with open(data_dir+'train_labeled_x_client_' + str(i), 'wb') as fp:
        pickle.dump(train_supervised_client[i], fp)
    with open(data_dir+'train_labeled_y_client_' + str(i), 'wb') as fp:
        pickle.dump(train_supervised_labels_client[i], fp)

    with open(data_dir+'test_x_client_' + str(i), 'wb') as fp:
        pickle.dump(test_supervised_client[i], fp)
    with open(data_dir+'test_y_client_' + str(i), 'wb') as fp:
        pickle.dump(test_supervised_labels_client[i], fp)



for SNR in np.arange(-10,10):
    test_supervised_client = []
    test_supervised_labels_client = []

    for i in range(num_clients):
        CFO_bin_probs = np.array(CFO_dist[i])


        test_supervised = []
        test_supervised_labels = []

        for j in range(num_mods):
            for ex in range(int(client_dists_supervised_test[i][j])):
                CFO_bin= np.random.choice(np.arange(len(CFO_bin_probs)),1,p=CFO_bin_probs)
                CFO= np.random.uniform(client_CFO_list_lower[CFO_bin[0]], client_CFO_list_upper[CFO_bin[0]])
                syms = Generate_symbols(modulation_mode=j, SNR_mode='other', num_symbols=n_symb, rayleigh_scale=0.35,
                                        sanity_check=False, SNR=SNR,del_f=CFO)
                syms = np.concatenate((np.expand_dims(syms.real, 0), np.expand_dims(syms.imag, 0)), 0)
                test_supervised += [syms]
                test_supervised_labels += [j]


        test_supervised_client += [np.array(test_supervised)];
        test_supervised_labels_client += [np.array(test_supervised_labels)];

    for i in range(num_clients):

        with open(data_dir+'test_x_client_' + str(i)+'_SNR_'+str(SNR), 'wb') as fp:
            pickle.dump(test_supervised_client[i], fp)
        with open(data_dir+'test_y_client_' + str(i)+'_SNR_'+str(SNR), 'wb') as fp:
            pickle.dump(test_supervised_labels_client[i], fp)
