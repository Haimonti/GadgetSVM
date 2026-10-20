
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import json
import torch
import numpy
import argparse
from collections import defaultdict
from scipy.spatial.distance import cdist
from numpy import dot
from numpy.linalg import norm
import scikit_wrappers_0 as scikit_wrappers
import pickle
from sklearn.model_selection import train_test_split
from sklearn.cluster import kmeans_plusplus
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import rand_score
from Common_functions import *
from sklearn.metrics import hinge_loss

import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):

    client_id = args.client_id;

    gpu = args.gpu
    in_channel = args.in_channel;
    cuda = args.cuda
    save_path = args.save_path
    encoder_weights_save_dir = args.save_path_encoder
    C=args.C
    SimCSE = args.SimCSE
    dropout_p = args.dropout_p
    data_dir = args.data_dir
    out_dir = args.out_dir
    


    with open(data_dir+'train_labeled_x_client_'+str(client_id), 'rb') as fp:
        local_train = pickle.load(fp)
    with open(data_dir+'train_labeled_y_client_'+str(client_id), 'rb') as fp:
        local_train_labels = pickle.load(fp)


    hyper = "default_hyperparameters.json"

    hf = open(os.path.join(hyper), 'r')
    params = json.load(hf)
    hf.close()
    # Check the number of input channels
    params['in_channels'] = in_channel
    params['cuda'] = cuda
    params['gpu'] = gpu
    params['Sim_CSE'] = SimCSE
    params['dropout_p'] = dropout_p
    params['compared_length'] = None

    local_model = scikit_wrappers.CausalCNNEncoderClassifier(**params)

    with open(encoder_weights_save_dir, 'rb') as fp:
        encoder_i_weight = pickle.load(fp)

    local_model.encoder.load_state_dict(encoder_i_weight)



    print("Encoding Feats")
    features = local_model.encode(local_train)
    print("features_obtained; encoder working")

    local_model.classifier = fit_classifier_hyperparameters(local_model, features, local_train_labels, C=C)
    torch.save(local_model, save_path + 'Personalized_SVM_for_client_' + str(client_id) + '.pt')

    local_model.encoder.eval()

    with open(data_dir+'test_x_client_'+str(client_id), 'rb') as fp:
        test = pickle.load(fp)
    with open(data_dir+'test_y_client_'+str(client_id), 'rb') as fp:
        test_labels = pickle.load(fp)


    test_label_dict = defaultdict(list)
    for k in range(4):
        test_label_dict[k] = []
        for j in range(len(test_labels)):
            if test_labels[j] == k:
                test_label_dict[k].append(j)

    feats = local_model.encode(test);
    acc = classifier_score_modded_feats(local_model, feats, test_labels)
    with open(out_dir + 'Fully_Personalized_SVM_acc_for_client_' + str(client_id)+'.pkl', 'wb') as fp:
        pickle.dump(acc, fp)

    for SNR in numpy.arange(-10,10):

        with open(data_dir+'test_x_client_' + str(client_id)+'_SNR_'+str(SNR), 'rb') as fp:
            test = pickle.load(fp)
        with open(data_dir+'test_y_client_' + str(client_id)+'_SNR_'+str(SNR), 'rb') as fp:
            test_labels = pickle.load(fp)

        test_label_dict = defaultdict(list)
        for k in range(4):
            test_label_dict[k] = []
            for j in range(len(test_labels)):
                if test_labels[j] == k:
                    test_label_dict[k].append(j)

        feats = local_model.encode(test);

        acc = classifier_score_modded_feats(local_model, feats, test_labels)
        with open(out_dir + 'Fully_Personalized_SVM_acc_for_client_' + str(client_id) + '_SNR_'+str(SNR)+ '.pkl', 'wb') as fp:
            pickle.dump(acc, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='Save_models/')
    parser.add_argument('--in_channel', type=int, default=2)
    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--client_id', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path_encoder', type=str, default='Save_models/Encoder_weights_saved_CFO_het')
    parser.add_argument('--C', type=float, default=float('inf'))
    parser.add_argument('--SimCSE', type=str2bool, default=False)
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--data_dir', type=str, default='./CFO_Het_Mixed_Data/')
    parser.add_argument('--out_dir', type=str, default='./Stats_and_Plot/')



    args = parser.parse_args()
    main(args)