import os
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
from Common_functions import fit_encoder_hyperparameters, fit_classifier_hyperparameters, classifier_score_modded_feats
from sklearn.model_selection import train_test_split
from sklearn.cluster import kmeans_plusplus
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import rand_score
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
    encoder_runs = args.encoder_runs;
    in_channel = args.in_channel;
    cuda = args.cuda
    save_path = args.save_path
    encoder_weights_save_dir = args.save_path_encoder
    comapared_length = args.compared_length
    SimCSE = args.SimCSE
    dropout_p = args.dropout_p
    data_dir = args.data_dir
    out_dir = args.out_dir

    print(['Client_id:', str(client_id), ' and gpu:',  str(gpu), 'and runs', str(encoder_runs)])

    with open(data_dir+'train_encoder_x_client_'+str(client_id), 'rb') as fp:
        train = pickle.load(fp)

    with open(data_dir+'train_encoder_y_client_'+str(client_id), 'rb') as fp:
        train_labels = pickle.load(fp)

    local_train=train.astype(numpy.float64);
    local_train_labels=train_labels;
    del train, train_labels
    hyper = "default_hyperparameters.json"

    hf = open(os.path.join(hyper), 'r')
    params = json.load(hf)
    hf.close()
    # Check the number of input channels
    params['in_channels'] = in_channel
    params['cuda'] = cuda
    params['gpu'] = gpu
    params['compared_length'] = comapared_length
    params['Sim_CSE'] = SimCSE
    params['dropout_p'] = dropout_p


    local_model = scikit_wrappers.CausalCNNEncoderClassifier(**params)

    # local_model.set_params(**params)
    local_model.encoder.train()

    if encoder_runs==0:
        encoder_i_weight = None
    else:

        with open(encoder_weights_save_dir, 'rb') as fp:
            encoder_i_weight=pickle.load(fp)

    local_model.encoder = fit_encoder_hyperparameters(local_model, local_train, encoder_i_weight, cuda,gpu)
    torch.save(local_model, save_path + '_Model_for_client_'+str(client_id)+'.pt')
    del local_train, local_train_labels

    if encoder_runs>0:
        
        with open(data_dir+'train_labeled_x_client_'+str(client_id), 'rb') as fp:
            local_train = pickle.load(fp)
        with open(data_dir+'train_labeled_y_client_'+str(client_id), 'rb') as fp:
            local_train_labels = pickle.load(fp)

        features = local_model.encode(local_train)
        local_model.classifier = fit_classifier_hyperparameters(local_model, features, local_train_labels, C=float('inf'))




        local_model.encoder.eval()

        with open(data_dir+'test_x_client_'+str(client_id), 'rb') as fp:
            test = pickle.load(fp)
        with open(data_dir+'test_y_client_'+str(client_id), 'rb') as fp:
            test_labels = pickle.load(fp)


        feats = local_model.encode(test);
        enc_to_del = local_model.encoder
        local_model.encoder = None
        del enc_to_del
        acc = classifier_score_modded_feats(local_model, feats, test_labels)
        print("Round: "+str(encoder_runs),", Client: ", client_id, ", SVM Acc: ", acc)
        with open(out_dir + 'Round_end_Fully_Personalized_SVM_acc_for_client_' + str(client_id)+'.pkl', 'wb') as fp:
            pickle.dump(acc, fp)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='Save_models/')
    parser.add_argument('--in_channel', type=int, default=2)
    parser.add_argument('--compared_length', type=int, default=None)
    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--client_id', type=int, default=  0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--encoder_runs', type=int, default=0)
    parser.add_argument('--save_path_encoder', type=str, default='Encoder_local_0.pt')
    parser.add_argument('--SimCSE', type=str2bool, default=True)
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--data_dir', type=str, default='./CFO_Het_Mixed_Data/')
    parser.add_argument('--out_dir', type=str, default='./Stats_and_Plot/')



    args = parser.parse_args()
    main(args)