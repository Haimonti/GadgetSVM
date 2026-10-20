import os
import json
import copy
import numpy
import torch.nn as nn
import torch
import scikit_wrappers_0 as scikit_wrappers
from scipy.optimize import linear_sum_assignment
from collections import Counter
from collections import defaultdict
from numpy import dot
from numpy.linalg import norm
import torch.nn.functional as F
import pickle


def fit_hyperparameters(file, train, train_labels, cuda, gpu,
                        save_memory=False):
    """
    Creates a classifier from the given set of hyperparameters in the input
    file, fits it and return it.

    @param file Path of a file containing a set of hyperparemeters.
    @param train Training set.
    @param train_labels Labels for the training set.
    @param cuda If True, enables computations on the GPU.
    @param gpu GPU to use if CUDA is enabled.
    @param save_memory If True, save GPU memory by propagating gradients after
           each loss term, instead of doing it after computing the whole loss.
    """
    classifier = scikit_wrappers.CausalCNNEncoderClassifier()

    # Loads a given set of hyperparameters and fits a model with those
    hf = open(os.path.join(file), 'r')
    params = json.load(hf)
    hf.close()
    # Check the number of input channels
    params['in_channels'] = numpy.shape(train)[1]
    params['cuda'] = cuda
    params['gpu'] = gpu
    classifier.set_params(**params)
    return classifier.fit(
        train, train_labels, save_memory=save_memory, verbose=True
    )


def classifier_agg(local_classifier_coef):
    classifier_coef = copy.deepcopy(local_classifier_coef[0])
    num = len(local_classifier_coef)
    for k in range(num):
        if k == 0:
            classifier_coef = numpy.asarray(local_classifier_coef[k])
            output = classifier_coef / num
        else:
            classifier_coef = numpy.asarray(local_classifier_coef[k])
            output += classifier_coef / num

    return output


def compute_weight_dist(w1, w2):
    output = []
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    for key in w1.keys():
        w1_weight = w1[key].reshape(-1, 1)
        w2_weight = w2[key].reshape(-1, 1)
        # output.append(torch.norm(w1_weight - w2_weight))
        dist = (cos(w1_weight, w2_weight) + 1) / 2
        output.append(dist[0].item())
    return sum(output)


def compute_numpy_dist(w1, w2):
    output = []
    w1 = w1.flatten()
    w2 = w2.flatten()
    cos_sim = dot(w1, w2) / (norm(w1) * norm(w2))
    output.append(cos_sim)
    return sum(output)


def k_means_dist(i, C_t, W_t):
    # similarity matrix W_t=[wij] consisting of all pairs of dot products
    C_t_count = Counter(C_t)
    card_C = C_t_count[C_t[i]]

    dist = []
    for k in list(set(C_t)):
        square_sum = 0
        block_idx = []
        for s in range(len(C_t)):
            if k == C_t[s]:
                block_idx.append(s)
        for j in block_idx:
            for s in block_idx:
                square_sum += W_t[j][s]

        dist.append(W_t[i][i] - sum([2 * W_t[i][j] if C_t[j] == k else 0 for j in range(len(W_t))]) / C_t_count[
            k] + square_sum / (C_t_count[k] ** 2))
    # dist = W_t[i][i] - sum([2*W_t[i][j] if C_t[i] == C_t[j] else 0 for j in range(len(W_t))])/card_C + square_sum/(card_C**2)
    return dist


def k_means_dist_mine(i, C_t, W_t):
    # similarity matrix W_t=[wij] consisting of all pairs of dot products
    C_t_count = Counter(C_t)
    card_C = C_t_count[C_t[i]]

    dist = []

    w_ii = W_t[i,i];
    for k in list(set(C_t)):
        cluster_members=list(numpy.where(numpy.array(C_t) == k)[-1]);
        second_term=-2*sum(W_t[i,cluster_members])/card_C;
        third_term=0
        for j in cluster_members:
            third_term+=sum(W_t[j,cluster_members])/(card_C**2);

        dist+=[w_ii+second_term+third_term];

    return dist


def block_sum(i, j, C_t, W_t, same_cluster):
    sum = 0

    if same_cluster == True:
        block_idx = []
        for k in range(len(C_t)):
            if C_t[k] == C_t[i]:
                block_idx.append(k)
        for k in block_idx:
            for s in block_idx:
                if k != s:
                    sum += W_t[k][s]
        return sum

    elif same_cluster == False:
        block_idx_i = []
        block_idx_j = []
        for k in range(len(C_t)):
            if C_t[k] == C_t[i]:
                block_idx_i.append(k)
            if C_t[k] == C_t[j]:
                block_idx_j.append(k)
        for k in block_idx_i:
            for s in block_idx_j:
                sum += W_t[k][s]
        return sum


def compute_E_hat_V_hat(C_t, W_t):
    C_t_count = Counter(C_t)
    E_hat_W_t = numpy.zeros_like(W_t)
    for i in range(len(W_t)):
        for j in range(len(W_t)):
            if C_t[i] == C_t[j]:
                card_C = C_t_count[C_t[i]]
                if i == j:
                    E_hat_W_t[i][j] = (1 / card_C) * sum(
                        [W_t[l][l] if C_t[l] == C_t[i] else 0 for l in range(len(W_t))])
                else:
                    E_hat_W_t[i][j] = block_sum(i, i, C_t, W_t, True) / (card_C * (card_C - 1))
            else:
                card_C = C_t_count[C_t[i]]
                card_D = C_t_count[C_t[j]]
                E_hat_W_t[i][j] = block_sum(i, j, C_t, W_t, False) / (card_C * card_D)

    V_hat_W_t = numpy.zeros_like(W_t)
    for i in range(len(W_t)):
        for j in range(len(W_t)):
            if C_t[i] == C_t[j]:
                card_C = C_t_count[C_t[i]]
                if i == j and card_C > 1:
                    V_hat_W_t[i][j] = (1 / (card_C - 1)) * sum(
                        [(W_t[l][l] - E_hat_W_t[l][l]) ** 2 if C_t[l] == C_t[i] else 0 for l in range(len(W_t))])
                else:
                    V_hat_W_t[i][j] = block_sum(i, i, C_t, (W_t - E_hat_W_t) ** 2, True) / (card_C * (card_C - 1) - 1)
            else:
                card_C = C_t_count[C_t[i]]
                card_D = C_t_count[C_t[j]]
                if card_C * card_D > 1:
                    V_hat_W_t[i][j] = block_sum(i, j, C_t, (W_t - E_hat_W_t) ** 2, False) / (card_C * card_D - 1)
                else:
                    V_hat_W_t[i][j] = block_sum(i, j, C_t, (W_t - E_hat_W_t) ** 2, False)
    return E_hat_W_t, V_hat_W_t


def assign_cluster_k_means(C_t, W_t):
    old_cluster = [C_t[i] for i in range(len(C_t))]
    for iter_t in range(100):
        new_cluster = []
        for i in range(len(C_t)):
            dist_c = numpy.asarray(k_means_dist(i, C_t, W_t))
            idx = numpy.argmin(dist_c)
            new_cluster.append(idx)
        if new_cluster == old_cluster:
            break
        else:
            old_cluster = new_cluster
    return new_cluster



def assign_cluster_k_means_mine(C_t, W_t, dist=numpy.nan):
    old_cluster_mine = [C_t[i] for i in range(len(C_t))]

    for iter_t in range(100):
        new_cluster_mine = []

        for i in range(len(C_t)):
            dist_c_mine = numpy.asarray(k_means_dist_mine(i, old_cluster_mine, W_t))

            idx_mine=numpy.argmin(dist_c_mine)
            new_cluster_mine.append(idx_mine)
            print(dist_c_mine)



        if new_cluster_mine == old_cluster_mine:
            print("Inside 'assign_cluster_k_means'. Clustering Converged after ", iter_t, " iterations")
            print("(Sanity Check) Old_cluster: ", old_cluster_mine)
            print("(Sanity Check) Old_cluster: ", new_cluster_mine)

            break
        else:
            old_cluster_mine = new_cluster_mine.copy();

    return new_cluster_mine


def Perform_agg_FLSC(clusters_list_main, cluster_aggs_main_dict, cluster_aggs_main_dict_intercept,
                C_t, alpha_1, alpha_2, local_models, num_clusters, num_clients):

    Flag=False;

    if cluster_aggs_main_dict==None:
        Flag=True

    if True:
        clusters_list = {j: [] for j in range(num_clusters)}
        for j in range(num_clients):
            C_t_idx_vec=C_t[j]
            for idx in C_t_idx_vec:
                clusters_list[idx].append(j)
        clusters_list_main = copy.deepcopy(clusters_list)


    cluster_aggs_t_dict = {};
    cluster_aggs_t_dict_intercept = {};


    for i in range(len(clusters_list_main)):
        if len(clusters_list_main[i])==0:
            continue
        weights_coef_list = [copy.deepcopy(local_models[j].classifier.coef_) for j in clusters_list_main[i]]
        weights_intercept_list = [copy.deepcopy(local_models[j].classifier.intercept_) for j in clusters_list_main[i]]

        # Update local classifier weights

        cluster_aggs_t_dict[i] = classifier_agg(weights_coef_list)
        cluster_aggs_t_dict_intercept[i] = classifier_agg(weights_intercept_list)

    if Flag:
        cluster_aggs_main_dict = copy.deepcopy(cluster_aggs_t_dict);
        cluster_aggs_main_dict_intercept = copy.deepcopy(cluster_aggs_t_dict_intercept)
    else:
        for i in range(len(clusters_list_main)):
            if len(clusters_list_main[i]) == 0:
                continue
            cluster_aggs_main_dict[i]=alpha_1*cluster_aggs_main_dict[i]+alpha_2*cluster_aggs_t_dict[i]
            cluster_aggs_main_dict_intercept[i] = alpha_1 * cluster_aggs_main_dict_intercept[i] + alpha_2 * cluster_aggs_t_dict_intercept[i]

    return clusters_list_main, cluster_aggs_main_dict, cluster_aggs_main_dict_intercept

def Perform_agg_IFCA(clusters_list_main, cluster_aggs_main_dict, cluster_aggs_main_dict_intercept,
                C_t, alpha_1, alpha_2, local_models, num_clusters):
    Flag=False;

    if cluster_aggs_main_dict==None:
        Flag=True


    if True:

        clusters_list = {j: [] for j in range(num_clusters)}
        for idx in range(len(C_t)):
            clusters_list[C_t[idx]].append(idx)
        clusters_list_main = copy.deepcopy(clusters_list)


    cluster_aggs_t_dict = {};
    cluster_aggs_t_dict_intercept = {};


    for i in range(len(clusters_list_main)):
        if len(clusters_list_main[i])==0:
            continue
        weights_coef_list = [copy.deepcopy(local_models[j].classifier.coef_) for j in clusters_list_main[i]]
        weights_intercept_list = [copy.deepcopy(local_models[j].classifier.intercept_) for j in clusters_list_main[i]]

        # Update local classifier weights

        cluster_aggs_t_dict[i] = classifier_agg(weights_coef_list)
        cluster_aggs_t_dict_intercept[i] = classifier_agg(weights_intercept_list)

    if Flag:
        cluster_aggs_main_dict = copy.deepcopy(cluster_aggs_t_dict);
        cluster_aggs_main_dict_intercept = copy.deepcopy(cluster_aggs_t_dict_intercept)
    else:
        for i in range(len(clusters_list_main)):
            if len(clusters_list_main[i]) == 0:
                continue
            cluster_aggs_main_dict[i]=alpha_1*cluster_aggs_main_dict[i]+alpha_2*cluster_aggs_t_dict[i]
            cluster_aggs_main_dict_intercept[i] = alpha_1 * cluster_aggs_main_dict_intercept[i] + alpha_2 * cluster_aggs_t_dict_intercept[i]

    return clusters_list_main, cluster_aggs_main_dict, cluster_aggs_main_dict_intercept

def Perform_agg(clusters_list_main, cluster_aggs_main_dict, cluster_aggs_main_dict_intercept, t, min_round_before_agg,
                C_t, local_models):
    if t >= min_round_before_agg:
        clusters = set(C_t)
        clusters_list = {j: [] for j in range(len(clusters))}
        for idx in range(len(C_t)):
            clusters_list[C_t[idx]].append(idx)

        cluster_aggs_t_dict = {};
        cluster_aggs_t_dict_intercept = {};
        feat_mat_combined = [];

        for i in range(len(clusters_list)):
            weights_coef_list = [copy.deepcopy(local_models[j].classifier.coef_) for j in clusters_list[i]]
            weights_intercept_list = [copy.deepcopy(local_models[j].classifier.intercept_) for j in clusters_list[i]]
            cluster_i_class_agg_t = classifier_agg(weights_coef_list)
            cluster_i_intercept_agg_t = classifier_agg(weights_intercept_list)
            # Update local classifier weights

            cluster_aggs_t_dict[i] = classifier_agg(weights_coef_list)
            cluster_aggs_t_dict_intercept[i] = classifier_agg(weights_intercept_list)
            feat_mat_combined += [list(numpy.concatenate(
                (local_models[i].classifier.coef_.flatten(), local_models[i].classifier.intercept_)))]

        feat_mat_combined = numpy.array(feat_mat_combined)

        if t == min_round_before_agg:

            cluster_aggs_main_dict = copy.deepcopy(cluster_aggs_t_dict);
            cluster_aggs_main_dict_intercept = copy.deepcopy(cluster_aggs_t_dict_intercept)
            clusters_list_main = copy.deepcopy(clusters_list)

        else:
            feat_mat_main = []
            for i in range(len(cluster_aggs_main_dict)):
                feat_mat_main += [list(numpy.concatenate(
                    (cluster_aggs_main_dict[i].flatten(), cluster_aggs_main_dict_intercept[i])))]
            feat_mat_main = numpy.array(feat_mat_main)
            for i in range(len(feat_mat_combined)):
                feat_mat_combined[i, :] = feat_mat_combined[i, :] / numpy.linalg.norm(feat_mat_combined[i, :])

            for i in range(len(feat_mat_main)):
                feat_mat_main[i, :] = feat_mat_main[i, :] / numpy.linalg.norm(feat_mat_main[i, :])

            assignment = linear_sum_assignment(numpy.matmul(feat_mat_main, feat_mat_combined.T), maximize=True)
            main_indicies = assignment[0]
            match_indicies = assignment[1]

            for i in range(len(main_indicies)):
                cluster_aggs_main_dict[main_indicies[i]] = ((t - min_round_before_agg) * cluster_aggs_main_dict[
                    main_indicies[i]] + cluster_aggs_t_dict[match_indicies[i]]) / (t - min_round_before_agg + 1)
                cluster_aggs_main_dict_intercept[main_indicies[i]] = ((t - min_round_before_agg) *
                                                                      cluster_aggs_main_dict_intercept[
                                                                          main_indicies[i]] +
                                                                      cluster_aggs_t_dict_intercept[
                                                                          match_indicies[i]]) / (
                                                                             t - min_round_before_agg + 1)



    return clusters_list_main, cluster_aggs_main_dict, cluster_aggs_main_dict_intercept




def Perform_agg_v2(clusters_list_main, cluster_aggs_main_dict, cluster_aggs_main_dict_intercept,
                C_t, alpha_1, alpha_2, local_models):
    Flag=False;


    if clusters_list_main == None:
        clusters = set(C_t)
        clusters_list = {j: [] for j in range(len(clusters))}
        for idx in range(len(C_t)):
            clusters_list[C_t[idx]].append(idx)
        clusters_list_main = copy.deepcopy(clusters_list)
        Flag=True

    cluster_aggs_t_dict = {};
    cluster_aggs_t_dict_intercept = {};


    for i in range(len(clusters_list_main)):
        weights_coef_list = [copy.deepcopy(local_models[j].classifier.coef_) for j in clusters_list_main[i]]
        weights_intercept_list = [copy.deepcopy(local_models[j].classifier.intercept_) for j in clusters_list_main[i]]

        # Update local classifier weights

        cluster_aggs_t_dict[i] = classifier_agg(weights_coef_list)
        cluster_aggs_t_dict_intercept[i] = classifier_agg(weights_intercept_list)

    if Flag:
        cluster_aggs_main_dict = copy.deepcopy(cluster_aggs_t_dict);
        cluster_aggs_main_dict_intercept = copy.deepcopy(cluster_aggs_t_dict_intercept)
    else:
        for i in range(len(clusters_list_main)):
            cluster_aggs_main_dict[i]=alpha_1*cluster_aggs_main_dict[i]+alpha_2*cluster_aggs_t_dict[i]
            cluster_aggs_main_dict_intercept[i] = alpha_1 * cluster_aggs_main_dict_intercept[i] + alpha_2 * cluster_aggs_t_dict_intercept[i]

    return clusters_list_main, cluster_aggs_main_dict, cluster_aggs_main_dict_intercept



def fit_encoder_hyperparameters(Classifier, train, encoder_weight, cuda, gpu,
                                save_memory=False):
    """
    Creates a classifier from the given set of hyperparameters in the input
    file, fits it and return it.

    @param file Path of a file containing a set of hyperparemeters.
    @param train Training set.
    @param train_labels Labels for the training set.
    @param cuda If True, enables computations on the GPU.
    @param gpu GPU to use if CUDA is enabled.
    @param save_memory If True, save GPU memory by propagating gradients after
           each loss term, instead of doing it after computing the whole loss.
    """
    # classifier = scikit_wrappers.CausalCNNEncoderClassifier()
    return Classifier.fit_encoder(
        train, encoder_weight, save_memory=save_memory, verbose=True
    )


def federated_avg(global_model, client_models):
    global_dict = global_model.state_dict()


    for key in global_dict.keys():
        global_dict[key] = 0. * global_dict[key];
        for i in range(len(client_models)):
            global_dict[key] += client_models[i].state_dict()[key] /len(client_models)
        # global_dict[key] = torch.stack([client_models[i].state_dict()[key] for i in range(len(client_models))],
        #                                dim=0).mean(dim=0)

    global_model.load_state_dict(global_dict)
    return global_model


@torch.no_grad()
def federated_avg_feddyn(global_model, client_models, encoder_runs, feddyn_alpha=0.01):
    global_dict = global_model.state_dict()
    
    diff_dict = copy.deepcopy(global_dict);
    if encoder_runs==0:
        h_dict=copy.deepcopy(global_dict);  
        for key in h_dict.keys():
            h_dict[key]=0*h_dict[key].to(client_models[0].state_dict()[key].device);
    
    else:
        with open('h_dict.pkl', 'rb') as fp:
            h_dict = pickle.load(fp)


    for key in global_dict.keys():
        
        diff_dict[key] = diff_dict[key].to(client_models[0].state_dict()[key].device);
        diff_dict[key]*=0;
        
        if global_dict[key].is_floating_point():    
            for i in range(len(client_models)):
                diff_dict[key]+=feddyn_alpha*1/len(client_models)*(client_models[i].state_dict()[key]-global_dict[key].to(client_models[i].state_dict()[key].device));
                
            h_dict[key]=h_dict[key]-diff_dict[key];
            global_dict[key] = 0. * global_dict[key].to(client_models[0].state_dict()[key].device);
            for i in range(len(client_models)):
                global_dict[key] += client_models[i].state_dict()[key] /len(client_models)
            # global_dict[key] = torch.stack([client_models[i].state_dict()[key] for i in range(len(client_models))],
            #                                dim=0).mean(dim=0)
            global_dict[key]-=(1/feddyn_alpha)*h_dict[key];
        else:
            global_dict[key]=client_models[0].state_dict()[key];

    with open('h_dict.pkl', 'wb') as fp:
        pickle.dump(h_dict, fp)
    global_model.load_state_dict(global_dict)
    return global_model

def modified_BCE_loss(output, target, num_classes=4):

    output=torch.exp(output)
    N_B_k_list=[];
    N_B_k=0;
    classes_present=[];
    for c in range(num_classes):
        N_B_k_m=0
        for k in range(len(target)):
            if target[k] == c:
                N_B_k_m+=1
        N_B_k+=N_B_k_m
        N_B_k_list.append(N_B_k_m)
        if N_B_k_m>0:
            classes_present.append(c)

    w_k_list=[];
    for c in range(num_classes):
        if c in classes_present:
            w_k_list.append(N_B_k/N_B_k_list[c])
        else:
            w_k_list.append(0)
    alpha_k_list=[w_k_list[c]/numpy.sum(w_k_list) for c in range(len(w_k_list))]



    loss=0;
    for k in range(len(target)):
        c=target[k]
        f=torch.tensor(alpha_k_list[c]).to(output[k,c].device);
        loss+=-1/len(target)*f*torch.log(output[k,c])



    return loss



def weights_agg(local_models_weights):
    agg_weights = copy.deepcopy(local_models_weights[0])
    for key in agg_weights.keys():
        agg_weights[key] = agg_weights[key] / len(local_models_weights)
    if len(local_models_weights) > 1:
        for i in range(1, len(local_models_weights)):
            for key in agg_weights.keys():
                agg_weights[key] += local_models_weights[i][key] / len(local_models_weights)
    return agg_weights


def fit_classifier_hyperparameters(Classifier, features, train_labels, C=None):
    """
    Creates a classifier from the given set of hyperparameters in the input
    file, fits it and return it.

    @param file Path of a file containing a set of hyperparemeters.
    @param train Training set.
    @param train_labels Labels for the training set.
    @param cuda If True, enables computations on the GPU.
    @param gpu GPU to use if CUDA is enabled.
    @param save_memory If True, save GPU memory by propagating gradients after
           each loss term, instead of doing it after computing the whole loss.
    """
    # classifier = scikit_wrappers.CausalCNNEncoderClassifier()

    # Loads a given set of hyperparameters and fits a model with those
    return Classifier.fit_classifier(
        features, train_labels, C=C
    )


def classifier_score(Classifier, test, test_labels):
    test_label_dict = defaultdict(list)
    for i in range(4):
        test_label_dict[i] = []
        for j in range(len(test_labels)):
            if test_labels[j] == i:
                test_label_dict[i].append(j)

    for i in range(4):
        test_per_class = numpy.asarray([test[j] for j in test_label_dict[i]])
        test_labels_per_class = numpy.asarray([test_labels[j] for j in test_label_dict[i]])
        print("Test accuracy per class: " + str(i) + " is " + str(
            Classifier.score(test_per_class, test_labels_per_class)))
    return


class CNN1D(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=128, kernel_size=16, padding=8)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=8, padding=4)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64 * (384+2), 256)  # Flattened after Conv layers
        self.bn3 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.dropout(x, p=0.1, training=self.training)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.bn4(self.fc2(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def classifier_score_modded(Classifier, test, test_labels):
    return Classifier.score(test, test_labels)

def classifier_score_modded_feats(Classifier, feats, test_labels):
    return Classifier.score_feats(feats, test_labels)

class Basic_LSTM_2(nn.Module):
    def __init__(self, input_size, feature_embedding_size, hidden_size=16, num_layers=1):
        super().__init__()
        self.input_size = input_size  # this is the number of features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=feature_embedding_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers
        )

        self.linear_1 = nn.Linear(in_features=self.input_size, out_features=feature_embedding_size)
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=10)

    def forward(self, x):
        x = self.linear_1(x)
        lstm_out, (h_out, _) = self.lstm(x)
        h_out = h_out.view(-1, self.hidden_size)
        out = self.linear_2(h_out)

        return out


def classifier_score(Classifier, test, test_labels):
    test_label_dict = defaultdict(list)
    for i in range(10):
        test_label_dict[i] = []
        for j in range(len(test_labels)):
            if test_labels[j] == i:
                test_label_dict[i].append(j)


    acc_to_return=[];

    for i in range(10):
        test_per_class = numpy.asarray([test[j] for j in test_label_dict[i]])
        if len(test_per_class)==0:
            acc_to_return+=[0];
            continue

        test_labels_per_class = numpy.asarray([test_labels[j] for j in test_label_dict[i]])
        print("Test accuracy per class: " + str(i) + " is " + str(
            Classifier.score(test_per_class, test_labels_per_class)))
        acc_to_return+=[Classifier.score(test_per_class, test_labels_per_class)];
    return acc_to_return

def classifier_score_modded(Classifier, test, test_labels):
    return Classifier.score(test, test_labels)