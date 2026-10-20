import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
import numpy
import argparse
from scipy.spatial.distance import cdist
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
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

import pickle

from sklearn.cluster import kmeans_plusplus

from sklearn.metrics import rand_score
from Common_functions import *
from multiprocessing import Pool
import subprocess
import numpy as np

def run_client(client_id, gpu=0, cuda=True, save_path_encoder='Encoder_weights_saved', C=float('inf'), SimCSE=False, dropout_p=0.1, data_dir='',save_path='', out_dir=''):
    # Command to run your client.py script with specified client ID and GPU
    cmd = f"python3 Output_Layer_Personalized_Client.py --client_id {client_id} --gpu {gpu}  --save_path_encoder {save_path_encoder} --cuda {cuda} --C {C} --SimCSE {SimCSE} --dropout_p {dropout_p} --data_dir {data_dir} --save_path {save_path} --out_dir {out_dir}"
    # Run the command
    subprocess.run(cmd, shell=True)


def main(args):
    cuda=args.cuda
    save_path=args.save_path
    save_path_encoder = args.save_path_encoder
    C=args.C
    SimCSE = args.SimCSE
    SimCSE_dpr = args.dropout_p
    out_dir = args.out_dir
    data_dir = args.data_dir
    SimCSE_dpr = args.dropout_p
    num_clients = args.num_clients
    gpu=args.gpu

    save_path_encoder=save_path+save_path_encoder



    if SimCSE:
        simcsestr='_SimCSE_dpr_'+str(SimCSE_dpr)+'_'
        
    else:
        simcsestr=''


    save_path_encoder=save_path_encoder+simcsestr
    save_string='_'+simcsestr

    if cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        cuda = False

    print("CUDA check:", torch.cuda.is_available())

    n_parallel_clients=1;
    for n in range(int(4/n_parallel_clients)):
        print(n)
        with Pool() as pool:
            tasks = [(client_id, gpu, cuda, save_path_encoder, C, SimCSE, SimCSE_dpr, data_dir, save_path, out_dir) for client_id in numpy.arange(n*n_parallel_clients,(n+1)*n_parallel_clients)]
            pool.starmap(run_client, tasks)

    acc_list=[];
    for client_id in numpy.arange(num_clients):
        with open(out_dir + 'Fully_Personalized_SVM_acc_for_client_' + str(client_id)+'.pkl', 'rb') as fp:
            acc=pickle.load(fp)
        acc_list.append(acc)


    torch.save(acc_list, out_dir +'acc_list_SVM'+save_string+'.pt')
    print("Accuracy: ", np.mean(acc_list))

    acc_list_SNR = [];
    for SNR in numpy.arange(-10,10):
        for client_id in numpy.arange(num_clients):
            with open(out_dir + 'Fully_Personalized_SVM_acc_for_client_' + str(client_id) + '_SNR_'+str(SNR)+'.pkl', 'rb') as fp:
                acc = pickle.load(fp)
            acc_list.append(acc)

        print("SNR: ",str(SNR)," Accuracy: ", np.mean(acc_list))
        acc_list_SNR.append(np.mean(acc_list))

    with open(out_dir + 'Fully_Personalized_SVM_acc_for_client_with_SNR_'+save_string +'.pkl',
              'wb') as fp:
        pickle.dump(acc_list_SNR, fp)

    plt.plot(np.arange(-10,10), acc_list_SNR)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title('Fully Personalized SVM Accuracy vs SNR')
    plt.grid()
    plt.savefig(out_dir + 'Fully_Personalized_SVM_acc_vs_SNR_averaged_across_clients_'+save_string+'.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='Save_models/')
    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--client_id', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path_encoder', type=str, default='Encoder_weights_saved_CFO_het')
    parser.add_argument('--data_dir', type=str, default='./CFO_Het_Mixed_Data/')
    parser.add_argument('--out_dir', type=str, default='./Stats_and_Plot/')
    parser.add_argument('--C', type=float, default=float('inf'))
    parser.add_argument('--SimCSE', type=str2bool, default=False)
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--num_clients', type=int, default=4)
    main(parser.parse_args())