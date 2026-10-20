import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import subprocess
import torch
from multiprocessing import Pool
from Common_functions import weights_agg
import pickle
import argparse
import numpy
os.environ["PYTHONWARNINGS"] = "ignore"
import matplotlib.pyplot as plt


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def run_client(client_id, gpu, encoder_runs, save_path_encoder='Encoder_weights_saved', cuda=True, compared_length=None, SimCSE=False, dropout_p=0.1, data_dir='', out_dir=''):
    # Command to run your client.py script with specified client ID and GPU
    cmd = f"python3 Train_Encoder_Client.py --client_id {client_id} --gpu {gpu} --encoder_runs {encoder_runs} --save_path_encoder {save_path_encoder} --cuda {cuda} --compared_length {compared_length} --SimCSE {SimCSE} --dropout_p {dropout_p} --data_dir {data_dir} --out_dir {out_dir}"
    # Run the command
    subprocess.run(cmd, shell=True)

def main(args):
    num_clients = args.num_clients
    gpu_list=args.gpu_list
    num_gpus = len(gpu_list)
    num_rounds=args.num_rounds;
    save_path=args.save_path
    start_from_saved_checkpoint=args.start_from_saved_checkpoint
    save_path_encoder=args.save_path_encoder
    compared_length=args.compared_length
    cuda=True
    SimCSE=args.SimCSE
    SimCSE_dpr=args.SimCSE_dpr
    data_dir=args.data_dir
    out_dir=args.out_dir     
    save_path_encoder=save_path+save_path_encoder;
    # Create a pool of workers, one for each GPU

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    acc_list_round = []
    encoder_runs_resume=0

    if SimCSE:
        simcsestr='_SimCSE_dpr_'+str(SimCSE_dpr)+'_'
    else:
        simcsestr=''

    save_path_encoder=save_path_encoder+simcsestr
    if start_from_saved_checkpoint:
        try:
            print("Starting from saved checkpoint")
            with open(out_dir + 'Round_end_Fully_Personalized_SVM_acc_compared_length_' + str(compared_length) + simcsestr+ '.pkl', 'rb') as fp:
                acc_list_round = pickle.load(fp)
            encoder_runs_resume = len(acc_list_round)+1
            print("Encoder runs resuming from: ", encoder_runs_resume)

        except:
            print("No saved checkpoint found, starting from scratch")
            encoder_runs_resume = 0

        
    
    for encoder_runs_ in range(num_rounds):
        encoder_runs=encoder_runs_+encoder_runs_resume;

        client_batches=num_clients//num_gpus+1
        rem=num_clients%num_gpus

        for i in range(client_batches):
            tasks=[];
            if i<client_batches-1:
                for g in range(num_gpus):
                    tasks+=[(num_gpus*i+g, gpu_list[g], encoder_runs, save_path_encoder, cuda, compared_length, SimCSE, SimCSE_dpr, data_dir, out_dir)]
            else:
                for g in range(rem):
                    tasks+=[(num_gpus*i+g, gpu_list[g], encoder_runs, save_path_encoder,  cuda, compared_length, SimCSE, SimCSE_dpr, data_dir, out_dir)]

            with Pool(num_gpus) as pool:
                pool.starmap(run_client, tasks)

        local_models=[];
        for client_id in range(num_clients):
            local_models+=[torch.load(save_path + '_Model_for_client_'+str(client_id)+'.pt', map_location='cuda:0', weights_only=False)]

        agg_weights = weights_agg([local_models[i].encoder.state_dict() for i in range(num_clients)])
        with open(save_path_encoder, 'wb') as fp:
            pickle.dump(agg_weights, fp)
        del local_models

        acc_list = [];
        if encoder_runs>0:
            for client_id in numpy.arange(num_clients):
                with open(out_dir + 'Round_end_Fully_Personalized_SVM_acc_for_client_' + str(client_id) + '.pkl', 'rb') as fp:
                    acc = pickle.load(fp)
                acc_list.append(acc)

            print("Round_End_Accuracy: ", numpy.mean(acc_list))
            acc_list_round+=[numpy.mean(acc_list)]

            plt.figure()
            plt.plot(numpy.arange(len(acc_list_round))+1, acc_list_round, marker='o')
            plt.savefig(out_dir + 'Round_end_Fully_Personalized_SVM_acc_compared_length_'+str(compared_length)+ simcsestr+'.png')
            plt.close()
        with open(out_dir + 'Round_end_Fully_Personalized_SVM_acc_compared_length_' + str(compared_length) + simcsestr +'.pkl', 'wb') as fp:
            pickle.dump(acc_list_round, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_from_saved_checkpoint', type=str2bool, default=False)
    parser.add_argument('--num_clients', type=int, default=4)
    parser.add_argument('--compared_length', type=int, default=192)
    parser.add_argument('--save_path', type=str, default='Save_models/')
    parser.add_argument('--num_rounds', type=int, default=10)
    parser.add_argument('--gpu_list', type=int, nargs='+', default=[1,4,5,6])
    parser.add_argument('--save_path_encoder', type=str, default='Encoder_weights_saved_CFO_het')
    parser.add_argument('--data_dir', type=str, default='./CFO_Het_Mixed_Data/')
    parser.add_argument('--SimCSE', type=str2bool, default=False)
    parser.add_argument('--SimCSE_dpr', type=float, default=0.1)
    parser.add_argument('--out_dir', type=str, default='./Stats_and_Plot/')
    args = parser.parse_args()
    main(args)
