import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os
import argparse
def load_gadget(folder, 
              fieldnames=["obj_value","wt_norm", 
                          "obj_value_difference", "accuracy", 
                          "zero_one_error", "train_time", "read_init_time"], 
              n=10):
    node_files = [os.path.join(folder, f) for f in os.listdir(folder) if "node" in f]
    assert n > 1
    ff = pd.read_csv(node_files[0], header=0)
    data_frames = [] # To hold all the data frames for individual nodes
    df_lengths = []
    df_lengths.append(len(ff))
    for i in range(1,n):
        f = pd.read_csv(node_files[i], header=0)
        data_frames.append(f)
        
        for fn in fieldnames:
            ff[fn] = ff[fn] + f[fn]
        df_lengths.append(len(f))
    for fn in fieldnames:
        ff[fn] = ff[fn]/n
    ff = ff[fieldnames]
    ff["total_time"] = ff["train_time"] + ff["read_init_time"]
    # Find the node that converged last
    last_node = np.argmax(df_lengths)
    gadget_train_time = data_frames[last_node]["train_time"].iloc[-1]
    ff.to_csv(os.path.join(folder, "agg_gadget_results.csv"))
    return ff, data_frames, gadget_train_time

  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--data_folder', type=str, required=True, help='folder where the node files and the pegasos result file is stored.')
    args = parser.parse_args()

    # Load both Pegasos and Gadget results into pandas dataframes.
    pegasos_df = pd.read_csv(os.path.join(args.data_folder, "cent_pegasos_results.csv"))
    pegasos_df["total_time"] = pegasos_df["train_time"] + pegasos_df["read_init_time"]
    pegasos_df.to_csv(os.path.join(args.data_folder, "cent_pegasos_results.csv"))
    agg_gadget_df, node_dfs, gadget_train_time = load_gadget(args.data_folder)

    # Now plot the figures and save them
    # Objective value vs Train time plot for Pegasos
    fig = plt.figure()
    #plt.plot(pegasos_df["train_time"][0:200], pegasos_df["obj_value"][0:200], label="Pegasos", color='r')
    plt.title("Objective Value vs Training Time")
    plt.xlabel("Training time elapsed (seconds)")
    print(pegasos_df["obj_value"], pegasos_df["train_time"])
    plt.ylabel("Objective Value")
    plt.plot(agg_gadget_df["train_time"][0:200], agg_gadget_df["obj_value"][0:200], label="Gadget", color='b')
    fig.savefig(os.path.join(args.data_folder, "peg_gadget_obj_value.png"))


