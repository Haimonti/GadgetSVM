import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os
import argparse
from copy import copy


def get_means_stds(array_of_dfs):
    """
    Takes an array of dataframes and returns their averages and standard deviations
    """
    assert type(array_of_dfs) == list
    df_array = copy(array_of_dfs)
    max_len = max(len(df) for df in array_of_dfs)
    df_concat = None 
    for j, df in enumerate(df_array):
        # Extend last row until the length of longest df 
        for i in range(len(df), max_len):
            new_data = pd.DataFrame(df[-1:].values, index=[i], columns=df.columns)
            df_array[j] = df_array[j].append(new_data)
        

        #df_array[j] = df_array[j].reset_index()
        #print(df_array[j])
        if df_concat is None:
            df_concat = copy(df_array[j])
        else:
            # concatenate them
            df_concat = pd.concat((df_concat, df_array[j]))
        
    # Now average each df and get a mean array
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()
    df_stds = by_row_index.std()
    print(df_means)
    print(df_stds)

    return df_means, df_stds

def load_gadget_run(data_folder, run):
    """ Loads the gadget data pertaining to one run"""
    run_folder = os.path.join(data_folder, "run" + str(run))
    if not os.path.exists(run_folder):
        raise FileNotFoundError("{} not found.".format(run_folder))
    
    # Need to obtain averages for obj_value, train_time, total_time, 
    # read_init_time, zero-one-error and accuracy 
    node_files = sorted([f for f in os.listdir(run_folder) if "node" in f])
    node_paths = [os.path.join(run_folder, f) for f in node_files]

    # Load the first file
    all_dfs = []
    for i, p in enumerate(node_paths):
        node_df = pd.read_csv(p, header=0)
        # Get the columns we need
        node_df = node_df[["obj_value","wt_norm", 
                          "obj_value_difference", "accuracy", 
                          "zero_one_error", "train_time", "read_init_time"]]
        # Define total time
        node_df["total_time"] = node_df["train_time"] + node_df["read_init_time"]
        all_dfs.append(node_df)
    return get_means_stds(all_dfs)
    # Call get_means_stds() to get the means and stds of all the dfs in the list

def get_gadget_results(data_folder, num_runs=3):

    dataset = data_folder.split("/")[-1].split('\\')[-1]
    gadget_run_dfs = []
    gadget_node_stds = []
    #data_folder = "../data/reuters"
    for run in range(num_runs):
        df_node_means, df_node_stds = load_gadget_run(data_folder, run)
        gadget_run_dfs.append(df_node_means)
        gadget_node_stds.append(df_node_stds)

    df_gadget_run_means, df_gadget_run_stds = get_means_stds(gadget_run_dfs)
    mean_save_folder = os.path.join(data_folder, dataset + "_gadget_means.csv")
    df_gadget_run_means.to_csv(mean_save_folder, index=False)
    
    # The new standard deviation is given by sqrt(Var(X) + Var(Y)) 
    # since there is no dependency between runs (Covariance = 0 assumption)

    node_variances = [np.square(g) for g in gadget_node_stds]
    average_run_variance, _  = get_means_stds(node_variances)
    average_run_std = np.sqrt(average_run_variance)
    std_save_folder = os.path.join(data_folder, dataset + "_gadget_stds.csv")
    average_run_std.to_csv(std_save_folder, index=False)
    print("Means and standard deviations stored in {} and {} respectively.".format(
            mean_save_folder, std_save_folder))
    return df_gadget_run_means, average_run_std

def plot_gadget_results(df_gadget_run_means, df_gadget_run_stds,fig=None,  ax=None, sample=10000000, type="train"):
    # Plot results for gadget
    if ax is None:
        fig, ax = plt.subplots(1,2, figsize=(10,5))
    
    if type == "train":
        ax[0].set_title("Objective Value vs Train Time")
        ax[0].set_xlabel("Train time")
    else:
        ax[0].set_title("Objective Value vs Total Time")
        ax[0].set_xlabel("Total time")
    ax[0].set_ylabel("Objective value")
    ax[0].plot(df_gadget_run_means[type+"_time"][0:sample],
                    df_gadget_run_means["obj_value"][0:sample]
                    , label="Gadget", color="blue"
            )
    
    """
    ax[0].fill_between(df_gadget_run_means[type+"_time"][0:sample], 
                     (df_gadget_run_means["obj_value"][0:sample]-df_gadget_run_stds["obj_value"][0:sample]).clip(lower=0), 
                     df_gadget_run_means["obj_value"][0:sample]+df_gadget_run_stds["obj_value"][0:sample],
                     facecolor="blue", alpha=0.3)
    """
    if type == "train":
        ax[1].set_title("Zero One Error vs Train Time")
        ax[1].set_xlabel("Train time")
    else:
        ax[1].set_title("Zero One Error vs Total Time")
        ax[1].set_xlabel("Total time")
    ax[1].set_ylabel("Zero One Error")
    ax[1].plot(df_gadget_run_means[type + "_time"][0:sample],
                    df_gadget_run_means["zero_one_error"][0:sample]
                    , label="Gadget", color="blue"
            )
    """ 
    ax[1].fill_between(df_gadget_run_means[type+"_time"][0:sample], 
                     (df_gadget_run_means["zero_one_error"][0:sample]-df_gadget_run_stds["zero_one_error"][0:sample]).clip(lower=0), 
                     df_gadget_run_means["zero_one_error"][0:sample]+df_gadget_run_stds["zero_one_error"][0:sample],
                     facecolor="blue", alpha=0.3)
    """


    return fig, ax


def get_pegasos_results(data_folder, num_runs=3):
    dataset = data_folder.split("/")[-1].split('\\')[-1]
    peg_run_paths = [os.path.join(args.data_folder, "run" + str(k), "cent_pegasos_results.csv") for k in range(num_runs)]
    pegasos_dfs = []
    for i, p in enumerate(peg_run_paths):
        temp_df = pd.read_csv(p, header=0)
        # Keep necessary columns
        temp_df = temp_df[["obj_value","wt_norm", 
                          "obj_value_difference", "accuracy", 
                          "zero_one_error", "train_time", "read_init_time"]]
        temp_df["total_time"] = temp_df["train_time"] + temp_df["read_init_time"]
        pegasos_dfs.append(temp_df)
    
    df_pegasos_run_means, df_pegasos_run_stds =  get_means_stds(pegasos_dfs)
    mean_save_folder = os.path.join(data_folder, dataset + "_pegasos_means.csv")
    df_pegasos_run_means.to_csv(mean_save_folder, index=False)
    std_save_folder = os.path.join(data_folder, dataset + "pegasos_stds.csv")
    df_pegasos_run_stds.to_csv(std_save_folder, index=False)
    return df_pegasos_run_means, df_pegasos_run_stds

def plot_pegasos_results(df_pegasos_run_means, df_pegasos_run_stds, fig=None, ax=None, sample=10000000, type="train"):

    if ax is None:
        fig, ax = plt.subplots(1,2, figsize=(10,5))

    ax[0].plot(df_pegasos_run_means[type + "_time"][0:sample], 
                df_pegasos_run_means["obj_value"][0:sample], 
                color="red", label="Pegasos")
    """
    ax[0].fill_between(df_pegasos_run_means[type + "_time"][0:sample], 
                     (df_pegasos_run_means["obj_value"][0:sample]-df_pegasos_run_stds["obj_value"][0:sample]).clip(lower=0), 
                     df_pegasos_run_means["obj_value"][0:sample]+df_pegasos_run_stds["obj_value"][0:sample],
                     facecolor="red", alpha=0.3)
    """
    ## On the second axes, plot the zero one errors

    

    ax[1].plot(df_pegasos_run_means[type +"_time"][0:sample], df_pegasos_run_means["zero_one_error"][0:sample], color="red", label="Pegasos")
    """
    ax[1].fill_between(df_pegasos_run_means[type + "_time"][0:sample], 
                     (df_pegasos_run_means["zero_one_error"][0:sample]-df_pegasos_run_stds["zero_one_error"][0:sample]).clip(lower=0), 
                     df_pegasos_run_means["zero_one_error"][0:sample]+df_pegasos_run_stds["zero_one_error"][0:sample],
                     facecolor="red", alpha=0.3)
    """
    
    return fig, ax

if __name__ == "__main__":
    sample=10000000000
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--data_folder', type=str, required=True, help='folder where the node files and the pegasos result file is stored.')
    parser.add_argument('--xtype', type=str, required=True, help='Either train or total time for x-axis.')
    parser.add_argument('--runs', type=int, required=True, help='Number of runs')
    args = parser.parse_args()

    assert args.xtype.lower() in ["train", "total"]
    args.xtype = args.xtype.lower()
    # Load both Pegasos and Gadget results into pandas dataframes.
    #data_folder = "../data/reuters"
    df_gadget_run_means, df_gadget_run_stds = get_gadget_results(args.data_folder, num_runs=args.runs)
    

    df_pegasos_run_means, df_pegasos_run_stds = get_pegasos_results(args.data_folder, num_runs=args.runs)

    
    # equalize the length of both the dataframes, according to train time.


    if len(df_pegasos_run_means) > len(df_gadget_run_means):
        max_len = len(df_pegasos_run_means)
        min_len = len(df_gadget_run_means)
        temp_mean_df = df_gadget_run_means.iloc[-1]
        temp_std_df = df_gadget_run_stds.iloc[-1]
        for i in range(min_len, max_len):
            df_gadget_run_means = df_gadget_run_means.append(temp_mean_df)
            df_gadget_run_stds = df_gadget_run_stds.append(temp_std_df)
        pegasos_end_time = df_pegasos_run_means[args.xtype + "_time"].iloc[-1]
        gadget_end_time = df_gadget_run_means[args.xtype + "_time"].iloc[min_len-1]
        steps_to_extend = max_len - min_len
        time_vals = np.linspace(gadget_end_time, pegasos_end_time, num=steps_to_extend)
        df_gadget_run_means[args.xtype + "_time"].iloc[min_len:max_len] = time_vals
    else:
        max_len = len(df_gadget_run_means)
        min_len = len(df_pegasos_run_stds)
        temp_mean_df = df_pegasos_run_means.iloc[-1]
        temp_std_df = df_pegasos_run_stds.iloc[-1]
        for i in range(len(df_pegasos_run_means), max_len):
            df_pegasos_run_means = df_pegasos_run_means.append(temp_mean_df)
            df_pegasos_run_stds = df_pegasos_run_stds.append(temp_std_df)
        gadget_end_time = df_gadget_run_means[args.xtype + "_time"].iloc[-1]
        pegasos_end_time = df_pegasos_run_means[args.xtype +  "_time"].iloc[min_len-1]
        steps_to_extend = max_len - min_len
        time_vals = np.linspace(pegasos_end_time, gadget_end_time, num=steps_to_extend)
        df_pegasos_run_means[args.xtype + "_time"].iloc[min_len:max_len] = time_vals
    assert len(df_gadget_run_means) == len(df_pegasos_run_means)
    assert len(df_gadget_run_stds) == len(df_pegasos_run_stds)

    # Plot results
    fig, ax = plot_gadget_results(df_gadget_run_means, df_gadget_run_stds, type=args.xtype)
    fig, ax = plot_pegasos_results(df_pegasos_run_means, df_pegasos_run_stds, fig, ax, type=args.xtype)
    save_path = os.path.join(args.data_folder, "Average_obj_value_plot_over_runs.png")
    
    #fig1.savefig(os.path.join(args.data_folder, "gadget_objective.png"))
    # Set the legend and scale

    ax[0].set_yscale('log')
    #ax[1].set_yscale('log')
    ax[0].set_xscale('log')
    #ax[1].set_xscale('log')
    
    ax[0].legend()
    ax[1].legend()
    fig.savefig(save_path)
    plt.show()
    
    
