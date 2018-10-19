import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os
import argparse
from copy import copy


def get_final_table(xtype, filename):
    data_folder = "../data"
    df = pd.DataFrame(columns = ["Dataset", "Gadget {} time (sec)".format(xtype), 
    	"Gadget Accuracy (%)", "Pegasos {} time (sec)".format(xtype), "Pegasos Accuracy (%)"])
    for dataset in ['adult', 'ccat', 'covertype', 'mnist', 'reuters', 'usps', 'webspam']:
        dataset_folder = os.path.join(data_folder, dataset)
        gadget_means = pd.read_csv(os.path.join(dataset_folder, dataset + "_gadget_means.csv"), header=0)
        gadget_stds = pd.read_csv(os.path.join(dataset_folder, dataset + "_gadget_stds.csv"), header=0)
        pegasos_means = pd.read_csv(os.path.join(dataset_folder, dataset + "_pegasos_means.csv"), header=0)
        pegasos_stds = pd.read_csv(os.path.join(dataset_folder, dataset + "_pegasos_stds.csv"), header=0)

        # Format as per table requirements
        gadget_time = "{} (+/-{})".format(round(gadget_means.iloc[-1][xtype+"_time"],3), round(gadget_stds.iloc[-1][xtype+"_time"],3))
        gadget_acc = "{} (+/-{})".format(round(gadget_means.iloc[-1]["accuracy"]*100,3), round(gadget_stds.iloc[-1]["accuracy"],3))
        pegasos_time = "{} (+/-{})".format(round(pegasos_means.iloc[-1][xtype+"_time"],3), round(pegasos_stds.iloc[-1][xtype+"_time"],3))
        pegasos_acc = "{} (+/-{})".format(round(pegasos_means.iloc[-1]["accuracy"]*100,3), round(pegasos_stds.iloc[-1]["accuracy"],3))
        df2 = pd.DataFrame(data = {"Dataset": dataset,
                                    "Gadget {} time (sec)".format(xtype): gadget_time,
                                  "Gadget Accuracy (%)": gadget_acc,
                                  "Pegasos {} time (sec)".format(xtype): pegasos_time,
                                  "Pegasos Accuracy (%)": pegasos_acc }, index=[0])
        df = df.append(df2)
        print(gadget_time, gadget_acc, pegasos_time, pegasos_acc)
    df = df.set_index("Dataset")
    if not filename.endswith(".csv"):
    	filename += ".csv"
    df.to_csv(os.path.join(data_folder, filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--xtype', type=str, required=True, help='Either train or total time for x-axis.')
    args = parser.parse_args()
    assert args.xtype.lower() in ["train", "total"]
    args.xtype = args.xtype.lower()
    get_final_table(xtype=args.xtype, filename="final_"+args.xtype+"_table.csv")