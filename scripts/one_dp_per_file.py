#Run on data and split it into n files


import os
import argparse
import shutil


#These prefixes will be used to name the split train and test files.
train_prefix = 't_'
test_prefix = 'tst_'

#Input dataset argument. NO default is provided. User has to input this argument.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",  help="Dataset name used for the data-specific folder.",
    type = str,required =True)
parser.add_argument("--datafolder" , help="Folder where the other dataset folders are found.",
    type = str,required =True)
parser.add_argument("--file",  help="Folder where the other dataset folders are found.",
    type = str,required =True)
parser.add_argument("--ext",  help="Extension of files to be saved.",
    type = str,default = 'dat')
args = parser.parse_args()

dataset = args.dataset
datafolder = args.datafolder
file = args.file
ext = args.ext

#peersim_path = '../peersim-pegasos/'
#pegasos_native_path = '../jni-pegasos/src/pegasos-native'


dataset_folder_path = os.path.join(datafolder,dataset)
save_folder_name = file.split(".")[0]
save_folder_path = os.path.join(dataset_folder_path, save_folder_name)
if os.path.exists(save_folder_path):
    shutil.rmtree(save_folder_path)

os.mkdir(save_folder_path)


filepath = os.path.join(dataset_folder_path, file)

def one_dp_per_file(filepath, save_folder_path):
    with open(filepath,'r') as f:
        contents = f.readlines()
        for i in range(len(contents)):
            if i%5000 == 0:
                print("{} done.".format(i))
            save_file_path = os.path.join(save_folder_path, str(i).zfill(8) + ".dat")
            with open(save_file_path, "w") as f1:
                print(contents[i]) 
                f1.write(contents[i].strip())
    print("Files written to {}".format(save_folder_path))

def one_dp_per_file2(filepath, save_folder_path):
    from sklearn.datasets import load_svmlight_file, dump_svmlight_file
    data = load_svmlight_file(filepath)
    
    for i in range(len(data[1])):
        if i%5000 == 0:
            print("{} done.".format(i))
        save_file_path = os.path.join(save_folder_path, str(i).zfill(8) + ".dat")    
        dump_svmlight_file(data[0][i].reshape(1, -1), [data[1][i]], save_file_path)
    print("Files written to {}".format(save_folder_path))

one_dp_per_file(filepath, save_folder_path)