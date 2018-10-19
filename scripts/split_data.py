#Run on data and split it into n files


import os
import argparse


#These prefixes will be used to name the split train and test files.
train_prefix = 't_'
test_prefix = 'tst_'

#Input dataset argument. NO default is provided. User has to input this argument.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",  help="Dataset name used for the data-specific folder.",
	type = str,required =True)
parser.add_argument("--datafolder" , help="Folder where the other dataset folders are found.",
	type = str,required =True)
parser.add_argument("--trainfile",  help="Name of train file.",
	type = str,required =True)
parser.add_argument("--testfile",  help="Name of test file.",
	type = str,required =True)
parser.add_argument("--splits",  help="Number of files to split the data into.",
	type = int,default = 10)
parser.add_argument("--ext",  help="Extension of files to be saved.",
	type = str,default = 'dat')
args = parser.parse_args()

dataset = args.dataset
datafolder = args.datafolder
trainfile, testfile = args.trainfile, args.testfile
n = args.splits
ext = args.ext

#peersim_path = '../peersim-pegasos/'
#pegasos_native_path = '../jni-pegasos/src/pegasos-native'


dataset_folder_path = os.path.join(datafolder,dataset)

#Find the train and test files automatically. THEY HAVE TO BE LABELED .trn and .tst

all_data_files = list(os.walk(dataset_folder_path))[0][2]


data_train_path = os.path.join(dataset_folder_path,trainfile)
data_test_path = os.path.join(dataset_folder_path, testfile)

#Read file
def write_to_file(filepath,writepath,prefix,n):
    with open(filepath,'r') as f:
        contents = f.read()
        datapoints = contents.split('\n')
        if len(datapoints[-1].split()) == 0:
            print("Last line is empty")
            print("Omitting last line")
            datapoints = datapoints[:-1]
        # Shuffle the data
        from random import shuffle
        shuffle(datapoints)
        print("Total number of datapoints: %d" %(len(datapoints)))
        pts_per_file = int(round((len(datapoints)-1)/n))
        print("Total number of datapoints per file: %d" %(pts_per_file))
    for i in range(n):
        batch = datapoints[i*pts_per_file:min(len(datapoints)-1,(i+1)*pts_per_file)]
        #Write to file
        with open(os.path.join(writepath,prefix + str(i) + '.' + ext), 'w') as wfile:
            wfile.write('\n'.join(batch))
    print("Split the data into " + str(n) +" files.")

write_to_file(data_train_path, dataset_folder_path,train_prefix,n)
write_to_file(data_test_path, dataset_folder_path,test_prefix,n)

