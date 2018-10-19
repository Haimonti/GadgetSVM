# GADGET SVM: A Gossip-bAseD sub-GradiEnT Solver for Linear SVMs

In the era of big data, an important weapon in a machine learning researcher’s arsenal is a scalable Support
Vector Machine (SVM) algorithm. SVMs are extensively used for solving classification problems. Traditional
algorithms for learning SVMs (such as chunking and Sequential Minimal Optimization) scale super linearly
with training set size which becomes infeasible very quickly for large data sets. In recent years, scalable
algorithms have been designed which study the primal or dual formulations of the problem. This often
suggests a way to decompose the problem and facilitates development of distributed algorithms. In this paper,
we present a distributed algorithm for learning linear Support Vector Machines in the primal form for binary
classification called Gossip-bAseD sub-GradiEnT (GADGET) SVM. The algorithm is designed such that
it can be executed locally on nodes of the distributed system; each node processes its local homogeneously
partitioned data and learns a primal SVM model; it then gossips with random neighbors about the classifier
learnt and uses this information to update the model. Extensive theoretical and empirical results suggest
that this anytime algorithm has performance comparable to its centralized and online counterparts.

### Prerequisites

Gadget requires the following to run if you are working from the source code:
1. Java(TM) SE Runtime Environment (build 1.8.0_181-b13)
2. Python 3.6 (Pandas, Numpy)

It also needs Peersim Simulator and Weka 3.6.11 to run. However, since we have modified the open source files as per the needs of this project, they can be found under ./src and no new installations will be required.

Since WEKA cannot load large datasets into memory, we modified the datasets in order to have a file for every data point in the dataset, split into appropriate folders. This leads to comparable disk storage space used, and now we select a random index from within the program and load that specific data point into memory, thereby being able to handle one example at a time. 

### Data Preparation

To prepare the dataset needed for our purpose, we went through the following steps:
1. Obtain train and test LIBSVM/SVMLight files. (Our code only works with LIBSVM/SVMLight format for now.)
2. Split them into k training datasets and k testing datasets using ./scripts/split_data.py. To split into 10 files as per our paper, run the following to create 10 train files (t_0.dat, t_1.dat,...) and 10 test files
(tst_0.dat, tst_1.dat, tst_2.dat,...):
```
cd scripts
python split_data.py --data_folder ../data --dataset mnist --trainfile mnist-train.dat --testfile mnist-test.dat --splits 10
```
3. We then create a file per data point, within the respectively named directories.
```
python one_dp_per_file.py --datafolder ../data --dataset mnist --file mnist-train.dat
python one_dp_per_file.py --datafolder ../data --dataset mnist --file mnist-test.dat

python one_dp_per_file.py --datafolder ../data --dataset mnist --file t_0.dat
python one_dp_per_file.py --datafolder ../data --dataset mnist --file t_1.dat
python one_dp_per_file.py --datafolder ../data --dataset mnist --file t_2.dat
...

python one_dp_per_file.py --datafolder ../data --dataset mnist --file tst_0.dat
python one_dp_per_file.py --datafolder ../data --dataset mnist --file tst_1.dat
python one_dp_per_file.py --datafolder ../data --dataset mnist --file tst_2.dat
...
```


### Running the code

Now that you have the data prepared, you can run the code using one of two JARs.
1. gadget_static_ooat_5.jar - Test accuracy is calculated once every 5 iterations.
2. gadget_static_ooat.jar - Test accuracy is calculated once every 50 iterations.

This distinction is made since test accuracy calculation takes quite a long time as according to our "one data point per file" strategy since Weka is not able to load the entire dataset into memory. Every test data point needs to serially accessed for accuracy to be accumulated over the test set.

To run GADGET, do the following:
```
java -jar gadget_static_oaat_5.jar <configfile_path> <error_file>
java -jar gadget_static_oaat_5.jar ./config/config-pegasosMnist0.cfg output.txt
```
Configuration parameters are found in ./config 

To run Centralized Pegasos on the same dataset (MNIST):

```
java -jar cent_pegasos_oaat_5.jar <train_directory> <test_directory> <lambda> <max_iterations> <run> <dimension>
java -jar cent_pegasos_oaat_5.jar ./data/mnist/mnist-train ./data/mnist/mnist-test 0.0000167 1000000 0 784
```

### Plotting graphs

Once you get results for 5 runs, you can aggregate them into one file for means and standard deviations using the following the Python script:

```
python plot_results.py --data_folder ./data/mnist --xtype train --runs 5 --gadget_acc_skip 50 --peg_obj_skip 40 --gadget_obj_skip 40
```
### Results

![alt text](data/adult_plot.png)
![alt text](data/ccat_plot.png)
![alt text](data/mnist_plot.png)
![alt text](data/reuters_plot.png)
![alt text](data/usps_plot.png)


### Built With

* [Java](https://www.java.com/en/)
* [Weka](https://www.cs.waikato.ac.nz/ml/weka/) - Base Machine Learning framework
* [Peersim Simulator](http://peersim.sourceforge.net/) - P2P simulator used simulate distributed computation

### License

Coming soon...

### Acknowledgments

Coming soon...