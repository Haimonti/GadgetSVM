code files

---

DATASET
-------
Name    : RCV1 (Reuters Corpus Volume I)
Task    : Binary text classification
Format  : libsvm / svmlight sparse format (.binary)
Labels  : {-1, +1}  (normalised via np.sign in data_loader.py)
Features: ~47,236 sparse features per sample (bag-of-words TF-IDF)

Files needed (NOT included in this zip — download manually):
  rcv1_train.binary.bz2   (~19 MB compressed, ~289 MB extracted)
  rcv1_test.binary.bz2    (~6 MB compressed, ~99 MB extracted)

Source: LIBSVM datasets — https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#rcv1.binary

Setup steps:
  1. Place the .bz2 files inside  data/raw/
  2. Run:  python data/extract_data.py
     → extracts to data/processed/rcv1_train/rcv1_train.binary
                   data/processed/rcv1_test/rcv1_test.binary
  3. Run:  python main.py

---

config.py
- having all file paths, configuration metrics, etc
	
extract_data.py
- raw data to structured and extracted to data folder


data_loader.py

 - must have class functions for functions related to splitting data
 - setting the features relevant to the data, normalizing the data
 - 
model.py
- must contain the sdca dumb algorothm implementation

network_topology.py
- must contain the networking toplogy mechanisms
- the gossip protocol and the methods for the decentralized network under p2p

main.py
- this file essentially integrates all other files [data_loader.py, model.py, topology.py]
- for data_loader, it must print out graphs and description about the data once normalizaed and structured	
- it must integrate the process in a streamlined manner, and in the end must run the training loop and the testing loop that is needed
- it must then printout graphs and save them in a folder containing the graphs needed
- everything in this file, must essentially save the results in a separate folder properly labelling what is being stored for easy inferencing.
	
here, everything must be logged, so make sure acorss all files, they are properly logged using logger from p2pfl
- model weights across multiple workers
- time taken for each iteration and rounds
- loss for each worker


project-root/
│
├── data/
│   ├── raw/                  # folder for raw data(if any)
│   │── processed/            # folder for extracted data
│   ├── extract_data.py
│   └── data_loader.py
│
├── src/
│   ├── __init__.py             
│   ├── model.py                 # Your main proprietary algorithm    
│   └── network_layer/           # distributed networking part       
│       ├── network_topology.py
│       ├── .....
│       └── ....
│
├── baselines/
│   ├── __init__.py
│   └── GADGET.py     # Abstract base class for all algorithms
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py            # Accuracy, runtime, memory calculators
│   └── visualizer.py         # Script to plot charts and graphs
│
├── results/
│   ├── logs/                 # Raw text execution logs/lightning logs/whatever loggings from that metric saving library
│   ├── metrics/              # 
│   └── plots/                # Saved comparison charts (PNG/PDF)
│
├── config.py  
├── main.py                     # Unified execution entry point
└── requirements.txt          # Environment dependencies











