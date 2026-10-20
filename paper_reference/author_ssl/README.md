**Acknowledgement**

This code for FedSSL-AMC is adapted from Yiyue Chen's work on Evolutionary Clustering in Federated Learning [https://arxiv.org/abs/2509.07198]

**Libraries and their versions used for simulation**

- joblib==1.5.1
- matplotlib==3.10.3
- multiprocess==0.70.18
- numpy==2.1.2
- scikit-learn==1.6.1
- scipy==1.15.3
- torch==2.7.0+rocm6.3

**Running the simulations**

- To generate the dataset, run python3 Generate_Dataset.py

- To Train the encoder, execute python3 Train_Encoder_Server.py
  + default_hyperparameters.json specifies the training parameters for the Causal CNN encoder. For example, depth sets the number of layers; nb_steps sets the total number of steps per round.  nb_random_samples sets the number of negative examples used for the cotnrastive loss per step.
  + -- num_clients sets the total number of clients, which should concur with the number of clients specified while generating the dataset. --gpu_list sets the list of gpus to utilize. --save_path_encoder specifies the name of the file the encoder weigths of the encoder are saved to. --num_rounds specifies the total number of rounds --save_path specifies the directory the models are saved to. --out_dir specifies the directory where the plots and accuracies are saved. 
  + Note that python3 Train_Encoder_Server.py --SimCSE True implements the Sim CSE baseline with --SimCSE_dpr specifying the corresponding dropout probability.
  + To resume from a checkpoint, add --start_from_saved_checkpoint True.
  + This file executes Train_Encoder_Client.py for each client in parallel according to the number of gpus specified. For example, four gpus are needed to run the client script for four clients concurrently. 

- To evaluate the train encoder and train the personalized SVMs, run python3 Output_layer_Personlized_Main
