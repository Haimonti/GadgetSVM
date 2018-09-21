package peersim.gossip;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.functions.SPegasos;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class CentPegasos {

	public static SPegasos trainPegasosClassifier(Instances trainingSet, double lambda) {

	    SPegasos cModel2 = new SPegasos();
	    // Set options
	    String[] options  = new String[8];
	    options[0] = "-L"; 
	    options[1] = Double.toString(lambda);
	    options[2] = "-N";
	    options[3] = "true";
	    options[4] = "-M";
	    options[5] = "true";
	    options[6] = "-E";
	    options[7] = "1";
	        try {
	        	cModel2.setOptions(options);
	        	System.out.println("Num converge iters: " + cModel2.num_converge_iters);
				cModel2.buildClassifier(trainingSet);
				System.out.println("Num converge iters: " + cModel2.num_converge_iters);
	        } catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	    return cModel2;
	}
	
	public static double getAccuracy(Classifier cModel, Instances testingSet) throws Exception {
		// Evaluate
        double[] spegasosPred=new double[testingSet.numInstances()];
		 double[] actual=new double[testingSet.numInstances()];
		 double acc=0; 
		 int clIndex = testingSet.classIndex();
		 System.out.println("Class index: " + clIndex);
		 System.out.println("Num test attributes: " + testingSet.numAttributes());
		 for (int i = 0; i < testingSet.numInstances(); i++)
		 {
			 spegasosPred[i]=cModel.classifyInstance(testingSet.instance(i));
			 actual[i] = testingSet.instance(i).classValue();
			 if(spegasosPred[i]==actual[i])
			 {
				 acc=acc+1;
			 }
        
        }
		 double testAccuracy = acc/(double)testingSet.numInstances();
		 
		 return testAccuracy;
		
		
	}
	
	public static void main(String[] args) throws Exception {
	
		// Parse arguments
		String globalTrainFilepath = args[0];
		String globalTestFilepath = args[1];
		System.out.println(globalTrainFilepath + " " + globalTestFilepath);
		double pegasosLambda = Double.parseDouble(args[2]);
		long epochs = Long.parseLong(args[3]);
		int numRun = Integer.parseInt(args[4]);
		
		// Init some variables
		double readInitTimeInDouble;
		double trainTimeInDouble = 0.0;
		long trainTimePerIter, startTime;
		int converged = 0;
		double accuracy = 0.0;
		final int NUM_CONVERGE_ITERS = 10;
	    
		// Read the data
		startTime = System.nanoTime();
		DataSource globalTrainSource = new DataSource(globalTrainFilepath);
		DataSource globalTestSource = new DataSource(globalTestFilepath);
		Instances globalTrainingSet = globalTrainSource.getDataSet();
	    Instances globalTestingSet = globalTestSource.getDataSet();
	    readInitTimeInDouble = (double)(System.nanoTime() - startTime)/(double)1e9;
	    
	    // Build the model
	    SPegasos cModel = trainPegasosClassifier(globalTrainingSet, pegasosLambda);
	    System.out.println("Num converge iters: " + cModel.num_converge_iters);
	    // Get the parent directory
	    File file = new File(globalTrainFilepath);
	    String parentPath = file.getAbsoluteFile().getParent();
	   
	    System.out.println(parentPath);
	    
	    // Create a new folder for run
	    File directory = new File(parentPath + "/" + "run" + numRun);
	    if (! directory.exists()){
	        directory.mkdir();
	        // If you require it to make the entire directory path including parents,
	        // use directory.mkdirs(); here instead.
	    }
	    
	    // Initialize the csv file to store the results
	    String csv_filename = parentPath + "/" + "run" + numRun +"/cent_pegasos_results" + ".csv";
		String headerString = "iter,obj_value,loss_value,wt_norm,obj_value_difference,converged,";
		headerString += "num_converge_iters,accuracy,zero_one_error,train_time,read_init_time,m_t\n";
		BufferedWriter bw = new BufferedWriter(new FileWriter(csv_filename));
  		bw.write(headerString);
  		bw.close();
  		
		String opString;
		
	    for (int iter = 0; iter < epochs; iter++) {
	    	System.out.println("Num converge iters: " + cModel.num_converge_iters);
	          if (converged == 0) {
	        	  globalTrainingSet.randomize(new Random(System.currentTimeMillis()));
	        	  startTime = System.nanoTime();
	        	  cModel.m_t = iter+2;
	        	  // Update the classifier
	        	  cModel.updateClassifier(globalTrainingSet.instance(0));
	        	  trainTimePerIter = System.nanoTime() - startTime;
	        	  trainTimeInDouble += (double)trainTimePerIter / (double)1e9;
	    
	        	  
		          //System.out.println("obj_value: " + cModel.m_obj_value);
		          
	        	// Get the accuracy of the test set
	      		accuracy = getAccuracy(cModel, globalTestingSet);
	      		
	          opString = "";
        	  opString += iter + "," + cModel.m_obj_value + "," + cModel.m_loss_value+","+cModel.wt_norm+","+cModel.m_obj_value_diff+",";
	          opString += converged + "," + cModel.num_converge_iters +"," + accuracy + "," + (1.0 - accuracy);
	          opString += ","+ trainTimeInDouble + "," + readInitTimeInDouble + "," + cModel.m_t + "\n"; 
	          System.out.println(opString);
	        // Write to file
	  		bw = new BufferedWriter(new FileWriter(csv_filename, true));
	  		bw.write(opString);
	  		
	  		bw.close();
	  		
	        	// Check if the algorithm has converged
	  				
				     if(cModel.num_converge_iters == NUM_CONVERGE_ITERS) {
				    	 converged = 1; 
				     }
				    
	          }
	          if(converged==1) {
	        	  break;
	          }
	         
	}
	}
}
