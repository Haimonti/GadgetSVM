package peersim.gossip;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

import weka.classifiers.Classifier;
import weka.classifiers.functions.SPegasos;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.SVMLightLoader;

public class CentPegasos {

	public static SPegasos trainPegasosClassifier(Instances trainingSet, double lambda) {

	    SPegasos cModel = new SPegasos();
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
	        	cModel.setOptions(options);
				cModel.buildClassifier(trainingSet);
			   
	        } catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	    return cModel;
	}
	
	public static double getAccuracy(Classifier cModel, Instances testingSet) throws Exception {
		// Evaluate
        double[] spegasosPred=new double[testingSet.numInstances()];
		 double[] actual=new double[testingSet.numInstances()];
		 int acc=0; 
		 int clIndex = testingSet.classIndex();
		 System.out.println("Class index: " + clIndex);
		 System.out.println("Num test attributes: " + testingSet.numAttributes());
		 for (int i = 0; i < testingSet.numInstances(); i++)
		 {
			 spegasosPred[i]=cModel.classifyInstance(testingSet.instance(i));
			 //System.out.println(sgdPred[i]);	
			// actual[i]=Double.parseDouble(testingSet.instance(i).getClass().toString());
			 actual[i] = testingSet.instance(i).classValue();
			 //System.out.println("Actual: "+actual[i]);
			 //System.out.println("Pred: "+spegasosPred[i]);
			 //System.out.println("=====");
			 if(spegasosPred[i]==actual[i])
			 {
				 acc=acc+1;
			 }
        
        }
		 double testAccuracy = (double)acc/testingSet.numInstances();
		 
		 return testAccuracy;
		
		
	}
	
	public static void main(String[] args) throws Exception {
	
		
		String globalTrainFilepath = args[0];
		String globalTestFilepath = args[1];
		System.out.println(globalTrainFilepath + " " + globalTestFilepath);
		double pegasosLambda = Double.parseDouble(args[2]);
		
		long epochs = Long.parseLong(args[3]);
		double readInitTimeInDouble;
		double trainTimeInDouble = 0.0;
		long trainTimePerIter, startTime;
		int converged = 0;
		double accuracy = 0.0;
	    
		startTime = System.nanoTime();
		DataSource globalTrainSource = new DataSource(globalTrainFilepath);
		DataSource globalTestSource = new DataSource(globalTestFilepath);
		Instances globalTrainingSet = globalTrainSource.getDataSet();
	    Instances globalTestingSet = globalTestSource.getDataSet();
	    readInitTimeInDouble = (double)(System.nanoTime() - startTime)/(double)1e9;
	    SPegasos cModel = trainPegasosClassifier(globalTrainingSet, pegasosLambda);
	    
	    // Get the parent directory
	    File file = new File(globalTrainFilepath);
	    String parentPath = file.getAbsoluteFile().getParent();
	   
	    System.out.println(parentPath);
	    String csv_filename = parentPath + "/" + "cent_pegasos_results.csv";
		
	    
		
		String headerString = "iter,obj_value,loss_value,wt_norm,obj_value_difference,converged,";
		headerString += "num_converge_iters,accuracy,zero_one_error,train_time,read_init_time\n";
		BufferedWriter bw = new BufferedWriter(new FileWriter(csv_filename));
  		bw.write(headerString);
  		bw.close();
  		
		String opString;
		
		
	    for (int iter = 0; iter < epochs; iter++) {
	        
	          if (converged == 0) {
	        	  startTime = System.nanoTime();
	        	  cModel.m_t = iter+2;
					if (cModel.m_t == 0) {
						cModel.m_t = 2;
					}
				//System.out.println("m_t: " + cModel.m_t);
	        	  cModel.updateClassifier(globalTrainingSet.instance(iter%globalTrainingSet.numInstances()));
	        	  trainTimePerIter = System.nanoTime() - startTime;
	        	  trainTimeInDouble += (double)trainTimePerIter / (double)1e9;
	        	  //System.out.println("Iter%num_examples: " + iter%globalTrainingSet.numInstances());
	        	  
		          //System.out.println("obj_value: " + cModel.m_obj_value);
		          
	        	// Get the accuracy of the test set
	      		accuracy = getAccuracy(cModel, globalTestingSet);
	      		
	          opString = "";
        	  opString += iter + "," + cModel.m_obj_value + "," + cModel.m_loss_value+","+cModel.wt_norm+","+cModel.m_obj_value_diff+",";
	          opString += converged + "," + cModel.num_converge_iters +"," + accuracy + "," + (1.0 - accuracy);
	          opString += ","+ trainTimeInDouble + "," + readInitTimeInDouble + "\n"; 
	          System.out.println(opString);
	        // Write to file
	  		bw = new BufferedWriter(new FileWriter(csv_filename, true));
	  		bw.write(opString);
	  		
	  		bw.close();
	  		
	        	// Check if the algorithm has converged
				     if(cModel.num_converge_iters == 10) {
				    	 converged = 1; 
				     }
				    
	          }
	          if(converged==1) {
	        	  break;
	          }
	         
	}
	}
}
