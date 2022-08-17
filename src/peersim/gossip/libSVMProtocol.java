package peersim.gossip;


import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.Arrays;
import java.lang.Integer;
import java.io.FileReader;
import java.io.LineNumberReader;

import peersim.gossip.PegasosNode;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import peersim.config.Configuration;
import peersim.config.FastConfig;
import peersim.core.*;
import peersim.cdsim.*;

import java.net.MalformedURLException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.GeneralSecurityException;
import java.text.ParseException;
import java.io.BufferedReader;

/**
 * Class GadgetProtocol
 * Implements a cycle based {@link CDProtocol}. It implements the Gadget algorithms
 * described in paper:
 * Chase Henzel, Haimonti Dutta
 * GADGET SVM: A Gossip-bAseD sub-GradiEnT SVM Solver   
 * 
 *  @author Nitin Nataraj, Deepak Nayak
 */


public class libSVMProtocol implements CDProtocol {
	/**
	 * New config option to get the learning parameter lambda for GADGET
	 * @config
	 */
	private static final String PAR_LAMBDA = "lambda";
	/**
	 * New config option to get the number of iteration for GADGET
	 * @config
	 */
	private static final String PAR_ITERATION = "iter";
	
	public static boolean flag = false;
	
	public static int t = 0;
	
	public static boolean optimizationDone = false;	
	
	public double EPSILON_VAL = 0.01;
	/** Linkable identifier */
	protected int lid;
	/** Learning parameter for GADGET, different from lambda parameter in pegasos */
	protected double lambda;
	/** Number of iteration (T in gadget)*/
	protected int T;
	
	private int pushsumflag = 0;
	
	public static double[][] optimalB;
	
	public static int end = 0;
	
	public static boolean pushsumobserverflag = false;
	public static final int CONVERGENCE_COUNT = 10;
	
	private double oldWeight;
	
	private boolean pushsum2_execute = true;
	
	private String protocol;
	private String protocolASG;

	private String resourcepath;
	
	private double peg_lambda;
	private int max_iter;
	private int exam_per_iter;
	private double[] weights;
	

	/**
	 * Default constructor for configurable objects.
	 */
	public libSVMProtocol(String prefix)
	{
		lambda = Configuration.getDouble(prefix + "." + PAR_LAMBDA, 0.01);
		T = Configuration.getInt(prefix + "." + PAR_ITERATION, 100);
		//T = 0;
		lid = FastConfig.getLinkable(CommonState.getPid());
		protocol = Configuration.getString(prefix + "." + "prot", "pushsum1");
		protocolASG = Configuration.getString(prefix + "." + "prot1", "pushSV");
		
	}

	/**
	 * Returns true if it possible to deliver a response to the specified node,
	 * false otherwise.
	 * Currently only checking failstate, but later may be we need to check
	 * the non-zero transition probability also
	 */
	protected boolean canDeliverRequest(Node node) {
		if (node.getFailState() == Fallible.DEAD)
			return false;
		return true;
	}
	/**
	 * Returns true if it possible to deliver a response to the specified node,
	 * false otherwise.
	 * Currently only checking failstate, but later may be we need to check
	 * the non-zero transition probability also
	 */
	protected boolean canDeliverResponse(Node node) {
		if (node.getFailState() == Fallible.DEAD)
			return false;
		return true;
	}

	/**
	 * Clone an existing instance. The clone is considered 
	 * new, so it cannot participate in the aggregation protocol.
	 */
	public Object clone() {
		libSVMProtocol gp = null;
		try { gp = (libSVMProtocol)super.clone(); }
		catch( CloneNotSupportedException e ) {} // never happens
		return gp;
	}
	
	public static double getAccuracy(Classifier cModel, String testPath) throws Exception
	{
		// Evaluate
	    // Files.listFiles() apparently does not work on Linux, and is unreliable.
	    
		File testFolder = new File(testPath);
	    File[] listOfFiles = testFolder.listFiles();
	    //int numtestfiles = listOfFiles.length;
		
		// We have to use the nio library to get the number of files in the directory
		//String[] listOfFiles = new File(testPath).list();
		
	    int numtestfiles = listOfFiles.length;
	    
	    //System.out.println("Num test files: " + numtestfiles);
        double[] spegasosPred=new double[numtestfiles];
		 double[] actual=new double[numtestfiles];
		 double acc=0; 
		 
		 DataSource testSource;
		 Instances testingSet;
		 for (int i = 0; i < numtestfiles; i++)
		 {
			 String testFilePath = listOfFiles[i].toString();
			 
				testSource = new DataSource(testFilePath);
				testingSet = testSource.getDataSet();
			 spegasosPred[i]=cModel.classifyInstance(testingSet.instance(0));
			 actual[i] = testingSet.instance(0).classValue();
			 if(spegasosPred[i]==actual[i])
			 {
				 acc=acc+1;
			 }
        //System.out.println("Acc: " + acc);
        }
		 double testAccuracy = acc/(double)numtestfiles;
		 System.out.println("Test Accuracy: " + testAccuracy);
		 return testAccuracy;
}
	
	public double getAccuracy2(double[] pred, Instances testingSet) throws Exception 
	{		
         double[] libSVMPred=new double[testingSet.numInstances()];
		 double[] actual=new double[testingSet.numInstances()];
		 double acc=0; 
		 int clIndex = testingSet.classIndex();
		 for (int i = 0; i < testingSet.numInstances(); i++)
		 {
			 libSVMPred[i]=pred[i];
			 actual[i] = testingSet.instance(i).classValue();
			 //System.out.println("Actual: "+actual[i]);
			 //System.out.println("Pred: "+spegasosPred[i]);
			 //System.out.println("=====");
			 if(libSVMPred[i]==actual[i])
			 {
				 acc=acc+1;
			 }       
        }
		 double testAccuracy = acc/(double)testingSet.numInstances();
		// System.out.println("Accuracy "+(double)acc/testingSet.numInstances());
		 return testAccuracy;		
	}
	
	private void pushsum1(Node node, LibLinearNode pn, int pid)
	{
		LibLinearNode peer = (LibLinearNode)selectNeighbor(pn, pid);
	
	    System.out.println("Node "+node.getID()+" is gossiping with Node "+peer.getID()+"....");
	    //Use the neighbors model to get the prediction on local model
	    double[] peerPred = new double[pn.testData.numInstances()];
	    try
	    {
	    	double[] retPred = new double[pn.testData.numClasses()];
			for(int h=0;h<pn.testData.numInstances();h++)
			{
				retPred=peer.libSVMClassifier.distributionForInstance(pn.trainData.instance(h));
				//System.out.println("Prob0 "+retPred[0]+" Pred1 " +retPred[1]);
			  //libSVMPred[h]=libsvmModel.classifyInstance(data.instance(h));
			  //pn.pred[h]=pn.libSVMClassifier.classifyInstance(pn.trainData.instance(h));	
				if(retPred[0]>retPred[1])
				{	
				  peer.pred[h]=0;
				}
				else
				{
					peer.pred[h]=1;
				}
			}
		    //for(int h=0;h<pn.testData.numInstances();h++)
			//{
		     //peerPred[h]=peer.libSVMClassifier.classifyInstance(pn.testData.instance(h));	
			//}
			System.out.println("Done using peer's libSVM model to predict local test set.");
			//Now do the gossip step
			//Assume that beta=1
			//Do we need to normalize the difference by number of nodes?
			for(int h=0;h<pn.testData.numInstances();h++)
			{
			pn.predTest[h]=pn.predTest[h] + Math.abs((pn.predTest[h]-peerPred[h]))/2;	
			}
	    }
	    catch(Exception e)
	    {
	    	e.printStackTrace();
	    }
	}
	
	private void pushSV(Node node, PegasosNode pn, int pid) 
		{
		PegasosNode peer = (PegasosNode)selectNeighbor(pn, pid);
	    //System.out.println("ASG SVM Algorithm: Node "+node.getID()+" is gossiping with Node "+peer.getID()+"....");
	    // Function to send selected Set
	    for(int h=0;h<pn.supportVecs.numInstances();h++)
		{
			boolean notInSet=true;
			for(int f=0;f<peer.updatedTrainData.numInstances();f++)
			{	
//			  if(pn.supportVecs.instance(h).equals(peer.updatedTrainData.instance(f)))
//			  {
//				  notInSet=false;
//			  }
			  boolean instEqual=true;	
			  for(int y=0;y<pn.numFeat;y++)
			  {
				  if(pn.supportVecs.instance(h).attribute(y)!=peer.updatedTrainData.instance(f).attribute(y))
				  {
				   instEqual=false; 
				  }
			  }
			  if(instEqual==true)
			  {
				  notInSet=false;
			  }			  
			}
			//If this support vector is not in the set, then add it to the training data
			if(notInSet==true)
			{
			 peer.updatedTrainData.add(pn.supportVecs.instance(h));	
			}
		}
	    for(int h1=0;h1<peer.supportVecs.numInstances();h1++)
		{
			boolean notInSetPeer=true;
			for(int f1=0;f1<pn.updatedTrainData.numInstances();f1++)
			{	
//			  if(peer.supportVecs.instance(h1).equals(pn.updatedTrainData.instance(f1)))
//			  {
//				  notInSetPeer=false;
//			  }
				  boolean instEqualASG=true;	
				  for(int y=0;y<pn.numFeat;y++)
				  {
					  if(peer.supportVecs.instance(h1).attribute(y)!=pn.updatedTrainData.instance(f1).attribute(y))
					  {
					   instEqualASG=false; 
					  }
				  }
				  if(instEqualASG==true)
				  {
					  notInSetPeer=false;
				  }			  		  	
			}
			//If this support vector is not in the set, then add it to the training data
			if(notInSetPeer==true)
			{
			 pn.updatedTrainData.add(peer.supportVecs.instance(h1));	
			}
		}
}
	
	protected List<Node> getPeers(Node node) {
		Linkable linkable = (Linkable) node.getProtocol(lid);
		if (linkable.degree() > 0) {
			List<Node> l = new ArrayList<Node>(linkable.degree());			
			for(int i=0;i<linkable.degree();i++) {
				l.add(linkable.getNeighbor(i));
			}
			return l;
		}
		else
			return null;						
	}			

	// Comment inherited from interface
	/**
	 * This is the method where actual algorithm is implemented. This method gets    
	 * called in each cycle for each node.
	 * NOTE: The Gadget algo's iteration corresponds to the inner loop, so call this 
	 * once only, i.e. keep simulation.cycles 1
	 * @throws Exception 
	 */
	public void nextCycle(Node node, int pid) 
	{	
		// Gets the current cycle of Neuro-LibSVM
		int itrLibSVM=CDState.getCycle();
		// Initializes the Pegasos Node
		LibLinearNode pn = (LibLinearNode)node;
		resourcepath = pn.getResourcePath();
		peg_lambda = pn.getPegasosLambda();
//		max_iter = pn.getMaxIter();
//		exam_per_iter = pn.getExamPerIter();
		
		long startTime = System.nanoTime();
		// Build the model
	 try{
	     // create Model
		pn.libSVMClassifier = new LibSVM();
		String[] options  = new String[3];
	    options[0] = "-S 0";
	    options[1] = "-K 0";
	    options[2] = "-C 0.001";
	    pn.libSVMClassifier.setOptions(options);
		pn.libSVMClassifier.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
		// train classifier
		pn.libSVMClassifier.buildClassifier(pn.trainData);
		
		// evaluate classifier
        //Evaluation eval = new Evaluation(pn.trainData);
        //eval.evaluateModel(pn.libSVMClassifier,pn.testData);
       
        pn.pred = new double[pn.trainData.numInstances()];
        pn.predTest = new double[pn.testData.numInstances()];
        
		//pn.libSVMClassifier.getProbabilityEstimates();
		//result.libSVMClassifier = libsvmModel;
	    //Store the local predictions on the train set
		double[] retPred = new double[pn.trainData.numClasses()];
		for(int h=0;h<pn.trainData.numInstances();h++)
		{
			retPred=pn.libSVMClassifier.distributionForInstance(pn.trainData.instance(h));
			//System.out.println("Prob0 "+retPred[0]+" Pred1 " +retPred[1]);
		  //libSVMPred[h]=libsvmModel.classifyInstance(data.instance(h));
		  //pn.pred[h]=pn.libSVMClassifier.classifyInstance(pn.trainData.instance(h));	
			if(retPred[0]>retPred[1])
			{	
			  pn.pred[h]=0;
			}
			else
			{
				pn.pred[h]=1;
			}
		}
		System.out.println("Done predicting local train set.");
		double[] retPredTest=new double[pn.testData.numClasses()];			
		//Store the local predictions on the test set
		for(int hTest=0;hTest<pn.testData.numInstances();hTest++)
		{
		  retPredTest=pn.libSVMClassifier.distributionForInstance(pn.testData.instance(hTest));
		  //System.out.println("Prob0 "+retPredTest[0]+" Pred1" +retPredTest[1]);
		  //libSVMPred[h]=libsvmModel.classifyInstance(data.instance(h));
		  if(retPredTest[0]>retPredTest[1])
		  {
		   pn.predTest[hTest]=0;
		  }
		  else {pn.predTest[hTest]=1;}
		  //pn.predTest[hTest]=pn.libSVMClassifier.classifyInstance(pn.testData.instance(hTest));	
		}
		System.out.println("Done predicting local test set.");
		//Ensure that the local models are all built in cycle 0
		//Only then they can be used in nextCycle for gossiping.
			if(itrLibSVM>0)
			{	
				pushsum1(node, pn, pid);
			}
			for(int hT=0;hT<pn.testData.numInstances();hT++)
			{
			 if(pn.predTest[hT]>0)
				pn.predTest[hT]=1;
			 else pn.predTest[hT]=-1;
			}
	    }
	    catch(Exception ex)
	    {
	    	ex.printStackTrace();
	    }
		// Get the accuracy of the test set. 
		
		if (itrLibSVM % 5 == 0) 
		{
			try
			{
				String testFilePath=pn.getResourcePath() + "/" + "tst_" + pn.getID()+"/" + "tst_" +pn.getID()+".arff";
				 FileReader rTest = new FileReader(testFilePath);
				 Instances dataTest = new Instances (rTest);
				 // Make the last attribute be the class
				 int classIndex = dataTest.numAttributes()-1;
				 dataTest.setClassIndex(classIndex); 
				 pn.accuracy = getAccuracy2(pn.predTest,dataTest);
				 System.out.println("Accuracy from Neurocomputing Algo : " + pn.accuracy);
			} 
			catch (Exception e1)
			{
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
		}
		double trainTimeInDouble = (double)(System.nanoTime() - startTime)/1e9;
		
		String csv_filename_libSVM = resourcepath + "/run" + pn.numRun + "/node_" + pn.getID()  + ".csv";
		System.out.println("Storing in " + csv_filename_libSVM);
		String opString = pn.getID() + "," + itrLibSVM + "," ;
		opString +=  + pn.accuracy + ","+ (1.0 - pn.accuracy); 
		opString += ","+ trainTimeInDouble + "\n"; 
		
		// Write to file
		try
		{
			BufferedWriter bw_libSVM= new BufferedWriter(new FileWriter(csv_filename_libSVM, true));
			bw_libSVM.write(opString);
			bw_libSVM.close();
		}
		catch(Exception e)
		{
		 e.printStackTrace();		
		}

	}

	/**
	 * Selects a random neighbor from those stored in the {@link Linkable} protocol
	 * used by this protocol.
	 */
	protected Node selectNeighbor(Node node, int pid) {
		Linkable linkable = (Linkable) node.getProtocol(lid);
		if (linkable.degree() > 0) 
			return linkable.getNeighbor(
					CommonState.r.nextInt(linkable.degree()));
		else
			return null;
	}

	public static void writeIntoFile(String millis) {
		File file = new File("exec-time.txt");
		 
		// if file doesnt exists, then create it
		if (!file.exists()) {
			try {
				file.createNewFile();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		FileWriter fw;
		try {
			fw = new FileWriter(file.getAbsoluteFile(),true);

		BufferedWriter bw = new BufferedWriter(fw);
		bw.write(millis+"\n");
		bw.close();
		} catch (IOException e)
		 {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		

	}
	
	public void writeWtVec(String fName, double db)
	{
		BufferedWriter out = null;
		try
		{
			FileWriter fstream = new FileWriter(fName, true);
			out = new BufferedWriter(fstream);
			//IlpVector vLoss = new IlpVector(ls);
		    out.write(String.valueOf(db)); 
		    out.write("\n");
	    	}
		catch (IOException ioe) 
		{ioe.printStackTrace();}
		finally
		{
		if (out != null) 
	    {try {out.close();} catch (IOException e) {e.printStackTrace();}
	    }
		}
	}

	
// Function to write weights into the file.
	public void writeWeightsToFile(PegasosNode pn, String modelfilename)
	{
		String opString = "";
		for (int i = 0; i < pn.wtvector.length;i++) {
			if (pn.wtvector[i] != 0.0) {
				opString += i;
				opString += ":";
				opString += pn.wtvector[i];
				opString += " ";	
				
			}
			
			
		}
		
		// Write to file
		try {
		BufferedWriter bw = new BufferedWriter(new FileWriter(modelfilename));
		bw.write(opString);
		bw.close();
		}
		catch(Exception e) {
			
		}
	}
	
	

}

