package FederatedML_Amazon;

import rice.environment.Environment;
import rice.p2p.commonapi.NodeHandle;
import rice.pastry.*;
import rice.pastry.socket.SocketPastryNodeFactory;
import rice.pastry.standard.RandomNodeIdFactory;

import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.util.HashSet;
import java.util.Set;
import java.util.Vector;
import java.util.zip.ZipEntry;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.HashMap;

import java.io.FileWriter;



public class MLSend {
	// apps stand for the vector space that contains all the scribe client applications
	// Vector<MyScribeClient> apps = new Vector<MyScribeClient>();
	// we have a parent client
	// Vector<MyScribeClient> apps = new Vector<MyScribeClient>();

	Vector<MyScribeClient> apps = new Vector<MyScribeClient>();
	Vector<PastryNode> nodes = new Vector<PastryNode>();
	Scanner scanner = new Scanner(System.in);

	int numIterations = 51;
	double[] iterationTimes = new double[numIterations];

	public MLSend(int bindport, InetSocketAddress bootaddress,
		int numNodes, Environment env) throws Exception {
		// generate a random node id
		NodeIdFactory nidFactory = new RandomNodeIdFactory(env);

		// construct the PastryNodeFactory; how we use socket
		PastryNodeFactory factory = new SocketPastryNodeFactory(nidFactory, bindport, env);
		

		// loop to construct # nodes
		// All the nodes are joining the ring
		for (int curNode = 0; curNode < numNodes; curNode++){
			// construct a new node
			PastryNode node = factory.newNode();
			nodes.add(node);
			node.boot(bootaddress);		
			// env.getTimeSource().sleep(10000);
			env.getTimeSource().sleep(10000);
			// the node may require sending several messages to fully boot into the ring
			synchronized(node){
				while(!node.isReady() && !node.joinFailed()){
					node.wait(3500);
					if (node.joinFailed()){
						throw new IOException("Could not join the FreePastry ring. Reason:"+node.joinFailedReason());
					}
				}
			}
		System.out.println("Finished creating new node: "+node);
		}

		// env.getTimeSource().sleep(15000);
		env.getTimeSource().sleep(15000);

		//construct a new scribe application
		// All the nodes are joining applications
		for (int curNode = 0; curNode < numNodes; curNode++){
			PastryNode curr_node = nodes.get(curNode);
			MyScribeClient app = new MyScribeClient(curr_node);
			// env.getTimeSource().sleep(10000);
			env.getTimeSource().sleep(10000);
			app.subscribe();
			apps.add(app);
		}

		// String[] app_ids = new String[apps.size()];
		ArrayList<String> app_ids = new ArrayList<String>();
		for (int curNode = 0; curNode < numNodes; curNode++){
			MyScribeClient app = apps.get(curNode);
			if (!app.isRoot() && app.getChildren().length == 0){ // child node
				app_ids.add(app.endpoint.getId().toString());
			}
		}

		float[] resource_percentages = new float[]{0.5F, 0.3F, 0.2F};
		// float[] resource_percentages = new float[]{0.6F, 0.4F};
		// float[] resource_percentages = new float[]{0.34F, 0.33F, 0.33F};
		HashMap<String, Float> app_ids_resource_percentages = new HashMap<String, Float>();
		for (int i = 0; i < app_ids.size();i++) 
		{ 		      
			app_ids_resource_percentages.put(app_ids.get(i), resource_percentages[i]); 		
		}   
		
		// System.out.println("APP IDs");
		// for (int i = 0; i < app_ids.size();i++) 
		// { 		      
		// 	System.out.println(app_ids.get(i)); 		
		// }   

		for (String name: app_ids_resource_percentages.keySet()) {
			String key = name.toString();
			String value = app_ids_resource_percentages.get(name).toString();
			System.out.println(key + " " + value);
		}

		// env.getTimeSource().sleep(10000);
		env.getTimeSource().sleep(10000);
		// System.out.println("Enter any character to continue:");
		// String line = scanner.nextLine();

		for (int iteration = 0; iteration < numIterations; iteration++){
			long startTime = System.currentTimeMillis();
			System.out.println("Iteration: "+Integer.toString(iteration)+" begins...");
			
			// if root node
			for (int curNode = 0; curNode < numNodes; curNode++){
				MyScribeClient app = apps.get(curNode);
				if (app.isRoot()){ // root node
					if (iteration == 0){
						// The first app (source node) saves the initial MLP model
						app.generatePartitions(app_ids_resource_percentages);
						app.buildModel(); // build initial model
					}
					else{
						app.combineModels(iteration, app_ids_resource_percentages);
					}
					app.convertByte("head", iteration); // convert the zip file to byte
					// long endTime = System.nanoTime();
					// long duration = (endTime - startTime);
					// System.out.println("First head node "+duration);
					app.sendMulticast(); 
					// long startTime = System.nanoTime();
					// System.out.println("Multicast starts at: "+startTime);
					}
				// send multicast (send model byte data to children nodes)
				// wait till the children nodes receive a model
				// else children node, wait
				// else{
				// 	env.getTimeSource().sleep(20000);
				// }

				// env.getTimeSource().sleep(20000);
			}

				// if (app.isRoot()){
				// 	app.sendMulticast(); 
				// 	env.getTimeSource().sleep(20000);
				// }	
				// else{
				// 	env.getTimeSource().sleep(20000);
				// }

			// go through children nodes the bottom nodes
			// train a seperate model and send the training results to the head node
			int count = 0;
			int start_index = 0;
			Set<Integer> hash_set = new HashSet<Integer>();
			while (count != numNodes){
				int real_index = start_index%numNodes;
				if (!hash_set.contains(real_index)){ // if not visited
					// hash_set.add(real_index);
					// count += 1;
					MyScribeClient app = apps.get(real_index);
					// root node
					if (app.isRoot()){
						// check all the traing results are gathered
						if (app.model_list.size() == app.getChildren().length){
							System.out.println("Head Node");
							long combiningTime = System.nanoTime();
							System.out.println("Multiple models combined at: "+combiningTime);

							// head node combines all the results and generates a new h5 file
							app.convertFromByte("head", iteration);

							// head node is visited
							hash_set.add(real_index);
							count += 1;
						}
						
					}
					else if (app.getChildren().length == 0){
						System.out.println("Child Node");
						// System.out.println(app);
						// if bytes are not received, lets wait until we receive it
						// System.out.println(app.bytes);
						while (app.bytes.length == 0){
							env.getTimeSource().sleep(50);
						}
						long receiveTime = System.nanoTime();
						System.out.println("Multicast received at: "+receiveTime);
						app.convertFromByte("child", iteration); // convert the byte file to h5 file and save it 
						NodeHandle parent_node = app.getParent();
						app.trainModel(iteration); // train the model for the child node			
						app.convertByte("child", iteration); // convert a file into a byte array
						app.routeMyMsg(parent_node); // send a byte array to the parent node

						// check this index is visited
						hash_set.add(real_index);
						count += 1;

					}
					
					// node in between (serve as a children and head node)
					else {
						if (app.model_list.size() == app.getChildren().length){
							System.out.println("Parent Node");
							app.convertFromByte("head", iteration);
							app.combineModels(iteration, app_ids_resource_percentages);
							app.convertByte("head", iteration);
							NodeHandle parent_node = app.getParent();
							app.routeMyMsg(parent_node); // send a byte array to the parent node

							// parent node is visited
							hash_set.add(real_index);
							count += 1;
						}

				}
			}
			start_index += 1;

		}
			
		long endTime = System.currentTimeMillis();
		double timeElapsed = (endTime - startTime) / 1000.0;
		iterationTimes[iteration] = timeElapsed;
	
	
	}	
	
	
	String csvFile = "timings.csv";
	String[] header = {"Iteration", "Time Elapsed (s)"};
	writeCSV(csvFile, header, iterationTimes);

	}


	
	public static void isSame(ZipEntry child, ZipEntry original) {
		boolean isHashcodeEquals = child.hashCode() == original.hashCode(); 
		if (isHashcodeEquals){
				System.out.println("Received File is the same as original!");
		}
		else{
			System.out.println("They are not the same");
			}
	}

	public static void writeCSV(String csvFile, String[] header, double[] iterationTimes) {
        FileWriter writer;

        try {
            writer = new FileWriter(csvFile);
            for (int i = 0; i < header.length - 1; i++) {
                writer.append(header[i]).append(",");
            }
            writer.append(header[header.length - 1]).append("\n");

            for (int i = 0; i < iterationTimes.length; i++) {
                writer.append(String.format("%d,%.3f\n", i+1, iterationTimes[i]));
            }

            writer.close();
            System.out.println("CSV file written successfully!");
        } catch (IOException e) {
            System.out.println("Error writing to CSV file: " + e.getMessage());
        }
    }


	// /**
 //   * Note that this function only works because we have global knowledge. Doing
 //   * this in an actual distributed environment will take some more work.
 //   * 
 //   * @param apps Vector of the applicatoins.
 //   */
	// public static void printTree(Vector<MyScribeClient> apps) {
	//     // build a hashtable of the apps, keyed by nodehandle
	//     Hashtable<NodeHandle, MyScribeClient> appTable = new Hashtable<NodeHandle, MyScribeClient>();
	//     Iterator<MyScribeClient> i = apps.iterator();
	//     while (i.hasNext()) {
	//       MyScribeClient app = (MyScribeClient) i.next();
	//       appTable.put(app.endpoint.getLocalNodeHandle(), app);
	//     }
	//     NodeHandle seed = ((MyScribeClient) apps.get(0)).endpoint
	//         .getLocalNodeHandle();

	//     // get the root
	//     NodeHandle root = getRoot(seed, appTable);

	//     // print the tree from the root down
	//     recursivelyPrintChildren(root, 0, appTable);
 //  }

 //  /**
 //   * Recursively crawl up the tree to find the root.
 //   */
	// public static NodeHandle getRoot(NodeHandle seed, Hashtable<NodeHandle, MyScribeClient> appTable) {
	//     MyScribeClient app = (MyScribeClient) appTable.get(seed);
	//     if (app.isRoot())
	//       return seed;
	//     NodeHandle nextSeed = app.getParent();
	//     return getRoot(nextSeed, appTable);
 //  }

 //  /**
 //   * Print's self, then children.
 //   */
	// public static void recursivelyPrintChildren(NodeHandle curNode,
 //      int recursionDepth, Hashtable<NodeHandle, MyScribeClient> appTable) {
	// 	// print self at appropriate tab level
	// 	String s = "";
	// 	for (int numTabs = 0; numTabs < recursionDepth; numTabs++) {
	// 	  s += "  ";
	// 	}
	// 	s += curNode.getId().toString();
	// 	System.out.println(s);

	// 	// recursively print all children
	// 	MyScribeClient app = (MyScribeClient) appTable.get(curNode);
	// 	NodeHandle[] children = app.getChildren();
	// 	for (int curChild = 0; curChild < children.length; curChild++) {
	// 	  recursivelyPrintChildren(children[curChild], recursionDepth + 1, appTable);
	// 	}
 //  }


	public static void main(String[] args) throws Exception {

    // Loads pastry configurations
    Environment env = new Environment();

    // disable the UPnP setting (in case you are testing this on a NATted LAN)
    env.getParameters().setString("nat_search_policy","never");
    
    try {
      // the port to use locally
      int bindport = Integer.parseInt(args[0]);

      // build the bootaddress from the command line args
      InetAddress bootaddr = InetAddress.getByName(args[1]);
      int bootport = Integer.parseInt(args[2]);
      InetSocketAddress bootaddress = new InetSocketAddress(bootaddr, bootport);

      // the port to use locally
      int numNodes = Integer.parseInt(args[3]);

      // launch our node!
      MLSend dt = new MLSend(bindport, bootaddress, numNodes,
          env);

    } catch (Exception e) {
      // remind user how to use
      System.out.println("Usage:");
      System.out
          .println("java [-cp FreePastry-<version>.jar] rice.tutorial.scribe.ScribeTutorial localbindport bootIP bootPort numNodes");
      System.out
          .println("example java rice.tutorial.scribe.ScribeTutorial 9001 pokey.cs.almamater.edu 9001 10");
      throw e;
    }
  }
}