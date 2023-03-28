package FederatedML_Amazon;

import rice.p2p.commonapi.*;
import rice.p2p.scribe.*;
import rice.pastry.commonapi.PastryIdFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class MyScribeClient implements ScribeClient, Application {
	int seqNum = 0;
	int int_random;
	CancellableTask publishTask;
	Scribe myScribe;
	Topic myTopic;
	List<byte[]> model_list;
	ArrayList<String> model_list_names;
	protected Endpoint endpoint;
	byte[] bytes;
	String my_path;
	public MyScribeClient(Node node){
		// make a list to save the received training results (for the head node)
		this.model_list = new ArrayList<byte[]>();
		this.model_list_names = new ArrayList<String>();

		// make a placeholder to save a model zip file
		this.endpoint = node.buildEndpoint(this, "myinstance");
		
		// model zip and bytes
		this.bytes = new byte[0];
		// this.my_path = "/Users/taehwan/Desktop/Research/pastry_amazon/src/rice/tutorial/FederatedML_Amazon";
		
		// this.my_path = "/home/ec2-user/FederatedML";
		// this.my_path = "/home/chirag/fl/MLP/FederatedML";
		this.my_path = "/home/chirag/fl/keras-gnn/cora_partition_metis/FederatedML";
		
		//construct Scribe
		myScribe = new ScribeImpl(node, "myScribeInstance");

		//construct the topic
		myTopic = new Topic(new PastryIdFactory(node.getEnvironment()), "example topic");
		System.out.println("myTopic = "+myTopic);

		endpoint.register();
	}


	public static String getResourcePercentageString(HashMap<String, Float> app_ids_resource_percentages){
		String app_ids_arguments_line = "";
		for (String name: app_ids_resource_percentages.keySet()) {
			String key = name.toString();
			String value = app_ids_resource_percentages.get(name).toString();
			app_ids_arguments_line += key + " " + value + " ";
		}

		// app_ids_arguments_line = "\"" + app_ids_arguments_line.trim() + "\"";
		return app_ids_arguments_line;
	}



	// generate partitions
	public void generatePartitions(HashMap<String, Float> app_ids_resource_percentages) throws InterruptedException{
		System.out.println("Data partitioning starts...");

		String app_ids_arguments_line = MyScribeClient.getResourcePercentageString(app_ids_resource_percentages);
		// for (String name: app_ids_resource_percentages.keySet()) {
		// 	String key = name.toString();
		// 	String value = app_ids_resource_percentages.get(name).toString();
		// 	app_ids_arguments_line += key + " " + value + " ";
		// }

		// app_ids_arguments_line = "\"" + app_ids_arguments_line.trim() + "\"";

		// System.out.println(app_ids_arguments_line);


		ProcessBuilder processBuilder = new ProcessBuilder();

		String[] command = {this.my_path+"/partition_data.sh", app_ids_arguments_line};
		processBuilder.command(command);
		try{
			Process process = processBuilder.start();
			StringBuilder output = new StringBuilder();
			BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
			String line;
			while ((line = reader.readLine()) != null){
				output.append(line+"\n");
			}
			int exitVal = process.waitFor();
			if (exitVal == 0) {
				System.out.println("Data partitioned successfully");
				System.out.println(output);
			} else {
				System.out.println("Something abnormal happened during data partitioning");
			}

		}catch (IOException e) {
			e.printStackTrace();
		}
	}

	// build initial model
	public void buildModel() throws InterruptedException{
		System.out.println("Initial model building starts...");
		ProcessBuilder processBuilder = new ProcessBuilder();
		String[] command = {this.my_path+"/init_job.sh"};
		processBuilder.command(command);
		try{
			Process process = processBuilder.start();
			StringBuilder output = new StringBuilder();
			BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
			String line;
			while ((line = reader.readLine()) != null){
				output.append(line+"\n");
			}
			int exitVal = process.waitFor();
			if (exitVal == 0) {
				System.out.println("Successfully save model as init_model_0.h5");
				System.out.println(output);
			} else {
				System.out.println("Something abnormal happened during init model building");
			}

		}catch (IOException e) {
			e.printStackTrace();
		}
}

	//convert a byte array to a file
	public void convertFromByte(String nodetype, int iteration) throws IOException, ClassNotFoundException{
		if (nodetype.equals("head")){
			System.out.println("Convert the received training results (byte data) into files....");
			int fileNum = 0;
			for (int index = 0; index < this.model_list.size(); index++){
				// Path path = Paths.get(this.my_path+"/Received_Models/model_"+fileNum+".h5");
				Path path = Paths.get(this.my_path+"/Received_Models/model_" + this.model_list_names.get(index) + "_" +fileNum+".h5");
				Files.write(path, this.model_list.get(index));
				fileNum++;
			}
			this.model_list = new ArrayList<byte[]>(); // empty the list
			this.model_list_names = new ArrayList<String>();
		}
		else if (nodetype.equals("child")){
			System.out.println("Convert the received byte array to a file... (child node, "+this.endpoint.getId()+")");
			Path path = Paths.get(this.my_path+"/init_model_"+this.endpoint.getId()+"_"+Integer.toString(iteration)+".h5");
			Files.write(path, this.bytes);
		}
		
		System.out.println("Convert is done...");
	}

	// convert a file to a byte array
	public void convertByte(String nodetype, int iteration) throws IOException{
		String filePath = "";
		if (nodetype.equals("head")){
			System.out.println("Convert a file to byte array... (head node, "+this.endpoint.getId()+")");
			filePath = this.my_path+"/init_model_"+Integer.toString(iteration)+".h5";
		}

		else if (nodetype.equals("child")){
			System.out.println("Convert a file to byte array... (child node, "+this.endpoint.getId()+")");
			filePath = this.my_path+"/new_model_"+this.endpoint.getId()+"_"+Integer.toString(iteration)+".h5";
		}
		//file to bytes[]
		this.bytes = Files.readAllBytes(Paths.get(filePath));
		System.out.println("Convert is done...");
	}

	// children node train the model and save the trained_model as new_model.h5
	public void trainModel(int iteration) throws InterruptedException{
		System.out.println("Model training starts for node "+this.endpoint.getId());
		ProcessBuilder processBuilder = new ProcessBuilder();
		// argument = endpoint id num

		String[] command = {this.my_path+"/train_model.sh", this.endpoint.getId().toString(), Integer.toString(iteration)};
		processBuilder.command(command);
		try {
			Process process = processBuilder.start();
			StringBuilder output = new StringBuilder();
			BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
			String line;
			while ((line = reader.readLine()) != null){
				output.append(line+"\n");
			}
			int exitVal = process.waitFor();
			if (exitVal == 0) {
				System.out.println("Success saving model as new_model_"+this.endpoint.getId().toString()+"_"+Integer.toString(iteration)+".h5");
				System.out.println(output);
				// this.zip = new ZipFile("/Users/taehwan/Desktop/Research/FreePastry/pastry/src/rice/tutorial/FederatedML/init_model.zip");
			} else {
				System.out.println("Something abnormal happened during training model for "+this.endpoint.getId().toString());
			}
		}catch (IOException e) {
			e.printStackTrace();
		}
	}

	// head node combines all the models (at the end of learning process)
	public void combineModels(int iteration, HashMap<String, Float> app_ids_resource_percentages) throws InterruptedException{
		System.out.println("Model combining starts (head node)");


		String app_ids_arguments_line = MyScribeClient.getResourcePercentageString(app_ids_resource_percentages);



		ProcessBuilder processBuilder = new ProcessBuilder();
		String[] command = {this.my_path+"/combine_model.sh", Integer.toString(iteration), app_ids_arguments_line};
		processBuilder.command(command);
		try {
			Process process = processBuilder.start();
			StringBuilder output = new StringBuilder();
			BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
			String line;
			while ((line = reader.readLine()) != null){
				output.append(line+"\n");
			}
			int exitVal = process.waitFor();
			if (exitVal == 0) {
				System.out.println("Successfully saving model as init_model_"+Integer.toString(iteration)+".h5");
				System.out.println(output);
				
			} else {
				System.out.println("Something abnormal happened during combining models... (Iteration :"+Integer.toString(iteration)+")");
			}
		}catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void subscribe(){
		myScribe.subscribe(myTopic, this);
	}

	// this gets called first
	public void startPublishTask(){
		publishTask = endpoint.scheduleMessage(new PublishContent(), 5000, 5000);  
	}
	// this gets called second
	// headnode received a message
	// sendmulticast
	public void deliver(Id id, Message message){
		System.out.println(this.endpoint.getId()+" received "+message); //this: headnode
		// if (message instanceof PublishContent){
		// 	sendMulticast();
		// }
		if (message instanceof MyMsg){
			System.out.println("Bytes training results transferred to Head Node");
			byte[] bytes = ((MyMsg)message).bytes;
			this.model_list.add(bytes);
			
			String message_string = message.toString();
			// System.out.println(message_string);
			// System.out.println("MyMsg from ".length());

			String from_model_name = message_string.substring(message_string.indexOf("MyMsg from ")+"MyMsg from ".length(), message_string.indexOf(" to "));
			// System.out.println(from_model_name);
			this.model_list_names.add(from_model_name);
		}
	}

	public void deliver(Topic topic, ScribeContent content){
		System.out.println("MyScribeClient.deliver("+topic+","+content+")");
		this.bytes = ((MyScribeContent)content).bytes;
		// System.out.println(this.bytes);
		System.out.println(this.endpoint.getId()+"Received Bytes!");
		if (((MyScribeContent)content).from == null) {
			new Exception("Stack Trace").printStackTrace();
		}
	}

	// PublishContent is used to publish a model from headnode to children nodes
	public void sendMulticast(){
		System.out.println("Node "+endpoint.getLocalNodeHandle()+" broadcasting "+seqNum); // headnode start broadcasting
		MyScribeContent myMessage = new MyScribeContent(endpoint.getLocalNodeHandle(), seqNum, this.bytes);
		myScribe.publish(myTopic, myMessage);
		seqNum++;
	}

	public void childAdded(Topic topic, NodeHandle child) {

	}

	public void subscribeFailed(Topic topic) {
		System.out.println("MyScribeClient.childFailed("+topic+")");
  	}
  	public void childRemoved(Topic topic, NodeHandle child) {
		System.out.println("MyScribeClient.childRemoved("+topic+","+child+")");
  	}

	public boolean forward(RouteMessage message) {
		return true;
	}

	public void update(NodeHandle handle, boolean joined) {

	}
	public boolean anycast(Topic topic, ScribeContent content) {
    boolean returnValue = myScribe.getEnvironment().getRandomSource().nextInt(3) == 0;
    System.out.println("MyScribeClient.anycast("+topic+","+content+"):"+returnValue);
    return returnValue;
  	}

	// class PublishContent implements Message {

	// 	byte[] bytes;
	// 	public PublishContent(byte[] bytes){
	// 		this.bytes = bytes;
	// 	}
	// 	public int getPriority(){
	// 		return MAX_PRIORITY;
	// 	}
	// 	public byte[] getBytes(){
	// 		return this.bytes;
	// 	}

	// }

	public boolean isRoot() {
		return myScribe.isRoot(myTopic);
	}
	public NodeHandle getParent(){
		return ((ScribeImpl)myScribe).getParent(myTopic);
	}

	public NodeHandle[] getChildren(){
		return myScribe.getChildren(myTopic);
	}

	class PublishContent implements Message {
	    public int getPriority() {
	      return MAX_PRIORITY;
	    }
  }
	// route message
	public void routeMyMsg(NodeHandle nh){
		System.out.println(this.endpoint.getId()+" sending to parent node training results.");
		Message msg = new MyMsg(this.endpoint.getId(), nh.getId(), this.bytes);
		this.endpoint.route(null, msg, nh);
		this.bytes = new byte[0]; // reset byte array
	}

}