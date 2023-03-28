package FederatedML_Amazon;

import rice.p2p.commonapi.NodeHandle;
import rice.p2p.scribe.ScribeContent;

public class MyScribeContent implements ScribeContent {
	NodeHandle from;

	int seq;
	byte[] bytes;

	public MyScribeContent(NodeHandle from, int seq, byte[] bytes){
		this.from = from;
		this.seq = seq;
		this.bytes = bytes;
	}

	public String toString(){
		return "MyScribeContent #"+seq+" from "+from+". BYTES TRANSFERED";
	}
}