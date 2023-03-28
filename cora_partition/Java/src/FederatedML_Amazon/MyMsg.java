package FederatedML_Amazon;

import rice.p2p.commonapi.Id;
import rice.p2p.commonapi.Message;

  // MyMsg is used to route a trained model from child node to headnode
public class MyMsg implements Message {
  /**
   * Where the Message came from.
   */
  Id from;
  /**
   * Where the Message is going.
   */
  Id to;
  byte[] bytes;
  /**
   * Constructor.
   */
  public MyMsg(Id from, Id to, byte[] bytes) {
    this.from = from;
    this.to = to;
    this.bytes = bytes;
  }
  
  public String toString() {
    return "MyMsg from "+from+" to "+to;
  }

  public byte[] getBytes(){
  	return this.bytes;
  }
  /**
   * Use low priority to prevent interference with overlay maintenance traffic.
   */
  public int getPriority() {
    return Message.LOW_PRIORITY;
  }
}