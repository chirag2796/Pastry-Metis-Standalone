import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# Load data
G = nx.karate_club_graph()
adj_matrix = nx.adjacency_matrix(G).toarray().astype(np.float32)
features = np.eye(adj_matrix.shape[0]).astype(np.float32)
labels = np.array([0 if G.nodes[i]['club']=='Mr. Hi' else 1 for i in range(G.number_of_nodes())]).astype(np.float32)

# Preprocess data
labels = tf.keras.utils.to_categorical(labels)

# Define GCN model
class GCN(Model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dense2 = layers.Dense(output_dim, activation='softmax')
        self.dropout = layers.Dropout(0.5)

    def call(self, inputs):
        x, A = inputs
        x = self.dropout(x)
        x = self.dense1(tf.matmul(A, x))
        x = self.dropout(x)
        x = self.dense2(tf.matmul(A, x))
        return x

# Define training loop
def train(model, x_train, y_train, epochs):
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            logits = model(x_train)
            loss_value = loss_fn(y_train, logits)
        
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_acc = np.mean(np.argmax(logits, axis=1) == np.argmax(y_train, axis=1))
        
        print(f"Epoch {epoch+1}, loss: {loss_value:.4f}, train accuracy: {train_acc:.4f}")

# Initialize and train model
input_dim = features.shape[1]
hidden_dim = 16
output_dim = labels.shape[1]
model = GCN(input_dim, hidden_dim, output_dim)
train(model, (features, adj_matrix), labels, epochs=200)
