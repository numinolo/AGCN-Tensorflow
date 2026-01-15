# Attentive-driven Graph Clustering Network

The original AGCN code was made with Torch from the author github [here](https://github.com/ZhihaoPENG-CityU/MM21---AGCN/).

The original paper could be found [here](https://arxiv.org/pdf/2108.05499).

---

## How to use
### 1. Preprocess Data
```python
import numpy as np
import networkx as nx
import tensorflow as tf

x = np.loadtxt('acm.txt') # import your data here
y = np.loadtxt('acm_label.txt') # import your data here
a = np.loadtxt('acm_graph.txt') # import your data here

N_CLUSTER = 3
LAYER = [500,500,2000,10]

graph = nx.Graph()
graph.add_nodes_from([i for i in range(y.shape[0])])
graph.add_edges_from(a)
a = nx.adjacency_matrix(graph).toarray()
a = np.array(a).astype(np.float32)
a = tf.cast(a, dtype=tf.float32) + tf.eye(tf.shape(a)[0])
rowsum = tf.reduce_sum(a, axis=1, keepdims=True)
rowsum = tf.where(tf.math.equal(rowsum, 0), tf.ones_like(rowsum), rowsum)
a = tf.divide(a, rowsum)

print('A:',a.shape)
print('X:',x.shape)
print('Y:',y.shape)
```
The adjacency matrix need to be normalize outside the model to reduce training time.

### 2. Pretrain AutoEncoder
```python
AE = AutoEncoder(x.shape[-1],layer=LAYER)
AE.compile('adam','mse')
AE.fit(x,x,batch_size=256,epochs=30)

emb = AE.encoder.predict(x)
kmeans = KMeans(n_clusters=N_CLUSTER,n_init=100)
pred = kmeans.fit_predict(emb)
AE.save('pretrained.keras')
```
### 3. Define model and training
```python
AE = tf.keras.models.load_model('pretrained.keras')
lambda1 = 0.1
lambda2 = 0.01
model = AGCN(N_CLUSTER,x.shape[-1],kmeans.cluster_centers_,LAYER,lambda1,lambda2,pretrained=AE)
model.compile(Adam(learning_rate=0.001))
model.fit([x,a],batch_size=x.shape[0],epochs=200)
```
