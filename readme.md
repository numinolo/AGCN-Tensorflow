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

# init
CLUSTER_N = 10
LAYERS = [500,500,2000,10]

# read sample data
X = np.loadtxt('usps.txt')

# read label data
LABEL = np.loadtxt('usps_label.txt')
LABEL = LABEL.reshape((LABEL.shape[0],1))

# read edges data
A = np.loadtxt('usps10_graph.txt')
graph = nx.Graph()
graph.add_nodes_from([i for i in range(LABEL.shape[0])])
graph.add_edges_from(A)

# create adjacency matrix and normalize it for GCN
A = nx.adjacency_matrix(graph).toarray()
A = np.array(A).astype(np.float32)
A = tf.cast(A,dtype=tf.float32) + tf.eye(tf.shape(A)[0])
D = tf.reduce_sum(A, axis=1)
D = tf.linalg.diag(D)
D = tf.math.pow(D,-0.5)
D = tf.where(tf.math.is_inf(D),0.0,D)
A = tf.matmul(tf.matmul(D, A), D)
```
The adjacency matrix need to be normalize outside the model to reduce training time. If you normalize inside the model, it could increase your training time by each epoch approximately 2x longer.

### 2. Pretrain AutoEncoder and get initial cluster center
```python
from sklearn.cluster import KMeans

# Auto Encoder for Pretrain
@tf.keras.utils.register_keras_serializable(package='AE')
class AE(Model):
    def __init__(self,n_layers,pre_trained=False,enc=None,**kwargs):
        super(AE, self).__init__(**kwargs)
        self.n_layers = n_layers # i.e. [256,500,500,2000,10]
        self.pre_trained = pre_trained

        # Encoder
        self.encoder = []
        if enc == None:
            for i,lyr in enumerate(self.n_layers[1:]): # i.e. [500,500,2000,10]
                en = Dense(lyr,name=f'encoder_{i+1}')
                self.encoder.append(en)
        else:
            self.encoder = enc

        # Decoder
        self.decoder = []
        for i,lyr in enumerate(list(reversed(self.n_layers))[1:]): # i.e. [2000,500,500,256]
            de = Dense(lyr,name=f'decoder_{i+1}')
            self.decoder.append(de)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,'n_layers':self.n_layers}

    def call(self,X):
        # Assume:
        # - X.shape = (5000,256)

        # variable
        _encoder = []
        _decoder = []

        # Encoder
        h = X
        for lyr in self.encoder:
            h = lyr(h)
            _encoder.append(h)

        # Decoder
        d = _encoder[-1]
        for lyr in self.decoder:
            d = lyr(d)
            _decoder.append(d)

        if self.pre_trained:
            return _decoder[-1]
        else:
            return _encoder,_decoder

pre_trained_model = AE([X.shape[-1]]+LAYERS,pre_trained=True)
pre_trained_model.compile(optimizer=Adam(learning_rate=0.001),loss='mse')
pre_trained_model.fit(X,X,epochs=30,batch_size=X.shape[0])

pre_trained_model.pre_trained = False
_enc,_dec = pre_trained_model.predict(X)

# initialize cluster centers mu using kmeans on AE final latent
kmeans = KMeans(n_clusters=CLUSTER_N, n_init=20)
kmeans.fit_predict(_enc[-1])
mu = kmeans.cluster_centers_
```
### 3. Define model and training
```python
l1 = 1000
l2 = 1000
lr = 0.001
model = AGCN(CLUSTER_N,mu,LAYERS,A,lambda1=l1,lambda2=l2,enc=pre_trained_model.encoder,dec=pre_trained_model.decoder,trainable=True)
model.compile(optimizer=Adam(learning_rate=lr))
model.fit(x,label,batch_size=x.shape[0],epochs=200)
```
Here's the hardest part, this model is sensitive to lambda1 and lambda2.

Another different dataset might have different lambda1 and lambda2. If it set wrong, the result will likely collapsed to 1 cluster only.

---

## Notes
However, seems like this code is not perfect yet. This code might have performance issue or wrong implementation.
Feel free to use or improve this code.
