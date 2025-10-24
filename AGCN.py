import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Input, LeakyReLU
from tensorflow.keras.losses import Loss, KLDivergence, MSE



class AutoEncoder(Layer):
    def __init__(self,n_layers,**kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.n_layers = n_layers # i.e. [256,500,500,2000,10]

    def build(self,ishape):
        # Encoder
        self.encoder = []
        for i,lyr in enumerate(self.n_layers[1:]): # i.e. [500,500,2000,10]
            en = Dense(lyr,name=f'encoder_{i+1}')
            self.encoder.append(en)

        # Decoder
        self.decoder = []
        for i,lyr in enumerate(list(reversed(self.n_layers))[1:]): # i.e. [2000,500,500,256]
            de = Dense(lyr,name=f'decoder_{i+1}')
            self.decoder.append(de)

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
        d = tf.cast(_encoder[-1],dtype=tf.float32)
        ii = 0
        for lyr in self.decoder:
            d = lyr(d)
            _decoder.append(d)

        return _encoder,_decoder



class GCN(Layer):
    def __init__(self,units,adj,active=True,**kwargs):
        super(GCN, self).__init__(**kwargs)
        self.units = units
        self.active = active
        self.dense = Dense(units)
        self.adj = adj

    def call(self,inputs):
        # Assume:
        # - X.shape = (5000,256)
        # - A.shape = (5000,5000)
        X = inputs

        res = tf.matmul(self.adj,self.dense(X)) # A(5000,5000) * X(5000,256) * W(256,units) -> (5000,units)
        if self.active:
            return tf.nn.leaky_relu(res,0.2) # Size (5000,units)
        else:
            return tf.nn.softmax(res, axis=-1) # Size (5000,units)



class AGCN_H(Layer):
    def __init__(self,units,adj,**kwargs):
        super(AGCN_H, self).__init__(**kwargs)
        self.units = units
        self.gcn = GCN(self.units,adj)
        self.dense = Dense(2)

    def call(self,inputs):
        # Assume:
        # - z.shape = (5000,500) | Neuron of each layer
        # - h.shape = (5000,500) | Neuron of each layer
        z,h = inputs

        # calculate attention of each z
        zh = tf.concat([z,h],-1) # Size (5000,1000)
        mi = self.dense(zh) # Size (5000,2)
        mi = tf.nn.leaky_relu(mi,0.2)
        mi = tf.nn.softmax(mi, axis=-1)
        mi = tf.math.l2_normalize(mi, axis=-1) # (5000,2)

        mi1 = tf.reshape(mi[:,0],(mi.shape[0],1)) # (5000,1)
        mi2 = tf.reshape(mi[:,1],(mi.shape[0],1)) # (5000,1)

        weighted_Z = mi1 * z
        weighted_H = mi2 * h

        Z = weighted_Z + weighted_H #  Size (5000,500)
        new_Z = self.gcn(Z) # Size (5000,units)
        return new_Z





class AGCN_S(Layer):
    def __init__(self,lyr,units,adj,**kwargs):
        super(AGCN_S, self).__init__(**kwargs)
        self.units = units # out neuron

        self.dense = Dense(lyr)
        self.gcn = GCN(self.units,adj,active=False)

    def call(self,inputs):
        Z = inputs
        # Assume 5000 sample data with 4 layers [500,500,2000,10] + Encoder Latest Layer [10]
        # Z = [z1,z2,z3,...] | Size (5,5000,(500/500/2000/10/10))
        U = tf.concat(Z,-1) # (5000,3020)
        U = self.dense(U) # (5000,5)
        U = tf.nn.leaky_relu(U,0.2)
        U = tf.nn.softmax(U, axis=-1)
        U = tf.math.l2_normalize(U, axis=-1) # attention weight of each layer -> (5000,5)

        wZ = [] # (5,5000,(500/500/2000/10/10))
        for i in range(len(Z)):
            ui = tf.reshape(U[:,i],(U.shape[0],1)) # (5000,1)
            wZ.append(ui * Z[i])
        ZZ = tf.concat(wZ,-1) # (5000,3020)
        nZ = self.gcn(ZZ) # (5000,units)
        return nZ



class AGCN(Model):
    def __init__(self, out_dim, mu, n_layers,adj,lambda1=1000,lambda2=1000,alpha=1,enc=None,dec=None,trainable=False,**kwargs):
        super(AGCN, self).__init__(**kwargs)
        self.n_layers = n_layers # [500,500,2000,10]
        self.out_dim = out_dim # total label
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.enc = enc
        self.dec = dec
        self.mu = mu # centroid
        self.agcns = AGCN_S(len(n_layers)+1,self.out_dim,adj,name='agcn_s')
        self.init_gcn = GCN(self.n_layers[0],adj)
        self.AGCNH = []
        self.adj = adj
        self.trainable = trainable # freeze the learning weight of auto encoder if set to false

    def build(self,ishape):
        # AGCN-H
        for i,lyr in enumerate(self.n_layers[1:]):
            ly = AGCN_H(lyr,self.adj,name=f'agcn_h_{i+1}')
            self.AGCNH.append(ly)
        # AE
        self.ae = AutoEncoder([ishape[-1]]+self.n_layers,name='auto_encoder')
        if self.enc != None:
            self.ae.encoder = self.enc
            for i in range(len(self.ae.encoder)):
                self.ae.encoder[i].trainable = self.trainable
        if self.dec != None:
            self.ae.decoder = self.dec
            for i in range(len(self.ae.decoder)):
                self.ae.decoder[i].trainable = self.trainable

    def kl_divergence(self,p, q):
        return tf.reduce_sum(p * tf.math.log(p / (q + 1e-8)), axis=1)

    def compute_q(self,features, centers):
        distances = tf.reduce_sum(tf.square(features - centers), axis=2)  # (n_samples, n_clusters)
        # Student's t-distribution
        q = tf.pow(1.0 + distances / self.alpha, -(self.alpha + 1.0) / 2.0)
        # Normalize to get probabilities
        q = q / tf.reduce_sum(q, axis=1, keepdims=True)
        return q

    def compute_p(self,q):
        numerator = tf.square(q) / tf.reduce_sum(q, axis=0, keepdims=True)  # Soft cluster frequencies
        p = numerator / tf.reduce_sum(numerator, axis=1, keepdims=True)     # Normalize per sample
        return p

    def call(self, inputs):
        X = inputs

        # variable
        _agcnh = []

        # AutoEncoder
        _encoder,_decoder = self.ae(X)

        # AGCN-H
        z = self.init_gcn(X)
        _agcnh.append(z)
        for i,lyr in enumerate(self.AGCNH):
            h = _encoder[i]
            z = lyr((z,h))
            _agcnh.append(z)

        # AGCN-S
        _agcnh.append(_encoder[-1])
        pred = self.agcns(_agcnh) # (5000,10)

        # Calc q
        # -> H_latest = (1000,10) -> (1000,1,10)
        # -> cl = (10,10) -> (1,10,10)
        hl = tf.expand_dims(_encoder[-1],axis=1)
        cl = tf.expand_dims(self.mu,axis=0)
        q = self.compute_q(hl,cl)

        # calculate p
        p = self.compute_p(q)

        # calculate loss
        kl_loss = tf.reduce_mean(self.kl_divergence(p, q))
        ce_loss = tf.reduce_mean(self.kl_divergence(p, pred))
        re_loss = tf.reduce_mean(MSE(X,_decoder[-1]))

        loss = self.lambda1 * kl_loss + self.lambda2 * ce_loss + re_loss
        self.add_loss(loss)
        return pred
