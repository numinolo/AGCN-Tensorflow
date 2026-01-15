import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam
import numpy as np



@tf.keras.utils.register_keras_serializable(package='AutoEncoder')
class AutoEncoder(Model):
    def __init__(self,input_dim,layer=[500,500,2000,10],**kwargs):
        super(AutoEncoder,self).__init__(**kwargs)
        self.layer = layer
        self.input_dim = input_dim

        tmp = [layers.Input(shape=(input_dim,))]
        for i in layer[:-1]:
            tmp.append(layers.Dense(i,activation='relu'))
        tmp.append(layers.Dense(layer[-1]))
        self.encoder = Sequential(tmp)

        tmp = [layers.Input(shape=(layer[-1],))]
        for i in layer[::-1][1:]:
            tmp.append(layers.Dense(i,activation='relu'))
        tmp.append(layers.Dense(input_dim))
        self.decoder = Sequential(tmp)

    def build(self,input_shape):
        super(AutoEncoder, self).build(input_shape)

    def get_config(self):
        base_config = super().get_config()
        config = {
            'input_dim':self.input_dim,
            'layer':self.layer,
        }
        return {**base_config, **config}

    def call(self,x):
        enc = self.encoder(x)
        return self.decoder(enc)



class GCN(layers.Layer):
    def __init__(self,units,lr=True):
        super().__init__()
        self.units = units
        self.lr = lr

    def build(self,input_shape):
        super(GCN, self).build(input_shape)
        self.w = self.add_weight(
            shape=(input_shape[0][-1],self.units),
            trainable=True,
            initializer='glorot_uniform',
        )

    def call(self,inputs):
        x,a = inputs
        res = tf.matmul(a,tf.matmul(x,self.w))
        if self.lr:
            return tf.nn.leaky_relu(res, alpha=0.2)
        return tf.nn.softmax(res,axis=-1)

class AGCN_H(layers.Layer):
    def __init__(self,units):
        super().__init__()
        self.units = units
        self.gcn = GCN(units)
        self.fc_att = layers.Dense(2)

    def build(self,input_shape):
        super(AGCN_H, self).build(input_shape)

    def call(self,inputs):
        z,h,a = inputs
        zh = tf.concat([z,h],-1)
        att = self.fc_att(zh)
        att = tf.nn.leaky_relu(att, alpha=0.2)
        att = tf.nn.softmax(att,axis=-1)
        att = tf.nn.l2_normalize(att,axis=1)
        m1,m2 = tf.expand_dims(att[:,0],-1),tf.expand_dims(att[:,1],-1)
        nz = m1 * z
        nh = m2 * h
        return self.gcn((nz+nh,a))

class AGCN_S(layers.Layer):
    def __init__(self,units,input_len):
        super().__init__()
        self.units = units
        self.input_len = input_len
        self.gcn = GCN(units,lr=False)
        self.fc_att = layers.Dense(input_len)

    def build(self,input_shape):
        super(AGCN_S, self).build(input_shape)

    def call(self,inputs):
        inp = tf.concat(inputs,-1)
        att = self.fc_att(inp)
        att = tf.nn.leaky_relu(att, alpha=0.2)
        att = tf.nn.softmax(att,axis=-1)
        att = tf.nn.l2_normalize(att,axis=1)

        nx = []
        for i in range(self.input_len):
            nx.append(inputs[i] * tf.expand_dims(att[:,i],-1))
        nx = tf.concat(nx,-1)
        return self.gcn((nx,a))

class AGCN(Model):
    def __init__(self,n_cluster,input_dim,centroid,layer,lambda1,lambda2,pretrained=None):
        super().__init__()
        self.n_cluster = n_cluster
        self.input_dim = input_dim
        self.centroid = centroid
        self.layer = layer
        self.pretrained = pretrained
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.centroid_shape = np.shape(centroid)
        self.eps = 1e-10

    def build(self,input_shape):
        # Auto Encoder
        if self.pretrained == None:
            self.auto_encoder = AutoEncoder(self.input_dim,layer=self.layer)
        else:
            self.auto_encoder = self.pretrained

        # AGCN-H
        self.init_gcn = GCN(self.layer[0])
        self.agcnh = []
        for i in self.layer[1:]:
            self.agcnh.append(AGCN_H(i))

        # AGCN-S
        self.agcns = AGCN_S(self.n_cluster,len(self.layer)+1)

        self.centroid_weight = self.add_weight(
            shape=self.centroid_shape,
            trainable=True,
            initializer=tf.constant_initializer(self.centroid),
        )
        super(AGCN, self).build(input_shape)

    def call(self,inputs):
        x,a = inputs

        # run encoder
        enc = []
        tmpx = x
        for l in self.auto_encoder.encoder.layers:
            tmpx = l(tmpx)
            enc.append(tmpx)
        dec = self.auto_encoder.decoder(tmpx)

        # run AGCN-H
        tmpx = self.init_gcn((x,a))
        _agcnh = [tmpx]
        for idx,l in enumerate(self.agcnh):
            tmpx = l((tmpx,enc[idx],a))
            _agcnh.append(tmpx)

        # run AGCN-S
        tmpx = _agcnh + [enc[-1]]
        result = self.agcns(tmpx)

        # calculate q
        q = 1.0 / (1.0 + tf.reduce_sum(tf.pow(tf.expand_dims(enc[-1],1) - self.centroid_weight, 2), 2) / 1.0)
        q = tf.pow(q,(1.0 + 1.0) / 2.0)
        q = q / tf.reduce_sum(q, axis=1,keepdims=True)

        # calculate p
        p = tf.pow(q,2) / tf.reduce_sum(q,0)
        p = p / tf.reduce_sum(p, axis=1,keepdims=True)
        p = tf.stop_gradient(p)

        # calculate loss
        q_loss = tf.reduce_mean(
            tf.reduce_sum(p * tf.math.log((p + self.eps) / (q + self.eps)), axis=1)
            ) * self.lambda1
        pred_loss = tf.reduce_mean(
            tf.reduce_sum(p * tf.math.log((p + self.eps) / (result + self.eps)), axis=1)
            ) * self.lambda2
        mse = tf.reduce_mean(losses.MSE(dec,x),0)

        self.add_loss(mse + q_loss + pred_loss)

        return tf.argmax(result,-1)
