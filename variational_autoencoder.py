import tensorflow as tf

class VariationalAutoencoder(object):

    def __init__(self, ndims=784, nlatent=2):
 

        self._ndims = ndims
        self._nlatent = nlatent

        # Create session
        self.session = tf.Session()
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])

        # Build graph.
        self.z_mean, self.z_log_var = self._encoder(self.x_placeholder)
        self.z = self._sample_z(self.z_mean, self.z_log_var)
        self.outputs_tensor = self._decoder(self.z)

        # Setup loss tensor, predict_tensor, update_op_tensor
        self.loss_tensor = self.loss(self.outputs_tensor, self.x_placeholder,
                                     self.z_mean, self.z_log_var)

        self.update_op_tensor = self.update_op(self.loss_tensor,
                                               self.learning_rate_placeholder)

        # Initialize all variables.
        self.session.run(tf.global_variables_initializer())
        
    def weight_variable(self,shape):
        initial=tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)
        #Function to define Bias variable
    def bias_variable(self,shape):
        initial=tf.constant(0.1,shape=shape)
        return tf.Variable(initial)

    def full_layer(self,input,size):
        in_size=int(input.get_shape()[1])
        W=self.weight_variable([in_size,size])
        b=self.bias_variable([size])
        return tf.matmul(input,W)+b

    def _sample_z(self, z_mean, z_log_var):
        z_shape=tf.shape(z_mean)
        E= tf.random_normal((z_shape), 0, 1, dtype=tf.float32) 
        z = tf.add(z_mean, tf.multiply(tf.exp(tf.sqrt(z_log_var)), E))
        return z

    def _encoder(self, x):
        f1=tf.nn.softplus(self.full_layer(x,100))
        f2=tf.nn.softplus(self.full_layer(f1,50))
        f3=tf.nn.softplus(self.full_layer(f2,4))
        z_mean=f3[:,:2]
        z_log_var=f3[:,2:]
        return z_mean, z_log_var

    def _decoder(self, z):
        f1=tf.nn.softplus(self.full_layer(z,50))
        f2=tf.nn.softplus(self.full_layer(f1,100))
        f3=tf.nn.softplus(self.full_layer(f2,784))   
        f = f3
        return f

    def _latent_loss(self, z_mean, z_log_var):
        latent_loss =-0.5*tf.reduce_sum(1 + z_log_var - tf.square(z_mean)  - tf.exp(z_log_var), axis=1)
        return latent_loss

    def _reconstruction_loss(self, f, y):
        return tf.reduce_sum(tf.pow(f-y,2),axis=1)
    
    def loss(self, f, y, z_mean, z_var):
        latent_loss=self._latent_loss(z_mean,z_var)
        reconstr_loss=self._reconstruction_loss(f,y)
        cost = tf.reduce_mean(reconstr_loss + latent_loss,axis=0)
        return cost

    def update_op(self, loss, learning_rate):
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)

    def generate_samples(self, z_np):
        return self.session.run(self.outputs_tensor, feed_dict={self.z: z_np})
        
    
