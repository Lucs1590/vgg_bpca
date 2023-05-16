import tensorflow as tf
from keras.layers import Layer

class BPCALayer(Layer):
    def __init__(self, pool_size=2, stride=2, n_components=1, **kwargs):
      super(BPCALayer, self).__init__(**kwargs)
      self.pool_size = pool_size
      self.stride = stride
      self.n_components = n_components

    def call(self, inputs):
      tf.debugging.assert_shapes([(inputs, (None, None, None, None))])

      def bpca_pooling(feature_map):
        # Compute the region of interest        
        feature_map_height = int(feature_map.shape[0])
        feature_map_width  = int(feature_map.shape[1])
        feature_map_channels = int(feature_map.shape[2])

        h = feature_map_height
        w = feature_map_width
        c = feature_map_channels

        # Create blocks (patches)
        data = tf.reshape(feature_map, [1, feature_map_height, feature_map_width, feature_map_channels])
        pool_size = self.pool_size
        strides = self.stride

        patch_size = [1, pool_size, pool_size, 1]
        strides = [1, strides, strides, 1]
        patches = tf.image.extract_patches(images=data, sizes=patch_size, strides=strides, rates=[1, 1, 1, 1], padding='VALID')

        # data = tf.reshape(patches, [h*w*8, 4])
        d = c // (self.pool_size * self.pool_size)
        data = tf.reshape(patches, [h*w*d, self.pool_size * self.pool_size])

        # Normalize the data by subtracting the mean and dividing by the standard deviation
        mean = tf.reduce_mean(data, axis=0)
        std = tf.math.reduce_std(data, axis=0)
        data = (data - mean) / std

        data = tf.where(tf.math.is_nan(data), 0.0, data)

        # Perform the Singular Value Decomposition (SVD) on the data
        s, u, v = tf.linalg.svd(data)

        # Extract the first n principal components from the matrix v
        pca_components = v[:, :self.n_components]

        # Perform the PCA transformation on the data
        transformed_data = tf.matmul(data, pca_components)

        return tf.reshape(transformed_data, [h // self.pool_size, w // self.pool_size, feature_map_channels]) 

      pooled = tf.vectorized_map(bpca_pooling, inputs)
      return pooled
