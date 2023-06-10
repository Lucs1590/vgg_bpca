import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.python import debug as tf_debug
from keras.layers import VersionAwareLayers
import keras.backend as K

layers = VersionAwareLayers()


class BPCAPooling(tf.keras.layers.Layer):
    def __init__(self, pool_size=2, stride=2, n_components=1, **kwargs):
        super(BPCAPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.stride = stride
        self.n_components = n_components

        self.patch_size = [1, self.pool_size, self.pool_size, 1]
        self.strides = [1, self.stride, self.stride, 1]

    def build(self, input_shape):
        super(BPCAPooling, self).build(input_shape)

    @tf.function
    def bpca_pooling(self, feature_map):
        # Compute the region of interest
        h, w, c = feature_map.shape  # block_height, block_width, block_channels

        # Create blocks (patches)
        data = tf.reshape(feature_map, [1, h, w, c])
        patches = tf.image.extract_patches(
            images=data,
            sizes=self.patch_size,
            strides=self.strides,
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        print('patches.shape before reshape', patches.shape)
        patches_size = patches.shape[1] * patches.shape[2] * patches.shape[3]
        patches = tf.reshape(
            patches,
            [
                patches_size // (self.pool_size * self.pool_size),
                self.pool_size * self.pool_size
            ]
        )
        print('patches.shape after reshape', patches.shape)

        # Normalize the data by subtracting the mean and dividing by the standard deviation
        mean = tf.reduce_mean(patches, axis=0)
        std = tf.math.reduce_std(patches, axis=0)
        patches = (patches - mean) / std
        patches = tf.where(tf.math.is_nan(patches), 0.0, patches)

        # Perform the Singular Value Decomposition (SVD) on the data
        _, _, v = tf.linalg.svd(patches)

        # Extract the first n principal components from the matrix v
        pca_components = v[:, :self.n_components]
        print('pca_components.shape', pca_components.shape)

        # Perform the PCA transformation on the data
        transformed_patches = tf.matmul(patches, pca_components)
        print('transformed_patches.shape', transformed_patches.shape)
        return tf.reshape(transformed_patches, [h // self.pool_size, w // self.pool_size, c])

    def call(self, inputs):
        pooled = tf.vectorized_map(self.bpca_pooling, inputs)
        pooled = tf.reshape(
            pooled, [-1, pooled.shape[-1]]
        )
        return pooled


tf.random.set_seed(42)
sess = K.get_session()
sess = tf_debug.LocalCLIDebugWrapperSession(sess)
K.set_session(sess)

maxpool = layers.GlobalMaxPooling2D()
avgpool = layers.GlobalAveragePooling2D()
# bpca_pool = GlobalBPCAPooling2D(n_components=32)
bpca_pool = BPCAPooling(pool_size=112, stride=112, n_components=1)

reference = tf.random.uniform(
    shape=(1, 112, 112, 32), minval=0, maxval=255, dtype=tf.float32
)

maxpool_out = maxpool(reference)
avgpool_out = avgpool(reference)
bpca_pool_out = bpca_pool(reference)

assert maxpool_out.shape == avgpool_out.shape == bpca_pool_out.shape
assert maxpool_out.shape == (1, 32)
assert avgpool_out.shape == (1, 32)
assert bpca_pool_out.shape == (1, 32)
