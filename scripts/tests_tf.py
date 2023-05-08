import tensorflow as tf
import numpy as np
from keras.layers import VersionAwareLayers

layers = VersionAwareLayers()

reference = np.random.randint(0, 255, size=(1, 1, 1,512))
reference = tf.convert_to_tensor(reference, dtype=tf.float32)

maxpool = layers.GlobalMaxPooling2D()
avgpool = layers.GlobalAveragePooling2D()

maxpool_out = maxpool(reference)
avgpool_out = avgpool(reference)
reshaped = tf.reshape(reference, (None, 512))

print(maxpool_out.shape)
print(avgpool_out.shape)
print(reshaped.shape)