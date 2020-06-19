import tensorflow as tf
from kerasom import Kerasom

model = Kerasom(input_dim=2, map_size=(3,3))
model.initialize()
model.compile(optimizer='adam')

converter = tf.lite.TFLiteConverter.from_keras_model(model.model)
converted_model = converter.convert()

# Save the TF Lite model.
with tf.io.gfile.GFile('model.tflite', 'wb') as f:
  f.write(converted_model)
