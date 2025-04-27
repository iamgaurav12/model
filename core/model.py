import tensorflow as tf
from tensorflow.keras import layers, Model
from config.settings import MODEL_INPUT_SHAPE, MODEL_OUTPUT_CLASSES

class FLProtoNet(Model):
    def __init__(self):
        super(FLProtoNet, self).__init__()
        self.dense1 = layers.Dense(8, activation='relu')
        self.dense2 = layers.Dense(MODEL_OUTPUT_CLASSES, activation='softmax')
        
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
    
    def build_model(self):
        """Ensure weights are initialized"""
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=MODEL_INPUT_SHAPE),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(MODEL_OUTPUT_CLASSES, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model