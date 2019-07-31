'''This module contains models that can be used in classifier.py'''

import tensorflow as tf

class LogisticRegression(tf.keras.Model):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.layer1 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        return self.layer1(inputs)


class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = tf.keras.layers.Dense(64, activation='relu')
        self.layer2 = tf.keras.layers.Dense(64, activation='relu')
        self.layer3 = tf.keras.layers.Dense(64, activation='relu')
        self.out_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.out_layer(x)
