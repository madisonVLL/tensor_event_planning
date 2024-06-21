from typing import *
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

time_features = []
time_labels = []

time_file = open("wedding_data/time_variations.csv")
for row in time_file:
    time_string, time_float = row.split(",")
    time_features.append(time_string)
    time_labels.append(float(time_float))
time_file.close()

time_features = np.array(time_features, dtype= str)
time_labels = np.array(time_labels, dtype= float)

time_model = tf.keras.Sequential([
  tf.keras.layers.Dense(units=1)
  #don't need to include shape if the unit is one, kindof implied
  #its one dementional
])

time_model.compile(loss="time period accuracy", optimizer=tf.keras.optimizers.Adam(0.1))
model_history = time_model.fit(time_features, time_labels, epochs=500, verbose=False)
print("done training time model")
'''
#just going to comment some of this out while I use the tutorial to sift things
#out
time_triggers: list[str] = np.array(["day", "month", "year", "AM", "PM"], str)
time_trig_alt: list[str] = np.array(["days", "months", "years", "am", "pm"], str)

time_learning = tf.keras.layers.Dense(units = 1, input_shape = [1])
time_model = tf.keras.Sequential([time_learning])
time_model.compile(loss="error", optimizer=tf.keras.optimizers.Adam(0.1))
#this model can have a lot of error, just need to generally relate uppercase
#to lowercase, singulars to plurals, etc

learning_mode = time_model(time_triggers, time_trig_alt, epoch = 500, Verbose = False)
#epoch is the amount of trials to learn the model

print(time_model.predict(["Hour"]))


#input = features, labels = output

#two d image into a vector is called flattening

#nn = nureal network
tfds.disable_progress_bar()
dataset, info = tfds.load("wedding_data/tpv.csv", with_info=True, as_supervised=True)

teach_dataset, test_dataset = dataset['train'], dataset['test']

'''
