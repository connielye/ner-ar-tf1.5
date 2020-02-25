import tensorflow as tf
import numpy as np

export_path = "path to saved_model"

session = tf.Session(graph=tf.Graph())

tf.saved_model.loader.load(session, ["serve"], export_path)

sentence = "any test sentence"



