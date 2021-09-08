import numpy as np
from typing import Dict, Text
import pandas as pd
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_recommenders as tfrs

coverages_url = "coverages.csv"
coverages_df = pd.read_csv(coverages_url, dtype = str)
coverages_dataset = tf.data.Dataset.from_tensor_slices(dict(coverages_df))
coverages_dataset = coverages_dataset.map(lambda x: x["ID"])

print("Local copy of the coverages dataset file: {}".format(coverages_dataset))

employees_url = "employees.csv"
employees_df = pd.read_csv(employees_url, dtype = str)
employees_dataset = tf.data.Dataset.from_tensor_slices(dict(employees_df))
employees_dataset = employees_dataset.map(lambda x: {
    "ID": x["ID"],
    "coverage_id": x["Coverage ID"],
    "age": x["Age"],
    "gender": x["Gender"]
})

print("Local copy of the employees dataset file: {}".format(employees_dataset))

employee_ids_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
employee_ids_vocabulary.adapt(employees_dataset.map(lambda x: x["ID"]))

coverage_ids_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
coverage_ids_vocabulary.adapt(coverages_dataset)

employee_model = tf.keras.Sequential([
    employee_ids_vocabulary,
    tf.keras.layers.Embedding(employee_ids_vocabulary.vocabulary_size(), 64)
])
coverage_model = tf.keras.Sequential([
    coverage_ids_vocabulary,
    tf.keras.layers.Embedding(coverage_ids_vocabulary.vocabulary_size(), 64)
])

# Define your objectives.
task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
    coverages_dataset.batch(2).map(coverage_model)
  )
)

class CoverageModel(tfrs.Model):

  def __init__(
      self,
      coverage_model: tf.keras.Model,
      employee_model: tf.keras.Model,
      task: tfrs.tasks.Retrieval):
    super().__init__()

    self.coverage_model = coverage_model
    self.employee_model = employee_model

    # Set up a retrieval task.
    self.task = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # Define how the loss is computed.

    employee_embeddings = self.employee_model(features["ID"])
    # query_embeddings = self.query_model({
    #     "ID": features["ID"],
    #     ""
    # })
    coverage_embeddings = self.coverage_model(features["coverage_id"])

    return self.task(employee_embeddings, coverage_embeddings)

model = CoverageModel(employee_model, coverage_model, task)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

# Train for 3 epochs.
model.fit(employees_dataset.batch(4), epochs=3)

# Use brute-force search to set up retrieval using the trained representations.
index = tfrs.layers.factorized_top_k.BruteForce(model.employee_model)
index.index_from_dataset(
    coverages_dataset.batch(4).map(lambda ID: (ID, model.coverage_model(ID))))

# Get some recommendations.
_, titles = index(np.array(["1"]))
print(f"Top 3 recommendations for user 42: {titles[0, :3]}")


# DATA_URL = "D:/ratings.csv"
# df = pd.read_csv(DATA_URL)
# ratings = tf.data.Dataset.from_tensor_slices(dict(df)).map(lambda x: {
#     "user_id": str(x["user_id"]),
#     "item_id": str(x["item_id"]),
#     "rating": float(x["rating"])
# })

