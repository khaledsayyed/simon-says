from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import tensorflow_recommenders as tfrs
import matplotlib.pyplot as plt
import tensorflowjs as tfjs

def train():
  url = "train.csv"
  plans_df = pd.read_csv(url, usecols=["PlanName"], dtype=str).drop_duplicates()
  employees_df = pd.read_csv(url, dtype=str, usecols=["EmployeeId", "eemGender", "eemBirthDate", "eemState"]).drop_duplicates()
  enrollment_df = pd.read_csv(url, dtype=str, usecols=["eemState", "PlanName", "eemGender", "eemZipCode"])
  enrollment_df = enrollment_df.groupby(["PlanName", "eemGender", "eemState", "eemZipCode"]).size().reset_index(name="count")

  enrollment_ds = tf.data.Dataset.from_tensor_slices(dict(enrollment_df))

  enrollment_ds = enrollment_ds.map(lambda x: {
    "count": x["count"],
    "gender": x["eemGender"],
    "state": x["eemState"],
    "zip_code": x["eemZipCode"],
    "plan_name": x["PlanName"]
  })
  print(enrollment_df)

  tf.random.set_seed(42)
  shuffled = enrollment_ds.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

  train = shuffled.take(10000)
  test = shuffled.skip(10000).take(20_000)

  feature_names = ["plan_name", "gender", "state", "zip_code"]

  vocabularies = {}

  for feature_name in feature_names:
    vocab = enrollment_ds.batch(1_000_000).map(lambda x: x[feature_name])
    vocabularies[feature_name] = np.unique(np.concatenate(list(vocab)))

  class DCN(tfrs.Model):

    def __init__(self, use_cross_layer, deep_layer_sizes, projection_dim=None):
      super().__init__()

      self.embedding_dimension = 32

      str_features = ["plan_name", "gender", "state", "zip_code"]
      int_features = []

      self._all_features = str_features + int_features
      self._embeddings = {}

      # Compute embeddings for string features.
      for feature_name in str_features:
        vocabulary = vocabularies[feature_name]
        self._embeddings[feature_name] = tf.keras.Sequential(
            [tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=vocabulary, mask_token=None),
            tf.keras.layers.Embedding(len(vocabulary) + 1,
                                      self.embedding_dimension)
      ])

      # Compute embeddings for int features.
      for feature_name in int_features:
        vocabulary = vocabularies[feature_name]
        self._embeddings[feature_name] = tf.keras.Sequential(
            [tf.keras.layers.experimental.preprocessing.IntegerLookup(
                vocabulary=vocabulary, mask_value=None),
            tf.keras.layers.Embedding(len(vocabulary) + 1,
                                      self.embedding_dimension)
      ])

      if use_cross_layer:
        self._cross_layer = tfrs.layers.dcn.Cross(
            projection_dim=projection_dim,
            kernel_initializer="glorot_uniform")
      else:
        self._cross_layer = None

      self._deep_layers = [tf.keras.layers.Dense(layer_size, activation="relu")
        for layer_size in deep_layer_sizes]

      self._logit_layer = tf.keras.layers.Dense(1)

      self.task = tfrs.tasks.Ranking(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError("RMSE")]
      )

    def call(self, features):
      # Concatenate embeddings
      embeddings = []
      for feature_name in self._all_features:
        embedding_fn = self._embeddings[feature_name]
        embeddings.append(embedding_fn(features[feature_name]))
      x = tf.keras.layers.Concatenate(axis=1)(embeddings)

      # Build Cross Network
      if self._cross_layer is not None:
        x = self._cross_layer(x)

      # Build Deep Network
      for deep_layer in self._deep_layers:
        x = deep_layer(x)

      return self._logit_layer(x)

    def compute_loss(self, features, training=False):
      labels = features.pop("count")
      scores = self(features)
      return self.task(
          labels=labels,
          predictions=scores,
      )

  cached_train = train.shuffle(100_000).batch(8192).cache()
  cached_test = test.batch(4096).cache()

  epochs = 8
  learning_rate = 0.01


  model = DCN(use_cross_layer=False,
                  deep_layer_sizes=[192,192],
                  projection_dim=20)
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))

  model.fit(cached_train, epochs=epochs, verbose=False)
  return model

class handler(BaseHTTPRequestHandler):
    recommendationModel = None

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()

        message = "Hello, World!!"
        path, _, query_string = self.path.partition('?')
        query = parse_qs(query_string)
        details = query["details"][0]
        print(details)
        detailsJSON = json.loads(details)

        if(handler.recommendationModel == None):
            handler.recommendationModel = train()

        weight_by_plan = {}
        for plan_name in detailsJSON["planNames"]:
            weight_by_plan[plan_name] = handler.recommendationModel({
                "plan_name": np.array([plan_name]),
                "gender": np.array([detailsJSON["gender"]]),
                "state": np.array([detailsJSON["state"]]),
                "zip_code": np.array([detailsJSON["zipCode"]])
            })

        array = sorted(weight_by_plan.items(), key=lambda x: x[1], reverse=True)
        recommended = array[0][0] 

        detailsJSON = json.loads(details)

        self.wfile.write(bytes(recommended, "utf8"))

        print(u"[START]: Received GET for %s with query: %s" % (path, detailsJSON))

with HTTPServer(('', 7800), handler) as server:
    server.serve_forever()