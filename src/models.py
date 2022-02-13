import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, Text


class RankingModel(tf.keras.Model):
    def __init__(self, movie_titles, user_ids):
        """
        [summary]

        Args:
            movie_titles ([type]): [description]
            user_ids ([type]): [description]
        """
        super().__init__()
        embedding_dimension = 32
        self.unique_movie_titles = np.unique(
            np.concatenate(list(movie_titles))
        )
        self.unique_user_ids = np.unique(
            np.concatenate(list(user_ids))
        )

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=self.unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(
                len(self.unique_user_ids) + 1, embedding_dimension)
        ])

        # Compute embeddings for movies.
        self.movie_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=self.unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(
                len(self.unique_movie_titles) + 1, embedding_dimension)
        ])

        # Compute predictions.
        self.ratings = tf.keras.Sequential([
            # Learn multiple dense layers.
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            # Make rating predictions in the final layer.
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):
        """
        [summary]

        Args:
            inputs ([type]): [description]

        Returns:
            [type]: [description]
        """
        user_id, movie_title = inputs
        user_embedding = self.user_embeddings(user_id)
        movie_embedding = self.movie_embeddings(movie_title)

        return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))


class MovielensModel(tfrs.models.Model):

    def __init__(self, movie_titles, user_ids):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel(
            movie_titles, user_ids
        )
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        return self.ranking_model(
            (features["user_id"], features["movie_title"]))

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        labels = features.pop("user_rating")

        rating_predictions = self(features)

        # The task computes the loss and the metrics.
        return self.task(labels=labels, predictions=rating_predictions)
