import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

import utils
import models

SEED = 42
BATCH_SIZE = 8192
LEARNING_RATE = 0.1


def main():
    logger = utils.setLogging()
    tf.random.set_seed(SEED)

    ratings = tfds.load("movielens/100k-ratings", split="train")
    ratings = ratings.map(lambda r: {
        "movie_title": r["movie_title"],
        "user_id": r["user_id"],
        "user_rating": r["user_rating"]
    })

    shuffled = ratings.shuffle(len(ratings),
                               seed=SEED,
                               reshuffle_each_iteration=False)

    tr_size, te_size = int(len(ratings) * 0.8), int(len(ratings) * 0.2)
    train = shuffled.take(tr_size)
    test = shuffled.skip(tr_size).take(te_size)

    movie_titles = ratings.batch(len(ratings)).map(lambda x: x["movie_title"])
    user_ids = ratings.batch(len(ratings)).map(lambda x: x["user_id"])

    model = models.MovielensModel(movie_titles, user_ids)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(
        learning_rate=LEARNING_RATE
    ))

    cached_train = train.shuffle(len(ratings)).batch(BATCH_SIZE).cache()
    cached_test = test.batch(BATCH_SIZE // 2).cache()

    model.fit(cached_train, epochs=3)
    logger.debug(model.evaluate(cached_test, return_dict=True))


if __name__ == "__main__":
    main()
