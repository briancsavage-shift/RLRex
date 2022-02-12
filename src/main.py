import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import utils


def main():
    logger = utils.setLogging()

    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    # Load data on movie ratings.
    ratings = tfds.load("movielens/100k-ratings", split="train")
    movies = tfds.load("movielens/100k-movies", split="train")

    # Build flexible representation models.
    user_model = tf.keras.Sequential([...])
    movie_model = tf.keras.Sequential([...])

    # Define your objectives.
    task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
        movies.batch(128).map(movie_model)
    ))

    # Create a retrieval model.
    model = tf.MovielensModel(user_model, movie_model, task)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

    # Train.
    model.fit(ratings.batch(4096), epochs=3)

    # Set up retrieval using trained representations.
    index = tfrs.layers.ann.BruteForce(model.user_model)
    index.index_from_dataset(
        movies.batch(100).map(lambda title: (title, model.movie_model(title)))
    )

    # Get recommendations.
    _, titles = index(np.array(["42"]))
    print(f"Recommendations for user 42: {titles[0, :3]}")


if __name__ == "__main__":
    main()
