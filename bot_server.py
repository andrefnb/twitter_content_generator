import schedule
import tensorflow as tf
import time
import twitter_functions
import content_creator_model

TESTING = True


def get_tweets():
    api = twitter_functions.authenticate()

    # Get new tweets
    twitter_functions.get_store_tweets(api)


def tweet():
    api = twitter_functions.authenticate()

    # Generate tweet
    twitter_functions.generate_and_tweet(api)


def train_model():

    # Train model
    print("Loading data")
    unique_chars, int_to_char, char_to_int, X, Y, x_modified, y_modified = content_creator_model.load_data()
    #print("Training began")
    #content_creator_model.train_model(x_modified, y_modified, load_checkpoint = False)

    # testing
    text = twitter_functions.generate()


schedule.every(1).minute.do(tweet)
schedule.every(1).minute.do(get_tweets)
# schedule.every(3).minute.do(train_model)
# schedule.every(1).minutes.do(job)

if TESTING:

    # TEST
    #api = twitter_functions.authenticate()
    #twitter_functions.generate_and_tweet(api)


    # new
    print(tf.test.is_gpu_available())
    print(tf.config.list_physical_devices('GPU'))
    train_model()


else:
    while True:
        schedule.run_pending()
        time.sleep(1)
