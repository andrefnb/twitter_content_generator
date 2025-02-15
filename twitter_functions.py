import tweepy
import config
import content_creator_model
from pathlib import Path

def authenticate():
    # Authenticate to Twitter
    auth = tweepy.OAuthHandler(config["API_key"], config["API_secret_key"])
    auth.set_access_token(config["access_token"], config["access_token_secret"])

    api = tweepy.API(auth)
    return api


def get_tweets(api):

    try:
        timeline = api.home_timeline()
        return timeline
    except:
        print("Error during get")


def get_store_tweets(api):
    filename = content_creator_model.TWEETS_FILE_NAME
    my_file = Path(filename)

    file = open(filename, 'a')

    timeline = get_tweets(api)
    for tweet in timeline:
        print(tweet)
        print(f"{tweet.user.name} said {tweet.text}")
        file.write("\n" + tweet.text)


def tweet(api, text):

    try:
        api.update_status(text)
    except:
        print("Error during tweet")


def generate():
    filename = content_creator_model.FILE_MODEL_NAME
    model = content_creator_model.load_model_custom(filename)
    # 280 is the max tweet character number
    text = content_creator_model.generate_content(model, nr_chars=280)

    print(f"Text: {text}")
    return text


def generate_and_tweet(api):
    text = generate()
    tweet(api, text)


configfile = config
config = configfile.config


# Authenticate to Twitter
auth = tweepy.OAuthHandler(config["API_key"], config["API_secret_key"])
auth.set_access_token(config["access_token"], config["access_token_secret"])

api = tweepy.API(auth)

# unique_chars, int_to_char, char_to_int, X, Y, x_modified, y_modified = content_creator_model.load_data()
# content_creator_model.train_model(x_modified, y_modified)




# filename = content_creator_model.FILE_MODEL_NAME
# model = content_creator_model.load_model_custom(filename)
# text = content_creator_model.generate_content(model, nr_chars = 280)
#
# tweet(api, text)

# try:
#     api.verify_credentials()
#     print("Authentication OK")
# except:
#     print("Error during authentication")


