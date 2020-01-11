# !pip install emoji
import nltk
nltk.download('stopwords')
import re
import pprint
import pickle
import emoji as emoji
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, log_loss
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
# import time


def load_dict_smileys():
    return {
        ":‑)": "smiley",
        ":-]": "smiley",
        ":-3": "smiley",
        ":->": "smiley",
        "8-)": "smiley",
        ":-}": "smiley",
        ":)": "smiley",
        ":]": "smiley",
        ":3": "smiley",
        ":>": "smiley",
        "8)": "smiley",
        ":}": "smiley",
        ":o)": "smiley",
        ":c)": "smiley",
        ":^)": "smiley",
        "=]": "smiley",
        "=)": "smiley",
        ":-))": "smiley",
        ":‑D": "smiley",
        "8‑D": "smiley",
        "x‑D": "smiley",
        "X‑D": "smiley",
        ":D": "smiley",
        "8D": "smiley",
        "xD": "smiley",
        "XD": "smiley",
        ":‑(": "sad",
        ":‑c": "sad",
        ":‑<": "sad",
        ":‑[": "sad",
        ":(": "sad",
        ":c": "sad",
        ":<": "sad",
        ":[": "sad",
        ":-||": "sad",
        ">:[": "sad",
        ":{": "sad",
        ":@": "sad",
        ">:(": "sad",
        ":'‑(": "sad",
        ":'(": "sad",
        ":‑P": "playful",
        "X‑P": "playful",
        "x‑p": "playful",
        ":‑p": "playful",
        ":‑Þ": "playful",
        ":‑þ": "playful",
        ":‑b": "playful",
        ":P": "playful",
        "XP": "playful",
        "xp": "playful",
        ":p": "playful",
        ":Þ": "playful",
        ":þ": "playful",
        ":b": "playful",
        "<3": "love",
        "\o/": "cheer"
    }

# self defined contractions
def load_dict_contractions():
    return {
        "ain't": "is not",
        "amn't": "am not",
        "aren't": "are not",
        "b/c": "because",
        "can't": "cannot",
        "'cause": "because",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "could've": "could have",
        "cya": "see you",
        "daren't": "dare not",
        "daresn't": "dare not",
        "dasn't": "dare not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "e'er": "ever",
        "em": "them",
        "everyone's": "everyone is",
        "finna": "fixing to",
        "fml":"fuck my life",
        "fb": "facebook",
        "gimme": "give me",
        "gonna": "going to",
        "gon't": "go not",
        "gotta": "got to",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "he've": "he have",
        "how'd": "how would",
        "how'll": "how will",
        "how're": "how are",
        "how's": "how is",
        "I'd": "I would",
        "I'll": "I will",
        "I'm": "I am",
        "I'm'a": "I am about to",
        "I'm'o": "I am going to",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "I've": "I have",
        "kinda": "kind of",
        "let's": "let us",
        "lol": "laughing out loud",
        "mayn't": "may not",
        "may've": "may have",
        "mightn't": "might not",
        "might've": "might have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "must've": "must have",
        "needn't": "need not",
        "ne'er": "never",
        "nite":"night",
        "o'": "of",
        "o'er": "over",
        "ol'": "old",
        "oughtn't": "ought not",
        "rofl": "rolling on the floor laughing",
        "shalln't": "shall not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "should've": "should have",
        "somebody's": "somebody is",
        "someone's": "someone is",
        "something's": "something is",
        "that'd": "that would",
        "that'll": "that will",
        "that're": "that are",
        "that's": "that is",
        "there'd": "there would",
        "there'll": "there will",
        "there're": "there are",
        "there's": "there is",
        "these're": "these are",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "this's": "this is",
        "those're": "those are",
        "'tis": "it is",
        "'twas": "it was",
        "u": "you",
        "w/": "with",
        "w/o": "without",
        "wanna": "want to",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we're": "we are",
        "weren't": "were not",
        "we've": "we have",
        "what'd": "what did",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "where'd": "where did",
        "where're": "where are",
        "where's": "where is",
        "where've": "where have",
        "which's": "which is",
        "who'd": "who would",
        "who'd've": "who would have",
        "who'll": "who will",
        "who're": "who are",
        "who's": "who is",
        "who've": "who have",
        "why'd": "why did",
        "why're": "why are",
        "why's": "why is",
        "won't": "will not",
        "wouldn't": "would not",
        "would've": "would have",
        "y'all": "you all",
        "ya": "you",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have",
        "Whatcha": "What are you",
        "luv": "love",
        "sux": "sucks"
    }

SMILEY = load_dict_smileys()
CONTRACTIONS = load_dict_contractions()

def tweet_cleaner(text):
    remove_tags = r'@[A-Za-z0-9]+'
    remove_links = r'https?://[A-Za-z0-9./]+'
    remove_tags_and_links_re = r'|'.join((remove_tags, remove_links))
    # Discard any html stuff in tweets
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    # Remove tags and links
    stripped = re.sub(remove_tags_and_links_re, '', souped)
    try:
        # Remove weird characters
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped

    words = clean.split()
    reformed = [SMILEY[word] if word in SMILEY else word for word in words]
    tweet = " ".join(reformed)

    tweet = tweet.replace("’", "'")
    words = tweet.split()
    reformed = [CONTRACTIONS[word] if word in CONTRACTIONS else word for word in words]
    tweet = " ".join(reformed)

    tweet = emoji.demojize(tweet)
    tweet = tweet.replace(":", " ")
    tweet = ' '.join(tweet.split())
    lower_case = tweet.lower()
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = tokenizer.tokenize(lower_case)
    return (" ".join(words)).strip()

def tweet_cleaner_none(text):
    remove_tags = r'@[A-Za-z0-9]+'
    remove_links = r'https?://[A-Za-z0-9./]+'
    remove_tags_and_links_re = r'|'.join((remove_tags, remove_links))
    # Discard any html stuff in tweets
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    # Remove tags and links
    stripped = re.sub(remove_tags_and_links_re, '', souped)
    try:
        # Remove weird characters
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped

    # words = clean.split()
    # reformed = [SMILEY[word] if word in SMILEY else word for word in words]
    # tweet = " ".join(reformed)

    # tweet = tweet.replace("’", "'")
    # words = tweet.split()
    # reformed = [CONTRACTIONS[word] if word in CONTRACTIONS else word for word in words]
    # tweet = " ".join(reformed)

    # tweet = emoji.demojize(tweet)
    # tweet = tweet.replace(":", " ")
    # tweet = ' '.join(tweet.split())
    lower_case = text.lower()
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = tokenizer.tokenize(lower_case)
    return (" ".join(words)).strip()

def tweet_cleaner_smiley(text):
  remove_tags = r'@[A-Za-z0-9]+'
  remove_links = r'https?://[A-Za-z0-9./]+'
  remove_tags_and_links_re = r'|'.join((remove_tags, remove_links))
  # Discard any html stuff in tweets
  soup = BeautifulSoup(text, 'lxml')
  souped = soup.get_text()
  # Remove tags and links
  stripped = re.sub(remove_tags_and_links_re, '', souped)
  try:
      # Remove weird characters
      clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
  except:
      clean = stripped

  words = clean.split()
  reformed = [SMILEY[word] if word in SMILEY else word for word in words]
  tweet = " ".join(reformed)

  # tweet = tweet.replace("’", "'")
  # words = tweet.split()
  # reformed = [CONTRACTIONS[word] if word in CONTRACTIONS else word for word in words]
  # tweet = " ".join(reformed)

  # tweet = emoji.demojize(tweet)
  # tweet = tweet.replace(":", " ")
  # tweet = ' '.join(tweet.split())
  lower_case = tweet.lower()
  # During the letters_only process two lines above, it has created unnecessay white spaces,
  # I will tokenize and join together to remove unneccessary white spaces
  words = tokenizer.tokenize(lower_case)
  return (" ".join(words)).strip()

def tweet_cleaner_contract(text):
    remove_tags = r'@[A-Za-z0-9]+'
    remove_links = r'https?://[A-Za-z0-9./]+'
    remove_tags_and_links_re = r'|'.join((remove_tags, remove_links))
    # Discard any html stuff in tweets
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    # Remove tags and links
    stripped = re.sub(remove_tags_and_links_re, '', souped)
    try:
        # Remove weird characters
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped

    # words = clean.split()
    # reformed = [SMILEY[word] if word in SMILEY else word for word in words]
    # tweet = " ".join(reformed)

    tweet = clean.replace("’", "'")
    words = tweet.split()
    reformed = [CONTRACTIONS[word] if word in CONTRACTIONS else word for word in words]
    tweet = " ".join(reformed)

    # tweet = emoji.demojize(tweet)
    # tweet = tweet.replace(":", " ")
    # tweet = ' '.join(tweet.split())
    lower_case = tweet.lower()
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = tokenizer.tokenize(lower_case)
    return (" ".join(words)).strip()

def tweet_cleaner_smiley_contract(text):
    remove_tags = r'@[A-Za-z0-9]+'
    remove_links = r'https?://[A-Za-z0-9./]+'
    remove_tags_and_links_re = r'|'.join((remove_tags, remove_links))
    # Discard any html stuff in tweets
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    # Remove tags and links
    stripped = re.sub(remove_tags_and_links_re, '', souped)
    try:
        # Remove weird characters
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped

    words = clean.split()
    reformed = [SMILEY[word] if word in SMILEY else word for word in words]
    tweet = " ".join(reformed)

    tweet = tweet.replace("’", "'")
    words = tweet.split()
    reformed = [CONTRACTIONS[word] if word in CONTRACTIONS else word for word in words]
    tweet = " ".join(reformed)

    # tweet = emoji.demojize(tweet)
    # tweet = tweet.replace(":", " ")
    # tweet = ' '.join(tweet.split())
    lower_case = tweet.lower()
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = tokenizer.tokenize(lower_case)
    return (" ".join(words)).strip()

df = pd.read_csv("Train.csv", encoding='latin-1')
# # df['length'] = df['SentimentText'].str.len()
# exclaimer = lambda x: x.count("!")
# pointer = lambda x: x.count(".")
# count_upper = lambda message: sum(1 for c in message if c.isupper())
# get_lower_upper_ration = lambda message: float((sum(1 for c in message if c.islower())) + 1) / (sum(1 for c in message if c.isupper()) +1)


# def get_multiple(message):
#     count = 0
#     for i in range(len(message) - 1):
#         if message[i] == message[i +1]:
#             count += 1

#     return (float(count) + 1) / (float(len(message)) + 1)

# df['exclaim'] = df.SentimentText.map(exclaimer)
# df['point'] = df.SentimentText.map(pointer)
# df['capital'] = df.SentimentText.map(count_upper)
# df['capital_lower_ration'] = df.SentimentText.map(get_lower_upper_ration)
# df['multiple'] = df.SentimentText.map(get_multiple)

# # above line will be different depending on where you saved your data, and your file name
# print(df.groupby('Sentiment').agg({'exclaim':'mean', 'point':'mean', 'capital':'mean', 'capital_lower_ration':'mean', 'multiple':'mean'}))
#         # find the number of network type entries
#         # 'network_type': "count",
#         # minimum, first, and number of unique dates
# # np.std, ddof=1}))

tokenizer = WordPunctTokenizer()

# #
tfidfconverter = TfidfVectorizer(max_features=8000,
                                 min_df=5, max_df=0.9,
                                #  stop_words=stopwords.words('english')
                                ngram_range=(1,1)
                                 )
print("Fitting tfidfvectorizer...")
# X = tfidfconverter.fit_transform(df.processed).toarray()
X = df.SentimentText.values
y = df.iloc[:, 0].values
print("Splitting data...")
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# pickle.dump(X_train, open("xtrain",'wb'))
# pickle.dump(y_train, open("ytrain",'wb'))
# pickle.dump(X_test, open("xtest",'wb'))
# pickle.dump(y_test, open("ytest",'wb'))
# pickle.dump(tfidfconverter.vocabulary_, open("vocabulary",'wb'))
#
# X_train = pickle.load(open("xtrain",'rb'))
# y_train = pickle.load(open("ytrain",'rb'))
# X_test = pickle.load(open("xtest",'rb'))
# y_test = pickle.load(open("ytest",'rb'))


pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words=None, max_df=0.5, max_features=10000, ngram_range=(1,2), preprocessor=tweet_cleaner)),
    ('tfidf', TfidfTransformer(use_idf=True, norm='l2')),
    ('clf', LogisticRegression(C=1.623776739188721, solver='liblinear', penalty='l2')),
])

scores = cross_val_score(pipeline, X, y, cv=10, scoring='accuracy')
print(scores)


    # pipeline.fit(X_train, y_train)
    #
    # print("=" * 30)
    #
    # print('****Results****')
    # train_predictions = pipeline.predict(X_test)
    # acc = accuracy_score(y_test, train_predictions)
    # print("Accuracy: {:.4%}".format(acc))
    #
    # train_predictions = pipeline.predict_proba(X_test)
    # ll = log_loss(y_test, train_predictions)
    # print("Log Loss: {}".format(ll))

# log_entry = pd.DataFrame([[name, acc * 100, ll]], columns=log_cols)
# log = log.append(log_entry)
# vect__max_df: 0.5
# 	vect__max_features: 10000
# 	vect__ngram_range: (1, 2)
# 	vect__preprocessor: None
# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__max_df': (0.5, 0.75, 0.9),
    'vect__max_features': (5000, 8000, 10000),
    'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),  # unigrams, bigrams ot trigrams
    'vect__preprocessor':  (None, tweet_cleaner_smiley_contract, tweet_cleaner_contract, tweet_cleaner_smiley, tweet_cleaner_none),
}
# grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
#
# print("Performing grid search...")
# print("pipeline:", [name for name, _ in pipeline.steps])
# print("parameters:")
# print(parameters)
# # t0 = time()
# grid_search.fit(X_train, y_train)
# # print("done in %0.3fs" % (time() - t0))
# print()
#
# print("Best score: %0.3f" % grid_search.best_score_)
# print("Best parameters set:")
# best_parameters = grid_search.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))
