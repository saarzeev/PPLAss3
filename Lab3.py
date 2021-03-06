# !pip install emoji
import re
import emoji as emoji
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import FunctionTransformer,KBinsDiscretizer,OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
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

def try_model_1():
    pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words=None, max_df=0.5, max_features=10000, ngram_range=(1, 2),
                                 preprocessor=tweet_cleaner)),
        ('tfidf', TfidfTransformer(use_idf=True, norm='l2')),
    ])

    pipeline.steps.append(('clf', LogisticRegression(C=1.623776739188721, solver='liblinear', penalty='l2')))
    scores = cross_validate(pipeline, X, y, cv=10, scoring=['accuracy', 'precision', 'recall'])
    print('LogisticRegression')
    print('accuracy %s average %f' % (scores['test_accuracy'], np.average(scores['test_accuracy'])))
    print('precision %s average %f' % (scores['test_precision'], np.average(scores['test_precision'])))
    print('recall %s average %f' % (scores['test_recall'], np.average(scores['test_recall'])))
    pipeline.steps.pop()
    pipeline.steps.append(('clf', DecisionTreeClassifier(min_samples_split=5)))
    scores = cross_validate(pipeline, X, y, cv=10, scoring=['accuracy', 'precision', 'recall'])
    print('DecisionTreeClassifier')
    print('accuracy %s average %f' % (scores['test_accuracy'], np.average(scores['test_accuracy'])))
    print('precision %s average %f' % (scores['test_precision'], np.average(scores['test_precision'])))
    print('recall %s average %f' % (scores['test_recall'], np.average(scores['test_recall'])))
    pipeline.steps.pop()
    pipeline.steps.append(('clf', KNeighborsClassifier()))
    scores = cross_validate(pipeline, X, y, cv=10, scoring=['accuracy', 'precision', 'recall'])
    print('KNeighborsClassifier')
    print('accuracy %s average %f' % (scores['test_accuracy'], np.average(scores['test_accuracy'])))
    print('precision %s average %f' % (scores['test_precision'], np.average(scores['test_precision'])))
    print('recall %s average %f' % (scores['test_recall'], np.average(scores['test_recall'])))
    pipeline.steps.pop()

def try_model_2():
    df_modified = df.copy()
    df_modified['SentimentText'] = pd.Series(data=[tweet_cleaner(x) for x in df['SentimentText'].values])
    df_modified['length'] = df_modified['SentimentText'].str.len()

    def get_bins(X):
        return np.transpose(np.array([X['length'].values]))

    def getSentimentText(X):
        return X['SentimentText'].values

    pipeline = Pipeline(
        [
            ('fetures',
             FeatureUnion(
                 [
                     ('later',
                      Pipeline(
                          [
                              ('Sentiments', FunctionTransformer(getSentimentText)),
                              ('vect',
                               CountVectorizer(stop_words=None, max_df=0.5, max_features=10000, ngram_range=(1, 2),
                                               preprocessor=tweet_cleaner)),
                              ('tfidf', TfidfTransformer(use_idf=True, norm='l2'))
                          ]
                      )),
                     ('bins',
                      Pipeline(
                          [('selector', FunctionTransformer(get_bins)),
                           ('bins', KBinsDiscretizer(n_bins=4))]
                      )
                      )
                 ]
             )
             )
        ]
    )

    pipeline.steps.append(('clf', LogisticRegression(C=1.623776739188721, solver='liblinear', penalty='l2')))
    scores = cross_validate(pipeline, df_modified[['SentimentText', 'length']], y, cv=10,
                            scoring=['accuracy', 'precision', 'recall'])
    print('LogisticRegression')
    print('accuracy %s average %f' % (scores['test_accuracy'], np.average(scores['test_accuracy'])))
    print('precision %s average %f' % (scores['test_precision'], np.average(scores['test_precision'])))
    print('recall %s average %f' % (scores['test_recall'], np.average(scores['test_recall'])))
    pipeline.steps.pop()
    pipeline.steps.append(('clf', DecisionTreeClassifier(min_samples_split=5)))
    scores = cross_validate(pipeline, df_modified[['SentimentText', 'length']], y, cv=10,
                            scoring=['accuracy', 'precision', 'recall'])
    print('DecisionTreeClassifier')
    print('accuracy %s average %f' % (scores['test_accuracy'], np.average(scores['test_accuracy'])))
    print('precision %s average %f' % (scores['test_precision'], np.average(scores['test_precision'])))
    print('recall %s average %f' % (scores['test_recall'], np.average(scores['test_recall'])))
    pipeline.steps.pop()
    pipeline.steps.append(('clf', KNeighborsClassifier()))
    scores = cross_validate(pipeline, df_modified[['SentimentText', 'length']], y, cv=10,
                            scoring=['accuracy', 'precision', 'recall'])
    print('KNeighborsClassifier')
    print('accuracy %s average %f' % (scores['test_accuracy'], np.average(scores['test_accuracy'])))
    print('precision %s average %f' % (scores['test_precision'], np.average(scores['test_precision'])))
    print('recall %s average %f' % (scores['test_recall'], np.average(scores['test_recall'])))
    pipeline.steps.pop()

def try_model_3():
    df_modified = df.copy()
    count_upper = lambda message: sum(1 for c in message if c.isupper())
    df_modified['capital'] = df.SentimentText.map(count_upper)
    df_modified['bin_capital'] = pd.qcut(df_modified['capital'], 4, labels=[1, 2, 3, 4])

    def get_bins(X):
        return np.transpose(np.array([X['capital'].values]))

    def getSentimentText(X):
        return X['SentimentText'].values

    pipeline = Pipeline(
        [
            ('fetures',
             FeatureUnion(
                 [
                     ('later',
                      Pipeline(
                          [
                              ('Sentiments', FunctionTransformer(getSentimentText)),
                              ('vect',
                               CountVectorizer(stop_words=None, max_df=0.5, max_features=10000, ngram_range=(1, 2),
                                               preprocessor=tweet_cleaner)),
                              ('tfidf', TfidfTransformer(use_idf=True, norm='l2'))
                          ]
                      )),
                     ('bins',
                      Pipeline(
                          [('selector', FunctionTransformer(get_bins)),
                           ('bins', KBinsDiscretizer(n_bins=3))]
                      )
                      )
                 ]
             )
             )
        ]
    )

    pipeline.steps.append(('clf', LogisticRegression(C=1.623776739188721, solver='liblinear', penalty='l2')))
    scores = cross_validate(pipeline, df_modified[['SentimentText', 'capital']], y, cv=10,
                            scoring=['accuracy', 'precision', 'recall'])
    print('LogisticRegression')
    print('accuracy %s average %f' % (scores['test_accuracy'], np.average(scores['test_accuracy'])))
    print('precision %s average %f' % (scores['test_precision'], np.average(scores['test_precision'])))
    print('recall %s average %f' % (scores['test_recall'], np.average(scores['test_recall'])))
    pipeline.steps.pop()
    pipeline.steps.append(('clf', DecisionTreeClassifier(min_samples_split=5)))
    scores = cross_validate(pipeline, df_modified[['SentimentText', 'capital']], y, cv=10,
                            scoring=['accuracy', 'precision', 'recall'])
    print('DecisionTreeClassifier')
    print('accuracy %s average %f' % (scores['test_accuracy'], np.average(scores['test_accuracy'])))
    print('precision %s average %f' % (scores['test_precision'], np.average(scores['test_precision'])))
    print('recall %s average %f' % (scores['test_recall'], np.average(scores['test_recall'])))
    pipeline.steps.pop()
    pipeline.steps.append(('clf', KNeighborsClassifier()))
    scores = cross_validate(pipeline, df_modified[['SentimentText', 'capital']], y, cv=10,
                            scoring=['accuracy', 'precision', 'recall'])
    print('KNeighborsClassifier')
    print('accuracy %s average %f' % (scores['test_accuracy'], np.average(scores['test_accuracy'])))
    print('precision %s average %f' % (scores['test_precision'], np.average(scores['test_precision'])))
    print('recall %s average %f' % (scores['test_recall'], np.average(scores['test_recall'])))
    pipeline.steps.pop()

def try_model_4():
    def get_multiple(message):
        count = 0
        for i in range(len(message) - 1):
            if message[i] == message[i + 1]:
                count += 1

        return (float(count) + 1) / (float(len(message)) + 1)

    df_modified = df.copy()
    df_modified['multiple'] = df_modified.SentimentText.map(get_multiple)

    def get_bins(X):
        return np.transpose(np.array([X['multiple'].values]))

    def getSentimentText(X):
        return X['SentimentText'].values

    pipeline = Pipeline(
        [
            ('fetures',
             FeatureUnion(
                 [
                     ('later',
                      Pipeline(
                          [
                              ('Sentiments', FunctionTransformer(getSentimentText)),
                              ('vect',
                               CountVectorizer(stop_words=None, max_df=0.5, max_features=10000, ngram_range=(1, 2),
                                               preprocessor=tweet_cleaner)),
                              ('tfidf', TfidfTransformer(use_idf=True, norm='l2'))
                          ]
                      )),
                     ('bins',
                      Pipeline(
                          [('selector', FunctionTransformer(get_bins)),
                           ('bins', KBinsDiscretizer(n_bins=4))]
                      )
                      )
                 ]
             )
             )
        ]
    )
    pipeline.steps.append(('clf', LogisticRegression(C=1.623776739188721, solver='liblinear', penalty='l2')))
    scores = cross_validate(pipeline, df_modified[['SentimentText', 'multiple']], y, cv=10,
                            scoring=['accuracy', 'precision', 'recall'])
    print('LogisticRegression')
    print('accuracy %s average %f' % (scores['test_accuracy'], np.average(scores['test_accuracy'])))
    print('precision %s average %f' % (scores['test_precision'], np.average(scores['test_precision'])))
    print('recall %s average %f' % (scores['test_recall'], np.average(scores['test_recall'])))
    pipeline.steps.pop()
    pipeline.steps.append(('clf', DecisionTreeClassifier(min_samples_split=5)))
    scores = cross_validate(pipeline, df_modified[['SentimentText', 'multiple']], y, cv=10,
                            scoring=['accuracy', 'precision', 'recall'])
    print('DecisionTreeClassifier')
    print('accuracy %s average %f' % (scores['test_accuracy'], np.average(scores['test_accuracy'])))
    print('precision %s average %f' % (scores['test_precision'], np.average(scores['test_precision'])))
    print('recall %s average %f' % (scores['test_recall'], np.average(scores['test_recall'])))
    pipeline.steps.pop()
    pipeline.steps.append(('clf', KNeighborsClassifier()))
    scores = cross_validate(pipeline, df_modified[['SentimentText', 'multiple']], y, cv=10,
                            scoring=['accuracy', 'precision', 'recall'])
    print('KNeighborsClassifier')
    print('accuracy %s average %f' % (scores['test_accuracy'], np.average(scores['test_accuracy'])))
    print('precision %s average %f' % (scores['test_precision'], np.average(scores['test_precision'])))
    print('recall %s average %f' % (scores['test_recall'], np.average(scores['test_recall'])))
    pipeline.steps.pop()


def tweet_cleaner(text):
    tokenizer = WordPunctTokenizer()
    SMILEY = load_dict_smileys()
    CONTRACTIONS = load_dict_contractions()
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

df = pd.read_csv("Train.csv", encoding='latin-1')



X = df.SentimentText.values
y = df.iloc[:, 0].values


pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words=None, max_df=0.5, max_features=10000, ngram_range=(1,2), preprocessor=tweet_cleaner)),
    ('tfidf', TfidfTransformer(use_idf=True, norm='l2')),
    ('clf', LogisticRegression(C=1.623776739188721, solver='liblinear', penalty='l2')),
])
scores = cross_validate(pipeline, X, y, cv=10, scoring=['accuracy','precision','recall'])
print('LogisticRegression')
print('accuracy %s average %f' %(scores['test_accuracy'],np.average(scores['test_accuracy'])))
print('precision %s average %f' %(scores['test_precision'],np.average(scores['test_precision'])))
print('recall %s average %f' %(scores['test_recall'],np.average(scores['test_recall'])))
