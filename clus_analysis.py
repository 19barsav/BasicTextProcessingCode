from operator import itemgetter

import pandas as pd
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor
from nltk import word_tokenize, sent_tokenize, FreqDist
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

"""
Analyzes the clusters after the k-means script is run
Shows top words, sentiment, and a summary of the cluster
"""

# CHANGE THIS FOR NEW FILES
df = pd.read_csv('output_clusters.csv',
                 names=[ '', "text_raw", "instrument", "text_clean", "text_freq", 'cluster'])
unique_clusters = sorted(df.cluster.unique())
list_of_df = [df[df['cluster'] == i] for i in unique_clusters[:-1]]



for i, df in enumerate(list_of_df):
    print("TOP WORDS FOR CLUSTER " + unique_clusters[i])
    data = list(df["text_freq"])
    data2 = [word for item in data for word in item.split()]

    fdist = FreqDist(data2)
    count = 0
    for l, k in sorted(fdist.items(), key=itemgetter(1), reverse=True):
        print(l, k)
        count += 1
        if count == 7:
            print("\n\n\n")
            break

    sid = SentimentIntensityAnalyzer()

    df['sent_scores'] = df["text_freq"].apply(lambda review: sid.polarity_scores(review))
    df['compound'] = df['sent_scores'].apply(lambda score_dict: score_dict['compound'])

    df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c >= 0 else 'neg')

    ratio = sum(df['compound'].tolist()[1:]) / len(df['compound'].tolist()[1:])
    ratio_pos = sum([1 if i == 'pos' else 0 for i in df['comp_score'].tolist()]) / len(df['compound'].tolist())

    print("AVG COMP")
    print(ratio)
    print(ratio_pos)

    print(df.head())

    print("\n\n\n")
    document = '. '.join(df["text_freq"].tolist())
    print(type(document))

    # Summarize document.
    # Object of automatic summarization.
    auto_abstractor = AutoAbstractor()
    # Set tokenizer.
    auto_abstractor.tokenizable_doc = SimpleTokenizer()
    # Set delimiter for making a list of sentence.
    auto_abstractor.delimiter_list = [".", "\n"]
    # Object of abstracting and filtering document.
    abstractable_doc = TopNRankAbstractor()
    # Summarize document.
    result_dict = auto_abstractor.summarize(document, abstractable_doc)
    count2 = 0
    result_dict = auto_abstractor.summarize(document, abstractable_doc)
    for sentence in result_dict['summarize_result']:
        print(sentence)
        if count2 == 3:
            break
        count2 += 1
