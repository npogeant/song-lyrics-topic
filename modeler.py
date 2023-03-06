import os
from itertools import chain
from collections import Counter

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import string

import spacy
from octis.models.LDA import LDA
from octis.dataset.dataset import Dataset
from octis.preprocessing.preprocessing import Preprocessing
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity

#!spacy download en_core_web_sm
spacy_nlp = spacy.load("en_core_web_sm")
import en_core_web_sm
from datawrapper import Datawrapper
from spacy.lang.en.stop_words import STOP_WORDS

API_TOKEN = os.getenv('DW_TOKEN')
dw = Datawrapper(access_token=API_TOKEN)


# Define a TopicModeler class
class TopicModeler:
    # Define the constructor for the class
    def __init__(self):
        # Initialize an LDA model with 5 topics
        self.model = LDA(num_topics=5)

        # Add some additional stop words to the spacy stop word list
        spacy_nlp.Defaults.stop_words |= {
            'la',
            'th',
            'aja',
            'feat',
            'cruz',
            'doo',
            'ah',
            'aha',
            'ahem',
            'ahh',
            'ahoy',
            'alas',
            'arg',
            'aw',
            'bam',
            'bingo',
            'blah',
            'boo',
            'bravo',
            'brrr',
            'cheers',
            'dang',
            'dart',
            'darn',
            'duh',
            'eek',
            'eh',
            'gee',
            'golly',
            'gosh',
            'ha',
            'hallelujah',
            'hey',
            'hmm',
            'huh',
            'humph',
            'hurray',
            'oh',
            'ooh',
            'ohh',
            'oops',
            'ouch',
            'ow',
            'phew',
            'phooey',
            'pooh',
            'pow',
            'shh',
            'shoo',
            'uh',
            'ugh',
            'wahoo',
            'well',
            'whoa',
            'whoops',
            'wow',
            'yeah',
            'yes',
            'yikes',
            'yippee',
            'yo',
            'yuck',
        }

        # Initialize a Preprocessing object with some default settings
        self.preprocessor = Preprocessing(
            vocabulary=None,
            max_features=None,
            remove_punctuation=True,
            punctuation=string.punctuation,
            lemmatize=True,
            stopword_list=list(STOP_WORDS),
            min_chars=2,
            min_words_docs=1,
            min_df=0.1,
            max_df=0.8,
        )

        # Initialize empty DataFrames and dictionaries for later use
        self.dataset = pd.DataFrame()
        self.top_words_topics = pd.DataFrame()
        self.topics = {}
        self.topic_word_matrix = {}
        self.topic_document_matrix = {}
        self.diversity_score = 0
        self.coherence_score = 0

    # Method to train the LDA model on a given corpus
    def modelize(self, corpus_path: str):
        # Preprocess the corpus and store it in the dataset attribute
        self.dataset = self._preprocess_corpus(corpus_path)

        # Train the LDA model on the preprocessed corpus and store the output
        output = self.model.train_model(self.dataset)

        # Store the topic word matrix and topic document matrix in their respective attributes
        self.topics = output['topics']
        self.topic_word_matrix = output['topic-word-matrix']
        self.topic_document_matrix = output['topic-document-matrix']

        # Evaluate the topics using coherence and diversity scores
        self._evaluate_topics(self.dataset, output)

    # Method to visualize the top words and frequencies of the trained topics for a given genre
    def visualize(self, genre):
        # Check if the topics have been trained before
        if not self.topics:
            print('No topics available, use modelize before.')
        else:
            # Get the top words for each topic and store them in the top_words_topics attribute
            self._get_top_words_topics()

            # Build the topics chart and frequencies chart using the top_words_topics attribute
            self._build_topics_chart(genre)
            self._build_frequencies_chart(genre)

    # This method preprocesses the given corpus by calling preprocess_dataset method of the Preprocessing class and returns the preprocessed dataset
    def _preprocess_corpus(self, corpus_path):
        return self.preprocessor.preprocess_dataset(corpus_path)

    # This method evaluates the topics by calculating coherence and diversity scores and assigning them to corresponding class attributes
    def _evaluate_topics(self, dataset, output):
        npmi = Coherence(texts=dataset.get_corpus())
        topic_diversity = TopicDiversity(topk=10)

        self.diversity_score = topic_diversity.score(output)
        self.coherence_score = npmi.score(output)

    # This method extracts top words for each topic and creates a DataFrame that shows the top words and their weights
    def _get_top_words_topics(self):
        # Get the vocabulary of the preprocessed dataset
        vocabulary = self.dataset.get_vocabulary()

        # For each topic in the topic-word matrix, create a DataFrame that shows the top words and their weights
        for topic_idx, topic in enumerate(self.topic_word_matrix):
            dataframe = (
                pd.DataFrame(
                    {
                        'Word': [vocabulary[i] for i, k in enumerate(topic)],
                        'Weights': topic,
                    }
                )
                .sort_values(by='Weights', ascending=False)
                .head(5)
            )

            # Add the topic number to the DataFrame
            dataframe['Topic'] = f"Topic {topic_idx+1}"

            # Concatenate the DataFrame with the class attribute that stores top words for each topic
            self.top_words_topics = pd.concat([self.top_words_topics, dataframe])

    # Method to build topics chart with Datawrapper
    def _build_topics_chart(self, genre):
        # Define the chart properties
        properties = {
            'visualize': {
                'dark-mode-invert': True,
                'highlighted-series': [],
                'highlighted-values': [],
                'sharing': {'enabled': False},
                'rules': False,
                'thick': False,
                'sort-by': 'Weights',
                'overlays': [],
                'sort-bars': False,
                'background': False,
                'base-color': 0,
                'force-grid': False,
                'mirror-bars': False,
                'swap-labels': False,
                'block-labels': False,
                'custom-range': ['', ''],
                'range-extent': 'nice',
                'thick-arrows': False,
                'reverse-order': False,
                'tick-position': 'top',
                'totals-labels': False,
                'chart-type-set': True,
                'color-category': {
                    'map': {
                        'Topic 1': '#721817',
                        'Topic 2': '#fa9f42',
                        'Topic 3': '#2b4162',
                        'Topic 4': '#0b6e4f',
                        'Topic 5': '#e0e0e2',
                    },
                    'palette': [],
                    'categoryOrder': [
                        'Topic 1',
                        'Topic 2',
                        'Topic 3',
                        'Topic 4',
                        'Topic 5',
                    ],
                    'categoryLabels': {},
                },
                'show-color-key': False,
                'color-by-column': True,
                'group-by-column': True,
                'label-alignment': 'right',
                'value-label-mode': 'left',
                'custom-grid-lines': '',
                'date-label-format': 'YYYY',
                'show-group-labels': True,
                'show-value-labels': True,
                'stack-percentages': False,
                'independent-scales': False,
                'space-between-cols': 17,
                'value-label-format': '0,0.[00]',
                'compact-group-labels': True,
                'show-category-labels': True,
                'value-label-alignment': 'left',
                'totals-labels-position': 'before',
                'value-label-visibility': 'always',
            },
            'axes': {'colors': 'Topic', 'groups': 'Topic'},
        }

        # Create the chart with the given properties
        chart = dw.create_chart(
            title=f"{genre} - Topic by top terms",
            chart_type='d3-bars-split',
            data=self.top_words_topics,
            folder_id='147354',
            metadata=properties,
        )

        # Update the chart's description with information about the data source and topic representation
        dw.update_description(
            chart['id'],
            source_name='Genius',
            intro='The topics found by the <b>LDA Model</b>, represented by their <b>top 5 weighted terms</b>.',
        )

    # Method to build words by their frequency chart with Datawrapper
    def _build_frequencies_chart(self, genre):
        # Get all words from the corpus and count their frequency
        joined_list = list(chain(*self.dataset.get_corpus()))
        freq_dict = Counter(joined_list)

        # Select the top 10 most frequent terms and create a DataFrame with their frequencies
        freq_df = (
            pd.DataFrame.from_dict(freq_dict, orient='index', columns=['Frequency'])
            .reset_index()
            .sort_values(by='Frequency', ascending=False)
            .head(10)
        )

        # Define the chart properties
        properties = {
            'visualize': {
                'dark-mode-invert': True,
                'highlighted-series': [],
                'highlighted-values': [],
                'sharing': {'enabled': False},
                'rules': True,
                'thick': False,
                'overlays': [
                    {
                        'to': '--zero-baseline--',
                        'from': 'Frequency',
                        'type': 'value',
                        'color': 0,
                        'title': '',
                        'opacity': 0.6,
                        'pattern': 'solid',
                        'labelDirectly': True,
                    }
                ],
                'sort-bars': False,
                'background': False,
                'base-color': 0,
                'force-grid': False,
                'swap-labels': False,
                'block-labels': False,
                'custom-range': ['', ''],
                'range-extent': 'nice',
                'thick-arrows': False,
                'reverse-order': False,
                'tick-position': 'top',
                'chart-type-set': True,
                'color-category': {'map': {}, 'palette': [], 'categoryLabels': {}},
                'show-color-key': True,
                'color-by-column': False,
                'group-by-column': False,
                'label-alignment': 'left',
                'custom-grid-lines': '',
                'date-label-format': 'YYYY',
                'show-group-labels': True,
                'show-value-labels': True,
                'value-label-format': '0,0.[00]',
                'compact-group-labels': False,
                'show-category-labels': True,
                'value-label-alignment': 'left',
                'value-label-visibility': 'always',
            },
            'axes': {'groups': 'index'},
        }

        # Create the chart with the given properties
        chart = dw.create_chart(
            title=f"{genre} - Most frequent terms",
            chart_type='d3-bars',
            data=freq_df,
            folder_id='147354',
            metadata=properties,
        )

        # Update the chart's description with information about the data source and topic representation
        dw.update_description(
            chart['id'],
            source_name='Genius',
            intro='Top <b>10 terms</b> according to their <b>frequency</b> in the whole text corpus.',
        )


if __name__ == "__main__":
    # List of music genres
    genres = [
        'all',
        'electronic',
        'rock',
        'Hip-Hop',
        'indie',
        'jazz',
        'reggae',
        'british',
        'punk',
        '80s',
        'acoustic',
        'rnb',
        'hardcore',
        'country',
        'blues',
        'alternative',
        'rap',
        'metal',
    ]

    # Initializing lists to hold diversity and coherence scores for each genre
    diversity_scores = []
    coherence_scores = []

    # Looping through each genre and performing topic modeling
    for genre in genres:
        # Initializing an instance of the TopicModeler class
        modeler = TopicModeler()

        # Performing topic modeling on lyrics data for the specified genre
        modeler.modelize(
            f"./data/{genre}_lyrics.csv",
        )

        # Appending the diversity and coherence scores to their respective lists
        diversity_scores.append(modeler.diversity_score)
        coherence_scores.append(modeler.coherence_score)

        # Creating a visualization of the topics for the current genre
        modeler.visualize(genre.upper())

    # Creating a dataframe to hold the metrics scores for each genre
    dataframe = pd.DataFrame(
        {
            'Genre': genres,
            'Diversity Score': diversity_scores,
            'Coherence Score': coherence_scores,
        }
    )

    # Saving the metrics scores dataframe as a CSV file
    dataframe.to_csv('./results/metrics_scores.csv', index=None)
