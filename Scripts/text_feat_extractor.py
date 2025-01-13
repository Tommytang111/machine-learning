#Purpose: Different methods/algorithms for extracting features from text
#Models: Bag of Words, TF-IDF, Character n-gram, Word n-gram, Count Vectorizer
#Tommy Tang
#Nov 13th, 2024

#Libraries
import nltk  
import numpy as np  
import random  
import string
import bs4 as bs  
import urllib.request  
import re  
import pandas as pd

def get_web_text(url):
    """Extracts text from a given URL"""
    raw_html = urllib.request.urlopen(url)
    raw_html = raw_html.read()
    article_html = bs.BeautifulSoup(raw_html, 'lxml')
    article_paragraphs = article_html.find_all('p')
    article_text = ''
    for para in article_paragraphs:  
        article_text += para.text
    return article_text

def bag_of_words(text):
    """Bag of Words Model"""

    # Use Sentence Tokenizer to split the text into sentences
    nltk.download('punkt')
    corpus = nltk.sent_tokenize(text)

    #Remove special characters and extra spaces
    for i in range(len(corpus)):
        corpus [i] = corpus [i].lower()
        corpus [i] = re.sub(r'\W',' ',corpus [i])
        corpus [i] = re.sub(r'\s+',' ',corpus [i])

    #Create a dictionary of word frequencies
    wordfreq = {}
    for sentence in corpus:
        for token in nltk.word_tokenize(sentence):
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1

    #Select the most frequent words
    import heapq
    most_freq = heapq.nlargest(200, wordfreq, key=wordfreq.get)

    #Create sentence vectors
    sentence_vectors = []
    for sentence in corpus:
        sentence_tokens = nltk.word_tokenize(sentence)
        sent_vec = []
        for token in most_freq:
            if token in sentence_tokens:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        sentence_vectors.append(sent_vec)

    return np.asarray(sentence_vectors), most_freq

def tf_idf(text):
    """TF-IDF Model"""

    # Use Sentence Tokenizer to split the text into sentences
    nltk.download('punkt')
    corpus = nltk.sent_tokenize(text)

    #Remove special characters and extra spaces
    for i in range(len(corpus)):
        corpus [i] = corpus [i].lower()
        corpus [i] = re.sub(r'\W',' ',corpus [i]) 
        corpus [i] = re.sub(r'\s+',' ',corpus [i]) 

    #Create a dictionary of word frequencies
    wordfreq = {}
    for sentence in corpus:
        for token in nltk.word_tokenize(sentence):
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1

    #Select the most frequent words
    import heapq
    most_freq = heapq.nlargest(200, wordfreq, key=wordfreq.get)

    #Compute idf values
    word_idf_values = {}
    for token in most_freq:
        doc_containing_word = 0
        for document in corpus:
            if token in nltk.word_tokenize(document):
                doc_containing_word += 1
        word_idf_values[token] = np.log(len(corpus)/(doc_containing_word + 1)) 
        #Why +1 here? To avoid division by zero error
        #But then the number of sentences containing the word will be falsely inflated by 1. What do I do about that?
        #I can add a print statement to see the values of doc_containing_word and len(corpus)
    word_idf_values

    #Compute tf values
    word_tf_values = {}
    for token in most_freq:
        sent_tf_vector = []
        for document in corpus:
            freq = 0
            doc_words = nltk.word_tokenize(document)
            for word in doc_words:
                if word == token:
                    freq += 1
            sent_tf_vector.append(freq/len(doc_words))
        word_tf_values[token] = sent_tf_vector

    #Compute tf-idf values
    tfidf_values = []
    for token in word_tf_values.keys():
        tfidf_sentences = []
        for tf_sentence in word_tf_values[token]:
            tf_idf_score = tf_sentence * word_idf_values[token]
            tfidf_sentences.append(tf_idf_score)
        tfidf_values.append(tfidf_sentences)

    #Transpose and convert to numpy array
    tf_idf_model = np.transpose(np.asarray(tfidf_values))
    return tf_idf_model, most_freq

def char_n_gram(text, chars):
    """Character n-gram model with character length of 'chars'"""
    ngrams = {}

    #Create the n-grams
    for i in range(len(text)-chars):
        seq = text[i:i+chars]
        if seq not in ngrams.keys():
            ngrams[seq] = []
        ngrams[seq].append(text[i+chars])

    #Generate text using probable next characters from n-grams
    curr_sequence = text[0:chars]
    output = curr_sequence
    for i in range(200):
        if curr_sequence not in ngrams.keys():
            break
        possible_chars = ngrams[curr_sequence]
        next_char = possible_chars[random.randrange(len(possible_chars))]
        output += next_char
        curr_sequence = output[len(output)-chars:len(output)]

    return(output, ngrams)

def word_n_gram(text, words):
    """Word n-gram model with word length of 'words'"""
    words_tokens = nltk.word_tokenize(text)
    ngrams = {}

    #Create the n-grams
    for i in range(len(words_tokens)-words):
        seq = ' '.join(words_tokens[i:i+words])
        if seq not in ngrams.keys():
            ngrams[seq] = []
        ngrams[seq].append(words_tokens[i+words])
    
    #Generate text using probable next words from n-grams
    curr_sequence = ' '.join(words_tokens[0:words])
    output = curr_sequence
    for i in range(50):
        if curr_sequence not in ngrams.keys():
            break
        possible_words = ngrams[curr_sequence]
        next_word = possible_words[random.randrange(len(possible_words))]
        output += ' ' + next_word
        seq_words = nltk.word_tokenize(output)
        curr_sequence = ' '.join(seq_words[len(seq_words)-words:len(seq_words)])

    return(output, ngrams)

def count_vectorizer(text):
    """Count Vectorizer Model"""
    # Use Sentence Tokenizer to split the text into sentences
    nltk.download('punkt')
    corpus = nltk.sent_tokenize(text)

    #Remove special characters and extra spaces
    for i in range(len(corpus)):
        corpus [i] = corpus [i].lower()
        corpus [i] = re.sub(r'\W',' ',corpus [i])
        corpus [i] = re.sub(r'\s+',' ',corpus [i])

    #Count Vectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X.toarray(), vectorizer.get_feature_names_out()

if __name__ == '__main__':
    #print(bag_of_words(get_web_text('https://en.wikipedia.org/wiki/Artificial_intelligence'))[1])
    #print(tf_idf(get_web_text('https://en.wikipedia.org/wiki/Artificial_intelligence'))[0])
    #print(char_n_gram(get_web_text('https://en.wikipedia.org/wiki/Artificial_intelligence'), 3)[0])
    #print(word_n_gram(get_web_text('https://en.wikipedia.org/wiki/Artificial_intelligence'), 3)[0])
    #print(count_vectorizer(get_web_text('https://en.wikipedia.org/wiki/Artificial_intelligence'))[1])s