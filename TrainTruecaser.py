"""
This script trains the TrueCase System
"""
from nltk.corpus import brown
from nltk.corpus import reuters
import pickle as cPickle

from TrainFunctions import *
from EvaluateTruecaser import defaultTruecaserEvaluation
import sys


def down_nltk():
    nltk.download('brown')
    nltk.download('reuters')
    nltk.download('semcor')
    nltk.download('conll2000')
    nltk.download('state_union')
    nltk.download('punkt')


def init():
    uniDist = nltk.FreqDist()
    backwardBiDist = nltk.FreqDist()
    forwardBiDist = nltk.FreqDist()
    trigramDist = nltk.FreqDist()
    wordCasingLookup = {}
    return uniDist, backwardBiDist, forwardBiDist, trigramDist, wordCasingLookup


def save_model(uniDist, backwardBiDist, forwardBiDist, trigramDist, wordCasingLookup, out_file):
    f = open('out_file', 'wb')
    cPickle.dump(uniDist, f, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(backwardBiDist, f, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(forwardBiDist, f, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(trigramDist, f, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(wordCasingLookup, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

        
"""
There are three options to train the true caser:
1) Use the sentences in NLTK
2) Use the train.txt file. Each line must contain a single sentence. Use a large corpus, for example Wikipedia
3) Use Bigrams + Trigrams count from the website http://www.ngrams.info/download_coca.asp

The more training data, the better the results
"""


def train_nltk(out_file):
    down_nltk()
    uniDist, backwardBiDist, forwardBiDist, trigramDist, wordCasingLookup = init()
    # :: Option 1: Train it based on NLTK corpus ::
    print("Update from NLTK Corpus")
    NLTKCorpus = brown.sents()+reuters.sents()+nltk.corpus.semcor.sents()+nltk.corpus.conll2000.sents()+nltk.corpus.state_union.sents()
    NLTKCorpus = [x[1:] for x in NLTKCorpus if len(x) > 1]
    updateDistributionsFromSentences(NLTKCorpus, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
    print(f"training sentence number: {len(NLTKCorpus)}")
    defaultTruecaserEvaluation(wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
    save_model(uniDist, backwardBiDist, forwardBiDist, trigramDist, wordCasingLookup, out_file)


def train(in_file, out_file):
    uniDist, backwardBiDist, forwardBiDist, trigramDist, wordCasingLookup = init()

    # :: Option 2: Train it based the train.txt file ::
    #Uncomment, if you want to train from train.txt
    print(f"Update from {in_file} file")
    sentences = []
    for line in open(in_file):
        sentences.append(line.strip())

    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    tokens = [x[1:] for x in tokens if len(x) > 1]
    updateDistributionsFromSentences(tokens, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
    print(f"training sentence number: {len(tokens)}")
    defaultTruecaserEvaluation(wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
    save_model(uniDist, backwardBiDist, forwardBiDist, trigramDist, wordCasingLookup, out_file)
   
# :: Option 3: Train it based ngrams tables from http://www.ngrams.info/download_coca.asp ::    
'''#Uncomment, if you want to train from train.txt
print("Update Bigrams / Trigrams")
updateDistributionsFromNgrams('ngrams/w2.txt', 'ngrams/w3.txt', wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)'''

        
if __name__ == "__main__":
    if sys.argv[1] == "nltk":
        train_nltk(sys.argv[2])
    else:
        train(sys.argv[1], sys.argv[2])
