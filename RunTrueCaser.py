from Truecaser import *
import pickle as cPickle


class RunTrueCaser:

    def __init__(self, model_file):
        f = open(model_file, 'rb')
        self.uniDist = cPickle.load(f)
        self.backwardBiDist = cPickle.load(f)
        self.forwardBiDist = cPickle.load(f)
        self.trigramDist = cPickle.load(f)
        self.wordCasingLookup = cPickle.load(f)
        f.close()

    def predict(self, sentence):
        if isinstance(sentence, str):
            sentence = sentence.split(" ")
        tokens = [x.lower() for x in sentence]
        tokensTrueCase = getTrueCase(tokens, 'title',
                                     self.wordCasingLookup,
                                     self.uniDist,
                                     self.backwardBiDist,
                                     self.forwardBiDist,
                                     self.trigramDist)
        return tokensTrueCase


if __name__ == "__main__":
    print("input model file")  # mnt/nlp/search/query_understanding/truecaszser/distributions.obj
    model_file = input()
    true_caser = RunTrueCaser(model_file)
    while True:
        print("input one sentence")
        sentence = input()
        output = true_caser.predict(sentence.split(' '))
        print("output results")
        print(output)
