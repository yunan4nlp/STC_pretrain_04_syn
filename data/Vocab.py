import sys
sys.path.extend(["../","./"])
import numpy as np

class NMTVocab:
    PAD, BOS, EOS, UNK = 0, 1, 2, 3
    S_PAD, S_BOS, S_EOS, S_UNK = '<pad>', '<s>', '</s>', '<unk>'
    def __init__(self, word_list):
        """
        :param word_list: list of words
        """
        self.i2w = [self.S_PAD, self.S_BOS, self.S_EOS, self.S_UNK] + word_list

        reverse = lambda x: dict(zip(x, range(len(x))))
        self.w2i = reverse(self.i2w)
        if len(self.w2i) != len(self.i2w):
            print("serious bug: words dumplicated, please check!")

        print("Vocab info: #words %d" % (self.vocab_size))

        self.i2extw = [self.S_PAD, self.S_BOS, self.S_EOS, self.S_UNK] # 4 word embedding vocab



    def word2id(self, xs):
        if isinstance(xs, list):
            return [self.w2i.get(x, self.UNK) for x in xs]
        return self.w2i.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self.i2w[x] for x in xs]
        return self.i2w[xs]

    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self.extw2i.get(x, self.UNK) for x in xs]
        return self.extw2i.get(xs, self.UNK)

    def id2extword(self, xs):
        if isinstance(xs, list):
            return [self.i2extw[x] for x in xs]
        return self.i2extw[xs]

    def save2file(self, outfile):
        with open(outfile, 'w', encoding='utf8') as file:
            for id, word in enumerate(self.i2w):
                if id > self.UNK: file.write(word + '\n')
            file.close()

    def load_pretrained_embs(self, embfile):
        embedding_dim = -1
        word_count = 0
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                if word_count < 1:
                    values = line.split()
                    embedding_dim = len(values) - 1
                word_count += 1
        print('Total words: ' + str(word_count) + '\n')
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        index = len(self.i2extw)
        embeddings = np.zeros((word_count + index, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                self.i2extw.append(values[0])
                vector = np.array(values[1:], dtype='float64')
                embeddings[self.UNK] += vector
                embeddings[index] = vector
                index += 1

        embeddings[self.UNK] = embeddings[self.UNK] / word_count
        embeddings = embeddings / np.std(embeddings)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self.extw2i = reverse(self.i2extw)

        if len(self.extw2i) != len(self.i2extw):
            print("serious bug: extern words dumplicated, please check!")

        return embeddings

    @property
    def vocab_size(self):
        return len(self.i2w)
    @property
    def extvocab_size(self):
        return len(self.i2extw)

import argparse
import pickle

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--infile', default='ch-en-model/tgt_vocab')
    argparser.add_argument('--outfile', default='ch-en-model/tgt_vocab.txt')

    args, extra_args = argparser.parse_known_args()

    vocab = pickle.load(open(args.infile, 'rb'))
    vocab.save2file(args.outfile)
