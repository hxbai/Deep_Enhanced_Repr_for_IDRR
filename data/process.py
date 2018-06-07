import sys
sys.path.append('../')
import os
import numpy as np
import torch
import gensim
import pickle
import argparse

import codecs
from bpe.learn_bpe import main as learn_bpe
from bpe.apply_bpe import BPE

from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('func', type=str)
parser.add_argument('splitting', type=int)

conf = Config()

class PreData(object):
    def __init__(self, corpus_splitting, need_bpe=True, need_we=True, need_data=True):
        self.corpus_splitting = corpus_splitting
        self._read_text()
        self._write_train_sent()
        self._get_vocab()
        if need_bpe:
            self._bpe()
        if need_we:
            self._get_we()
            self._make_char_table()
            self._make_subword_table()
        if need_data:
            self._pre_data()

    def _read_text(self):
        print('reading text...')
        if self.corpus_splitting == 1:
            path_pre = './interim/lin/'
        elif self.corpus_splitting == 2:
            path_pre = './interim/ji/'
        elif self.corpus_splitting == 3:
            path_pre = './interim/l/'
        with open(path_pre + 'train.pkl', 'rb') as f:
            self.arg1_train_r = pickle.load(f)
            self.arg2_train_r = pickle.load(f)
            self.conn_train_r = pickle.load(f)
            self.sense_train_r = pickle.load(f)
        with open(path_pre + 'dev.pkl', 'rb') as f:
            self.arg1_dev_r = pickle.load(f)
            self.arg2_dev_r = pickle.load(f)
            self.sense1_dev_r = pickle.load(f)
            self.sense2_dev_r = pickle.load(f)
        with open(path_pre + 'test.pkl', 'rb') as f:
            self.arg1_test_r = pickle.load(f)
            self.arg2_test_r = pickle.load(f)
            self.sense1_test_r = pickle.load(f)
            self.sense2_test_r = pickle.load(f)

    def _write_train_sent(self):
        print('writing training sent...')
        if self.corpus_splitting == 1:
            fname = './bpe/train_sent_lin.txt'
        elif self.corpus_splitting == 2:
            fname = './bpe/train_sent_ji.txt'
        elif self.corpus_splitting == 3:
            fname = './bpe/train_sent_l.txt'
        sentlist = []
        for l in [self.arg1_train_r, self.arg2_train_r]:
            for sent in l:
                sentlist.append(' '.join(sent))
        with open(fname, 'w') as f:
            for sent in sentlist:
                f.write(sent.lower() + '\n')

    def _get_vocab(self):
        print('getting vocab...')
        self.word2i = {'<pad>':0, '</s>':1}
        self.v_size = 2
        for sentlist in [
            self.arg1_train_r, self.arg2_train_r,
            self.arg1_dev_r, self.arg2_dev_r, self.arg1_test_r, self.arg2_test_r,
        ]:
            for sent in sentlist:
                for word in sent:
                    if word not in self.word2i:
                        self.word2i[word] = self.v_size
                        self.v_size += 1
        self._vocab_write('vocab', self.word2i)

    def _vocab_write(self, fname, dict):
        if self.corpus_splitting == 1:
            s = '_lin.txt'
        elif self.corpus_splitting == 2:
            s = '_ji.txt'
        elif self.corpus_splitting == 3:
            s = '_l.txt'
        with open(fname+s, 'w') as f:
            for (word, idx) in dict.items():
                tmp = str(idx) + ' ' + word + '\n'
                f.write(tmp)
        if fname == 'vocab':
            with open('./bpe/'+fname+s, 'w') as f:
                for (word, idx) in dict.items():
                    f.write(word+'\n')

    def _bpe(self):
        # if sys.version_info < (3, 0):
        #     sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
        #     sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
        #     sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
        # else:
        #     sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
        #     sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
        #     sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)
        if self.corpus_splitting == 1:
            s = '_lin'
        elif self.corpus_splitting == 2:
            s = '_ji'
        elif self.corpus_splitting == 3:
            s = '_l'

        print('learning bpe...')
        with codecs.open('./bpe/train_sent'+s+'.txt', encoding='utf-8') as input:
            with codecs.open('./bpe/codes'+s+'.txt', 'w', encoding='utf-8') as output:
                learn_bpe(input, output, 1000)

        print('applying bpe...')
        with codecs.open('./bpe/codes'+s+'.txt', encoding='utf-8') as codes:
            with codecs.open('./bpe/vocab'+s+'.txt', encoding='utf-8') as input:
                with codecs.open('./bpe/bpe'+s+'.txt', 'w', encoding='utf-8') as output:
                    bpe = BPE(codes)
                    for line in input:
                        output.write(bpe.segment(line).strip())
                        output.write('\n')

        with open('./bpe/vocab'+s+'_1k.txt', 'w') as new:
            with codecs.open('./bpe/vocab'+s+'.txt', encoding='utf-8') as origin:
                with codecs.open('./bpe/bpe'+s+'.txt', encoding='utf-8') as seg:
                    origin_l = [word.replace('\n', '') for word in origin]
                    seg_l = [word.replace('\n', '') for word in seg]
                    assert len(origin_l) == len(seg_l)
                    for i in range(len(origin_l)):
                        new.write(origin_l[i] + ' ' + seg_l[i].replace('@@', '') + '\n')

    def _get_we(self):
        print('reading pretrained w2v...')
        w2v = gensim.models.KeyedVectors.load_word2vec_format(conf.wordvec_path, binary=True)
        pretrained_vocab = w2v.vocab.keys()
        print('making we...')
        we = np.zeros((self.v_size, conf.wordvec_dim))
        n = 0
        for word, idx in self.word2i.items():
            if word in pretrained_vocab:
                we[idx, :] = w2v[word]
                n += 1
            elif word.lower() in pretrained_vocab:
                we[idx, :] = w2v[word.lower()]
                n += 1
        self.we = torch.from_numpy(we)
        if self.corpus_splitting == 1:
            torch.save(self.we, './processed/lin/we.pkl')
        elif self.corpus_splitting == 2:
            torch.save(self.we, './processed/ji/we.pkl')
        elif self.corpus_splitting == 3:
            torch.save(self.we, './processed/l/we.pkl')
        print('{}/{} words found'.format(n, self.v_size))

    def _make_char_table(self):
        print('making char lookup/len table...')
        lookup_table = np.zeros((self.v_size, 50))
        len_table = np.zeros((self.v_size, 1))
        for word, idx in self.word2i.items():
            if idx == 0 or idx == 1:
                len_table[idx, 0] = 1
                continue
            len_table[idx, 0] = min(len(word) + 2, 50)
            word_encoded = word.encode('utf-8', 'ignore')[:48]
            char_ids = [260] * 50
            char_ids[0] = 258
            for k, chr_id in enumerate(word_encoded, start=1):
                char_ids[k] = chr_id
            char_ids[len(word_encoded) + 1] = 259
            res = [c + 1 for c in char_ids]
            lookup_table[idx, :] = np.array(res)
        lookup_table = torch.from_numpy(lookup_table)
        len_table = torch.from_numpy(len_table)
        if self.corpus_splitting == 1:
            torch.save([lookup_table, len_table], './processed/lin/char_table.pkl')
        elif self.corpus_splitting == 2:
            torch.save([lookup_table, len_table], './processed/ji/char_table.pkl')
        elif self.corpus_splitting == 3:
            torch.save([lookup_table, len_table], './processed/l/char_table.pkl')
            
    def _make_subword_table(self):
        print('making subword lookup/len table...')
        lookup_table = np.zeros((self.v_size, 12))
        len_table = np.zeros((self.v_size, 1))
        self.sub2i = {'<pad>':0, '</s>':1, '<sos>':2, '<eos>':3}
        self.sub_size = 4
        if self.corpus_splitting == 1:
            fname = './bpe/vocab_lin_1k.txt'
        elif self.corpus_splitting == 2:
            fname = './bpe/vocab_ji_1k.txt'
        elif self.corpus_splitting == 3:
            fname = './bpe/vocab_l_1k.txt'
        w2sub = {}
        with open(fname, 'r') as f:
            for line in f:
                splited_line = line.strip().split(' ')
                word = splited_line[0]
                sub = [s for s in splited_line[1:]]
                w2sub[word] = sub
        w2sub['<pad>'] = ['<pad>']
        w2sub['</s>'] = ['</s>']
        for (word, sub) in w2sub.items():
            for s in sub:
                if s not in self.sub2i:
                    self.sub2i[s] = self.sub_size
                    self.sub_size += 1
        self._vocab_write('sub', self.sub2i)
        n = 0
        for w, idx in self.word2i.items():
            if w in w2sub:
                word = w
                n += 1
            elif w.lower() in w2sub:
                word = w.lower()
                n += 1
            else:
                continue
            len_table[idx, 0] = min(len(w2sub[word])+2, 12)
            sub_ids = [0] * 12
            sub_ids[0] = 2
            for k, sub in enumerate(w2sub[word][:10], start=1):
                sub_ids[k] = self.sub2i[sub]
            sub_ids[len(w2sub[word][:10]) + 1] = 3
            lookup_table[idx, :] = np.array(sub_ids)
        lookup_table = torch.from_numpy(lookup_table)
        len_table = torch.from_numpy(len_table)
        if self.corpus_splitting == 1:
            torch.save([lookup_table, len_table], './processed/lin/sub_table.pkl')
        elif self.corpus_splitting == 2:
            torch.save([lookup_table, len_table], './processed/ji/sub_table.pkl')
        elif self.corpus_splitting == 3:
            torch.save([lookup_table, len_table], './processed/l/sub_table.pkl')
        print('{}/{} words found'.format(n, self.v_size))

    def _text2i(self, texts):
        l = len(texts)
        tensor = torch.LongTensor(l, conf.max_sent_len).zero_()
        for i in range(l):
                s = texts[i] + ['</s>']
                minlen = min(len(s), conf.max_sent_len)
                for j in range(minlen):
                    tensor[i][j] = self.word2i[s[j]]
        return tensor

    def _sense2i(self, senses):
        if self.corpus_splitting == 1 or self.corpus_splitting == 2:
            sense2i = conf.sense2i
            senseidx = 0
        elif self.corpus_splitting == 3:
            sense2i = conf.senseclass2i
            senseidx = 1
        l = len(senses)
        tensor = torch.LongTensor(l)
        for i in range(l):
            if senses[i][senseidx] is None:
                tensor[i] = -1
            else:
                tensor[i] = sense2i[senses[i][senseidx]]
        return tensor

    def _makeconnlist(self):
        print('make conn list...')
        self.conn2i = {}
        self.conn_size = 0
        for conn in self.conn_train_r:
            if conn not in self.conn2i:
                self.conn2i[conn] = self.conn_size
                self.conn_size += 1
        print('{} conns.'.format(self.conn_size))
        self._vocab_write('conn', self.conn2i)

    def _conn2i(self, conns):
        l = len(conns)
        tensor = torch.LongTensor(l)
        for i in range(l):
            tensor[i] = self.conn2i[conns[i]]
        return tensor

    def _pre_data(self):
        self._makeconnlist()
        print('pre training data...')
        train_data = [
            self._text2i(self.arg1_train_r),
            self._text2i(self.arg2_train_r),
            self._sense2i(self.sense_train_r),
            self._conn2i(self.conn_train_r)
        ]
        print('pre dev/test data...')
        dev_data = [
            self._text2i(self.arg1_dev_r),
            self._text2i(self.arg2_dev_r),
            self._sense2i(self.sense1_dev_r),
            self._sense2i(self.sense2_dev_r),
        ]
        test_data = [
            self._text2i(self.arg1_test_r),
            self._text2i(self.arg2_test_r),
            self._sense2i(self.sense1_test_r),
            self._sense2i(self.sense2_test_r),
        ]
        print('saving data...')
        if self.corpus_splitting == 1:
            pre = './processed/lin/'
        elif self.corpus_splitting == 2:
            pre = './processed/ji/'
        elif self.corpus_splitting == 3:
            pre = './processed/l/'
        torch.save(train_data, pre+'train.pkl')
        torch.save(dev_data, pre+'dev.pkl')
        torch.save(test_data, pre+'test.pkl')
        print('predata done.')

def testpredata(corpus_splitting):
    if corpus_splitting == 1:
        pre = './processed/lin/'
    elif corpus_splitting == 2:
        pre = './processed/ji/'
    elif corpus_splitting == 3:
        pre = './processed/l/'
    we = torch.load(pre+'we.pkl')
    lookup_table, len_table = torch.load(pre+'char_table.pkl')
    sub_lookup, sub_len = torch.load(pre+'sub_table.pkl')
    train_data = torch.load(pre+'train.pkl')
    dev_data = torch.load(pre+'dev.pkl')
    test_data = torch.load(pre+'test.pkl')
    print(we)
    print(lookup_table)
    print(len_table)
    print(sub_lookup)
    print(sub_len)
    for data in [train_data, dev_data, test_data]:
        for d in data:
            print(d.size())

def main():
    A = parser.parse_args()
    if A.func == 'pre':
        if A.splitting == 1:
            d = PreData(1)
        elif A.splitting == 2:
            d = PreData(2)
        elif A.splitting == 3:
            d = PreData(3)
    elif A.func == 'test':
        if A.splitting == 1:
            testpredata(1)
        elif A.splitting == 2:
            testpredata(2)
        elif A.splitting == 3:
            testpredata(3)
    else:
        raise Exception('wrong args')

if __name__ == '__main__':
    main()