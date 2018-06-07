import torch
from datetime import datetime

class Config(object):
    def __init__(self, classnum=11, splitting=2):
        self.i2sense = [
            'Temporal.Asynchronous', 'Temporal.Synchrony', 'Contingency.Cause',
            'Contingency.Pragmatic cause', 'Comparison.Contrast', 'Comparison.Concession',
            'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.Restatement',
            'Expansion.Alternative','Expansion.List'
        ]
        self.sense2i = {
            'Temporal.Asynchronous':0, 'Temporal.Synchrony':1, 'Contingency.Cause':2,
            'Contingency.Pragmatic cause':3, 'Comparison.Contrast':4, 'Comparison.Concession':5,
            'Expansion.Conjunction':6, 'Expansion.Instantiation':7, 'Expansion.Restatement':8,
            'Expansion.Alternative':9,'Expansion.List':10
        }
        self.i2senseclass = ['Temporal', 'Contingency', 'Comparison', 'Expansion']
        self.senseclass2i = {'Temporal':0, 'Contingency':1, 'Comparison':2, 'Expansion':3}

        self.four_or_eleven = classnum         # 11, 4, 2
        self.corpus_splitting = splitting        # 1 for Lin, 2 for Ji, 3 for 4-way and binary
        if self.four_or_eleven == 4 or self.four_or_eleven == 2:
            self.corpus_splitting = 3
        self.binclass = 0         # 0, 1, 2, 3 self.senseclass2i

        self.wordvec_path = '~/Projects/GoogleNews-vectors-negative300.bin.gz'
        self.wordvec_dim = 300
        self.max_sent_len = 100

        ################################################################################
        # attention
        self.attn_topk = 2
        self.attn_dropout = 0
        
        ###############################################################################
        # char/sub
        self.need_char = False
        self.need_sub = True
        if self.need_sub:
            self.need_char = False
        self.char_num = 262
        self.char_padding_idx = 261
        if self.need_sub:
            if self.corpus_splitting == 1:
                self.char_num = 982
            elif self.corpus_splitting == 2:
                self.char_num = 982
            elif self.corpus_splitting == 3:
                self.char_num = 982
            self.char_padding_idx = 0
        self.char_embed_dim = 50
        self.char_enc_dim = 50
        self.char_filter_num = 2
        self.char_filter_dim = [2, 3]
        self.char_dropout = 0
        self.char_hid_dim = self.char_enc_dim * self.char_filter_num

        ###############################################################################
        # elmo
        self.need_elmo = True
        self.elmo_options = '~/Projects/ELMo/elmo_2x4096_512_2048cnn_2xhighway_options.json'
        self.elmo_weights = '~/Projects/ELMo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
        self.elmo_dropout = 0
        self.elmo_labmda = 0.001
        self.elmo_dim = 300

        ################################################################################
        # CNNLayer RNNLayer
        self.use_rnn = False

        if self.corpus_splitting == 1:
            self.embed_dropout = 0.4
        elif self.corpus_splitting == 2:
            self.embed_dropout = 0.4
        elif self.corpus_splitting == 3:
            self.embed_dropout = 0.4
            
        self.cnn_dim = self.wordvec_dim

        if self.need_char or self.need_sub:
            self.cnn_dim += self.char_hid_dim
        if self.need_elmo:
            self.cnn_dim += self.elmo_dim

        if self.corpus_splitting == 1:
            self.cnn_layer_num = 5
        elif self.corpus_splitting == 2:
            self.cnn_layer_num = 4
        elif self.corpus_splitting == 3:
            self.cnn_layer_num = 5

        if self.corpus_splitting == 1:
            self.cnn_kernal_size = [5, 5, 5, 5, 5]
        elif self.corpus_splitting == 2:
            self.cnn_kernal_size = [5, 5, 5, 5]
        elif self.corpus_splitting == 3:
            self.cnn_kernal_size = [3, 3, 3, 3, 3]

        if self.corpus_splitting == 1:
            self.cnn_dropout = 0.4
        elif self.corpus_splitting == 2:
            self.cnn_dropout = 0.4
        elif self.corpus_splitting == 3:
            self.cnn_dropout = 0.4

        self.attned_dim = self.cnn_dim * self.attn_topk
        self.pair_rep_dim = self.attned_dim * 2 * self.cnn_layer_num

        ################################################################################
        # Classifier
        self.clf_class_num = self.four_or_eleven

        if self.corpus_splitting == 1:
            self.clf_fc_num = 0
            self.clf_fc_dim = 2048
        elif self.corpus_splitting == 2:
            self.clf_fc_num = 0
        elif self.corpus_splitting == 3:
            self.clf_fc_num = 0
        
        if self.corpus_splitting == 1:
            self.clf_dropout = 0.3
        elif self.corpus_splitting == 2:
            self.clf_dropout = 0.3
        elif self.corpus_splitting == 3:
            self.clf_dropout = 0.3
        
        if self.corpus_splitting == 1:
            self.conn_num = 94
        elif self.corpus_splitting == 2:
            self.conn_num = 92
        elif self.corpus_splitting == 3:
            self.conn_num = 93
        
        ################################################################################
        self.seed = 666
        self.batch_size = 128
        self.shuffle = True

        if self.corpus_splitting == 1:
            self.lr = 0.001
        elif self.corpus_splitting == 2:
            self.lr = 0.001
        elif self.corpus_splitting == 3:
            self.lr = 0.001
        
        self.l2_penalty = 0
        self.grad_clip = 1
        self.epochs = 10000

        self.is_mttrain = True
        self.lambda1 = 1

        self.logdir = './res/' + datetime.now().strftime('%B%d-%H:%M:%S')
