import time
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score

from data import Data
from model import ArgEncoder, Classifier

class ModelBuilder(object):
    def __init__(self, use_cuda, conf):
        self.cuda = use_cuda
        self.conf = conf
        self._pre_data()
        self._build_model()
    
    def _pre_data(self):
        print('pre data...')
        self.data = Data(self.cuda, self.conf)

    def _build_model(self):
        print('loading embedding...')
        if self.conf.corpus_splitting == 1:
            pre = './data/processed/lin/'
        elif self.conf.corpus_splitting == 2:
            pre = './data/processed/ji/'
        elif self.conf.corpus_splitting == 3:
            pre = './data/processed/l/'
        we = torch.load(pre+'we.pkl')
        char_table = None
        sub_table = None
        if self.conf.need_char or self.conf.need_elmo:
            char_table = torch.load(pre+'char_table.pkl')
        if self.conf.need_sub:
            sub_table = torch.load(pre+'sub_table.pkl')
        print('building model...')
        self.encoder = ArgEncoder(self.conf, we, char_table, sub_table, self.cuda)
        self.classifier = Classifier(self.conf.clf_class_num, self.conf)
        if self.conf.is_mttrain:
            self.conn_classifier = Classifier(self.conf.conn_num, self.conf)
        if self.cuda:
            self.encoder.cuda()
            self.classifier.cuda()
            if self.conf.is_mttrain:
                self.conn_classifier.cuda()
        self.criterion = torch.nn.CrossEntropyLoss()
        para_filter = lambda model: filter(lambda p: p.requires_grad, model.parameters())
        self.e_optimizer = torch.optim.Adagrad(para_filter(self.encoder), self.conf.lr, weight_decay=self.conf.l2_penalty)
        self.c_optimizer = torch.optim.Adagrad(para_filter(self.classifier), self.conf.lr, weight_decay=self.conf.l2_penalty)
        if self.conf.is_mttrain:
            self.con_optimizer = torch.optim.Adagrad(para_filter(self.conn_classifier), self.conf.lr, weight_decay=self.conf.l2_penalty)

    def _print_train(self, epoch, time, loss, acc):
        print('-' * 80)
        print(
            '| end of epoch {:3d} | time: {:5.2f}s | loss: {:10.5f} | acc: {:5.2f}% |'.format(
                epoch, time, loss, acc * 100
            )
        )
        print('-' * 80)

    def _print_eval(self, task, loss, acc, f1):
        print(
            '| ' + task + ' loss {:10.5f} | acc {:5.2f}% | f1 {:5.2f}%'.format(loss, acc * 100, f1*100)
        )
        print('-' * 80)

    def _save_model(self, model, filename):
        torch.save(model.state_dict(), './weights/' + filename)

    def _load_model(self, model, filename):
        model.load_state_dict(torch.load('./weights/' + filename))

    def _train_one(self):
        self.encoder.train()
        self.classifier.train()
        if self.conf.is_mttrain:
            self.conn_classifier.train()
        total_loss = 0
        correct_n = 0
        train_size = self.data.train_size
        for a1, a2, sense, conn in self.data.train_loader:
            if self.conf.four_or_eleven == 2:
                mask1 = (sense == self.conf.binclass)
                mask2 = (sense != self.conf.binclass)
                sense[mask1] = 1
                sense[mask2] = 0
            if self.cuda:
                a1, a2, sense, conn = a1.cuda(), a2.cuda(), sense.cuda(), conn.cuda()
            a1, a2, sense, conn = Variable(a1), Variable(a2), Variable(sense), Variable(conn)
            repr = self.encoder(a1, a2)
            output = self.classifier(repr)
            _, output_sense = torch.max(output, 1)
            assert output_sense.size() == sense.size()
            tmp = (output_sense == sense).long()
            correct_n += torch.sum(tmp).data
            loss = self.criterion(output, sense)

            if self.conf.is_mttrain:
                conn_output = self.conn_classifier(repr)
                loss2 = self.criterion(conn_output, conn)
                loss = loss + loss2 * self.conf.lambda1

            self.e_optimizer.zero_grad()
            self.c_optimizer.zero_grad()
            if self.conf.is_mttrain:
                self.con_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), self.conf.grad_clip)
            torch.nn.utils.clip_grad_norm(self.classifier.parameters(), self.conf.grad_clip)
            if self.conf.is_mttrain:
                torch.nn.utils.clip_grad_norm(self.conn_classifier.parameters(), self.conf.grad_clip)
            self.e_optimizer.step()
            self.c_optimizer.step()
            if self.conf.is_mttrain:
                self.con_optimizer.step()
            
            total_loss += loss.data * sense.size(0)
        return total_loss[0] / train_size, correct_n[0] / train_size

    def _train(self, pre):
        for epoch in range(self.conf.epochs):            
            start_time = time.time()
            loss, acc = self._train_one()
            self._print_train(epoch, time.time()-start_time, loss, acc)
            self.logwriter.add_scalar('loss/train_loss', loss, epoch)
            self.logwriter.add_scalar('acc/train_acc', acc*100, epoch)

            dev_loss, dev_acc, dev_f1 = self._eval('dev')
            self._print_eval('dev', dev_loss, dev_acc, dev_f1)
            self.logwriter.add_scalar('loss/dev_loss', dev_loss, epoch)
            self.logwriter.add_scalar('acc/dev_acc', dev_acc*100, epoch)
            self.logwriter.add_scalar('f1/dev_f1', dev_f1*100, epoch)

            test_loss, test_acc, test_f1 = self._eval('test')
            self._print_eval('test', test_loss, test_acc, test_f1)
            self.logwriter.add_scalar('loss/test_loss', test_loss, epoch)
            self.logwriter.add_scalar('acc/test_acc', test_acc*100, epoch)
            self.logwriter.add_scalar('f1/test_f1', test_f1*100, epoch)

    def train(self, pre):
        print('start training')
        self.logwriter = SummaryWriter(self.conf.logdir)
        self._train(pre)
        print('training done')

    def _eval(self, task):
        self.encoder.eval()
        self.classifier.eval()
        total_loss = 0
        correct_n = 0
        if task == 'dev':
            data = self.data.dev_loader
            n = self.data.dev_size
        elif task == 'test':
            data = self.data.test_loader
            n = self.data.test_size
        else:
            raise Exception('wrong eval task')
        output_list = []
        gold_list = []
        for a1, a2, sense1, sense2 in data:
            if self.conf.four_or_eleven == 2:
                mask1 = (sense1 == self.conf.binclass)
                mask2 = (sense1 != self.conf.binclass)
                sense1[mask1] = 1
                sense1[mask2] = 0
                mask0 = (sense2 == -1)
                mask1 = (sense2 == self.conf.binclass)
                mask2 = (sense2 != self.conf.binclass)
                sense2[mask1] = 1
                sense2[mask2] = 0  
                sense2[mask0] = -1              
            if self.cuda:
                a1, a2, sense1, sense2 = a1.cuda(), a2.cuda(), sense1.cuda(), sense2.cuda()
            a1 = Variable(a1, volatile=True)
            a2 = Variable(a2, volatile=True)
            sense1 = Variable(sense1, volatile=True)
            sense2 = Variable(sense2, volatile=True)

            output = self.classifier(self.encoder(a1, a2))
            _, output_sense = torch.max(output, 1)
            assert output_sense.size() == sense1.size()
            gold_sense = sense1
            mask = (output_sense == sense2)
            gold_sense[mask] = sense2[mask]
            tmp = (output_sense == gold_sense).long()
            correct_n += torch.sum(tmp).data

            output_list.append(output_sense)
            gold_list.append(gold_sense)

            loss = self.criterion(output, gold_sense)
            total_loss += loss.data * gold_sense.size(0)
        
        output_s = torch.cat(output_list)
        gold_s = torch.cat(gold_list)
        if self.conf.four_or_eleven == 2:
            f1 = f1_score(gold_s.cpu().data.numpy(), output_s.cpu().data.numpy(), average='binary')
        else:
            f1 = f1_score(gold_s.cpu().data.numpy(), output_s.cpu().data.numpy(), average='macro')
        return total_loss[0] / n, correct_n[0] / n, f1

    def eval(self, pre):
        print('evaluating...')
        self._load_model(self.encoder, pre+'_eparams.pkl')
        self._load_model(self.classifier, pre+'_cparams.pkl')
        test_loss, test_acc, f1 = self._eval('test')
        self._print_eval('test', test_loss, test_acc, f1)
