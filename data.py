import torch
import torch.utils.data as torchdata

class Dataset(torchdata.Dataset):
    def __init__(self, data_path):
        super(Dataset, self).__init__()
        self.d1, self.d2, self.d3, self.d4 = torch.load(data_path)

    def __getitem__(self, index):
        return self.d1[index], self.d2[index], self.d3[index], self.d4[index]

    def __len__(self):
        return len(self.d4)

class Data(object):
    def __init__(self, use_cuda, conf, batch_size=None):
        if batch_size is None:
            batch_size = conf.batch_size
        kwargs = {'batch_size':batch_size, 'shuffle':conf.shuffle, 'drop_last':False}
        if use_cuda:
            kwargs['pin_memory'] = True
        if conf.corpus_splitting == 1:
            pre = './data/processed/lin/'
        elif conf.corpus_splitting == 2:
            pre = './data/processed/ji/'
        elif conf.corpus_splitting == 3:
            pre = './data/processed/l/'
        train_data = Dataset(pre+'train.pkl')
        dev_data = Dataset(pre+'dev.pkl')
        test_data = Dataset(pre+'test.pkl')
        self.train_size = len(train_data)
        self.dev_size = len(dev_data)
        self.test_size = len(test_data)
        self.train_loader = torchdata.DataLoader(train_data, **kwargs)
        self.dev_loader = torchdata.DataLoader(dev_data, **kwargs)
        self.test_loader = torchdata.DataLoader(test_data, **kwargs)

def test():
    from config import Config
    data = Data(False, Config())
    print(data.train_size)
    print(data.dev_size)
    print(data.test_size)
    loaders = [data.train_loader, data.dev_loader, data.test_loader]
    for loader in loaders:
        res = {}
        for d in loader:
            l = []
            for i in d:
                l.append(i.size())
            if tuple(l) not in res:
                res[tuple(l)] = 1
            else:
                res[tuple(l)] += 1
        for i in res:
            print(i, '*', res[i])
        print('-' * 100)

if __name__ == '__main__':
    test()