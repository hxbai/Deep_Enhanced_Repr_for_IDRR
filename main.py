import argparse
import torch

from config import Config
from builder import ModelBuilder

parser = argparse.ArgumentParser()
parser.add_argument('func', type=str)
parser.add_argument('splitting', type=int)

def main(conf, is_train=True, pre=None):
    havecuda = torch.cuda.is_available()
    torch.manual_seed(conf.seed)
    if havecuda:
        torch.cuda.manual_seed(conf.seed)

    model = ModelBuilder(havecuda, conf)
    if is_train:
        model.train(pre)
    else:
        model.eval(pre)
    
if __name__ == '__main__':
    A = parser.parse_args()
    if A.splitting == 1:
        conf = Config(11, 1)
        if A.func == 'train':
            main(conf, True, 'lin')
        elif A.func == 'eval':
            main(conf, False, 'lin')
        else:
            raise Exception('wrong func')
    if A.splitting == 2:
        conf = Config(11, 2)
        if A.func == 'train':
            main(conf, True, 'ji')
        elif A.func == 'eval':
            main(conf, False, 'ji')
        else:
            raise Exception('wrong func')
    if A.splitting == 3:
        conf = Config(4, 3)
        if A.func == 'train':
            main(conf, True, 'four')
        elif A.func == 'eval':
            main(conf, False, 'four')
        else:
            raise Exception('wrong func')