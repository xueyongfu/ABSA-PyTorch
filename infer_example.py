import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn.functional as F

from data_utils import build_tokenizer, build_embedding_matrix
from models import IAN, MemNet, ATAE_LSTM, AOA, LSTM, TD_LSTM


class Inferer:
    """A simple inference example"""
    def __init__(self, opt):
        self.opt = opt
        self.tokenizer = build_tokenizer(
            fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
            max_seq_len=opt.max_seq_len,
            dat_fname='{}/{}_tokenizer.dat'.format(opt.output_path, opt.dataset))

        embedding_matrix = build_embedding_matrix(
            word2idx=self.tokenizer.word2idx,
            embed_dim=opt.embed_dim,
            dat_fname='{}/{}_{}_embedding_matrix.dat'.format(opt.output_path, str(opt.embed_dim), opt.dataset),
            w2v_file=opt.w2v_file)
        self.model = opt.model_class(embedding_matrix, opt)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model.to(opt.device)
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, raw_texts):
        context_seqs = [self.tokenizer.text_to_sequence(raw_text.lower().strip()) for raw_text in raw_texts]
        aspect_seqs = [self.tokenizer.text_to_sequence('null')] * len(raw_texts)
        context_indices = torch.tensor(context_seqs, dtype=torch.int64).to(self.opt.device)
        aspect_indices = torch.tensor(aspect_seqs, dtype=torch.int64).to(self.opt.device)

        t_inputs = [context_indices, aspect_indices]
        t_outputs = self.model(t_inputs)

        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()
        return t_probs


if __name__ == '__main__':
    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'aoa': AOA,
    }
    # set your trained models here
    model_state_dict_paths = {
        'lstm':'state_dict/lstm_mams_val_acc0.651',
        'atae_lstm': 'state_dict/',
        'ian': 'state_dict/',
        'memnet': 'state_dict/',
        'aoa': 'state_dict/',
    }

    # 配置inference模型参数
    class Option(object): pass
    opt = Option()
    opt.model_name = 'lstm'
    opt.model_class = model_classes[opt.model_name]
    opt.w2v_file = '/root/models/english/Glove/glove.840B.300d.txt'
    opt.output_path = 'output'
    opt.dataset = 'mams'
    opt.dataset_file = {
            'train': 'datasets/MAMS/train_data/train.txt',
            'test': 'datasets/MAMS/train_data/dev.txt'
        }

    opt.state_dict_path = model_state_dict_paths[opt.model_name]
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.max_seq_len = 120
    opt.polarities_dim = 3
    opt.hops = 3
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inf = Inferer(opt)
    t_probs = inf.evaluate(['From the speed to the multi touch gestures this operating system beats Windows easily .'])
    print(t_probs.argmax(axis=-1) - 1)
