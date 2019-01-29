import pickle
from biaffine.driver.Config import Configurable
from biaffine.driver.Model import ParserModel
from biaffine.driver.Parser import BiaffineParser
import sys
import os
import torch


def load_parser(args, extra_args):
    from biaffine.data import Vocab
    sys.modules['data.Vocab'] = Vocab ## for pickle load model
    os.chdir("biaffine")
    print('#####Loading Biaffine Parser######')
    config = Configurable(args.parser_config_file, extra_args)
    vocab = pickle.load(open(config.load_vocab_path, 'rb'))
    vec = vocab.create_pretrained_embs(config.pretrained_embeddings_file)
    model = ParserModel(vocab, config, vec)
    model.load_state_dict(torch.load(config.load_model_path))
    print('#####Biaffine Parser Loading OK######')
    os.chdir("..")
    from data import Vocab
    sys.modules['data.Vocab'] = Vocab ## for pickle load model
    return model, vocab, config

