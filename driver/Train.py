# -*- coding: utf-8 -*-
import sys
sys.path.extend(["../","./"])
import argparse
import random
from driver.Config import *
from driver.Optim import *
from driver.NMTHelper import *
from model.DL4MT import DL4MT
from model.Transformer import Transformer
from module.Criterions import *
import pickle
import os
import re
from biaffine.load_parser import load_parser

def train(nmt, train_srcs, train_tgts, config):
    global_step = 0
    nmt.prepare_training_data(train_srcs, train_tgts)
    valid_files = config.dev_files.strip().split(' ')
    valid_srcs = read_corpus(valid_files[0])
    valid_tgts = read_corpus(valid_files[1])
    valid_data = nmt.prepare_data(valid_srcs, valid_tgts)
    eval_valid_data = nmt.prepare_eval_data(valid_srcs)

    test_files = config.test_files.strip().split(' ')
    test_srcs = read_corpus(test_files[0])
    test_tgts = read_corpus(test_files[1])
    test_data = nmt.prepare_data(test_srcs, test_tgts)
    eval_test_data = nmt.prepare_eval_data(test_srcs)

    optim = Optimizer(name=config.learning_algorithm,
                      model=nmt.model,
                      lr=config.learning_rate,
                      grad_clip=config.clip
                      )


    print('start training...')
    best_ppl = 1e30
    best_step = -1
    bad_step = 0
    new_lr = config.learning_rate
    eval_time = int(0)

    for iter in range(config.train_iters):
        # dynamic adjust lr
        total_stats = Statistics()
        batch_num = nmt.batch_num
        batch_iter = 0
        for batch in create_train_batch_iter(nmt.train_data, nmt.batch_size, shuffle=True):
            stat = nmt.train_one_batch(batch)
            total_stats.update(stat)
            batch_iter += 1
            total_stats.print_out(global_step, iter, batch_iter, batch_num)

            model_updated = False
            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                optim.step()
                nmt.model.zero_grad()
                global_step += 1
                model_updated = True

            if model_updated and global_step > config.eval_start and global_step % config.validate_every == 0:
                dev_start_time = time.time()
                dev_state = nmt.valid(valid_data)
                dev_ppl = dev_state.ppl()
                during_time = float(time.time() - dev_start_time)
                print("step %d, epoch %d: dev ppl: %.2f, time %.2f" \
                      % (global_step, iter, dev_ppl, during_time))
                #predict(nmt, eval_valid_data, valid_files, global_step)


                test_start_time = time.time()
                test_state = nmt.valid(test_data)
                test_ppl = test_state.ppl()
                during_time = float(time.time() - test_start_time)
                print("step %d, epoch %d: dev ppl: %.2f, time %.2f" \
                      % (global_step, iter, test_ppl, during_time))
                #predict(nmt, eval_test_data, test_files, global_step)

                if dev_ppl < best_ppl:
                    lrs = [lr for lr in optim.get_lrate()]
                    print("Exceed best ppl: history = %.2f, current = %.2f, lr_ratio = %.6f" % \
                          (best_ppl, dev_ppl, lrs[0]))
                    best_ppl = dev_ppl
                    bad_step = 0
                    if global_step > config.save_after:
                        torch.save(nmt.model.state_dict(), config.save_model_path + '.' + str(global_step))
                        best_step = global_step
                elif eval_time >= config.start_decay_at:
                    bad_step += 1
                    if bad_step >= config.max_patience and new_lr > config.min_lrate:
                        bad_step = 0
                        model_path = config.load_model_path + '.' + str(best_step)
                        nmt_model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
                        new_lr = max(new_lr * config.decay_scale, config.min_lrate)
                        optim.set_lrate(new_lr)
                        lrs = [lr for lr in optim.get_lrate()]
                        print("Decaying the learning ratio to %.6f" % (lrs[0]))
                    else:
                        lrs = [lr for lr in optim.get_lrate()]
                        print("Current the learning ratio is %.6f, bad_step = %d" % (lrs[0], bad_step))
                else:
                    bad_step += 1
                    lrs = [lr for lr in optim.get_lrate()]
                    print("Current the learning ratio is %.6f, bad_step = %d" % (lrs[0], bad_step))

                eval_time += 1

def predict(nmt, eval_data, eval_files, global_step):
    result = nmt.translate(eval_data)

    outputFile = eval_files[0] + '.' + str(global_step)
    output = open(outputFile, 'w', encoding='utf-8')
    ordered_result = []
    for idx, instance in enumerate(eval_data):
        src_key = '\t'.join(instance[1])
        cur_result = result.get(src_key)
        if cur_result is not None:
            ordered_result.append(cur_result)
        else:
            print("Strange, miss one sentence")
            ordered_result.append([''])

        sentence_out = ' '.join(ordered_result[idx])
        sentence_out = sentence_out.replace(' <unk>', '')
        sentence_out = sentence_out.replace('@@ ', '')

        output.write(sentence_out + '\n')

    output.close()


def evaluate(nmt, eval_files, config, global_step):
    valid_srcs = read_corpus(eval_files[0])
    eval_data = nmt.prepare_eval_data(valid_srcs)
    result = nmt.translate(eval_data)

    outputFile = eval_files[0] + '.' + str(global_step)
    output = open(outputFile, 'w', encoding='utf-8')
    ordered_result = []
    for idx, instance in enumerate(eval_data):
        src_key = '\t'.join(instance[1])
        cur_result = result.get(src_key)
        if cur_result is not None:
            ordered_result.append(cur_result)
        else:
            print("Strange, miss one sentence")
            ordered_result.append([''])

        sentence_out = ' '.join(ordered_result[idx])
        sentence_out = sentence_out.replace(' <unk>', '')
        sentence_out = sentence_out.replace('@@ ', '')

        output.write(sentence_out + '\n')

    output.close()

    command = 'perl %s %s < %s' % (config.bleu_script, ' '.join(eval_files[1:]), outputFile)
    bleu_exec = os.popen(command)
    bleu_exec = bleu_exec.read()
    # Get bleu value
    bleu_val = re.findall('BLEU = (.*?),', bleu_exec, re.S)[0]
    bleu_val = float(bleu_val)

    return bleu_val

if __name__ == '__main__':
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    random.seed(666)
    np.random.seed(666)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)


    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='default.cfg')
    argparser.add_argument('--parser_config_file', default='parser.cfg')
    argparser.add_argument('--tgt_word_file', default=None)
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)
    argparser.add_argument('--model_id', type=str, default='8000')
    argparser.add_argument('--use-pretrain', action='store_true', default=False)

    args, extra_args = argparser.parse_known_args()

    parser_model, parser_vocab, parser_config = load_parser(args, extra_args) ## load biaffine parser, vocab, config

    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)


    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = False
    print("\nGPU using status: ", config.use_cuda)

    train_files = config.train_files.strip().split(' ')
    train_srcs, train_tgts = read_training_corpus(train_files[0], train_files[1], \
                                                  config.min_src_length, config.min_tgt_length,
                                                  config.max_src_length, config.max_tgt_length)
    if not args.use_pretrain:
        src_vocab, tgt_vocab = creat_vocabularies(train_srcs, train_tgts, config.src_vocab_size, config.tgt_vocab_size)

        ext_src_emb = src_vocab.load_pretrained_embs(config.src_emb)
        ext_tgt_emb = tgt_vocab.load_pretrained_embs(config.tgt_emb)

        if args.tgt_word_file is not None:
            tgt_words = read_tgt_words(args.tgt_word_file)
            tgt_vocab = NMTVocab(tgt_words)
            ext_tgt_emb = tgt_vocab.load_pretrained_embs(config.tgt_emb)

        pickle.dump(src_vocab, open(config.save_src_vocab_path, 'wb'))
        pickle.dump(tgt_vocab, open(config.save_tgt_vocab_path, 'wb'))

        nmt_model = eval(config.model_name)(config, parser_config, src_vocab, tgt_vocab,
                                            ext_src_emb, ext_tgt_emb, config.use_cuda)

    else:
        model_path = config.load_model_path + '.' + args.model_id
        print("####LOAD pretrain model from: " + model_path + "####")
        src_vocab = pickle.load(open(config.save_src_vocab_path, 'rb+'))
        tgt_vocab = pickle.load(open(config.save_tgt_vocab_path, 'rb+'))

        ext_src_emb = src_vocab.load_pretrained_embs(config.src_emb)
        ext_tgt_emb = tgt_vocab.load_pretrained_embs(config.tgt_emb)
        nmt_model = eval(config.model_name)(config, parser_config, src_vocab.size, tgt_vocab.size,
                                            ext_src_emb, ext_tgt_emb, config.use_cuda)

        nmt_model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    print("Sentence Number: #train = %d" %(len(train_srcs)))

    # model
    critic = NMTCritierion(label_smoothing=config.label_smoothing)


    if config.use_cuda:
        #torch.backends.cudnn.enabled = False
        parser_model = parser_model.cuda()
        nmt_model = nmt_model.cuda()
        critic = critic.cuda()

    nmt = NMTHelper(nmt_model, parser_model, critic, src_vocab, tgt_vocab,  parser_vocab, config, parser_config)


    train(nmt, train_srcs, train_tgts, config)

