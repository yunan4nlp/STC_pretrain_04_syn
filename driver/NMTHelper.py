import torch.nn.functional as F
from torch.autograd import Variable
from data.DataLoader import *
from module.Utils import *
from data.Vocab import NMTVocab


class NMTHelper(object):
    def __init__(self, model, parser, critic, src_vocab, tgt_vocab, dep_vocab, config, parser_config):
        self.model = model
        self.critic = critic
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.config = config

        self.parser = parser
        self.dep_vocab = dep_vocab
        self.parser_config = parser_config

    def dep_input_id(self, src_input):
        lower_src_input = [cur_word.lower() for cur_word in src_input]
        word_ids = [self.dep_vocab.ROOT] + self.dep_vocab.word2id(lower_src_input)
        extword_ids = [self.dep_vocab.ROOT] + self.dep_vocab.extword2id(lower_src_input)
        return word_ids, extword_ids

    def prepare_training_data(self, src_inputs, tgt_inputs):
        self.train_data = []
        #for idx in range(self.config.max_train_length):
        self.train_data.append([])
        for src_input, tgt_input in zip(src_inputs, tgt_inputs):
            word_ids, extword_ids = self.dep_input_id(src_input)

            #idx = int(len(src_input) - 1)
            self.train_data[0].append((self.src_data_id(src_input), self.tgt_data_id(tgt_input),
                                       self.ext_src_data_id(src_input), self.ext_tgt_data_id(tgt_input),
                                       word_ids, extword_ids))
        self.train_size = len(src_inputs)
        self.batch_size = self.config.train_batch_size
        batch_num = 0
        #for idx in range(self.config.max_train_length):
        train_size = len(self.train_data[0])
        batch_num += int(np.ceil(train_size / float(self.batch_size)))
        self.batch_num = batch_num

    def prepare_training_data_backup(self, src_inputs, tgt_inputs):
        self.train_data = []
        for idx in range(self.config.max_train_length):
            self.train_data.append([])
        for src_input, tgt_input in zip(src_inputs, tgt_inputs):
            idx = int(len(src_input) - 1)
            self.train_data[idx].append((self.src_data_id(src_input), self.tgt_data_id(tgt_input)))
        self.train_size = len(src_inputs)
        self.batch_size = self.config.train_batch_size
        batch_num = 0
        for idx in range(self.config.max_train_length):
            train_size = len(self.train_data[idx])
            batch_num += int(np.ceil(train_size / float(self.batch_size)))
        self.batch_num = batch_num

    def prepare_valid_data(self, src_inputs, tgt_inputs):
        self.valid_data = []
        for src_input, tgt_input in zip(src_inputs, tgt_inputs):
            self.valid_data.append((self.src_data_id(src_input), self.tgt_data_id(tgt_input)))
        self.valid_size = len(self.valid_data)

    def prepare_data(self, src_inputs, tgt_inputs):
        data = []
        for src_input, tgt_input in zip(src_inputs, tgt_inputs):
            data.append((self.src_data_id(src_input), self.tgt_data_id(tgt_input),
                         self.ext_src_data_id(src_input), self.ext_tgt_data_id(tgt_input)))
        return data

    def src_data_id(self, src_input):
        result = self.src_vocab.word2id(src_input)
        return result + [self.src_vocab.EOS]

    def tgt_data_id(self, tgt_input):
        result = self.tgt_vocab.word2id(tgt_input)
        return [self.tgt_vocab.BOS] + result + [self.tgt_vocab.EOS]

    def ext_src_data_id(self, src_input):
        result = self.src_vocab.extword2id(src_input)
        return result + [self.src_vocab.EOS]

    def ext_tgt_data_id(self, tgt_input):
        result = self.tgt_vocab.extword2id(tgt_input)
        return [self.tgt_vocab.BOS] + result + [self.tgt_vocab.EOS]

    def prepare_eval_data(self, src_inputs):
        eval_data = []
        for src_input in src_inputs:
            eval_data.append((self.src_data_id(src_input), src_input))

        return eval_data

    def pair_data_variable(self, batch):
        batch_size = len(batch)

        src_lengths = [len(batch[i][0]) for i in range(batch_size)]
        max_src_length = int(np.max(src_lengths))

        tgt_lengths = [len(batch[i][1]) for i in range(batch_size)]
        max_tgt_length = int(np.max(tgt_lengths))

        src_words = Variable(torch.LongTensor(batch_size, max_src_length).fill_(NMTVocab.PAD), requires_grad=False)
        tgt_words = Variable(torch.LongTensor(batch_size, max_tgt_length).fill_(NMTVocab.PAD), requires_grad=False)

        ext_src_words = Variable(torch.LongTensor(batch_size, max_src_length).fill_(NMTVocab.PAD), requires_grad=False)
        ext_tgt_words = Variable(torch.LongTensor(batch_size, max_tgt_length).fill_(NMTVocab.PAD), requires_grad=False)

        dep_words = Variable(torch.LongTensor(batch_size, max_src_length).zero_(), requires_grad=False)
        dep_extwords = Variable(torch.LongTensor(batch_size, max_src_length).zero_(), requires_grad=False)
        dep_masks = Variable(torch.Tensor(batch_size, max_src_length).zero_(), requires_grad=False)

        for b, instance in enumerate(batch):
            assert len(instance[0]) == len(instance[2])
            assert len(instance[1]) == len(instance[3])

            assert len(instance[0]) == len(instance[4])
            assert len(instance[0]) == len(instance[5])

            for index, word in enumerate(instance[0]):
                src_words[b, index] = word
            for index, word in enumerate(instance[1]):
                tgt_words[b, index] = word
            for index, word in enumerate(instance[2]):
                ext_src_words[b, index] = word
            for index, word in enumerate(instance[3]):
                ext_tgt_words[b, index] = word

            for index, word in enumerate(instance[4]):
                dep_words[b, index] = word
                dep_masks[b, index] = 1
            for index, word in enumerate(instance[5]):
                dep_extwords[b, index] = word

            b += 1

        if self.use_cuda:
            src_words = src_words.cuda(self.device)
            tgt_words = tgt_words.cuda(self.device)
            ext_src_words = ext_src_words.cuda(self.device)
            ext_tgt_words = ext_tgt_words.cuda(self.device)

            dep_words = dep_words.cuda(self.device)
            dep_extwords = dep_extwords.cuda(self.device)
            dep_masks = dep_masks.cuda(self.device)

        return src_words, tgt_words, ext_src_words, ext_tgt_words, src_lengths, tgt_lengths, dep_words, dep_extwords, dep_masks

    def source_data_variable(self, batch):
        batch_size = len(batch)
        src_lengths = [len(batch[i][0]) for i in range(batch_size)]
        max_src_length = int(src_lengths[0])

        src_words = Variable(torch.LongTensor(batch_size, max_src_length).fill_(NMTVocab.PAD), requires_grad=False)
        for b, instance in enumerate(batch):
            for index, word in enumerate(instance[0]):
                src_words[b, index] = word
            b += 1

        if self.use_cuda:
            src_words = src_words.cuda(self.device)
        return src_words, src_lengths

    def parse_one_batch(self, dep_words, dep_extwords, dep_masks, bTrain):
        if bTrain and self.config.parser_tune == 1:
            self.parser.train()
        else:
            self.parser.eval()
        parser_outputs = self.parser.lstm_hidden(dep_words, dep_extwords, dep_masks)
        chunks = torch.split(parser_outputs, 1, dim=0)
        parser_outputs = torch.cat(chunks[1:] + chunks[0:1], 0)
        return parser_outputs.transpose(0, 1)

    def compute_forward(self, seqs_x, seqs_y, ext_seqs_x, ext_seqs_y, dep_words, dep_extwords, dep_masks, xlengths, normalization=1.0):
        """
        :type model: Transformer

        :type critic: NMTCritierion
        """
        synx = self.parse_one_batch(dep_words, dep_extwords, dep_masks, True)

        y_inp = seqs_y[:, :-1].contiguous()
        ext_y_inp = ext_seqs_y[:, :-1].contiguous()
        y_label = seqs_y[:, 1:].contiguous()

        dec_outs = self.model(seqs_x, synx, ext_seqs_x, y_inp, ext_y_inp, lengths=xlengths)

        loss = self.critic(generator=self.model.generator,
                      normalization=normalization,
                      dec_outs=dec_outs,
                      labels=y_label)

        mask = y_label.data.ne(NMTVocab.PAD)
        pred = self.model.generator(dec_outs).data.max(2)[1]  # [batch_size, seq_len]
        num_correct = y_label.data.eq(pred).float().masked_select(mask).sum() / normalization
        num_total = mask.sum().type(torch.FloatTensor)

        stats = Statistics(loss.data[0], num_total, num_correct)

        return loss, stats


    def train_one_batch(self, batch):
        self.model.train()
        self.model.zero_grad()
        src_words, tgt_words, ext_src_words, ext_tgt_words, src_lengths, tgt_lengths, dep_words, dep_extwords, dep_masks = self.pair_data_variable(batch)
        loss, stat = self.compute_forward(src_words, tgt_words, ext_src_words, ext_tgt_words,  dep_words, dep_extwords, dep_masks, src_lengths)
        loss = loss / self.config.update_every
        loss.backward()

        return stat

    def valid(self, data):
        valid_stat = Statistics()
        self.model.eval()
        for batch in create_batch_iter(data, self.config.test_batch_size):
            src_words, tgt_words, ext_src_words, ext_tgt_words, src_lengths, tgt_lengths, dep_words, dep_extwords, dep_masks  = self.pair_data_variable(batch)
            loss, stat = self.compute_forward(src_words, tgt_words, ext_src_words, ext_tgt_words, dep_words, dep_extwords, dep_masks, src_lengths)
            valid_stat.update(stat)
        return valid_stat

    def translate(self, eval_data):
        self.model.eval()
        result = {}
        for batch in create_batch_iter(eval_data, self.config.test_batch_size):
            batch_size = len(batch)
            src_words, src_lengths = self.source_data_variable(batch)
            allHyp = self.translate_batch(src_words, src_lengths)
            all_hyp_inds = [beam_result[0] for beam_result in allHyp]
            for idx in range(batch_size):
                if all_hyp_inds[idx][-1] == self.tgt_vocab.EOS:
                    all_hyp_inds[idx].pop()
            all_hyp_words = [self.tgt_vocab.id2word(idxs) for idxs in all_hyp_inds]
            for idx, instance in enumerate(batch):
                result['\t'.join(instance[1])] = all_hyp_words[idx]

        return result


    def translate_batch(self, src_inputs, src_input_lengths):
        word_ids = self.model(src_inputs, lengths=src_input_lengths, mode="infer", beam_size=self.config.beam_size)
        word_ids = word_ids.cpu().numpy().tolist()

        result = []
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if wid != NMTVocab.PAD] for line in sent_t]
            result.append(sent_t)

        return result
