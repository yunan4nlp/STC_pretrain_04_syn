import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from module.Basic import BottleLinear as Linear
from module.Sublayers import LayerNorm, PositionwiseFeedForward, MultiHeadedAttention
from module.Embeddings import Embeddings
from module.Utils import *
from data.Vocab import NMTVocab


def get_attn_causal_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.

    :param seq: Input sequence.
        with shape [batch_size, time_steps, dim]
    '''
    assert seq.dim() == 3
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask

class EncoderBlock(nn.Module):

    def __init__(self, d_model, d_inner_hid, n_head, dropout=0.1):
        super(EncoderBlock, self).__init__()

        self.layer_norm = LayerNorm(features=d_model)

        self.slf_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(size=d_model, hidden_size=d_inner_hid, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, slf_attn_mask=None):

        input_norm = self.layer_norm(enc_input)
        context, _, _ = self.slf_attn(input_norm, input_norm, input_norm, slf_attn_mask)
        out = self.dropout(context) + enc_input

        return self.pos_ffn(out)

class Encoder(nn.Module):

    def __init__(
            self, src_vocab, ext_src_emb, syn_hidden, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):

        super().__init__()
        self.extword_embed = nn.Embedding(src_vocab.extvocab_size, d_word_vec, padding_idx=0)
        self.extword_embed.weight.data.copy_(torch.from_numpy(ext_src_emb))
        self.extword_embed.weight.requires_grad = False

        self.syn_linear = nn.Linear(in_features=syn_hidden, out_features=d_word_vec, bias=True)

        #self.emb_linear = nn.Linear(100, d_word_vec, True)
        #nn.init.xavier_uniform_(self.emb_linear.weight)

        self.num_layers = n_layers
        self.embeddings = Embeddings(num_embeddings=src_vocab.vocab_size,
                                     embedding_dim=d_word_vec,
                                     dropout=dropout,
                                     add_position_embedding=True
                                     )
        self.block_stack = nn.ModuleList(
            [EncoderBlock(d_model=d_model, d_inner_hid=d_inner_hid, n_head=n_head, dropout=dropout)
             for _ in range(n_layers)])

        self.layer_norm = LayerNorm(d_model)

    def forward(self, src_seq, synx, ext_src_seq):
        # Word embedding look up
        assert src_seq.size() == ext_src_seq.size()
        batch_size, src_len = src_seq.size()

        synx = self.syn_linear(synx)

        ext_emb = self.extword_embed(ext_src_seq)

        #ext_emb = self.emb_linear(ext_emb)

        emb = self.embeddings(src_seq)

        emb = emb + ext_emb + synx

        enc_mask = src_seq.data.eq(NMTVocab.PAD)
        enc_slf_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, src_len, src_len)

        out = emb

        for i in range(self.num_layers):
            out = self.block_stack[i](out, enc_slf_attn_mask)

        out = self.layer_norm(out)

        return out, enc_mask

class DecoderBlock(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner_hid, n_head, dropout=0.1):
        super(DecoderBlock, self).__init__()

        self.slf_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout)
        self.ctx_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(size=d_model, hidden_size=d_inner_hid)

        self.layer_norm_1 = LayerNorm(d_model)
        self.layer_norm_2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def compute_cache(self, enc_output):

        return self.ctx_attn.compute_cache(enc_output, enc_output)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None,
                enc_attn_cache=None, self_attn_cache=None):
        # Args Checks
        input_batch, input_len, _ = dec_input.size()

        contxt_batch, contxt_len, _ = enc_output.size()

        input_norm = self.layer_norm_1(dec_input)
        all_input = input_norm

        query, _, self_attn_cache = self.slf_attn(all_input, all_input, input_norm,
                                     mask=slf_attn_mask, self_attn_cache=self_attn_cache)

        query = self.dropout(query) + dec_input

        query_norm = self.layer_norm_2(query)
        mid, attn, enc_attn_cache = self.ctx_attn(enc_output, enc_output, query_norm,
                                      mask=dec_enc_attn_mask, enc_attn_cache=enc_attn_cache)

        output = self.pos_ffn(self.dropout(mid) + query)

        return output, attn, self_attn_cache, enc_attn_cache

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(
            self, tgt_vocab, ext_tgt_emb, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):

        super(Decoder, self).__init__()

        self.extword_embed = nn.Embedding(tgt_vocab.extvocab_size, d_word_vec, padding_idx=0)
        self.extword_embed.weight.data.copy_(torch.from_numpy(ext_tgt_emb))
        self.extword_embed.weight.requires_grad = False


        self.n_head = n_head
        self.num_layers = n_layers
        self.d_model = d_model

        self.embeddings = Embeddings(tgt_vocab.vocab_size, d_word_vec,
                                       dropout=dropout, add_position_embedding=True)

        self.block_stack = nn.ModuleList([
            DecoderBlock(d_model=d_model, d_inner_hid=d_inner_hid, n_head=n_head, dropout=dropout)
            for _ in range(n_layers)])

        self.out_layer_norm = LayerNorm(d_model)

    @property
    def dim_per_head(self):
        return self.d_model // self.n_head

    def forward(self, tgt_seq, ext_tgt_seq, enc_output, enc_mask, enc_attn_caches=None, self_attn_caches=None):

        assert tgt_seq.size() == ext_tgt_seq.size()
        batch_size, tgt_len = tgt_seq.size()

        query_len = tgt_len
        key_len = tgt_len

        src_len = enc_output.size(1)

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt_seq)

        ext_emb = self.extword_embed(ext_tgt_seq)

        if self_attn_caches is not None:
            emb = emb[:,-1:].contiguous()
            query_len = 1

        # Decode mask
        dec_slf_attn_pad_mask = tgt_seq.data.eq(NMTVocab.PAD).unsqueeze(1).expand(batch_size, query_len, key_len)
        dec_slf_attn_sub_mask = get_attn_causal_mask(emb)

        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)
        dec_enc_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, query_len, src_len)

        output = emb
        new_self_attn_caches = []
        new_enc_attn_caches = []
        for i in range(self.num_layers):

            output, attn, self_attn_cache, enc_attn_cache \
                = self.block_stack[i](output,
                                      enc_output,
                                      dec_slf_attn_mask,
                                      dec_enc_attn_mask,
                                      enc_attn_cache=enc_attn_caches[i] if enc_attn_caches is not None else None,
                                      self_attn_cache=self_attn_caches[i] if self_attn_caches is not None else None)

            new_self_attn_caches = new_self_attn_caches + [self_attn_cache]
            new_enc_attn_caches = new_enc_attn_caches + [enc_attn_cache]

        output = self.out_layer_norm(output)

        return output, new_self_attn_caches, new_enc_attn_caches

class Generator(nn.Module):

    def __init__(self, n_words, hidden_size, shared_weight=None, padding_idx=-1):

        super(Generator, self).__init__()

        self.n_words = n_words
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

        self.proj = Linear(self.hidden_size, self.n_words, bias=False)
        self.actn = nn.LogSoftmax(dim=-1)

        if shared_weight is not None:
            self.proj.linear.weight = shared_weight


    def forward(self, input):
        """
        input == > Linear == > LogSoftmax
        """
        return self.actn(self.proj(input))

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, config, parser_config, src_vocab, tgt_vocab, ext_src_emb, ext_tgt_emb, use_gpu=True):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab, ext_src_emb, syn_hidden=parser_config.lstm_hiddens * 2, n_layers=config.num_layers, n_head=config.num_heads,
            d_word_vec=config.embed_size, d_model=config.embed_size,
            d_inner_hid=config.attention_size, dropout=config.dropout_hidden)

        self.decoder = Decoder(
            tgt_vocab, ext_tgt_emb, n_layers=config.num_layers, n_head=config.num_heads,
            d_word_vec=config.embed_size, d_model=config.embed_size,
            d_inner_hid=config.attention_size, dropout=config.dropout_hidden)

        self.dropout = nn.Dropout(config.dropout_hidden)


        if config.proj_share_weight:
            self.generator = Generator(n_words=tgt_vocab.vocab_size,
                                       hidden_size=config.embed_size,
                                       shared_weight=self.decoder.embeddings.embeddings.weight,
                                       padding_idx=NMTVocab.PAD)

        else:
            self.generator = Generator(n_words=tgt_vocab.vocab_size, hidden_size=config.embed_size, padding_idx=NMTVocab.PAD)

        self.use_gpu = use_gpu

    def forward(self, src_seq, synx, ext_src_seq, tgt_seq=None, ext_tgt_seq=None, mode="train", **kwargs):
        if mode == "train":
            assert tgt_seq is not None and ext_tgt_seq is not None
            return self.force_teaching(src_seq, synx, ext_src_seq, tgt_seq, ext_tgt_seq, **kwargs)
        elif mode == "infer":
            assert tgt_seq is None and ext_tgt_seq is None
            return self.batch_beam_search(src_seq=src_seq, **kwargs)

    def force_teaching(self, src_seq, synx, ext_src_seq, tgt_seq, ext_tgt_seq, lengths):

        enc_output, enc_mask = self.encoder(src_seq, synx, ext_src_seq)
        dec_output, _, _ = self.decoder(tgt_seq, ext_tgt_seq, enc_output, enc_mask)

        return dec_output

    def batch_beam_search(self, src_seq, ext_src_seq, lengths, beam_size=5, max_steps=150):

        batch_size = src_seq.size(0)

        enc_output, enc_mask = self.encoder(src_seq, ext_src_seq) # [batch_size, seq_len, dim]

        # dec_caches = self.decoder.compute_caches(enc_output)

        # Tile beam_size times
        enc_mask = tile_batch(enc_mask, multiplier=beam_size, batch_dim=0)
        enc_output = tile_batch(enc_output, multiplier=beam_size, batch_dim=0)

        final_word_indices = src_seq.data.new(batch_size, beam_size, 1).fill_(NMTVocab.BOS) # Word indices in the beam
        final_lengths = enc_output.data.new(batch_size, beam_size).fill_(0.0) # length of the sentence
        beam_mask = enc_output.data.new(batch_size, beam_size).fill_(1.0) # Mask of beams
        beam_scores = enc_output.data.new(batch_size, beam_size).fill_(0.0) # Accumulated scores of the beam


        self_attn_caches = None # Every element has shape [batch_size * beam_size, num_heads, seq_len, dim_head]
        enc_attn_caches = None

        for t in range(max_steps):

            inp_t = Variable(final_word_indices.view(-1, final_word_indices.size(-1)), volatile=True)

            dec_output, self_attn_caches, enc_attn_caches \
                = self.decoder(tgt_seq=inp_t,
                               enc_output=enc_output,
                               enc_mask=enc_mask,
                               enc_attn_caches=enc_attn_caches,
                               self_attn_caches=self_attn_caches) # [batch_size * beam_size, seq_len, dim]

            next_scores = - self.generator(dec_output[:,-1].contiguous()).data # [batch_size * beam_size, n_words]
            next_scores = next_scores.view(batch_size, beam_size, -1)
            next_scores = mask_scores(next_scores, beam_mask=beam_mask)

            beam_scores = next_scores + beam_scores.unsqueeze(2) # [B, Bm, N] + [B, Bm, 1] ==> [B, Bm, N]

            vocab_size = beam_scores.size(-1)
            if t == 0:
                beam_scores = beam_scores[:,0,:].contiguous()

            beam_scores = beam_scores.view(batch_size, -1)

            # Get topK with beams【
            beam_scores, indices = torch.topk(beam_scores, k=beam_size, dim=-1, largest=False, sorted=False)
            next_beam_ids = torch.div(indices, vocab_size)
            next_word_ids = indices % vocab_size

            # Re-arrange by new beam indices
            beam_mask = tensor_gather_helper(gather_indices=next_beam_ids,
                                             gather_from=beam_mask,
                                             batch_size=batch_size,
                                             beam_size=beam_size,
                                             gather_shape=[-1],
                                             use_gpu=self.use_gpu)

            final_word_indices = tensor_gather_helper(gather_indices=next_beam_ids,
                                                      gather_from=final_word_indices,
                                                      batch_size=batch_size,
                                                      beam_size=beam_size,
                                                      gather_shape=[batch_size * beam_size, -1],
                                                      use_gpu=self.use_gpu)

            final_lengths = tensor_gather_helper(gather_indices=next_beam_ids,
                                                 gather_from=final_lengths,
                                                 batch_size=batch_size,
                                                 beam_size=beam_size,
                                                 gather_shape=[-1],
                                                 use_gpu=self.use_gpu)

            self_attn_caches = map_structure(
                lambda t: Variable(tensor_gather_helper(gather_indices=next_beam_ids,
                                               gather_from=t.data,
                                               batch_size=batch_size,
                                               beam_size=beam_size,
                                               gather_shape=[batch_size * beam_size, self.decoder.n_head,
                                                             -1, self.decoder.dim_per_head],
                                                use_gpu=self.use_gpu), volatile=True), self_attn_caches)

            # If next_word_ids is EOS, beam_mask_ should be 0.0
            beam_mask_ = 1.0 - next_word_ids.eq(NMTVocab.EOS).float()
            next_word_ids.masked_fill_((beam_mask_ + beam_mask).eq(0.0), NMTVocab.PAD) # If last step a EOS is already generated, we replace the last token as PAD
            beam_mask = beam_mask * beam_mask_

            # # If an EOS or PAD is encountered, set the beam mask to 0.0
            # beam_mask_ = next_word_ids.gt(Vocab.EOS).float()
            # beam_mask = beam_mask * beam_mask_

            final_lengths += beam_mask

            final_word_indices = torch.cat((final_word_indices, next_word_ids.unsqueeze(2)), dim=2)

            if beam_mask.eq(0.0).all():
                break


        scores = beam_scores / (final_lengths + 1e-2)

        _, reranked_ids = torch.sort(scores, dim=-1, descending=False)

        return tensor_gather_helper(gather_indices=reranked_ids,
                                    gather_from=final_word_indices[:, :, 1:].contiguous(),
                                    batch_size=batch_size,
                                    beam_size=beam_size,
                                    gather_shape=[batch_size * beam_size, -1],
                                    use_gpu=self.use_gpu)