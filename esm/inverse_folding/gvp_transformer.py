# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from secrets import token_urlsafe
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from scipy.spatial import transform

from esm.data import Alphabet
from esm.inverse_folding.sampling import greedy_sample, multinomial_sample, nucleus_sample, topk_sample

from .features import DihedralFeatures
from .gvp_encoder import GVPEncoder
from .gvp_utils import unflatten_graph
from .gvp_transformer_encoder import GVPTransformerEncoder
from .transformer_decoder import TransformerDecoder
from .util import rotate, CoordBatchConverter 


class GVPTransformerModel(nn.Module):
    """
    GVP-Transformer inverse folding model.

    Architecture: Geometric GVP-GNN as initial layers, followed by
    sequence-to-sequence Transformer encoder and decoder.
    """

    def __init__(self, args, alphabet):
        super().__init__()
        encoder_embed_tokens = self.build_embedding(
            args, alphabet, args.encoder_embed_dim,
        )
        decoder_embed_tokens = self.build_embedding(
            args, alphabet, args.decoder_embed_dim, 
        )
        encoder = self.build_encoder(args, alphabet, encoder_embed_tokens)
        decoder = self.build_decoder(args, alphabet, decoder_embed_tokens)
        self.args = args
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = GVPTransformerEncoder(args, src_dict, embed_tokens)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
        )
        return decoder

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.padding_idx
        emb = nn.Embedding(num_embeddings, embed_dim, padding_idx)
        nn.init.normal_(emb.weight, mean=0, std=embed_dim ** -0.5)
        nn.init.constant_(emb.weight[padding_idx], 0)
        return emb

    def forward(
        self,
        coords,
        padding_mask,
        confidence,
        prev_output_tokens,
        return_all_hiddens: bool = False,
        features_only: bool = False,
    ):
        encoder_out = self.encoder(coords, padding_mask, confidence,
            return_all_hiddens=return_all_hiddens)
        logits, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            return_all_hiddens=return_all_hiddens,
        )
        return logits, extra

    def prep_for_sample(self, coords, partial_seq=None, confidence=None, device=None):
        """
        Prepares inputs for incremental decoding and sampling

        Args:
            coords: L x 3 x 3 list representing one backbone
            partial_seq: Optional, partial sequence with mask tokens if part of
            confidence: Optional, length L list of confidence scores for coordinates
            device: Optional, device to use for sampling
        """
        L = len(coords)
        # Convert to batch format
        batch_converter = CoordBatchConverter(self.decoder.dictionary)
        batch_coords, confidence, _, _, padding_mask = (
            batch_converter([(coords, confidence, None)], device=device)
        )
        
        # Start with prepend token
        mask_idx = self.decoder.dictionary.get_idx('<mask>')
        starting_tokens = torch.full((1, 1+L), mask_idx, dtype=int)
        starting_tokens = starting_tokens.to(device)
        starting_tokens[0, 0] = self.decoder.dictionary.get_idx('<cath>')

        if partial_seq is not None:
            for i, c in enumerate(partial_seq):
                starting_tokens[0, i+1] = self.decoder.dictionary.get_idx(c)
            
        # Save incremental states for faster sampling
        incremental_state = dict()
        
        # Run encoder only once
        encoder_out = self.encoder(batch_coords, padding_mask, confidence)

        return starting_tokens, encoder_out, incremental_state

    def greedy_sample(self, coords, partial_seq=None, confidence=None, device=None, temperature=1):

        starting_tokens, encoder_out, incremental_state = self.prep_for_sample(coords, partial_seq, confidence, device)

        mask_idx = self.decoder.dictionary.get_idx('<mask>')
        # Decode one token at a time
        for i in range(1, starting_tokens.shape[1]):
            if starting_tokens[0, i] != mask_idx:
                continue
            logits, _ = self.decoder(
                starting_tokens[:, :i], 
                encoder_out,
                incremental_state=incremental_state,
            )
            logits = logits[0].transpose(0, 1)
            logits /= temperature
            probs = F.softmax(logits, dim=-1)
            starting_tokens[:, i] = greedy_sample(probs)
        sampled_seq = starting_tokens[0, 1:]
        
        # Convert back to string via lookup
        return ''.join([self.decoder.dictionary.get_tok(a) for a in sampled_seq])


    def multinomial_sample(self, coords, partial_seq=None, confidence=None, device=None, temperature=1):
        starting_tokens, encoder_out, incremental_state = self.prep_for_sample(coords, partial_seq, confidence, device)
        
        mask_idx = self.decoder.dictionary.get_idx('<mask>')
        # Decode one token at a time
        for i in range(1, starting_tokens.shape[1]):
            if starting_tokens[0, i] != mask_idx:
                continue
            logits, _ = self.decoder(
                starting_tokens[:, :i], 
                encoder_out,
                incremental_state=incremental_state,
            )
            logits = logits[0].transpose(0, 1)
            logits /= temperature
            probs = F.softmax(logits, dim=-1)
            starting_tokens[:, i] = multinomial_sample(probs)
        sampled_seq = starting_tokens[0, 1:]
        
        # Convert back to string via lookup
        return ''.join([self.decoder.dictionary.get_tok(a) for a in sampled_seq])
    
    def topk_sample(self, coords, k, partial_seq=None, confidence=None, device=None, temperature=1):
        
        starting_tokens, encoder_out, incremental_state = self.prep_for_sample(coords, partial_seq, confidence, device)

        mask_idx = self.decoder.dictionary.get_idx('<mask>')
        # Decode one token at a time
        for i in range(1, starting_tokens.shape[1]):
            if starting_tokens[0, i] != mask_idx:
                continue
            logits, _ = self.decoder(
                starting_tokens[:, :i], 
                encoder_out,
                incremental_state=incremental_state,
            )
            logits = logits[0].transpose(0, 1)
            logits /= temperature
            probs = F.softmax(logits, dim=-1)
            starting_tokens[:, i] = topk_sample(probs, k=k)
        sampled_seq = starting_tokens[0, 1:]
        
        # Convert back to string via lookup
        return ''.join([self.decoder.dictionary.get_tok(a) for a in sampled_seq])
    
    def nucleus_sample(self, coords, partial_seq=None, confidence=None, device=None, temperature=1, p=0.92):

        starting_tokens, encoder_out, incremental_state = self.prep_for_sample(coords, partial_seq, confidence, device)

        mask_idx = self.decoder.dictionary.get_idx('<mask>')
        # Decode one token at a time
        for i in range(1, starting_tokens.shape[1]):
            if starting_tokens[0, i] != mask_idx:
                continue
            logits, _ = self.decoder(
                starting_tokens[:, :i], 
                encoder_out,
                incremental_state=incremental_state,
            )
            logits = logits[0].transpose(0, 1)
            logits /= temperature
            probs = F.softmax(logits, dim=-1)
            starting_tokens[:, i] = nucleus_sample(probs, p=0.92)
        sampled_seq = starting_tokens[0, 1:]
        
        # Convert back to string via lookup
        return ''.join([self.decoder.dictionary.get_tok(a) for a in sampled_seq])

    
    def beam_sample(self, coords, beam_size, partial_seq=None, confidence=None, device=None, temperature=1):

        starting_tokens, encoder_out, incremental_state = self.prep_for_sample(coords, partial_seq, confidence, device)

        #Beams are stored as a dictionary of tuples
        #Keys are integer indices, tuples contain the tensor itself and its log probability
        beams = {}

        #Initialize beams with the most likely first tokens
        logits, _ = self.decoder(
                starting_tokens[:, :1], 
                encoder_out,
                incremental_state=incremental_state,
            )
        logits = logits[0].transpose(0, 1)
        logits /= temperature
        probs = F.softmax(logits, dim=-1)
        values, indices = torch.topk(probs, k=beam_size)

        for idx, (residue, probability) in enumerate(zip(indices[0], values[0])):
            beam_starting_tensor = starting_tokens
            beam_starting_tensor[0, 1] = residue
            beams[idx] = (beam_starting_tensor, torch.log(probability))
        
        #Decode one token at a time, across all beams
        for i in range(2, starting_tokens.shape[1]):
            
            #Store all possible beams at this step in a dictionary for retrieval later
            all_possible_beams = {}

            for beam in beams.keys():

                prev_tokens = beams[beam][0]
                log_prob = beams[beam][1]
                logits, _ = self.decoder(
                prev_tokens[:, :i], 
                encoder_out,
                incremental_state=incremental_state)

                logits = logits[0].transpose(0, 1)
                logits /= temperature
                probs = F.softmax(logits, dim=-1)
                sorted_probabilities, indices = torch.sort(probs, descending=True)

                for residue, probability in zip(indices[0], sorted_probabilities[0]):
                    new_beam_tensor = prev_tokens
                    new_beam_tensor[:, i] = residue
                    new_log_prob = log_prob + torch.log(probability)

                    all_possible_beams[new_log_prob] = new_beam_tensor

            sorted_beams = dict(sorted(all_possible_beams.items(), reverse=True))[:5]

            for beam, (log_probability, tokens) in enumerate(sorted_beams.items()):
                beams[beam] = (tokens, log_probability)
        
        out_seqs = []

        for key, (tokens, log_prob) in beams.items():
            out_seqs.append(''.join([self.decoder.dictionary.get_tok(a) for a in tokens]))

        return(out_seqs)