# Interpreting the ESM Inverse Folding Model

This repo contains my personal edits to the ESM Inverse Folding protein design model, to allow visualization of the model's attention mechanisms and general interpretation of how the model arrives at a generated sequence given an input backbone structure.

Fork of the original model available at https://github.com/facebookresearch/esm


# Summary of my changes so-far (Updated 22nd June 2022)

ESM-IF1 is made up of a GVP block to extract features from the protein backbone, a transformer encoder block, and a transformer decoder block.

These encoder and decoder blocks make use of `MultiHeadAttention` layers (extended from `nn.Module`). While these layers do have an optional argument to return attention weights for each head, by default this behaviour is disabled!

I have updated `TransformerEncoderLayer` and  `TransformerDecoderLayer` to return the attention weights for all 8 heads. I have also subsequently updated `GVP TransformerEncoder` and `TransformerDecoder` in order to return the weights for all encoder/decoder layers in a dictionary.
