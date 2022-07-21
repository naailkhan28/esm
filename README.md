# Interpreting the ESM Inverse Folding Model

This repo contains my personal additions to the ESM Inverse Folding protein design model, to allow visualization of the model's attention mechanisms and general interpretation of how the model arrives at a generated sequence given an input backbone structure.

Fork of the original model available at https://github.com/facebookresearch/esm


# Attention Weights

ESM-IF1 is made up of a GVP block to extract features from the protein backbone, a transformer encoder block, and a transformer decoder block.

These encoder and decoder blocks make use of `MultiHeadAttention` layers (extended from `nn.Module`). While these layers do have an optional argument to return attention weights for each head, by default this behaviour is disabled!

I have updated `TransformerEncoderLayer` and  `TransformerDecoderLayer` to return the attention weights for all 8 heads. I have also subsequently updated `GVP TransformerEncoder` and `TransformerDecoder` in order to return the weights for all encoder/decoder layers in a dictionary.

# Weight Gradients
Interpretability methods often use not only the attention weights, but their gradients with respect to an output as well (for example, see the Encoder-Decoder method from https://github.com/hila-chefer/Transformer-MM-Explainability). Here, I have added a PyTorch `reverse_hook` to the attention weight matrices at each encoder and decoder layer. 

Following the forward pass through the model with a set of structure coordinates, you can backpropagate a one-hot Tensor corresponding to the choice of amino acid at a given output position. You can then call recover gradients like so:

    for layer in model.encoder.layers:
        attention_gradients = layer.self_attn.get_attn_gradients()
        
    for layer in model.decoder.layers:
        self_attention_gradients = layer.self_attn.get_attn_gradients()
        encoder_attention_gradients = layer.encoder_attn.get_attn_gradients()
