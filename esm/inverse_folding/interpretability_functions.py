import torch


def get_encoder_attributions(model, input_coords):    
    with torch.autocast("cuda"):

      #We initialize the encoder self-attention matrix as the identity matrix - we initially assume each token attends only to itself
      Rii = torch.eye(input_coords.shape[1], input_coords.shape[1]).to(input_coords.device)


      for layer in model.encoder.layers:
        #Retrieve encoder self-attention weights, and their gradients with respect to our backpropagated tensor
        encoder_weights = layer.self_attn.get_attn().detach()
        encoder_gradients = layer.self_attn.get_attn_gradients().detach()
        
        encoder_weights = encoder_weights.reshape(-1, encoder_weights.shape[-2], encoder_weights.shape[-1])
        encoder_gradients = encoder_gradients.reshape(-1, encoder_gradients.shape[-2], encoder_gradients.shape[-1])

        #Process our encoder self-attention weights = re-weight them by their gradients and average across all heads, and remove any negative values
        encoder_weights = encoder_weights * encoder_gradients
        encoder_weights = encoder_weights.clamp(min=0).mean(dim=0)

        Rii += torch.matmul(encoder_weights, Rii)

        return Rii

def get_decoder_relevancy_matrix(model, logits, input_coords, Rii):
    with torch.autocast("cuda"):

        #The decoder self-attention matrix is the identity matrix, just like the encoder self-attention
        Rqq = torch.eye(logits.shape[2], logits.shape[2]).to(input_coords.device)

        #Decoder cross-attention between encoder and decoder is initialized as zeros - we initially assume no relationship
        Rqi = torch.zeros(logits.shape[2], input_coords.shape[1]).to(input_coords.device)

        for layer in model.decoder.layers:
          #Retrieve decoder self-attention weights and their gradients, plus cross-attention weights and gradients
          decoder_weights = layer.self_attn.get_attn().detach()
          decoder_gradients = layer.self_attn.get_attn_gradients().detach()
          cross_weights = layer.encoder_attn.get_attn().detach()
          cross_gradients = layer.encoder_attn.get_attn_gradients().detach()
          
          decoder_weights = decoder_weights.reshape(-1, decoder_weights.shape[-2], decoder_weights.shape[-1])
          decoder_gradients = decoder_gradients.reshape(-1, decoder_gradients.shape[-2], decoder_gradients.shape[-1])
          cross_weights = cross_weights.reshape(-1, cross_weights.shape[-2], cross_weights.shape[-1])
          cross_gradients = cross_gradients.reshape(-1, cross_gradients.shape[-2], cross_gradients.shape[-1])

          #Process all weights as before
          decoder_weights = decoder_weights * decoder_gradients
          decoder_weights = decoder_weights.clamp(min=0).mean(dim=0)
          cross_weights = cross_weights * cross_gradients
          cross_weights = cross_weights.clamp(min=0).mean(dim=0)

          #Update relevancy matrices with decoder self-attentions
          Rqq += torch.matmul(decoder_weights, Rqq)
          Rqi += torch.matmul(decoder_weights, Rqi)

          #Normalize self-attention relevancy matrices - subtract identity, divide by sum, add identity again
          Rqq_normalized = Rqq.clone()
          Rii_normalized = Rii.clone()

          qq_diag = range(Rqq_normalized.shape[-1])
          ii_diag = range(Rii_normalized.shape[-1])

          Rqq_normalized = Rqq_normalized - torch.eye(Rqq_normalized.shape[-1]).to(input_coords.device)
          Rii_normalized = Rii_normalized - torch.eye(Rii_normalized.shape[-1]).to(input_coords.device)

          assert Rqq_normalized[qq_diag, qq_diag].min() >= 0
          assert Rii_normalized[ii_diag, ii_diag].min() >= 0

          Rqq_normalized = Rqq_normalized / Rqq_normalized.sum(dim=-1, keepdim=True)
          Rii_normalized = Rii_normalized / Rii_normalized.sum(dim=-1, keepdim=True)

          Rqq_normalized = Rqq_normalized + torch.eye(Rqq_normalized.shape[-1]).to(input_coords.device)
          Rii_normalized = Rii_normalized + torch.eye(Rii_normalized.shape[-1]).to(input_coords.device)

          #Now we can update the cross-attention relevancy matrix finally
          Rqi_addition = torch.matmul(Rqq_normalized.t(), torch.matmul(cross_weights, Rii_normalized))
          Rqi_addition[torch.isnan(Rqi_addition)] = 0

          Rqi = Rqi + Rqi_addition

        Rqi = (Rqi - Rqi.min()) / (Rqi.max() - Rqi.min())
        return Rqi