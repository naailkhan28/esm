from multiprocessing.spawn import prepare
import torch
import numpy as np
from esm.inverse_folding.util import CoordBatchConverter, extract_coords_from_structure

def get_encoder_attributions(model, input_coords):
  #Use autocast to switch to float16 and save memory   
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
  #Use autocast to switch to float16 and save memory
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

def prepare_previous_tokens(alphabet, sampled_sequence, sequence_position, device):
    #We use our generated sequence as the input to the decoder
    #We also need to prepend a <cath> token
    tokenized_sequence = torch.tensor([alphabet.get_idx(residue) for residue in sampled_sequence])
    cath_token = torch.tensor([alphabet.get_idx('<cath>')])

    prev_tokens = torch.cat((cath_token, tokenized_sequence), 0)
    prev_tokens = prev_tokens.unsqueeze(0)
    prev_tokens = prev_tokens[:, :sequence_position+1]
    prev_tokens = prev_tokens.to(device)
    return prev_tokens

def get_model_inputs(alphabet, coords, device):
    #The coordinates extracted from the PDB file need to be processed by the Batch Converter to get them into the correct format
    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, None)]
    input_coords, confidence, _, _, padding_mask = batch_converter(batch, device=device)
    return input_coords, confidence, padding_mask

#Get relevancy matrix for a given position in a sampled sequence with respect to an input structure
def get_sequence_position_attributions(model, alphabet, sampled_sequence, sequence_position, coords, device):

  prev_tokens = prepare_previous_tokens(alphabet, sampled_sequence, sequence_position, device)

  input_coords, confidence, padding_mask = get_model_inputs(alphabet, coords, device)

  #Forward pass through the model - use autocast to switch to float16 and save memory
  with torch.autocast("cuda"):
    logits, _ = model.forward(input_coords, padding_mask, confidence, prev_tokens)
  
  #Get the amino acid at our chosen sequence position
  amino_acid = sampled_sequence[sequence_position-1]
  token = alphabet.get_idx(amino_acid)

  #Build a one-hot tensor representing this token and backpropagate it
  one_hot = torch.zeros_like(logits, dtype=float)
  one_hot[0, token, sequence_position] = 1
  one_hot.requires_grad_(True)
  one_hot = torch.sum(one_hot * logits)

  model.zero_grad(set_to_none=True)
  one_hot.backward(retain_graph=True)

  encoder_relevancy_matrix = get_encoder_attributions(model, input_coords)
  Rqi = get_decoder_relevancy_matrix(model, logits, input_coords, encoder_relevancy_matrix)

  return(Rqi)

def bhattacharya(p, q):
	return -1 * np.log(sum([np.sqrt(p[i] * q[i]) for i in range(len(p))]))

def get_averaged_bhattacharyya_distances(model, alphabet, sampled_sequence, device, structure, descending=False):
  distances = {}

  for j in range(1, len(sampled_sequence)+1):
    if j % 10 == 0:
      print(f"Residue {j} / {len(sampled_sequence+1)}")
    
    masked_coords, native_seq = extract_coords_from_structure(structure)

    sequence_position = j
    
    Rqi = get_sequence_position_attributions(model, alphabet, sampled_sequence, sequence_position, masked_coords, device)
    relevancies = Rqi[-1][1:-1]
    values, indices = torch.sort(relevancies, descending=descending)

    prev_tokens = prepare_previous_tokens(alphabet, sampled_sequence, sequence_position, device)
    input_coords, confidence, padding_mask = get_model_inputs(alphabet, masked_coords, device)

    #Forward pass through the model - use autocast to switch to float16 and save memory
    with torch.autocast("cuda"):
      logits, _ = model.forward(input_coords, padding_mask, confidence, prev_tokens)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    wt_probs = probs[-1].to("cpu").detach().numpy()

    distances[j] = {}

    for i, residue in enumerate(indices):

      masked_coords[residue.item(), :] = float("inf")

      #The coordinates extracted from the PDB file need to be processed by the Batch Converter to get them into the correct format
      masked_input_coords, masked_confidence, masked_padding_mask = get_model_inputs(alphabet, masked_coords, device)

      #Forward pass through the model - use autocast to switch to float16 and save memory
      with torch.autocast("cuda"):
        logits, _ = model.forward(masked_input_coords, masked_padding_mask, masked_confidence, prev_tokens)

      masked_probs = torch.nn.functional.softmax(logits, dim=-1)
      masked_probs = masked_probs[-1].to("cpu").detach().numpy()

      distances[j][i+1] = bhattacharya(wt_probs[0], masked_probs[0])
    
  return(distances)