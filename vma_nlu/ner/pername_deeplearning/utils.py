import os
import gdown
import torch
from vma_nlu.ner.pername_deeplearning import MODEL_DIR, BASE_URL, CHECKPOINT, CHAR_VOCAB, LABEL_SET

from torch import Tensor, device, dtype, nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


def get_mask(max_seq_len, seq_len):
    mask = [[1]*seq_len[i]+[0]*(max_seq_len-seq_len[i]) for i in range(len(seq_len))]
    mask = torch.tensor(mask)
    mask = mask.unsqueeze(1).expand(-1, mask.shape[-1], -1)
    mask = torch.triu(mask)
    return mask

def get_useful_ones(out, label, mask):
    # get mask, mask the padding and down triangle

    mask = mask.reshape(-1)
    tmp_out = out.reshape(-1, out.shape[-1])
    tmp_label = label.reshape(-1)
    # index select, for gpu speed
    indices = mask.nonzero(as_tuple=False).squeeze(-1).long()
    tmp_out = tmp_out.index_select(0, indices)
    tmp_label = tmp_label.index_select(0, indices)

    return tmp_out, tmp_label

def load_model(path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def initializeFolder():
	if not os.path.exists(MODEL_DIR):
		os.mkdir(MODEL_DIR)
		print("Directory ", MODEL_DIR," created")

def download_model():
    checkpoint = MODEL_DIR + CHECKPOINT
    label_set_path = MODEL_DIR + LABEL_SET
    char_vocab_path = MODEL_DIR + CHAR_VOCAB

    if os.path.isfile(checkpoint) != True:
        print(f"{CHECKPOINT} will be downloaded...")
        gdown.download(BASE_URL + CHECKPOINT, checkpoint, quiet=False)

    if os.path.isfile(label_set_path) != True:
        print(f"{LABEL_SET} will be downloaded...")
        gdown.download(BASE_URL + LABEL_SET, label_set_path, quiet=False)

    if os.path.isfile(char_vocab_path) != True:
        print(f"{CHAR_VOCAB} will be downloaded...")
        gdown.download(BASE_URL + CHAR_VOCAB, char_vocab_path, quiet=False)

    return checkpoint, label_set_path, char_vocab_path

def get_extended_attention_mask(attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
      """
      Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

      Arguments:
          attention_mask (:obj:`torch.Tensor`):
              Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
          input_shape (:obj:`Tuple[int]`):
              The shape of the input to the model.
          device: (:obj:`torch.device`):
              The device of the input to the model.

      Returns:
          :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
      """
      # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
      # ourselves in which case we just need to make it broadcastable to all heads.
      if attention_mask.dim() == 3:
          extended_attention_mask = attention_mask[:, None, :, :]
      elif attention_mask.dim() == 2:
          # Provided a padding mask of dimensions [batch_size, seq_length]
          # - if the model is a decoder, apply a causal mask in addition to the padding mask
          # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
          extended_attention_mask = attention_mask[:, None, None, :]
      else:
          raise ValueError(
              f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
          )

      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -10000.0 for masked positions.
      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)  # fp16 compatibility
      extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
      return extended_attention_mask