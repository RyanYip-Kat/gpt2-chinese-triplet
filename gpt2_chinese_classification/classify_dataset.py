import io
import os
import torch
import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader

from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
from tokenizations import tokenization_bert


class CustomDataset(Dataset):

  def __init__(self,text_file, use_tokenizer):
      self.text_file = text_file
      self.use_tokenizer = use_tokenizer
      self.load_data()

  def load_data(self):
      texts = []
      labels = []
      for i,line in enumerate(open(self.text_file,encoding="utf-8")):
          line = line.rstrip()
          ll = line.split("\t")
          text,label=  ll[0],ll[1]
          texts.append(text)
          labels.append(label)
      self.texts = texts
      self.labels = labels
      self.n_examples = len(texts)
      self.classes = np.unique(labels).tolist()
      self.labels_ids = {x:i for i,x in enumerate(self.classes)}

  def __len__(self):
    r"""When used `len` return the number of examples.

    """
    
    return self.n_examples

  def __getitem__(self, item):
    r"""Given an index return an example from the position.
    
    Arguments:

      item (:obj:`int`):
          Index position to pick an example to return.

    Returns:
      :obj:`Dict[str, str]`: Dictionary of inputs that contain text and 
      asociated labels.

    """

    return {'text':self.texts[item],
            'label':self.labels[item]}

class Gpt2ClassificationCollator(object):
    r"""
    Data Collator used for GPT2 in a classificaiton rask. 
    
    It uses a given tokenizer and label encoder to convert any text and labels to numbers that 
    can go straight into a GPT2 model.

    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

    Arguments:

      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.

      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to 
          labels names and Values map to number associated to those labels.

      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.

    """

    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):

        # Tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        # Check max sequence length.
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        # Label encoder used inside the class.
        self.labels_encoder = labels_encoder

        return

    def __call__(self, sequences):
        r"""
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this 
        class as a function.

        Arguments:

          item (:obj:`list`):
              List of texts and labels.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """

        # Get all texts from sequences list.
        texts = [sequence['text'] for sequence in sequences]
        # Get all labels from sequences list.
        labels = [sequence['label'] for sequence in sequences]
        # Encode all labels using label encoder.
        labels = [self.labels_encoder[label] for label in labels]
        # Call tokenizer on all texts to convert into tensors of numbers with 
        # appropriate padding.
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        # Update the inputs with the associated encoded labels as tensor.
        inputs.update({'labels':torch.tensor(labels)})

        return inputs




if __name__=="__main__":
    model_path="gpt2-chinese-youlai/final_model/"
    text_file = "/path/Bert-Chinese-Text-Classification-Pytorch/THUCNews/data/train.txt"
    max_length = 60
    batch_size = 16

    print('Loading tokenizer...')
    tokenizer = tokenization_bert.BertTokenizer(vocab_file=model_path+"vocab.txt")
    #tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    dataset = CustomDataset(text_file,tokenizer)
    n_labels = len(dataset.classes)
    labels_ids = dataset.labels_ids
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_path, num_labels=n_labels)
    # default to left padding

    gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer,
                                                          labels_encoder=labels_ids,
                                                          max_sequence_len=max_length)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
    true_labels = []
    for batch in dataloader:
        #true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long) for k,v in batch.items()}
        print(true_labels)
