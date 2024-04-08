import torch
import torch.nn as nn
import torch.utils.data as Data
from transformers import BertTokenizer, BertModel

pretrained_model = 'BertRCNN'
hidden_size = 768
n_class = 2
maxlen = 8
tokenizer = BertTokenizer.from_pretrained(pretrained_model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(Data.Dataset):
  def __init__(self, sentences, labels=None, with_labels=True,):
    self.tokenizer = tokenizer
    self.with_labels = with_labels
    self.sentences = sentences
    self.labels = labels

  def __getitem__(self, index):
    # Selecting sentence1 and sentence2 at the specified index in the data frame
    sent = self.sentences[index]

    # Tokenize the pair of sentences to get token ids, attention masks and token type ids
    encoded_pair = self.tokenizer(sent,
                    padding='max_length',  # Pad to max_length
                    truncation=True,       # Truncate to max_length
                    max_length=maxlen,  
                    return_tensors='pt')  # Return torch.Tensor objects

    token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
    attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
    token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

    if self.with_labels:  # True if the dataset has labels
      label = self.labels[index]
      return token_ids, attn_masks, token_type_ids, label
    else:
      return token_ids, attn_masks, token_type_ids


# model
class BertClassify(nn.Module):
  def __init__(self):
    super(BertClassify, self).__init__()
    self.bert = BertModel.from_pretrained(pretrained_model, output_hidden_states=True, return_dict=True)
    self.linear = nn.Linear(hidden_size, n_class)
    self.dropout = nn.Dropout(0.5)

  def forward(self, X):
    input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) # 返回一个output字典
    logits = self.linear(self.dropout(outputs.pooler_output))

    return logits

bc = BertClassify().to(device)


# test
bc.eval()
with torch.no_grad():
  test_text = ['砖家证实接吻也可怀孕，哥表示鸭梨很大！']
  test = MyDataset(test_text, labels=None, with_labels=False)
  x = test.__getitem__(0)
  x = tuple(p.unsqueeze(0).to(device) for p in x)
  pred = bc([x[0], x[1], x[2]])
  pred = pred.data.max(dim=1, keepdim=True)[1]
  if pred[0][0] == 0:
    print('谣言 ')
  else:
    print('非谣言')