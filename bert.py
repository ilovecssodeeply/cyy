import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import transformers
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
train_curve = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 定义一些参数，模型选择了最基础的bert中文模型
batch_size = 16
epoches = 100
model = "bert-base-chinese"
hidden_size = 768
n_class = 7
maxlen = 12


# data，构造一些训练数据
# sentences = ["我喜欢打篮球", "这个相机很好看", "今天玩的特别开心", "我不喜欢你", "太糟糕了", "真是件令人伤心的事情"]
# sentences = ["2022 12 31 manly ma man manl an anl anly nly ly",
#              "2022 12 30 molar mo mol mola ol ola olar la lar ar",
#              "2022 12 29 impel im imp impe mp mpe mpel pe pel el"]
            #  "condo co con cond on ond ondo nd ndo do"]
    
# sentences = np.load('encode.npy')[0:3]
sentences = np.load('encode.npy')
print(sentences)
print(sentences.shape)

# labels = np.load('scores.npy', allow_pickle=True).astype(float)[0:3]
labels = np.load('scores.npy', allow_pickle=True).astype(float)
print(labels)
print(labels.shape)

# labels = np.array([
#             [0.0, 0.02, 0.17, 0.37, 0.29, 0.12, 0.03], 
#           [0.0, 0.02, 0.16, 0.38, 0.30, 0.12, 0.02],
#           [0.0, 0.03, 0.21, 0.40, 0.25, 0.10, 0.01]])
labels = torch.tensor(labels)
# print(labels)
# 0	,0.02,	0.17,	0.35,	0.29,	0.14,	0.03

# labels = np.array(labels)
# print(labels.sum(axis=1))
  # 1积极, 0消极.
# word_list = ' '.join(sentences).split()
# word_list = list(set(word_list))
# word_dict = {w: i for i, w in enumerate(word_list)}
# num_dict = {i: w for w, i in word_dict.items()}
# vocab_size = len(word_list)
# 将数据构造成bert的输入格式
# inputs_ids: token的字典编码
# attention_mask:长度与inputs_ids一致，真实长度的位置填充1，padding位置填充0
# token_type_ids: 第一个句子填充0，第二个句子句子填充1
class MyDataset(Data.Dataset):
  def __init__(self, sentences, labels=None, with_labels=True,):
    self.tokenizer = AutoTokenizer.from_pretrained(model)
    self.with_labels = with_labels
    self.sentences = sentences
    self.labels = labels
  def __len__(self):
    return len(sentences)
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
train = Data.DataLoader(dataset=MyDataset(sentences, labels), batch_size=batch_size, shuffle=True, num_workers=1)
# model
class BertClassify(nn.Module):
  def __init__(self):
    super(BertClassify, self).__init__()
    self.bert = AutoModel.from_pretrained(model, output_hidden_states=True, return_dict=True)
    self.linear = nn.Linear(hidden_size, n_class) # 直接用cls向量接全连接层分类
    self.dropout = nn.Dropout(0.5)
    self.softmax = nn.Softmax(dim=1)
    self.double()
  def forward(self, X):
    input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) # 返回一个output字典
    # 用最后一层cls向量做分类
    # outputs.pooler_output: [bs, hidden_size]
    logits = self.linear(self.dropout(outputs.pooler_output))
    # print(logits)
    out = self.softmax(logits)
    return out
bc = BertClassify().to(device)
optimizer = optim.Adam(bc.parameters(), lr=1e-5, weight_decay=1e-4)
loss_fn = nn.MSELoss()
# loss_fn = nn.L1Loss()
# train
sum_loss = 0
total_step = len(train)
for epoch in range(epoches):
  for i, batch in enumerate(train):
    optimizer.zero_grad()
    # print(batch)
    batch = tuple(p.to(device) for p in batch)
    # token_ids, attn_masks, token_type_ids, label
    pred = bc([batch[0], batch[1], batch[2]])
    loss = loss_fn(pred, batch[3])
    sum_loss += loss.item()
    loss.backward()
    optimizer.step()
    if epoch % 1 == 0:
      print('[{}|{}] step:{}/{} loss:{:.4f}'.format(epoch+1, epoches, i+1, total_step, loss.item()))
  train_curve.append(sum_loss)
  sum_loss = 0
# test
bc.eval()
with torch.no_grad():
  test_text = ["condo co con cond on ond ondo nd ndo do"]
  test = MyDataset(test_text, labels=None, with_labels=False)
  x = test.__getitem__(0)
  x = tuple(p.unsqueeze(0).to(device) for p in x)
  pred = bc([x[0], x[1], x[2]])
  # pred = pred.data.max(dim=1, keepdim=True)[1]
  # if pred[0][0] == 0:
  #   print('消极')
  # else:
  #   print('积极')
  print(pred.data)
fig = pd.DataFrame(train_curve).plot() # loss曲线
fig.figure.savefig('pic.png')
