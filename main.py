# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)
import numpy as np
embed_dim = 32


class attention(nn.Module):
    """
    The purpose of this class is to map words to numbers using the attention mechanism
    Attention:
    Helps determine affinities between words using weights
    the weights determines the amount of say a word has on the final representation
    of another word

    Mathematically weights are calculated by multiplying query and key vectors
    """
    def __init__(self,heads, masking = None):
        super().__init__()
        self.dim_lower = embed_dim//heads
        self.query = nn.Linear(embed_dim, self.dim_lower) # Created query vector
        self.key = nn.Linear(embed_dim, self.dim_lower) # Created key vector
        self.value = nn.Linear(embed_dim, self.dim_lower) # Created value vector
        self.masking = masking

    def forward(self,x):
        B,T,C = self.key(x).shape
        self.weights = torch.matmul(self.query(x), self.key(x).view(B,C,T))###(number_of tokens,number_of_tokens)

        if self.masking:
            self.weights = torch.tril(self.weights, diagonal = 0)
            self.weights.masked_fill_(self.weights == 0, float('-inf'))

        self.weights = F.softmax(self.weights, dim=1)/np.sqrt(self.dim_lower)

        return self.weights @ self.value(x)

class mha(nn.Module):
    """
    Implements multiple attentions
    """

    def __init__(self,heads):
        super().__init__()
        self.attn_block = [attention(heads,masking = True) for _ in range(heads)]

    def forward(self,x):
        return torch.cat([self.attn_block[i](x) for i in range(len(self.attn_block))], dim = 2)
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, heads, seq_len):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size,embed_dim)
        self.positional_encoding = nn.Embedding(seq_len,embed_dim)
        self.pos_embed = self.positional_encoding(torch.tensor([i for i in range(seq_len)]))
        ### Attention part
        self.mha = mha(heads)
        self.ll1 = nn.Linear(embed_dim,vocab_size)

    def forward(self, x, targets = None):
        x = self.embedding_table(x) # B T C
        x = x + self.pos_embed
        x = self.mha(x)
        x = self.ll1(x)

        if targets == None:
            loss = None
        else:
            B,T,C = x.shape
            x  = x.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(x,targets)
        return x,loss

    def generate(self,idx, max_new_tokens):
        """
        idx is batch,time dimensions

        :param idx:
        :return:
        """
        for _ in range(max_new_tokens):

            logits,loss = bigram(idx)  ##(B,T,C)
            next_logits = logits[:, -1 ,:] ## taking the least elemnet
            probs = F.softmax(next_logits, dim = -1)
            idx_next  = torch.multinomial(probs,num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx

with open('input.txt', 'r') as tiny_h:
    data = tiny_h.read()

### Map characters to integers
chars =  set(list(data))

encode = {char:idx for idx,char in enumerate(chars)}

decode = {idx:char for idx,char in enumerate(chars)}

encoded_data = torch.tensor([encode[char] for char in data], dtype = torch.long)

#encde = {for }

block_size = 8
batch_size = 4

N = 0.9

train_data = encoded_data[: int(N * len(data))]
valid_data = encoded_data[int(N * len(data)) :]



def get_batch(split):
    data = train_data if split == 'train' else valid_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in  idx])
    return x,y


xb,yb = get_batch(split = 'train')

print(xb.shape)

heads = 4
seq_len = 8
bigram = BigramLanguageModel(len(encode),heads, seq_len)
logits,loss = bigram(xb,yb)


generation = bigram.generate(torch.zeros((1,1), dtype = torch.long), 250)


decoded_generation = ''.join([decode[int(idx)] for idx in generation[0]])

print(decoded_generation)

optimizer = torch.optim.AdamW(bigram.parameters(), lr = 1e-3)

### Training code

batch_size = 32
iterations = 20000

for steps in range(iterations):
    xb,yb  = get_batch('train')
    logits, loss = bigram(xb,yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training Loss")
print(loss)

print("After Training")

generation = bigram.generate(torch.zeros((1,1), dtype = torch.long), 250)

decoded_generation = ''.join([decode[int(idx)] for idx in generation[0]])

print(decoded_generation)

class attention(nn.Module):
    def __init__(self,heads, masking = None):
        self.dim_lower = embed_dim//heads
        self.query = nn.Linear(embed_dim, self.dim_lower) # Created query vector
        self.key = nn.Linear(embed_dim, self.dim_lower) # Created key vector
        self.value = nn.Linear(embed_dim, self.dim_lower) # Created value vector
        self.masking = masking

    def forward(self,x):
        self.weights = torch.matmul(self.query(x), self.key(x).T)  ###(number_of tokens,number_of_tokens)

        if self.masking:
            self.weights = torch.tril(self.weights, diagonal = 0)
            self.weights.masked_fill_(self.weights == 0, float('-inf'))

        self.weights = F.softmax(self.weights, dim=1)/np.sqrt(self.dim_lower)

        return self.weights @ self.value(x)

#
#
# class Encoder():
#
#     def __init__(self):
#
#     # First step is implement attention
#     #
#
# def attention(q,k,v):


# class Decoder():
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    with open('input.txt', 'r') as tiny_h:
        text = tiny_h.read()
    # print(text[:90])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
