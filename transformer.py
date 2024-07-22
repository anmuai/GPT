# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)
import numpy as np
embed_dim = 64
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#device = 'cpu'
print(device)
iterations = 600 ## Number training iterations
eval_iters = 200 ## Number of times model loss is calulated
heads = 8
seq_len = 64
batch_size = 64
n_layer = 1
import time

class attention(nn.Module):
    """
    The purpose of this class is to map words to numbers using the attention mechanism
    Attention:
    Helps determine affinities between words using weights
    the weights determines the amount of say a word has on the final representation
    of another word

    Mathematically weights are calculated by multiplying query and key vectors
    after weights are calculated they are multiplied by the value vectoors
    to get the final represnatation of each word
    """

    def __init__(self, heads, masking=True):
        super().__init__()
        self.dim_lower = embed_dim // heads
        self.query = nn.Linear(embed_dim, self.dim_lower)  # Created query vector
        self.key = nn.Linear(embed_dim, self.dim_lower)  # Created key vector
        self.value = nn.Linear(embed_dim, self.dim_lower)  # Created value vector
        self.masking = masking

    def forward(self, x):
        B, T, C = self.key(x).shape
        self.weights = torch.matmul(self.query(x), self.key(x).view(B, C, T))  ###(number_of tokens,number_of_tokens)

        if self.masking:
            self.weights = torch.tril(self.weights, diagonal=0)
            self.weights.masked_fill_(self.weights == 0, float('-inf'))

        self.weights = F.softmax(self.weights / np.sqrt(self.dim_lower), dim=-1)

        return self.weights @ self.value(x)


class mha(nn.Module):
    """
    Implements multiple attentions
    """

    def __init__(self, heads):
        super().__init__()
        self.attn_block = nn.ModuleList([attention(heads, masking=True) for _ in range(heads)])

    def forward(self, x):
        return torch.cat([self.attn_block[i](x) for i in range(len(self.attn_block))], dim=2)


class Block(nn.Module):
    def __init__(self, heads, embed_dim):
        super().__init__()
        self.mha = mha(heads).to(device)
        self.ffwd = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4),
                                  nn.ReLU(),
                                  nn.Linear(embed_dim * 4, embed_dim))

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self,x):

        x = x + self.ln1(self.mha(x))

        x = x + self.ffwd(self.ln2(x))

        return x



class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, heads, seq_len, n_layer):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size,embed_dim)
        self.positional_encoding = nn.Embedding(seq_len,embed_dim)
        ### Attention part

        self.block = nn.Sequential(
            Block(heads,embed_dim),
            Block(heads, embed_dim),
            Block(heads, embed_dim),
            Block(heads, embed_dim)
        )

        ### This acts on a per token basis

        self.ll1 = nn.Linear(embed_dim,vocab_size)

    def forward(self, x, targets = None):
        x = self.embedding_table(x) # B T C
        B,T,C = x.shape
        self.pos_embed = self.positional_encoding(torch.tensor([i for i in range(T)]).to(device))
        x = x + self.pos_embed
        x = self.block(x)
        x = self.ll1(x)

        if targets == None:
            loss = None
        else:
            B,T,C = x.shape
            logits  = x.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return x,loss

    def generate(self,idx, max_new_tokens):
        """
        idx is batch,time dimensions

        :param idx:
        :return:
        """
        for _ in range(max_new_tokens):

            logits,loss = self(idx[:,-seq_len:])  ##(B,T,C)
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

N = 0.9

train_data = encoded_data[: int(N * len(data))]
valid_data = encoded_data[int(N * len(data)) :]

def get_batch(split):
    """
    samples a batch of indices from the data
    with dimensions (batch_size,seq_len)
    here seq_len is equal to the sequence length
    :param split:
    :return: tuple of sampled indices and targets
    """
    data = train_data if split == 'train' else valid_data
    idx = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in idx])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in idx])
    x = x.to(device)
    y = y.to(device)
    return x,y


xb,yb = get_batch(split = 'train')
print(xb.shape)


bigram = BigramLanguageModel(len(encode),heads, seq_len, n_layer)
m = bigram.to(device)
logits,loss = m(xb,yb)

print(logits.shape)
print(loss)
generation = bigram.generate(torch.zeros((1,1), dtype = torch.long).to(device), 250)


decoded_generation = ''.join([decode[int(idx)] for idx in generation[0]])

print(decoded_generation)

optimizer = torch.optim.AdamW(bigram.parameters(), lr = 1e-3)
#
# ### Training code
#
# batch_size = 32
# iterations = 20000
#
#
#
@torch.no_grad()
def compute_loss(model,number_times):
    """
    @torch.no_grad disables the tensor.backward
    It disables the calculation of computation graphs
    which saves memory
    model.eval(): Puts the model in evaluation mode
    so dropout and other layers are disabled especially
    during training

    :param model:
    :return:
    """
    model.eval()
    train_loss_array = []
    valid_loss_array = []
    for i in range(number_times):
        train_xb,train_yb = get_batch(split = 'train')
        val_xb, val_yb = get_batch(split ='valid')
        _, train_loss = model(train_xb,train_yb)
        train_loss_array.append(train_loss.item())
        _, valid_loss = model(val_xb, val_yb)
        valid_loss_array.append(valid_loss.item())
    model.train()
    return torch.mean(train_loss),torch.mean(valid_loss)

start = time.time()
for steps in range(iterations):
    xb,yb  = get_batch('train')
    if steps >= 100 and steps % 100 == 0:
        train_loss, valid_loss = compute_loss(m.to(device), eval_iters)
        print(f"Iteration no: {steps} >>> Training Loss{train_loss} >>> Validation loss {valid_loss}")
    logits, loss = bigram(xb,yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

end = time.time()

print('############## Time Taken ############')
print(end - start)

print("After Training")

generation = bigram.generate(torch.zeros((1,1), dtype = torch.long).to(device), 250)

decoded_generation = ''.join([decode[int(idx)] for idx in generation[0]])

print(decoded_generation)
#
# compute_loss(bigram)

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
