# Preparations

import pandas as pd
import numpy as np
import re
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures


# Reading in the csv file from the url and creating a dataframe
url = "https://raw.githubusercontent.com/bethancunningham/nlp_2026/main/noms_10s.csv"
new_names = pd.read_csv(url, sep = ";", skiprows = 8, header=0, decimal=",", thousands=".")
df = new_names[new_names["Sexe"] == "H"]
df = df.head(250)
print(df.tail())

# Removing alternative versions of names after slash
df["Nom"] = df["Nom"].str.replace(r"/.*", "", regex=True)

# Renaming frequency column
df.rename(columns={'Rànquing. Freqüència': 'Frequency'}, inplace=True)

# Creating list of names with number of appearances proportionate to frequency/100 and rounded up 
names_list = df.loc[
    df.index.repeat(np.ceil(df["Frequency"] / 100).astype(int)),
    "Nom"
].tolist()

print(len(names_list))


# Getting all unique characters
chars = sorted(list(set(''.join(names_list))))

# Mapping characters to index of integers, with . as 0, and creating reverse mapping of index to character
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)


# Building dataset of blocks of 3 characters
block_size = 3

def build_dataset(words):
  X, Y = [], []
  for w in words:

    #print(w)
    context = [0] * block_size # Padding out context window
    for ch in w + '.':
      ix = stoi[ch] # Converting character to index
      X.append(context) # Storing 3-character context
      Y.append(ix) # Storing next character
      #print(''.join(itos[i] for i in context), '--->', itos[ix])
      context = context[1:] + [ix] # Sliding context window forward by 1 character

  X = torch.tensor(X) # Converting contexts to tensor
  Y = torch.tensor(Y) # Converting next characters to tensor
  print(X.shape, Y.shape)
  return X, Y

# Building training, dev and test sets
import random
random.seed(42)
random.shuffle(names_list)
n1 = int(0.8*len(names_list))
n2 = int(0.9*len(names_list)) # 80 10 10

Xtr, Ytr = build_dataset(names_list[:n1])
Xdev, Ydev = build_dataset(names_list[n1:n2])
Xte, Yte = build_dataset(names_list[n2:])

# Creating initial embedding matrix
C = torch.randn((len(stoi), 2))

g = torch.Generator().manual_seed(123456) # for reproducibility

# Initialising model parameters
C = torch.randn((39, 10), generator=g) # Character embedding matrix - 39 characters by 10 embedding size
W1 = torch.randn((30, 40), generator=g) # Concatenated embeddings (3*10) by 40 neurons - hidden layer weights
b1 = torch.randn(40, generator=g) # 40 neurons - hidden layer biases
W2 = torch.randn((40, 39), generator=g) # 40 neurons, 39 output layer size - hidden layer weights
b2 = torch.randn(39, generator=g) # 39 output layer size - hidden layer biases
parameters = [C, W1, b1, W2, b2]

# Gradient tracking
for p in parameters:
  p.requires_grad = True

# Log-scale range for learning rate search
lre = torch.linspace(-3, 0, 1000) # between 0.001 and 1

# Converting log scale to actual rates
lrs = 10**lre

# Lists for tracking training stats
lri = []
lossi = []
stepi = []
dev_lossi = []
dev_steps = [] 

for i in range(80000): # 80000 iterations

  # Constructing minibatch
  ix = torch.randint(0, Xtr.shape[0], (32,))

  # Forward pass
  emb = C[Xtr[ix]]
  h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, Ytr[ix])
  #print(loss.item())

  # Backward pass (resetting gradients first)
  for p in parameters:
    p.grad = None
  loss.backward()

  # Update
  #lr = lrs[i]
  lr = 0.1 if i < 100000 else 0.01
  for p in parameters:
    p.data += -lr * p.grad

  # Print loss every 1000 steps
  if i % 1000 == 0:
      print(f'step{i}: loss {loss.item():.4f}')
    
  # Evaluating on dev set every 1000 steps
  if i % 1000 == 0:
          with torch.no_grad():
              emb_dev = C[Xdev]
              h_dev = torch.tanh(emb_dev.view(-1, block_size*10) @ W1 + b1)
              logits_dev = h_dev @ W2 + b2
              loss_dev = F.cross_entropy(logits_dev, Ydev)

              dev_lossi.append(loss_dev.log10().item())  # Saving dev loss
              dev_steps.append(i)                        # Saving step number


  # Tracking stats for plotting
  #lri.append(lre[i])
  stepi.append(i)
  lossi.append(loss.log10().item()) # Log loss - easier to visualist

print(loss.item())

# Plotting train vs dev loss to check for overfitting
plt.plot(stepi, lossi, label='Train loss')
plt.plot(dev_steps, dev_lossi, label='Dev loss') 
plt.xlabel('Step')
plt.ylabel('log10(Loss)')
plt.legend()
plt.title('Train vs Dev Loss')
plt.show()

# Evaluating final loss on training set
emb = C[Xtr] 
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2 
loss = F.cross_entropy(logits, Ytr)
print(loss)

# Evaluate final loss on dev set
emb = C[Xdev]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) 
logits = h @ W2 + b2 
loss = F.cross_entropy(logits, Ydev)
print(loss)

# Visualising dimensions 0 and 1 of the embedding matrix C for all characters
plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color='white')
plt.grid('minor')
plt.show()

# Initialising context window with padding tokens for sampling
context = [0] * block_size
C[torch.tensor([context])].shape

# Sampling from the model
g = torch.Generator().manual_seed(12345 + 10)

for _ in range(50):

    out = []
    context = [0] * block_size # Initialising with all ...
    while True:
      emb = C[torch.tensor([context])]
      h = torch.tanh(emb.view(1, -1) @ W1 + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0: # Ending at full stop
        break

    print(''.join(itos[i] for i in out)) # Decoding and printing generated names