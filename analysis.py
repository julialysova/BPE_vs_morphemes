import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
import pandas as pd
import random

from transformers import AutoTokenizer
random.seed(42)

# LOAD MORPHEMES
# Loading of data with lemmas and morphemes
data = pd.read_csv('data.csv')
data = data.dropna()
data['morphemes'] = data['morphemes'].apply(lambda x: x.split("/"))
print(data.head(5))

# Creation of lists with morphemes
words_full = data['morphemes'].tolist() # list of words divided into morphemes
morphemes = set([item for sublist in words_full for item in sublist]) # list of unique morphemes
print("Number of morphemes:", len(morphemes))


# LOAD TOKENIZER
# Load pretrained multilingual BERT
model_name = "bert-base-multilingual-cased"
tokenizer2 = AutoTokenizer.from_pretrained(model_name)

# Getting segmentation of words with a tokenizes
predictions_bert = []
for lemma in data['lemma']:
    encoded = tokenizer2.tokenize(lemma) # get segments
    encoded2 = []
    for sub in encoded:
      sub = sub.replace("##", "") # getting rid of extra signs
      encoded2.append(sub)
    predictions_bert.append(encoded2)
print(predictions_bert[:8])

# List of morphemes
morphemes_bert = set([item for sublist in predictions_bert for item in sublist])
print("Number of segments:", len(morphemes_bert))

# COMPARISON OF SEGMENTS AND MORPHEMES
# Function for finding how similar is the tokenizer segmentation to morphemes division in Russian
def exact_match_accuracy(ground_truth, predictions):
    matches = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == pred)
    total = len(ground_truth)
    return matches / total * 100

ground_truth = words_full

accuracy_bert = exact_match_accuracy(ground_truth, predictions_bert)
print(f"Similarity (exact match) between morphemes and BERT tokenization: {accuracy_bert:.2f}%")


# TRAINING MODELS ON DIFFERENT TYPES OF TOKENS

# Function for building a dataset
def build_dataset(words, block_size, stoi):
    """
    Constructs training data for a character-level language model.
    
    Args:
        words (list of str): List of words to process.
        block_size (int): The context window size.
        stoi (dict): Dictionary mapping characters to integer indices.

    Returns:
        X (torch.Tensor): Input tensor of shape (num_samples, block_size).
        Y (torch.Tensor): Target tensor of shape (num_samples,).
    """
    X, Y = [], []
    
    for w in words:
        context = [0] * block_size  # Initialize context with padding (assuming 0 is a special token)
        
        w.append('.')
        for ch in w: # Append '.' as an end-of-word marker
            ix = stoi[ch]  # Convert character to index
            X.append(context)  # Store the current context as input
            Y.append(ix)  # Store the target character index
            
            # Slide the context window forward
            context = context[1:] + [ix]  

    X = torch.tensor(X)  # Convert list to tensor
    Y = torch.tensor(Y)  
    
    return X, Y


# Function for computing final losses on training, validation, and test sets
def compute_loss(X, Y, C, W1, b1, W2, b2):
    emb = C[X]  
    h = torch.tanh(emb.view(-1, block_size * embedding_dim) @ W1 + b1)
    logits = h @ W2 + b2
    return F.cross_entropy(logits, Y)


# MORPHEMES MODEL

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(morphemes))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)

# build the dataset
block_size = 2 # context length: how many characters do we take to predict the next one?
embedding_dim = 10 # this needs to multiply with the block size to give 20



random.shuffle(words_full)
n1 = int(0.8*len(words_full))
n2 = int(0.9*len(words_full))

Xtr, Ytr = build_dataset(words_full[:n1], block_size, stoi)
Xdev, Ydev = build_dataset(words_full[n1:n2], block_size, stoi)
Xte, Yte = build_dataset(words_full[n2:], block_size, stoi)

g = torch.Generator().manual_seed(42) # for reproducibility
C = torch.randn((len(chars) + 1, embedding_dim), generator=g) # Adjust embedding dimension as needed
W1 = torch.randn((block_size * embedding_dim, 200), generator=g) # Input dimension adjusted for block size
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, len(chars) + 1), generator=g) # Output dimension adjusted to match vocabulary size
b2 = torch.randn(len(chars) + 1, generator=g)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
  p.requires_grad = True

lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre

lri = []
lossi = []
stepi = []

for i in range(15000):

  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (64,))

  # forward pass
  emb = C[Xtr[ix]] 
  h = torch.tanh(emb.view(-1, block_size * embedding_dim) @ W1 + b1) 
  logits = h @ W2 + b2 
  loss = F.cross_entropy(logits, Ytr[ix])

  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()

  # update
  lr = 0.1 if i < 100000 else 0.001
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  stepi.append(i)
  lossi.append(loss.log10().item())


plt.plot(stepi, lossi)
plt.show()


loss_te = compute_loss(Xte, Yte, C, W1, b1, W2, b2)
print("Morhpeme model loss:", loss_te)

# sample from the model
g = torch.Generator().manual_seed(42 + 10)

for _ in range(20):

    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      emb = C[torch.tensor([context])] 
      h = torch.tanh(emb.view(1, -1) @ W1 + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break

    print(''.join(itos[i] for i in out))


# SEGMENTS MODEL
    
# build the vocabulary of characters and mappings to/from integers
chars_bert = sorted(list(morphemes_bert))
stoi = {s:i+1 for i,s in enumerate(chars_bert)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)


random.shuffle(predictions_bert)
n1 = int(0.8*len(predictions_bert))
n2 = int(0.9*len(predictions_bert))

Xtr, Ytr = build_dataset(predictions_bert[:n1], block_size, stoi)
Xdev, Ydev = build_dataset(predictions_bert[n1:n2], block_size, stoi)
Xte, Yte = build_dataset(predictions_bert[n2:], block_size, stoi)


g = torch.Generator().manual_seed(42) # for reproducibility
C = torch.randn((len(chars_bert) + 1, embedding_dim), generator=g) # Adjust embedding dimension as needed
W1 = torch.randn((block_size * embedding_dim, 200), generator=g) # Input dimension adjusted for block size
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, len(chars_bert) + 1), generator=g) # Output dimension adjusted to match vocabulary size
b2 = torch.randn(len(chars_bert) + 1, generator=g)
parameters = [C, W1, b1, W2, b2]


for p in parameters:
  p.requires_grad = True

lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre

lri = []
lossi = []
stepi = []


for i in range(15000):

  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (64,))

  # forward pass
  emb = C[Xtr[ix]]
  h = torch.tanh(emb.view(-1, block_size * embedding_dim) @ W1 + b1) 
  logits = h @ W2 + b2 
  loss = F.cross_entropy(logits, Ytr[ix])

  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()

  # update
  lr = 0.1 if i < 100000 else 0.001
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  stepi.append(i)
  lossi.append(loss.log10().item())


plt.plot(stepi, lossi)
plt.show()


loss_te = compute_loss(Xte, Yte, C, W1, b1, W2, b2)
print("BPE model loss:", loss_te)

# GENERATION
# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):

    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      emb = C[torch.tensor([context])] 
      h = torch.tanh(emb.view(1, -1) @ W1 + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break

    print(''.join(itos[i] for i in out))
