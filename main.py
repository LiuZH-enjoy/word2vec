import torch
import argparse
import train
import preprocess
import module
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, help='batch大小(默认为8)')
parser.add_argument('--epochs', type=int, default=2000, help='epoch数(默认为2000)')
parser.add_argument('--embedding_size', type=int, default=2, help='隐藏层维度(默认为2)')
parser.add_argument('--window_size', type=int, default=2, help='背景词长度(默认为2)')
args = parser.parse_args()
args.device = torch.device('cpu')
args.cuda = False

dtype = torch.FloatTensor

sentences = ["jack like dog", "jack like cat", "jack like animal",
  "dog cat animal", "banana apple cat dog like", "dog fish milk like",
  "dog cat animal like", "jack like apple", "apple like", "jack like banana",
  "apple banana jack movie book music like", "cat dog hate", "cat dog like"]

word_sequence = " ".join(sentences).split() # ['jack', 'like', 'dog', 'jack', 'like', 'cat', 'animal',...]
vocab = list(set(word_sequence)) # build words vocabulary
word2idx = {w: i for i, w in enumerate(vocab)} # {'jack':0, 'like':1,...}
args.vocab_size=len(vocab)

model = module.Word2vec(args)
loader = preprocess.make_data(word_sequence, word2idx, args)
train.train(loader, model, args)


for i, label in enumerate(vocab):
  W, WT = model.parameters()
  x,y = float(W[i][0]), float(W[i][1])
  plt.scatter(x, y)
  plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()