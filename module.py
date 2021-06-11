import torch.nn as nn
import torch


class Word2vec(nn.Module):
    def __init__(self, args):
        super(Word2vec, self).__init__()

        self.W = nn.Parameter(torch.randn(args.vocab_size, args.embedding_size)).type(torch.FloatTensor)
        self.V = nn.Parameter(torch.randn(args.embedding_size, args.vocab_size)).type(torch.FloatTensor)
        
    def forward(self, X):
        # X : [batch_size, voc_size] one-hot
        # torch.mm only for 2 dim matrix, but torch.matmul can use to any dim
        hidden_layer = torch.mm(X, self.W) # hidden_layer : [batch_size, embedding_size]
        output_layer = torch.mm(hidden_layer, self.V) # output_layer : [batch_size, vocab_size]
        return output_layer





