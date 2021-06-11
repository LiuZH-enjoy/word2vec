import torch
import torch.utils.data as Data
import numpy as np


def make_data(word_sequence, word2idx, args):
    skip_grams = []
    for idx in range(args.window_size, len(word_sequence) - args.window_size):
        center = word2idx[word_sequence[idx]]  # center word
        context_idx = list(range(idx - args.window_size, idx)) + list(range(idx + 1, idx + args.window_size + 1))  # context word idx
        context = [word2idx[word_sequence[i]] for i in context_idx]
        for w in context:
            skip_grams.append([center, w])
    input_data = []
    output_data = []
    for i in range(len(skip_grams)):
        input_data.append(np.eye(args.vocab_size)[skip_grams[i][0]])
        output_data.append(skip_grams[i][1])
    input_data, output_data = torch.Tensor(input_data), torch.LongTensor(output_data)
    dataset = Data.TensorDataset(input_data, output_data)
    loader = Data.DataLoader(dataset, args.batch_size, True)
    return loader
