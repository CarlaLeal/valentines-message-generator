import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from model import Model
from valentines_dataset import ValentinesMessages

def train(dataset, model, batch_size, max_epochs, sequence_length):
    model.train()
    data_loader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(max_epochs):
        state_h, state_c = model.init_state(sequence_length)
        for batch, (beginning_words,next_word) in enumerate(data_loader):
            optimizer.zero_grad()
            y_pred, (state_h, state_c) = model(beginning_words, (state_h, state_c))
            #what is the format of the y prediction? Is this returned by the forward function?
            loss = criterion(y_pred.transpose(1,2),next_word)
            state_h = state_h.detach()
            state_c = state_c.detach()
            loss.backward()
            optimizer.step()
            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })


def predict(dataset, model, text, next_words=10):
    model.eval()
    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))
    for i in range(0, next_words):
        input_words = torch.tensor([[dataset.words_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(input_words, (state_h, state_c))
        last_word_logits = y_pred[0][-1]
        # don't understand, I guess for this to run then p has to be an integer
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])
    return words

parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', dest='max_epochs', type=int, default=1)
parser.add_argument('--batch-size', dest='batch_size', type=int, default=256)
parser.add_argument('--sequence-length', dest='sequence_length', type=int, default=4)
args = parser.parse_args()
max_epochs = args.max_epochs
batch_size = args.batch_size
sequence_length = args.sequence_length

dataset = ValentinesMessages(sequence_length)
training_size = round(len(dataset)*0.8)
testing_size = len(dataset)-training_size

train_subset, test_subset = random_split(dataset, [training_size, testing_size])
model = Model(dataset)
train(dataset, model, batch_size, max_epochs, sequence_length)
print(predict(dataset, model, "You are my girl", next_words=20))
