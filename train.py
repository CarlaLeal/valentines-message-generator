import numpy
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from valentines_dataset import ValentinesMessages

def train(dataset, model, batch_size, max_epochs, sequence_length):
    model.train()
    data_loader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(max_epochs):
        state_h, state_c = model.init_state(sequence_length)
        for batch, (beginning_words,next_words) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred, (state_h, state_c) = model(beginning_words, (state_h, state_c))
            #what is the format of the y prediction? Is this returned by the forward function?
            loss = criterion(y_pred.transpose(1,2),y)
            state_h = state_h.detatch()
            state_c = state_c.detatch()
            loss.backward()
            optimizer.setp()
            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })


def predict(dataset, model, text, next_words=10):
    model.eval()
    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))
    for i in range(0, next_words):
        return

