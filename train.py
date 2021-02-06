import argparse
import numpy as np
from model import Model
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from valentines_dataset import ValentinesMessages

MODEL_PATH = "./model_1.pth"
def train(train_loader, validation_loader, model, batch_size, max_epochs, sequence_length):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = []
    train_losses = []
    validation_losses = []
    for epoch in range(max_epochs):
        state_h, state_c = model.init_state(sequence_length)
        train_loss = 0
        val_loss = 0
        for batch, (beginning_words,next_word) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred, (state_h, state_c) = model(beginning_words, (state_h, state_c))
            #what is the format of the y prediction? Is this returned by the forward function?
            loss = criterion(y_pred.transpose(1,2),next_word)
            train_loss+=loss.item()
            state_h = state_h.detach()
            state_c = state_c.detach()
            loss.backward()
            optimizer.step()
            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })
        model.eval()
        with torch.no_grad():
            for batch, (beginning_words, next_word) in enumerate(validation_loader):
                y_pred, (state_h, state_c) = model(beginning_words, (state_h, state_c))
                loss = criterion(y_pred.transpose(1,2),next_word)
                val_loss+=loss.item()
        epochs.append(epoch)
        train_losses.append(train_loss)
        validation_losses.append(val_loss)
    torch.save(model.state_dict(), MODEL_PATH)
    return epochs, train_losses, validation_losses



parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', dest='max_epochs', type=int, default=25)
parser.add_argument('--batch-size', dest='batch_size', type=int, default=256)
parser.add_argument('--sequence-length', dest='sequence_length', type=int, default=4)
args = parser.parse_args()
max_epochs = args.max_epochs
batch_size = args.batch_size
sequence_length = args.sequence_length

dataset = ValentinesMessages(sequence_length)
training_size = round(len(dataset)*0.8)
testing_size = len(dataset)-training_size

train_subset, validation_subset = random_split(dataset, [training_size, testing_size])
train_loader = DataLoader(train_subset, batch_size=100)
validation_loader = DataLoader(validation_subset, batch_size = 100)
model = Model(dataset)
epochs, training_losses, validation_losses = train(train_loader, validation_loader, model, batch_size, max_epochs, sequence_length)
training_history = pd.DataFrame.from_dict({'epoch': epochs, 'loss': training_losses, 'dataset': "train"})
validation_history = pd.DataFrame.from_dict({'epoch': epochs, 'loss': validation_losses, 'dataset': "validation"})
total_loss = pd.concat([training_history, validation_history])
figure = plt.figure(figsize=(10,10))
sns.lineplot(data=total_loss, x="epoch", y="loss", hue="dataset")
plt.savefig('./total_loss.png')


