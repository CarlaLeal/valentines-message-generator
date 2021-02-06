import argparse
from model import Model
import numpy as np
import torch
from torch.utils.data import DataLoader
from valentines_dataset import ValentinesMessages
MODEL_PATH = "./model_1.pth"
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
parser.add_argument('--sequence-length', dest='sequence_length', type=int, default=4)
args = parser.parse_args()
sequence_length = args.sequence_length
dataset = ValentinesMessages(sequence_length)
model = Model(dataset)
model.load_state_dict(torch.load(MODEL_PATH))
print(predict(dataset, model, "You are my girl", next_words=20))

