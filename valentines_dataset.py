from torch.utils.data import Dataset
import torch
import pandas  as pd

class  ValentinesMessages(Dataset):
    def __init__(self, sequence_length, dataset_path='valentines_messages.csv'):
        self.sequence_length = sequence_length
        self.dataset = self.load_dataset(dataset_path)
        self.words = self.load_words()

        self.unique_words = self.get_unique_words()
        self.index_to_word = {index: word for index,word in enumerate(self.unique_words)}
        self.words_to_index = {word: index for index,word in enumerate(self.unique_words)}
        self.word_indexes = [self.words_to_index[w] for w in self.words]
        word_index = self.word_indexes[6]

    def load_dataset(self, dataset_path):
        def remove_bullets(valentines_message):
            start = valentines_message.split(' ')[0]
            if start == '🙂' or start == '♥' or start == '–':
                valentines_message = valentines_message.replace(start, '')
            return valentines_message
        dataset = pd.read_csv(dataset_path)
        dataset['message'] = dataset['message'].map(remove_bullets)
        return dataset

    def load_words(self):
        words = self.dataset['message'].str.cat(sep=' ')
        return words.split(' ')

    def get_unique_words(self):
        return list(set(self.words))

    def __len__(self):
        # Words that are input to the model  will not be repeated in the output
        return len(self.word_indexes) - self.sequence_length
    def __getitem__(self, idx):
        return (
            torch.tensor(self.word_indexes[idx:idx+self.sequence_length]),
            torch.tensor(self.word_indexes[idx+1:idx+self.sequence_length+1])
        )
