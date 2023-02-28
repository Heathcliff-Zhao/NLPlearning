import re
from random import *

import config


class data_processing:
    def __init__(self):
        self.segment_ids = None
        self.vocab_size = None
        self.input_ids = None
        self.batch = None
        self.token_list = None
        self.word_dict = None
        self.number_dict = None
        self.word_list = None
        self.sentences = None
        self.text = config.text

    def remove_special(self):
        self.sentences = re.sub("[.,!?\\-]", '', self.text.lower()).split('\n')

    def get_wordlist(self):
        self.word_list = list(set(" ".join(self.sentences).split()))

    def get_worddict(self):
        self.word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
        for index, word in enumerate(self.word_list):
            self.word_dict[word] = index + 4

    def get_vocabsize(self):
        self.vocab_size = len(self.word_dict)

    def get_numberdict(self):
        self.number_dict = {index: word for index, word in enumerate(self.word_dict)}

    def get_tokenlist(self):
        self.token_list = list()
        for sentence in self.sentences:
            self.token_list.append([self.word_dict[word] for word in sentence.split()])

    def make_batch(self):
        self.batch = []
        positive = negative = 0
        while positive != config.batch_size / 2 or negative != config.batch_size / 2:
            tokens_a_index, tokens_b_index = randrange(len(self.sentences)), randrange(
                len(self.sentences))
            tokens_a, tokens_b = self.token_list[tokens_a_index], self.token_list[tokens_b_index]
            self.input_ids = [self.word_dict['[CLS]']] + tokens_a + [self.word_dict['[SEP]']] \
                             + tokens_b + [self.word_dict['[SEP]']]
            self.segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)
            n_pred = min(config.max_pred, max(1, int(round(len(self.input_ids) * 0.15))))
            cand_masked_pos = [i for i, token in enumerate(self.input_ids)
                               if token != self.word_dict['[CLS]'] and token != self.word_dict['[SEP]']]
            shuffle(cand_masked_pos)
            masked_tokens, masked_pos = [], []
            for pos in cand_masked_pos[:n_pred]:
                masked_pos.append(pos)
                masked_tokens.append(self.input_ids[pos])
                chance=random()
                if chance < 0.8:
                    self.input_ids[pos] = self.word_dict['[MASK]']
                elif chance > 0.9:
                    index = randint(0, self.vocab_size - 1)
                    self.input_ids[pos] = self.word_dict[self.number_dict[index]]
            n_pad = config.maxlen - len(self.input_ids)
            self.input_ids.extend([0] * n_pad)
            self.segment_ids.extend([0] * n_pad)
            if config.max_pred > n_pred:
                n_pad = config.max_pred - n_pred
                masked_tokens.extend([0] * n_pad)
                masked_pos.extend([0] * n_pad)
            if tokens_a_index + 1 == tokens_b_index and positive < config.batch_size / 2:
                self.batch.append([self.input_ids, self.segment_ids, masked_tokens, masked_pos, True])
                positive += 1
            elif tokens_a_index + 1 != tokens_b_index and negative < config.batch_size / 2:
                self.batch.append([self.input_ids, self.segment_ids, masked_tokens, masked_pos, False])
                negative += 1

    def preprocess(self):
        self.remove_special()
        self.get_wordlist()
        self.get_worddict()
        self.get_vocabsize()
        self.get_numberdict()
        self.get_tokenlist()
        self.make_batch()
        return self.batch
