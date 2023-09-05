import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class NerDataset(Dataset):
    def __init__(self, examples, tokenizer, label_map, max_seq_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feat = self.example2feature(self.examples[idx], self.tokenizer, self.label_map, self.max_seq_length)
        return feat.input_ids, feat.input_mask, feat.segment_ids, feat.predict_mask, feat.label_ids

    @classmethod
    def example2feature(self, sample, tokenizer, label_map, max_seq_length):
        add_label = 'X'
        # tokenize_count = []
        tokens = ['[CLS]']
        predict_mask = [0]
        label_ids = [label_map['[CLS]']]
        for i, w in enumerate(sample.words):
            # use bertTokenizer to split words
            # 1996-08-22 => 1996 - 08 - 22
            # sheepmeat => sheep ##me ##at
            sub_words = tokenizer.tokenize(w)
            if not sub_words:
                sub_words = ['[UNK]']
            # tokenize_count.append(len(sub_words))
            tokens.extend(sub_words)
            for j in range(len(sub_words)):
                if j == 0:
                    predict_mask.append(1)
                    label_ids.append(label_map[sample.labels[i]])
                else:
                    # '##xxx' -> 'X' (see bert paper)
                    predict_mask.append(0)
                    label_ids.append(label_map[add_label])

        # truncate
        if len(tokens) > max_seq_length - 1:
            print('Example No.{} is too long, length is {}, truncated to {}!'.format(sample.guid, len(tokens),
                                                                                     max_seq_length))
            tokens = tokens[0:(max_seq_length - 1)]
            predict_mask = predict_mask[0:(max_seq_length - 1)]
            label_ids = label_ids[0:(max_seq_length - 1)]
        tokens.append('[SEP]')
        predict_mask.append(0)
        label_ids.append(label_map['[SEP]'])

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        feat = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            predict_mask=predict_mask,
            label_ids=label_ids)

        return feat

    @classmethod
    def pad(cls, batch):
        seqlen_list = [len(sample[0]) for sample in batch]
        maxlen = np.array(seqlen_list).max()

        f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: X for padding
        input_ids_list = torch.LongTensor(f(0, maxlen))
        input_mask_list = torch.LongTensor(f(1, maxlen))
        segment_ids_list = torch.LongTensor(f(2, maxlen))
        predict_mask_list = torch.ByteTensor(f(3, maxlen))
        label_ids_list = torch.LongTensor(f(4, maxlen))

        return input_ids_list, input_mask_list, segment_ids_list, predict_mask_list, label_ids_list


class InputExample(object):
    """A single training/test example for NER."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example(a sentence or a pair of sentences).
          words: list of words of sentence
          labels_a/labels_b: (Optional) string. The label seqence of the text_a/text_b. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        # list of words of the sentence,example: [EU, rejects, German, call, to, boycott, British, lamb .]
        self.words = words
        # list of label sequence of the sentence,like: [B-ORG, O, B-MISC, O, O, O, B-MISC, O, O]
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data.
    result of convert_examples_to_features(InputExample)
    """

    def __init__(self, input_ids, input_mask, segment_ids, predict_mask, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.predict_mask = predict_mask
        self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """
        Reads a BIO data.
        """
        with open(input_file) as f:
            out_lists = []
            entries = f.read().strip().split("\n\n")
            for entry in entries:
                words = []
                ner_labels = []
                pos_tags = []
                bio_pos_tags = []
                for line in entry.splitlines():
                    if not (line.startswith('#')):
                        pieces = line.strip().split()
                        if len(pieces) < 1:
                            continue
                        word = pieces[0]
                        words.append(word)
                        pos_tags.append(pieces[1])
                        bio_pos_tags.append(pieces[2])
                        ner_labels.append(pieces[-1])
                out_lists.append([words, pos_tags, bio_pos_tags, ner_labels])
        return out_lists


class CoNLLDataProcessor(DataProcessor):
    '''
    CoNLL-2003
    '''

    def __init__(self, list_of_labels):
        if list_of_labels:
            general_tokens = ['X', '[CLS]', '[SEP]']  # important to add these tokens on top the list
            list_of_labels[:0] = general_tokens
            self._label_types = list_of_labels
        else:
            self._label_types = ['X', '[CLS]', '[SEP]', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC',
                                 'B-LOC', 'B-ORG', 'B-CW', 'I-CW', 'B-PROD', 'I-PROD', 'B-CORP', 'I-CORP', 'B-GRP',
                                 'I-GRP']
        self._num_labels = len(self._label_types)
        self._label_map = {label: i for i, label in enumerate(sorted(self._label_types))}
        self.idlabel_map: dict = {v: k for v, k in enumerate(sorted(self._label_types))}

    def get_train_examples(self, data_dir, filename):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, filename)))

    def get_dev_examples(self, data_dir, filename):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, filename)))

    def get_test_examples(self, data_dir, filename):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, filename)))

    def get_labels(self):
        return self._label_types

    def get_num_labels(self):
        return self.get_num_labels

    def get_label_map(self):
        return self._label_map

    def get_start_label_id(self):
        return self._label_map['[CLS]']

    def get_stop_label_id(self):
        return self._label_map['[SEP]']

    def get_x_label_id(self):
        return self._label_map['X']

    def _create_examples(self, all_lists):
        examples = []
        for (i, one_lists) in enumerate(all_lists):
            guid = i
            words = one_lists[0]
            labels = one_lists[-1]
            examples.append(InputExample(
                guid=guid, words=words, labels=labels))
        return examples

    def map_id2lab(self, list_of_ids, is_tensor=False) -> list:
        """
        Mapping a list of ids into a list of labels
        """
        result = []
        for label_id in list_of_ids:
            label_id = label_id.item() if is_tensor else label_id
            result.append(self.idlabel_map[label_id])

        return result

    def map_lab2id(self, list_of_labels, is_tensor=False) -> list:
        """
        Mapping a list of labels into a list of label's id
        """
        result = []
        for label in list_of_labels:
            label = label.item() if is_tensor else label
            result.append(self._label_map[label] if label in self._label_map else self._label_map["O"])

        return result


class Datasets(object):
    def __init__(self, train, dev, test):
        self._train = train
        self._dev = dev
        self._test = test

    def get_train_data(self):
        return self._train

    def get_dev_data(self):
        return self._dev

    def get_test_data(self):
        return self._test


class LabelHandler(object):
    def __init__(self, label_list, start_label_id, stop_label_id, unique_labels):
        self._label_list = label_list
        self._start_label_id = start_label_id
        self._stop_label_id = stop_label_id
        self._unique_labels = unique_labels

    def get_label_list(self):
        return self._label_list

    def get_start_label_id(self):
        return self._start_label_id

    def get_stop_label_id(self):
        return self._stop_label_id

    def get_unique_labels(self):
        return self._unique_labels


def preprocess_dataset(input_file, filename="distribution_plot", title_label="Distribution Plot", plot_data=True):
    """Reads a BIO data."""
    count_ner_label, count_words, tags = [], [], []
    with open(input_file) as f:
        sentences = f.read().strip().split("\n\n")
        for entry in sentences:
            for line in entry.splitlines():
                if not (line.startswith('#')):
                    pieces = line.strip().split()
                    if len(pieces) < 1:
                        continue
                    tags.append(pieces[-1])
                    if not pieces[-1] == 'O':
                        count_ner_label.append(pieces[-1])
                        count_words.append(pieces[0])

    print('Number of Senteces:', len(sentences))

    data = {'Words': count_words, 'Tag': count_ner_label}
    data_distr = pd.DataFrame(data)
    unique_tag = set(tags)
    if plot_data:
        tag_distribution = data_distr.groupby(['Tag']).size()
        save_file = tag_distribution.plot(kind='bar', xlabel='Tag', ylabel='Frequency', title=title_label,
                                          figsize=[15, 10]).get_figure()

        save_file.savefig("dataset/preprocessing/" + filename + ".png")
    return list(unique_tag)


def load_datasets(conf, do_lower_case=False, evaluate=False):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(conf.bert, do_lower_case=do_lower_case)

    # pre elaborazione del file, con matplot
    train_label_list, dev_label_list = [], []
    if not evaluate:
        train_label_list = preprocess_dataset(os.path.join(conf.dataset_dir[0], conf.train_dataset[0]),
                                              "train_token_distribution", "NER Tags Distribution - Train Dataset")
        dev_label_list = preprocess_dataset(os.path.join(conf.dataset_dir[0], conf.dev_dataset[0]),
                                            "dev_token_distribution", "NER Tags Distribution - Dev Dataset")
    test_label_list = preprocess_dataset(os.path.join(conf.dataset_dir[0], conf.test_dataset[0]),
                                         "test_token_distribution", "NER Tags Distribution - Test Dataset")

    unique_set = set(train_label_list + dev_label_list + test_label_list)
    list_unique_labels = list(unique_set)

    data_processor = CoNLLDataProcessor(list_of_labels=list_unique_labels)
    label_list = data_processor.get_labels()
    label_map = data_processor.get_label_map()
    start_label_id = data_processor.get_start_label_id()
    stop_label_id = data_processor.get_stop_label_id()

    train_dataset, dev_dataset, test_dataset = Any, Any, Any
    if evaluate:
        test_examples = data_processor.get_test_examples(conf.dataset_dir[0], conf.test_dataset[0])
        test_dataset = NerDataset(test_examples, tokenizer, label_map, max_seq_length=128)
    else:
        train_examples = data_processor.get_train_examples(conf.dataset_dir[0], conf.train_dataset[0])
        dev_examples = data_processor.get_dev_examples(conf.dataset_dir[0], conf.dev_dataset[0])
        train_dataset = NerDataset(train_examples, tokenizer, label_map, max_seq_length=128)
        dev_dataset = NerDataset(dev_examples, tokenizer, label_map, max_seq_length=128)

    all_dataset = Datasets(train_dataset, dev_dataset, test_dataset)
    label_handler = LabelHandler(label_list, start_label_id, stop_label_id, list_unique_labels)

    return all_dataset, label_handler, data_processor
