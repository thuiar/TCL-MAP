import os
import csv
import sys
import pickle
from transformers import BertTokenizer
import numpy as np


def get_t_data(args, data_args):
    
    t_data, cons_text_feats, condition_idx = get_data(args, data_args)

    return t_data, cons_text_feats, condition_idx

def get_data(args, data_args):

    processor = DatasetProcessor(args)
    data_path = data_args['data_path']

    train_examples = processor.get_examples(data_path, 'train') 
    train_feats, train_cons_text_feats, train_condition_idx = get_backbone_feats(args, data_args, train_examples)

    dev_examples = processor.get_examples(data_path, 'dev')
    dev_feats, dev_cons_text_feats, dev_condition_idx = get_backbone_feats(args, data_args, dev_examples)

    test_examples = processor.get_examples(data_path, 'test')
    test_feats, test_cons_text_feats, test_condition_idx = get_backbone_feats(args, data_args, test_examples)
    outputs = {
        'train': train_feats,
        'dev': dev_feats,
        'test': test_feats
    }
    cons_text_feats = {
        'train': train_cons_text_feats,
        'dev': dev_cons_text_feats,
        'test': test_cons_text_feats
    }

    condition_idx = {
        'train': train_condition_idx,
        'dev': dev_condition_idx,
        'test': test_condition_idx
    }

    return outputs, cons_text_feats, condition_idx

def get_backbone_feats(args, data_args, examples):

    tokenizer = BertTokenizer.from_pretrained(args.text_backbone, do_lower_case=True)   
    
    data_args['prompt_len'] = args.prompt_len
    data_args['label_len'] = args.label_len
    
    features, cons_features, condition_idx, args.max_cons_seq_length = convert_examples_to_features(args, examples, data_args, tokenizer)     
    features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in features]
    cons_features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in cons_features]
    return features_list, cons_features_list, condition_idx

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    @classmethod
    def _read_pkl(cls, input_file, quotechar=None):
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
        lines = [] 
        for k, v in data.items():
            line = k.split('_')
            line.append(v)
            lines.append(line)
        return lines

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class DatasetProcessor(DataProcessor):

    def __init__(self, args):
        super(DatasetProcessor).__init__()
        
        if args.dataset in ['MIntRec']:
            self.select_id = 3
            self.label_id = 4
        elif args.dataset in ['MELD']:
            self.select_id = 2
            self.label_id = 3
        
    def get_examples(self, data_dir, mode):
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'dev':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
        elif mode == 'all':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "all.tsv")), "all")
        

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue

            guid = "%s-%s" % (set_type, i)
            text_a = line[self.select_id]
            label = line[self.label_id]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

def convert_examples_to_features(args, examples, data_args, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
        
    max_seq_length = data_args['max_seq_len']
    label_len = data_args['label_len']
    features = []
    cons_features = []
    condition_idx = []
    prefix = ['MASK'] * data_args['prompt_len']

    max_cons_seq_length = max_seq_length + len(prefix) + label_len
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        if args.dataset in ['MIntRec']:
            condition = tokenizer.tokenize(example.label)
        elif args.dataset in ['MELD']:
            condition = tokenizer.tokenize(data_args['bm']['label_maps'][example.label])

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # construct augmented sample pair
        cons_tokens = ["[CLS]"] + tokens_a + prefix + condition + (label_len - len(condition)) * ["MASK"] + ["[SEP]"]
        tokens = ["[CLS]"] + tokens_a + prefix + label_len * ["[MASK]"] + ["[SEP]"]

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        cons_inputs_ids = tokenizer.convert_tokens_to_ids(cons_tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_cons_seq_length - len(input_ids))
        input_ids += padding
        cons_inputs_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_cons_seq_length
        assert len(cons_inputs_ids) == max_cons_seq_length
        assert len(input_mask) == max_cons_seq_length
        assert len(segment_ids) == max_cons_seq_length
        # record the position of prompt
        condition_idx.append(1 + len(tokens_a) + len(prefix))


        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids)
                        )
        
        cons_features.append(
            InputFeatures(input_ids=cons_inputs_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids)
                        )
    return features, cons_features, condition_idx, max_cons_seq_length

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop()