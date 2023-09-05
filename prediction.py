import torch
from torch import IntTensor, BoolTensor
from transformers import BertTokenizerFast

from configuration.configuration import Configuration


def prediction_mask(words_ids: list) -> list:
    """
    List of boolean values, True if the sub-word is the first one
    :param words_ids:
    :return:
    """
    mask = [False] * len(words_ids)

    pred = None
    for idx, ids in enumerate(words_ids):
        if ids != pred and ids is not None:
            mask[idx] = True
        pred = ids
    return mask


def map_lab2id(list_of_labels, label2id) -> list:
    """
    Mapping a list of labels into a list of label's id
    """
    result = []
    for label in list_of_labels:
        result.append(label2id[label] if label in label2id else label2id["O"])

    return result


def map_id2lab(list_of_ids, id2label) -> list:
    """
    Mapping a list of ids into a list of labels
    """
    result = []
    for label_id in list_of_ids:
        result.append(id2label[label_id] if label_id in id2label else "O")

    return result


def predict(model, sentence: str, conf: Configuration, label2id: dict, id2label: dict):
    model.eval()

    tokenizer = BertTokenizerFast.from_pretrained(conf.bert, do_lower_case=False)
    token_text = tokenizer(sentence)
    input_ids = IntTensor(token_text["input_ids"]).unsqueeze(0)
    input_mask = IntTensor(token_text["attention_mask"]).unsqueeze(0)
    predict_mask = BoolTensor(prediction_mask(token_text.word_ids()))
    label_list = list(label2id)

    with torch.no_grad():
        _, predicted_label_seq_ids = model(input_ids, input_mask)
    valid_predicted = torch.masked_select(predicted_label_seq_ids, predict_mask)
    # valid_label_ids = torch.masked_select(label_ids, predict_mask)
    print(predicted_label_seq_ids[0].tolist())
    # result = [*zip(sentence.split(), predicted_label_seq_ids, predict_mask)]
    #
    # output = []
    # for (token, tag, mask) in result:
    #
    #     if tag:
    #         output.extend(["[", token, "]"])
    #     else:
    #         output.append(token)
    #
    #     if tag and mask:
    #         output.append(tag)
    #
    # print("\n" + " ".join(output) + "\n\n")

    print(valid_predicted)
    print(sentence)
    print(map_id2lab(predicted_label_seq_ids[0].tolist(), id2label))
