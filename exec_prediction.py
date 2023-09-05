import torch
from transformers import BertModel

from configuration.configuration import Configuration
from model.BERT_CRFClassifier import BERT_CRF_NER
from model.prediction import predict
from utils.parser_utils import parse_args

if __name__ == '__main__':

    args, _ = parse_args()

    conf = Configuration(args)
    conf.show_parameters(["bert"])

    if args.saved_model_name is None:
        raise Exception("Define models!")

    labels_set = {'B-AerospaceManufacturer': 0, 'B-AnatomicalStructure': 1, 'B-ArtWork': 2, 'B-Artist': 3,
                  'B-Athlete': 4, 'B-CarManufacturer': 5, 'B-Cleric': 6, 'B-Clothing': 7, 'B-Disease': 8,
                  'B-Drink': 9, 'B-Facility': 10, 'B-Food': 11, 'B-HumanSettlement': 12, 'B-MedicalProcedure': 13,
                  'B-Medication/Vaccine': 14, 'B-MusicalGRP': 15, 'B-MusicalWork': 16, 'B-ORG': 17, 'B-OtherLOC': 18,
                  'B-OtherPER': 19, 'B-OtherPROD': 20, 'B-Politician': 21, 'B-PrivateCorp': 22, 'B-PublicCorp': 23,
                  'B-Scientist': 24, 'B-Software': 25, 'B-SportsGRP': 26, 'B-SportsManager': 27, 'B-Station': 28,
                  'B-Symptom': 29, 'B-Vehicle': 30, 'B-VisualWork': 31, 'B-WrittenWork': 32,
                  'I-AerospaceManufacturer': 33, 'I-AnatomicalStructure': 34, 'I-ArtWork': 35, 'I-Artist': 36,
                  'I-Athlete': 37, 'I-CarManufacturer': 38, 'I-Cleric': 39, 'I-Clothing': 40, 'I-Disease': 41,
                  'I-Drink': 42, 'I-Facility': 43, 'I-Food': 44, 'I-HumanSettlement': 45, 'I-MedicalProcedure': 46,
                  'I-Medication/Vaccine': 47, 'I-MusicalGRP': 48, 'I-MusicalWork': 49, 'I-ORG': 50,
                  'I-OtherLOC': 51, 'I-OtherPER': 52, 'I-OtherPROD': 53, 'I-Politician': 54, 'I-PrivateCorp': 55,
                  'I-PublicCorp': 56, 'I-Scientist': 57, 'I-Software': 58, 'I-SportsGRP': 59, 'I-SportsManager': 60,
                  'I-Station': 61, 'I-Symptom': 62, 'I-Vehicle': 63, 'I-VisualWork': 64, 'I-WrittenWork': 65,
                  'O': 66, 'X': 67, '[CLS]': 68, '[SEP]': 69}

    # Give a label returns id : label --> id
    label2id: dict = {k: v for v, k in enumerate(sorted(labels_set))}
    # Give id returns a label : id --> label
    id2label: dict = {v: k for v, k in enumerate(sorted(labels_set))}

    paths = args.dataset_dir

    # load the saved model
    bert_model = BertModel.from_pretrained(conf.bert)
    model = BERT_CRF_NER(bert_model, label2id['[CLS]'], label2id['[SEP]'],
                         len(label2id), torch.device(conf.gpu))
    checkpoint = torch.load(conf.path_saved_model + conf.saved_model_name, map_location=conf.gpu)
    pretrained_dict = checkpoint['model_state']
    net_state_dict = model.state_dict()
    pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    net_state_dict.update(pretrained_dict_selected)
    model.load_state_dict(net_state_dict)

    model.to(torch.device(conf.gpu))

    while True:
        sentence = input("Please enter a sentence:\n")
        if sentence == "":
            continue
        if sentence == "exit":
            break
        else:
            predict(model, sentence, conf, label2id, id2label)
