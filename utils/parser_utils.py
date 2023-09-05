import argparse


def parse_args():
    p = argparse.ArgumentParser(description='Model configuration.', add_help=True)

    p.add_argument('--dataset_dir', type=str, nargs='+',
                   help='Dataset used for training', default='.')

    p.add_argument('--train_dataset', type=str, nargs='+',
                   help='Dataset used for training', default=None)

    p.add_argument('--dev_dataset', type=str, nargs='+',
                   help='Dataset used for validation', default=None)

    p.add_argument('--test_dataset', type=str, nargs='+',
                   help='Dataset used for test the final model', default=None)

    p.add_argument('--hyperparam_file', type=str, nargs='+',
                   help='Dataset used for training', default='.')

    p.add_argument('--saved_model', type=str, nargs='+',
                   help='Model trained ready to evaluate or use, if list, the order must follow the same of '
                        'datasets',
                   default=None)

    p.add_argument('--saved_model_name', type=str,
                   help='Name to give to a trained model', default=None)

    p.add_argument('--path_saved_model', type=str,
                   help='Directory to save the model', default=".")

    p.add_argument('--bert', type=str,
                   help='Bert model provided by Huggingface', default="bert-base-cased")

    p.add_argument('--save', type=int,
                   help='set 1 if you want save the model otherwise set 0', default=1)

    p.add_argument('--eval', type=str,
                   help='define the type of evaluation: conlleval or df', default="conlleval")

    p.add_argument('--hyperparam_tuning', type=str,
                   help='(y/n) Define if execute the hyperparameters tuning. If ''n'' use defualt params',
                   default='n')

    p.add_argument('--lr', type=float, help='Learning rate', default=0.001)

    p.add_argument('--momentum', type=float, help='Momentum', default=0.9)

    p.add_argument('--weight_decay', type=float, help='Weight decay', default=0.0002)

    p.add_argument('--batch_size', type=int, help='Batch size', default=2)

    p.add_argument('--max_epoch', type=int, help='Max number of epochs', default=20)

    p.add_argument('--patience', type=float, help='Patience in early stopping', default=3)

    p.add_argument('--refresh_rate', type=int, help='Refresh rate in tqdm', default=2)

    return p.parse_known_args()
