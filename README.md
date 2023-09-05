> University of Pisa, UNIPI \
> Academic year 2022/23 \
> Master Degree in Computer Science - Artificial Intelligence \
> Authors: [Gaetano Nicassio](https://github.com/nicas83) \
> September, 2023
>

# Named entity recognition.

As final project for Human Language Technologies (HLT) course, I developed a project that extracts knowledge 
from the MultiCoNER II competition. I also compared the quality of the project’s result with the results of MultiCoNER 
competition, available at this url: https://multiconer.github.io/results

The datasets used are too big to be uploaded on this repo. You can download the dataset following these instructions https://multiconer.github.io/dataset \
The default parameters (without execute the gridsearch) are stored in the file model/parameters/model_parameter.json \
The parameters for the grid search are stored in the file model/tuning/hyperparameters.json

### Running the Code

#### Arguments:

```
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
    p.add_argument('--refresh_rate', type=int, help='Refresh rate in tqdm', default=2)
``` 

#### Running

###### Train model

```
python train_model.py ---saved_model_name ner_model_trained.pt --dataset_dir dataset/multiconer/eng/ --train_dataset
en_train.conll --dev_dataset en_dev.conll --test_dataset en_test.conll
--path_saved_model saved_models/
```

###### Evaluate the trained model

```
python evaluate_models.py --saved_model_name ner_model_trained.pt --path_saved_model saved_models/ --dataset_dir dataset/multiconer/eng
--train_dataset en_train.conll --dev_dataset en_dev.conll --test_dataset en_test.conll
```

### Setting up the code environment

```
$ pip install -r requirements.txt
```
