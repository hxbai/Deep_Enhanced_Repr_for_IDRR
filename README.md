# Deep Enhanced Representation for Implicit Discourse Relation Recognition

This is the code for paper:

Deep Enhanced Representation for Implicit Discourse Relation Recognition

Hongxiao Bai, Hai Zhao    (COLING 2018)

## Usage

We use the processed data from https://github.com/cgpotts/pdtb2.

Put the `pdtb2.csv` to `./data/raw/` first.

Edit the paths of pre-trained word embedding file and ELMo files in `config.py`.

Then prepare the data:

        bash ./prepare_data.sh

For training and evaluating:

        python main.py func splitting

`func` can be `train` or `eval`, and `splitting` is 1 or 2 or 3,
1 for PDTB-Lin 11-way classification, 2 for PDTB-Ji 11-way classification and 3 for 4-way classification.

For example:

        python main.py train 1

means training for PDTB-Lin 11-way classification.

        python main.py eval 2

means evaluating with pre-trained parameters for PDTB-Ji 11-way classification.

## Pre-trained parameters

The pre-trained parameter weights can be downloaded by

[https://drive.google.com/file/d/1cYzVtgA82oZW5N9hz0yIPnH8z2MjHTDW/view?usp=sharing]

put the `weights` directory to `./`.

The results are higher than the reported results in the paper since the reported results are averaged.

## Requirements

        nltk == 3.2.5
        numpy == 1.14.2
        gensim == 3.1.0
        scikit-learn == 0.19.1
        pytorch == 0.3.1
        allennlp == 0.4.1
        tensorboardX == 1.0
