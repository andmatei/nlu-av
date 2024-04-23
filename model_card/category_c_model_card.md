---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/andmatei/nlu-av

---

# Model Card for y44694jk-x20023am-AV

<!-- Provide a quick summary of what the model is/does. -->

This is a classification model that was trained to
      detect whether two pieces of text were written by the same author.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based upon a BERT model that was fine-tuned
      on 30K pairs of texts.

- **Developed by:** Jungwoo Koo and Andrei Matei
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Transformers
- **Finetuned from model [optional]:** bert-base-uncased

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/google-bert/bert-base-uncased
- **Paper or documentation:** https://aclanthology.org/N19-1423.pdf

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

30K pairs of texts drawn from emails, news articles and blog posts. 256 tokens drawn from that texts. This was done with random sampling and this random sampling was conducted in every epochs, so that the train data could obtain the text that was not used in the previous epoch. This helps the model to train the unseen data.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 5e-5
      - train_batch_size: 32
      - eval_batch_size: 32
      - seed: 42
      - num_epochs: 3
      - tokens per text : 256

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 1 hour
      - duration per training epoch: 20 minutes
      - model size: 436MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

A subset of the development set provided, amounting to 6K pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Accuracy
      - Macro Precision
      - Macro Recall
      - Macro F1-score
      - Weighted Macro Precision
      - Weighted Macro Recall
      - Weighted F1-score
      - Matthews Correlation Coefficient

### Results

The model obtained the accuracy of 77.8%, Macro Precision of 78.1%, Macro Recall of 77.8%, Macro F1-score of 77.7%, Weighted Macro Precision of 78.0%, Weighted Macro Recall of 77.8%, Weighted Macro F1-score of 77.8%, and  Matthews Correlation Coefficient of 55.9%.

## Technical Specifications

### Hardware


      - System RAM: at most 12.7 GB
      - GPU RAM: at most 15.0 GB
      - Storage: at most 78.2GB,
      - GPU: T4

### Software


      - Transformers 4.40.0
      - Pytorch 2.2.1+cu121

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any inputs (concatenation of two sequences) longer than
      256 subwords might be truncated by the model.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The hyperparameters were determined by experimentation
      with different values. 
