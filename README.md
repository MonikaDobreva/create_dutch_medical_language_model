# Creating Dutch Medical Language Models
This directory contains the code for the creation and evaluation of domain-specific Dutch Medical Language models.

It is a modification of Stella Verkijk's repository: https://github.com/cltl-students/verkijk_stella_rma_thesis_dutch_medical_language_model

# Overview
The src folder contains all code and data. Per subfolder, a readme is provided. 
The structure is the following:
```
└───src
│   └───gather_traindata (provides the code used for gathering, filtering and preparing the data used for pre-training in train_lm)
│   └───train_lm (provides the code to pre-train two medical language models: from scratch and extending RobBERT
│   └───ICF_test (provides the code to fine-tune and test language models on a medical classification task)
│   └───similarity_test (provides the code to create a similarity test set from hospital notes and provides the code and data to test language models on this)
│   └───NER_test (provides the code to fine-tune and test language models on named entitiy recognition for general Dutch)
│   └───anonymization (provides the code to anonymize the vocabulary of a language model and test the level of anonymicity of a language model)
```


