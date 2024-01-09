# Collecting pre-training data
This folder contains all scripts that were used for the gathering and pre-processing of the pre-training data for the Dutch medical language models.

# Data
The data used in dummy_data.csv is fictional, created for testing purposes.

# Scripts
_get_txt_data_ gathers pre-training data for the creation of the domain-specific medical language models.
It loads csv's, adapts the row that contains the note by anonymizing it and dividing it in chunks and then exports it to a .txt file.
The file, created by Stella Verkijk is reworked to follow best software development practices and was made more abstract so the functionality can be reused.

The folder filter_out_unwanted_data contains two scripts that are not used for this research.
