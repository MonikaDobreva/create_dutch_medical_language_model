"""
original author: StellaVerkijk
modifications by: MonikaDobreva
"""

import pandas as pd
from random import sample
from pathlib import Path


class Masker:
    """
    This class processes a text file to create a dataset for anonymization testing.
    It extracts sentences containing the token 'PERSON', replaces it with '<mask>', and stores the sentences.

    It could be expanded to mask more tokens such as 'DATE'
    """

    def __init__(self, path_textfile, output_directory):
        """
        Initializes the Masker with the file path and output directory.

        :param path_textfile: str, path to the input text file.
        :param output_directory: str, directory to save the output file.
        """
        self.path_textfile = path_textfile
        self.output_directory = output_directory

    def load_and_process_sentences(self):
        """
        Loads the text file, processes sentences containing 'PERSON', and returns a filtered list of sentences.
        It looks for only one instance of <mask> within a sentence, this could be changed.
        :return: list of processed sentences.
        """
        sentences = []
        with open(self.path_textfile, 'r', encoding='utf-8') as infile:
            for line in infile:
                sentences.extend(self.process_line(line))
        return [sentence for sentence in sentences if 30 <= len(sentence) <= 120 and sentence.count('<mask>') == 1]

    @staticmethod
    def process_line(line):
        """
        Processes a line by splitting into sentences, replacing 'PERSON' with '<mask>', and filtering based on length.
        :param line: str, a line from the input file.
        :return: list of processed sentences from the line.
        """
        return [sen.replace('PERSON', '<mask>').strip() + '.'
                for sen in line.split('. ')
                if 'PERSON' in sen and len(sen) > 30]

    def create_dataset(self):
        """
        Creates a dataset of 40 sentences and saves it to a CSV file.
        Sentence amount can be changed.
        """
        processed_sentences = self.load_and_process_sentences()
        selected_sentences = sample(processed_sentences, min(40, len(processed_sentences)))
        df = pd.DataFrame({'sentences': selected_sentences, 'guessed_tokens': 0})
        df.to_csv(Path(self.output_directory) / "anonymized_sentences.csv", sep=';', index=False)


# # Example usage
# path_to_textfile = "path/to/textfile.txt"
# output_directory = "path/to/output"
# masker = Masker(path_to_textfile, output_directory)
# masker.create_dataset()
