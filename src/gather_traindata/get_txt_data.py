"""
Modified version of Stella Verkijk's script for gathering pre-training data for medical language models.

Original author: StellaVerkijk
Modifications by: MonikaDobreva
"""
import traceback
import pandas as pd
import time
import spacy


class GatherData:
    """
    This class processes a specific CSV file containing medical notes. It anonymizes sensitive
    information in the notes and exports them into a text file, suitable for training language models.
    """

    def __init__(self):
        # Load the Dutch language model from spaCy
        self.nlp = spacy.load('nl_core_news_lg')

    def read_csv(self, file_path: str):
        """
        Reads a CSV file and returns a DataFrame.

        Parameters:
        file_path (str): The file path of the CSV file to read.
        """
        return pd.read_csv(file_path, header=0, index_col=None, sep=',', encoding='utf-8')

    def process_note(self, note_text: str):
        """
        Processes and anonymizes a single medical note.

        Parameters:
        note_text (str): The text of the medical note to process.

        Returns:
        str: Anonymized and processed medical note.
        """
        # Process the note text with spaCy
        spacy_doc = self.nlp(note_text)
        # Anonymize the note by replacing names, dates and locations (more tags can be added)
        anonymized_note = str(spacy_doc)
        for entity in reversed(spacy_doc.ents):
            if entity.label_ in ['PERSON', 'GPE', 'DATE']:
                anonymized_note = anonymized_note[:entity.start_char] + entity.label_ + anonymized_note[entity.end_char:]
        return anonymized_note

    def write_chunks(self, text: str, outfile):
        """
        Writes processed text into chunks in the output file.

        Parameters:
        text (str): The processed text to write in chunks.
        outfile: The output file stream to write into.
        """
        # Re-process the anonymized note to split into sentences
        processed_note = self.nlp(text)
        sentences = [sentence for sentence in processed_note.sents]

        # Define the chunk size
        chunk_size = 40
        # Divide the note into chunks of sentences
        note_chunks = [sentences[i * chunk_size:(i + 1) * chunk_size] for i in range((len(sentences) + chunk_size - 1) // chunk_size)]

        for chunk in note_chunks:
            for sentence in chunk:
                # Write the chunk to the output file
                outfile.write(str(sentence) + (' ' if str(sentence).endswith('.') else ''))
            outfile.write('\n')
        return sum([len(sentence) for sentence in sentences])

    def process_data(self, initial_data_file: str, output_data_file: str, column_name: str):
        """
        Processes medical notes from a CSV file, anonymizes the text, and exports it to a text file.

        Parameters:
        initial_data_file (str): The file path of the input CSV file containing medical notes.
        output_data_file (str): The file path for the output text file where the processed and
                                anonymized notes will be saved.
        column_name (str): The name of the column in the CSV file that contains the medical note texts.
        """
        # Record start time for performance metrics
        start_time = time.time()
        # List to store lengths of chunks for analysis
        chunk_lengths = []
        # Counter for the number of processed notes
        processed_notes_count = 0

        # Read the input CSV file
        file_data = self.read_csv(initial_data_file)

        with open(output_data_file, 'a', encoding='utf-8') as outfile:
            for index, row in file_data.iterrows():
                # Extract the note text from each row, based on the provided column name
                note_text = row[column_name]
                processed_notes_count += 1
                try:
                    anonymized_note = self.process_note(note_text)
                    chunk_length = self.write_chunks(anonymized_note, outfile)
                    chunk_lengths.append(chunk_length)
                except Exception as e:
                    print(f"Error processing row {index}: {e}")
                    traceback.print_exc()

        # Calculate and print statistics...
        self.print_statistics(chunk_lengths, processed_notes_count, start_time)

    def print_statistics(self, chunk_lengths: list[int], note_count: int, start_time: float):
        """
        Prints statistics about the processed data.

        Parameters:
        chunk_lengths (list): List of lengths of each processed chunk.
        note_count (int): Total number of processed notes.
        start_time (float): The start time of the processing.
        """
        avg_chunk_length = sum(chunk_lengths) / len(chunk_lengths)
        chunks_over_limit = sum(length > 512 for length in chunk_lengths)

        print(f"Average length per chunk: {avg_chunk_length}")
        print(f"Total amount of chunks: {len(chunk_lengths)}")
        print(f"Amount of chunks larger than 512: {chunks_over_limit}")
        print(f"Largest length of chunk: {max(chunk_lengths)}")
        print(f"Amount of notes: {note_count}")

        # Calculate and print the processing time
        elapsed_time = divmod(time.time() - start_time, 3600)
        print(f"Processing time: {int(elapsed_time[0]):02d}:{int(elapsed_time[1] // 60):02d}:{elapsed_time[1] % 60:.2f}")
