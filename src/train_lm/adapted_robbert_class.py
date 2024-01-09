"""
@Author StellaVerkijk
This is a pytorch dataloader for a Roberta-based model. 
This dataloader was based on the dataloader from Pieter Delobelle for RobBERT but was adapted to be able to load bigger files.

Minor modifications and expanded documentation by Monika Dobreva
"""


import torch
from torch.utils.data.dataset import Dataset
from tokenizers.processors import RobertaProcessing


class LineByLineTextDatasetRobbert(Dataset):
    """
    A PyTorch Dataset class to read and encode text data line by line for language model training.
    This class is specifically adapted for the RoBERTa tokenizer.

    Attributes:
    block_size (int): Maximum length of the tokenized output. Longer sequences are truncated.
    tokenizer (Tokenizer): Tokenizer used to encode the text data.
    examples (List[str]): List of text lines from the input files.

    Parameters:
    tokenizer: The tokenizer used for encoding the text.
    file_paths (list): List of file paths containing the training data.
    block_size (int): Maximum length of the tokenized output (default: 512).
    """

    def __init__(self, tokenizer, file_paths: list, block_size=512):
        self.block_size = block_size
        self.tokenizer = tokenizer
        # Post-processing to add special tokens required by RoBERTa
        self.tokenizer.post_processor = RobertaProcessing(
            ("</s>", self.tokenizer.convert_tokens_to_ids("</s>")),
            ("<s>", self.tokenizer.convert_tokens_to_ids("<s>")),
        )

        self.examples = []
        # Read and store each line of the input files
        for file_path in file_paths:
            print("Processing file:", file_path)
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    if len(line) > 0 and not line.isspace():
                        self.examples.append(line.strip('\n'))

    def __len__(self):
        """
        Returns the number of examples in the dataset.
        """
        return len(self.examples)

    def __getitem__(self, i):
        """
        Retrieves the i-th item from the dataset and encodes it using the tokenizer.

        Parameters:
        i (int): Index of the item to retrieve.

        Returns:
        torch.Tensor: The encoded representation of the i-th text line.
        """
        return torch.tensor(self.tokenizer.encode(self.examples[i])[: self.block_size - 2], dtype=torch.long)
