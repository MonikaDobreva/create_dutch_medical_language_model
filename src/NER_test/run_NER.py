"""
4 arguments should be passed when running from commandline: modeltype, path_to_model, traindata, evaldata
Example of how to run to test BERTje:
python run_NER.py 'bert', 'GroNLP/bert-base-dutch-cased', 'data/ned_train_text.txt', 'data/ned_testb_text.txt'
"""
import sys
from simpletransformers.ner import NERModel, NERArgs


class Runner:
    """
    This class is designed to test a transformer language model on the CoNLL-2002 NERC task
    (https://www.cnts.ua.ac.be/conll2002/ner/) for Dutch.
    """

    def __init__(self, model_type, model_path, training_data, evaluation_data):
        """
        Initializes the NERTester with model specifications and data paths.

        :param model_type: str, type of the model (e.g., 'bert')
        :param model_path: str, path to the pre-trained model
        :param training_data: str, path to the training data
        :param evaluation_data: str, path to the evaluation data
        """
        self.model_type = model_type
        self.model_path = model_path
        self.training_data = training_data
        self.evaluation_data = evaluation_data

    def run(self):
        """
        Trains the model on the specified training data and evaluates it on the evaluation data.
        Returns the result of the evaluation.
        """
        model_args = NERArgs()
        model_args.labels_list = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        model_args.classification_report = True

        model = NERModel(self.model_type, self.model_path, args=model_args)
        model.train_model(self.training_data)
        result, model_outputs, wrong_preds = model.eval_model(self.evaluation_data)

        return result


# Command-line execution
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python run_NER.py <model_type> <model_path> <training_data> <evaluation_data>")
        sys.exit(1)

    tester = Runner(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    results = tester.run()
    print(results)
