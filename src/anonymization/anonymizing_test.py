import pandas as pd
from transformers import pipeline, RobertaTokenizer, RobertaForMaskedLM
from collections import Counter


class Masker:
    """
    This class uses a trained language model to predict masked names in sentences.
    """
    def __init__(self, model_path, data_file):
        """
        Initializes the predictor with the model path and data file.

        :param model_path: str, path to the pretrained model.
        :param data_file: str, path to the data file with sentences.
        """
        print("Loading model...")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForMaskedLM.from_pretrained(model_path)
        self.fill_mask = pipeline('fill-mask', model=self.model, tokenizer=self.tokenizer)
        self.data_file = data_file

    def get_sentences(self):
        """
        Loads sentences from the data file.
        :return: list of sentences.
        """
        print("Getting sentences...")
        df = pd.read_csv(self.data_file, delimiter=';')
        return df['sentences'].tolist()

    def make_predictions(self, sentences):
        """
        Makes predictions for each sentence.
        :param sentences: list of sentences.
        :return: list of dictionaries with sentences and predictions.
        """
        print("Making predictions...")
        predictions_list = []
        for sen in sentences:
            predictions = self.fill_mask(sen, top_k=40)
            tokens = [pred['token_str'] for pred in predictions]
            predictions_list.append({'sen': sen, 'predictions': tokens})
        return predictions_list

    def count_predictions(self, predictions_list):
        """
        Counts the frequency of each prediction.
        :param predictions_list: list of dictionaries with sentences and predictions.
        :return: DataFrame with predictions and their counts.
        """
        print("Counting predictions...")
        all_predictions = [item for pred_dict in predictions_list for item in pred_dict['predictions']]
        counts = Counter(all_predictions)
        return pd.DataFrame(list(counts.items()), columns=['Prediction', 'times predicted']).sort_values(by='times predicted', ascending=False)

    def save_predictions(self, df, output_file):
        """
        Saves the predictions to a CSV file.
        :param df: DataFrame with predictions.
        :param output_file: str, path to the output file.
        """
        df.to_csv(output_file, sep=';', index=None)
        print(f"Predictions saved to {output_file}")


# # Example Usage
# model_path = "../../processing/from_scratch_final_model_new_vocab"
# data_file = "anon_specific_testset_eval.csv"
# output_file = "predictions_from_scratch_unseen_data.csv"
#
# predictor = Masker(model_path, data_file)
# sentences = predictor.get_sentences()
# predictions_list = predictor.make_predictions(sentences)
# predictions_df = predictor.count_predictions(predictions_list)
# predictor.save_predictions(predictions_df, output_file)
