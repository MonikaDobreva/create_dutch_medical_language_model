import unittest
from io import StringIO
from gather_persons import GatherData
import pandas as pd


class TestGatherData(unittest.TestCase):
    def setUp(self):
        self.gather_data = GatherData()

    def test_read_csv_valid_file(self):
        test_csv_data = "ID,Note\n1,This is a test note."
        test_csv_file = StringIO(test_csv_data)
        result = self.gather_data.read_csv(test_csv_file)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['Note'], 'This is a test note.')

    def test_process_note_anonymization(self):
        test_note = "Dr. Smith visited Amsterdam on 2020-01-01."
        expected_anonymized_note = "Dr. PERSON visited GPE on DATE."
        anonymized_note = self.gather_data.process_note(test_note)
        self.assertEqual(anonymized_note, expected_anonymized_note)

    def test_write_chunks_output_format(self):
        test_text = "Sentence one. Sentence two. Sentence three."
        output_file = StringIO()
        self.gather_data.write_chunks(test_text, output_file)
        output_file.seek(0)
        content = output_file.read()
        self.assertIn("Sentence one.", content)
        self.assertIn("Sentence two.", content)

    def test_process_data_valid_input(self):
        input_csv_data = "ID,Note\n1,Dr. Smith visited Amsterdam on 2020-01-01."
        output_file = StringIO()
        input_csv_file = StringIO(input_csv_data)
        self.gather_data.process_data(input_csv_file, output_file, "Note")
        output_file.seek(0)
        content = output_file.read()
        self.assertIn("PERSON", content)
        self.assertIn("GPE", content)
        self.assertIn("DATE", content)

    def test_print_statistics_output(self):
        chunk_lengths = [10, 20, 30]
        note_count = 3
        start_time = 0
        with unittest.mock.patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            self.gather_data.print_statistics(chunk_lengths, note_count, start_time)
            output = mock_stdout.getvalue()
            self.assertIn("Average length per chunk: 20.0", output)
            self.assertIn("Total amount of chunks: 3", output)


if __name__ == '__main__':
    unittest.main()
