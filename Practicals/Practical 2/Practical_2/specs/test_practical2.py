import unittest

import pandas as pd


class TestPractical(unittest.TestCase):

    original_file = './specs/SensorData_question1.csv'
    
    def setUp(self):
        self.df_q1 = pd.read_csv('./output/question1_out.csv')
        self.df_q2 = pd.read_csv('./output/question2_out.csv')

    def tearDown(self):
        self.df_q1 = None
        self.df_q2 = None

    def test_question1_columns_ok(self):
        self.assertListEqual(['Input' + str(x) for x in range(1, 13)] + [
            'Original Input3', 'Original Input12', 'Average Input'],
                             list(self.df_q1.columns))

    def test_question1_shape_ok(self):
        self.assertEqual(199, self.df_q1.shape[0])
        self.assertEqual(15, self.df_q1.shape[1])

    def test_question1_original_input3(self):
        original_data = pd.read_csv(self.original_file)
        original_column = original_data['Input3'].values.tolist()

        self.assertListEqual(original_column,
                             self.df_q1['Original Input3'].values.tolist())

    def test_question1_original_input12(self):
        original_data = pd.read_csv(self.original_file)
        original_column = original_data['Input12'].values.tolist()

        self.assertListEqual(original_column,
                             self.df_q1['Original Input12'].values.tolist())

    def test_question1_input3_normalization(self):
        original_data = pd.read_csv(self.original_file)
        col = original_data['Input3']
        col = (col - col.mean()) / col.std()

        for i, j in zip(col.values.tolist(),
                        self.df_q1['Input3'].values.tolist()):
            self.assertAlmostEqual(i, j, places=5)

    def test_question1_input12_normalization(self):
        original_data = pd.read_csv(self.original_file)
        col = original_data['Input12']
        col = (col - col.min()) / (col.max() - col.min())

        col = col.round(decimals=3)
        self.df_q1 = self.df_q1.round(decimals=3)

        self.assertListEqual(col.values.tolist(),
                             self.df_q1['Input12'].values.tolist())

    def test_question2_columns_ok(self):
        cols = list(self.df_q2.columns)

        original_cols = [193913, 297392, 298062, 383188, 283315, 296448,
                         878280, 377461, 325182, 868304, 1469292, 1470048,
                         756401, 379708, 207274, 1435862, 461425, 357031,
                         769657, 812105, 47475, 767183, 241412, 810057,
                         183337, 839552, 786084, 785793, 796258, 486110,
                         489489, 1471841, 878652, 755145, 626502, 814526,
                         143306, 344134, 866702, 491565, 770059, 770394,
                         629896, 624360, 52076, 43733, 44563, 713922, 609663,
                         840942, 80109, 782811, 814260, 784224, 841641,
                         244618, 295985, 244637, 417226]

        produced_cols1 = ['pca' + str(i) + '_width' for i in range(22)]
        produced_cols2 = ['pca' + str(i) + '_freq' for i in range(22)]

        for c in original_cols:
            self.assertIn(str(c), cols)

        for c in produced_cols1:
            self.assertIn(c, cols)

        for c in produced_cols2:
            self.assertIn(c, cols)

    def test_question2_shape_ok(self):
        self.assertEqual(88, self.df_q2.shape[0])
        self.assertEqual(103, self.df_q2.shape[1])

    def test_question2_bins_freq_ok(self):
        for i in range(22):
            self.assertEqual(10, len(
                self.df_q2['pca' + str(i) + '_freq'].value_counts()))

    def test_question2_bins_width_ok(self):
        for i in [0, 2, 5, 6, 19, 21]:
            self.assertEqual(9,
                             len(self.df_q2[f'pca{i}_width'].value_counts()))

        for i in [1, 3, 4, 7, 8, 10, 12, 13, 15, 16, 17, 18, 20]:
            self.assertEqual(10,
                             len(self.df_q2[f'pca{i}_width'].value_counts()))

        self.assertEqual(7, len(self.df_q2['pca9_width'].value_counts()))
        self.assertEqual(8, len(self.df_q2['pca11_width'].value_counts()))
        self.assertEqual(7, len(self.df_q2['pca14_width'].value_counts()))


if __name__ == '__main__':
    unittest.main()
