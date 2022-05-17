import unittest

import pandas as pd


class TestPractical(unittest.TestCase):

    def setUp(self):
        self.df_q1 = pd.read_csv('./output/question1_out.csv')
        self.df_q2 = pd.read_csv('./output/question2_out.csv')

    def tearDown(self):
        self.df_q1 = None
        self.df_q2 = None

    def test_question1_columns_ok(self):
        """Columns in the output file should match the original columns

        This test method checks that the columns in the output file match the
        original columns. Also the order should match the original file!

        """
        cols = list(self.df_q1.columns)
        self.assertListEqual(['horsepower', 'mpg', 'cylinders',
                              'displacement', 'weight', 'acceleration',
                              'model year', 'origin', 'car name'], cols)

    def test_question1_shape_ok(self):
        """Shape of the output dataset should match the original shape

        This test method checks the shape of the output file matches the shape
        of the original file.

        """
        output_shape = self.df_q1.shape
        self.assertEqual(398, output_shape[0])
        self.assertEqual(9, output_shape[1])

    def test_question1_no_missing_horsepower_values(self):
        """Horsepower column should have no missing value

        This test method checks the output file for missing values on the
        'horsepower' column. No missing values should be found there!

        """
        missing = self.df_q1['horsepower'].isnull().sum()
        self.assertEqual(0, missing)

    def test_question1_no_missing_origin_values(self):
        """Origin column should have no missing value

        This test method checks the output file for missing values on the
        'origin' column. No missing values should be found there!

        """
        missing = self.df_q1['origin'].isnull().sum()
        self.assertEqual(0, missing)

    def test_question1_horsepower_mean(self):
        """Mean value for horsepower should be unchanged

        This test method computes the mean of horsepower from the output file
        and checks that it matches the expected mean value. It is, of course,
        equal to the mean of the column in the original file.

        """
        new_horsepower_mean = self.df_q1['horsepower'].mean()
        self.assertAlmostEqual(104.2670157, new_horsepower_mean)

    def test_question1_origin_mean(self):
        """Mean value for origin should be changed

        This test method computes the mean of origin from the output file and
        checks that it matches the expected mean value. It will differ from
        the mean of the column in the original file.
        """
        new_origin_mean = self.df_q1['origin'].mean()
        self.assertAlmostEqual(1.55527638, new_origin_mean, places=7)

    def test_question1_horsepower_indexes(self):
        expected_idx = [16, 32, 45, 73, 94, 114, 126, 147, 167, 189, 199, 222,
                        330, 336, 354,  374]

        for idx in expected_idx:
            self.assertAlmostEqual(104.2670157,
                                   self.df_q1.iloc[idx]['horsepower'])

    def test_question1_origin_indexes(self):
        expected_idx = [9, 28, 53, 75, 94, 109, 124, 142, 162, 178, 193, 209,
                        226, 241, 256]

        for idx in expected_idx:
            self.assertEqual(1, self.df_q1.iloc[idx]['origin'])

    def test_question2_columns_ok(self):
        self.assertListEqual(['horsepower', 'origin', 'mpg', 'cylinders',
                              'displacement', 'weight', 'acceleration',
                              'model year', 'car name', 'other'],
                             list(self.df_q2.columns))

    def test_question2_shape_ok(self):
        self.assertEqual(796, self.df_q2.shape[0])
        self.assertEqual(10, self.df_q2.shape[1])

    def test_question2_other_field(self):
        self.assertEqual(1.5, self.df_q2['other'].mean())

    def test_question2_concatenation_ok(self):
        first_row = self.df_q2.iloc[0]
        last_row = self.df_q2.iloc[self.df_q2.shape[0] - 1]

        self.assertEqual(130, first_row['horsepower'])
        self.assertEqual('chevrolet chevelle malibu', first_row['car name'])

        self.assertEqual(82, last_row['horsepower'])
        self.assertEqual('chevy s-10', last_row['car name'])


if __name__ == '__main__':
    unittest.main()
