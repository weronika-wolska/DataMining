import unittest

import pandas as pd


class TestPractical(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.read_csv('./output/question_1.csv')
        self.df2 = pd.read_csv('./output/question_2.csv')
        self.df3 = pd.read_csv('./output/question_3.csv')

    def tearDown(self):
        self.df1 = None
        self.df2 = None
        self.df3 = None

    def test_q1_columns_ok(self):
        cols = list(self.df1.columns)

        self.assertIn('x', cols)
        self.assertIn('y', cols)
        self.assertIn('cluster', cols)

    def test_q1_rows_ok(self):
        self.assertEqual(16, self.df1.shape[0])

    def test_q1_clusters_ok(self):
        self.assertSetEqual({0, 1, 2}, set(list(self.df1.cluster)))

    def test_q2_columns_ok(self):
        cols = list(self.df2.columns)

        self.assertIn('CALORIES', cols)
        self.assertIn('PROTEIN', cols)
        self.assertIn('FAT', cols)
        self.assertIn('SODIUM', cols)
        self.assertIn('FIBER', cols)
        self.assertIn('CARBO', cols)
        self.assertIn('SUGARS', cols)
        self.assertIn('POTASS', cols)
        self.assertIn('VITAMINS', cols)
        self.assertIn('SHELF', cols)
        self.assertIn('WEIGHT', cols)
        self.assertIn('CUPS', cols)

        self.assertIn('config1', cols)
        self.assertIn('config2', cols)
        self.assertIn('config3', cols)

    def test_q2_rows_ok(self):
        self.assertEqual(77, self.df2.shape[0])

    def test_q2_clusters_ok(self):
        self.assertSetEqual({0, 1, 2, 3, 4}, set(list(self.df2.config1)))
        self.assertSetEqual({0, 1, 2, 3, 4}, set(list(self.df2.config2)))
        self.assertSetEqual({0, 1, 2}, set(list(self.df2.config3)))

    def test_q3_columns_ok(self):
        cols = list(self.df3.columns)

        self.assertIn('x', cols)
        self.assertIn('y', cols)

        self.assertIn('kmeans', cols)
        self.assertIn('dbscan1', cols)
        self.assertIn('dbscan2', cols)

    def test_q3_rows_ok(self):
        self.assertEqual(322, self.df3.shape[0])

    def test_q3_clusters_ok(self):
        self.assertSetEqual({0, 1, 2, 3, 4, 5, 6}, set(list(self.df3.kmeans)))
        self.assertSetEqual({-1, 0, 1, 2, 3, 4, 5},
                            set(list(self.df3.dbscan1)))
        self.assertSetEqual({-1, 0, 1, 2, 3, 4}, set(list(self.df3.dbscan2)))


if __name__ == '__main__':
    unittest.main()
