import unittest
import numpy as np
from kbs import synthetic, validate, evaluate

class PrototypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.df=synthetic(n_firms=40)
    def test_reproducible(self):
        self.assertTrue(self.df.equals(synthetic(n_firms=40)))
    def test_complete_schema(self):
        self.assertEqual(len(validate(self.df)),600)
    def test_temporal_evaluation(self):
        result, choices, picks=evaluate(self.df)
        self.assertEqual(result['split']['test_year'],2025)
        self.assertTrue((picks.cost<=picks.budget).all())
        self.assertTrue((picks.digital>=picks.min_digital).all())
        self.assertTrue(np.isfinite(choices.estimated_probability).all())
    def test_duplicate_rejected(self):
        import pandas as pd
        with self.assertRaises(ValueError): validate(pd.concat([self.df,self.df.iloc[:1]]))

if __name__=='__main__': unittest.main()
