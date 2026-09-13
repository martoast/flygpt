import unittest
import numpy as np
from scripts.run_topology_confirmation import assign_test, analysis

class ConfirmationTests(unittest.TestCase):
    def test_unseen_split_excludes_every_old_category(self):
        old={name:[{'id':i} for i in ids] for name,ids in {'train':range(2048),'validation':range(2048,2304),'old_test':range(2304,2944)}.items()}
        ids,n=assign_test(old,128,'locked')
        self.assertEqual(n,1152);self.assertEqual(len(set(ids)),128)
        self.assertTrue(all(i>=2944 for i in ids))
        self.assertEqual(ids,assign_test(old,128,'locked')[0])
    def test_overlap_refused(self):
        with self.assertRaises(RuntimeError):assign_test({'a':[{'id':1}],'b':[{'id':1}]},128,'x')
    def test_analysis_requires_all_pairs(self):
        with self.assertRaises(ValueError):analysis([.9]*9,[.5]*9)
    def test_null_and_direction(self):
        self.assertFalse(analysis([.5]*10,[.5]*10)['primary_pass'])
        x=np.arange(10)/100+.7;y=np.ones(10)*.5
        positive=analysis(x,y);negative=analysis(y,x)
        self.assertTrue(positive['primary_pass']);self.assertFalse(negative['primary_pass'])
        self.assertAlmostEqual(positive['sign_flip_p'],2/1024)
        self.assertAlmostEqual(positive['primary_p'],negative['primary_p'])

if __name__=='__main__':unittest.main()
