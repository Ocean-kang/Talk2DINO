import unittest
from experiments.Head_caption_completeness_diagnosis.core import (
    coverage,comparison_mask,drift_decomposition,strict_groups,delta)


class CoverageTests(unittest.TestCase):
    def test_add_attribution_is_exact(self):
        v=[.9,.8,.2];old=[1.,1.,0.];new=[1.,0.,.5];mask=[True,True,True]
        old_d=coverage(v,old,mask)
        new_d=coverage(v,new,mask)
        d=drift_decomposition(v,old,new,mask,1)
        self.assertAlmostEqual(delta(new_d['d_add'],old_d['d_add']),d['sum_contribution'],places=12)
        self.assertGreater(d['target_d_add_contribution'],0)
        self.assertLess(d['off_target_d_add_contribution'],0)

    def test_union_bug_reproduced_only_s0(self):
        v=[.8,.5];ot=[1,0];nt=[0,0]
        ou=[False,False];nu=[True,False]
        legacy=comparison_mask(v,ot,nt,ou,nu,0,legacy=True)
        fixed=comparison_mask(v,ot,nt,ou,nu,0,legacy=False)
        self.assertTrue(legacy[0]);self.assertFalse(fixed[0])

    def test_unknown_excluded_s1(self):
        mask=comparison_mask([.8],[0],[0],[True],[True],0)
        self.assertEqual(mask,[False])
        self.assertIsNone(coverage([.8],[0],mask)['d_add'])

    def test_remove_not_universally_monotonic(self):
        v=[.1,.9]; t0=[1,0];t1=[1,1];valid=[True,True]
        a=coverage(v,t0,valid);b=coverage(v,t1,valid)
        self.assertLess(b['d_remove'],a['d_remove'])

    def test_synonym_sets_do_not_merge_hypernyms(self):
        vocab=['bike','bicycle','vehicle','car']
        groups=strict_groups(vocab,[['bike','bicycle']])
        self.assertEqual(len(groups),3)
        self.assertTrue(any(len(g)==2 for g in groups))
        self.assertTrue(any(set(g)=={2} for g in groups))

    def test_group_unknown_members_dropped(self):
        result=coverage([.8,.9],[1,1],[True,False],groups=[[0,1]])
        self.assertEqual(result['valid_count'],0)

if __name__=='__main__':unittest.main()
