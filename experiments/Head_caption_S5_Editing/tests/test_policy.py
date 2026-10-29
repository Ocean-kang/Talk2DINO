import unittest
from types import SimpleNamespace
from experiments.Head_caption_S5_Editing.editing import neutralize,append_visible,edit,mentioned,eligible
from experiments.Head_caption_S5_Editing.select import candidate_lists,score_change,admissible
from experiments.Head_caption_S5_Editing.build import plan

class EditingTests(unittest.TestCase):
    def test_exact_noun(self):
        self.assertEqual(neutralize('A man holding an umbrella near a bus.', 'umbrella').text,
                         'A man holding a thing near a bus.')
        self.assertEqual(neutralize('Two dogs chase a cat.', 'dog').text,'Two things chase a cat.')
        self.assertEqual(neutralize('A car near a traffic light.','traffic light').text,
                         'A car near a thing.')
        self.assertEqual(neutralize('An umbrella and umbrella.','umbrella').text,'A thing and thing.')
    def test_refusal_and_boundaries(self):
        self.assertEqual(neutralize('A woman running.','woman').applied,[])
        self.assertEqual(neutralize('A carpet is visible.','car').text,'A carpet is visible.')
        self.assertFalse(mentioned('a carpet','car'))
        self.assertFalse(eligible('riding'))
    def test_add(self):
        self.assertEqual(append_visible('A man holds an umbrella.',['bus','traffic light']).text,
                         'A man holds an umbrella, with a bus and a traffic light also visible.')
        self.assertEqual(append_visible('A car.',['car']).text,'A car.')
        self.assertEqual(edit('A car.',remove='car',add='bus')['caption'],
                         'A thing, with a bus also visible.')
    def test_top10(self):
        vocab=['car','dog','umbrella','train','bus']
        hi,lo,rem=candidate_lists(vocab,[.1,.7,.5,.3,-.9],'A car.',2,None,'caption_low10')
        self.assertEqual([vocab[i] for i in hi],['dog','umbrella'])
        self.assertEqual([vocab[i] for i in lo],['bus','car'])
        self.assertEqual([vocab[i] for i in rem],['car'])
    def test_s5_known(self):
        m=score_change([1.,0.], [0.,0.],[False,False], [1.,0.],[False,False],0)
        a=SimpleNamespace(max_off_drift=.1,min_target_effect=.2,min_aggregate_effect=.001,
           target_dominance=1.,max_remove_add_penalty=.01)
        self.assertTrue(admissible(m,'ADD',a)[0])
        self.assertAlmostEqual(m['delta_add'],-1.)
        unknown=score_change([1.],[0.],[True],[1.],[False],0)
        self.assertFalse(unknown['target_known'])
    def test_full_coverage(self):
        base={'images':[{'id':10}], 'annotations':[{'id':j,'image_id':10,'caption':'A car.','ann_feats':j} for j in range(5)]}
        manifest=[{'image_id':10,'annotation_id':j,'raw_caption':'A car.','edited_caption':'A car.',
                   'operation':'KEEP'} for j in range(5)]
        im,anns,changed,counts=plan(base,manifest)
        self.assertEqual((len(im),len(anns),len(changed)),(1,5,0))
        with self.assertRaises(ValueError):plan(base,manifest[:-1])

    def test_variable_caption_counts(self):
        # Different images can legitimately have 5, 6, or 7 captions.
        count_per_image = {10: 5, 20: 6, 30: 7}
        images = [{'id': iid} for iid in count_per_image]
        anns = []
        manifest = []
        for iid, n in count_per_image.items():
            for j in range(n):
                aid = iid * 100 + j
                ann = {'id': aid, 'image_id': iid,
                       'caption': 'A bus is visible.', 'ann_feats': j}
                anns.append(ann)
                manifest.append({'image_id': iid, 'annotation_id': aid,
                                 'raw_caption': ann['caption'],
                                 'edited_caption': ann['caption'],
                                 'operation': 'KEEP'})
        base = {'images': images, 'annotations': anns}
        actual_images, actual_anns, changed, _ = plan(base, manifest)
        self.assertEqual(len(actual_images), 3)
        self.assertEqual(len(actual_anns), 18)
        self.assertEqual(len(changed), 0)
        from collections import Counter
        self.assertEqual(dict(Counter(a['image_id'] for a in actual_anns)), count_per_image)
        # Selection must not accidentally discard the 6th or 7th caption.
        with self.assertRaises(ValueError):
            plan(base, [r for r in manifest if r['annotation_id'] != 3006])

if __name__=='__main__':unittest.main()
