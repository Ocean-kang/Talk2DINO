import unittest
import torch
from experiments.Head_caption_completeness_diagnosis.text_evidence import (
    aliases_for_vocab,lexical_support,phrases,TextScorer,clip_null)
from experiments.Head_caption_completeness_diagnosis.perturbations import make_cases,remove_mention


class TinyTeacher:
    def embed(self,texts,template='{}'):
        out=[]
        for text in texts:
            x=torch.zeros(512)
            x[sum(map(ord,text))%512]=1
            out.append(x)
        return torch.stack(out)


class TextTests(unittest.TestCase):
    def test_alias_and_negation(self):
        vocab=['bicycle','person','car']
        alias=aliases_for_vocab(vocab)
        yes,no=lexical_support('A man rides a bike without a car.',vocab,alias)
        self.assertEqual(yes,{0,1})
        self.assertEqual(no,{2})

    def test_phrase_locality(self):
        raw=phrases('A man riding a red bicycle.')
        new=phrases('A man riding.')
        self.assertIn('man',raw);self.assertIn('man',new)
        self.assertIn('red bicycle',raw)
        self.assertNotIn('red bicycle',new)

    def test_edit_and_cases(self):
        vocab=['bicycle','car','tree']
        alias=aliases_for_vocab(vocab)
        cases=make_cases('A man rides a bicycle on the road.',vocab,alias,
                         [.9,.1,.0],[.8,.0,.0],[.1,.1,.0],
                         min_high=.6,min_image=.3,max_low=.2,max_other=.3,max_low_image=.2)
        types={x['kind'] for x in cases}
        self.assertIn('delete_supported',types)
        self.assertIn('add_low_evidence',types)
        self.assertIn('neutral_prefix',types)
        self.assertNotIn('bicycle',remove_mention('A man rides a bicycle on the road.','bicycle'))

    def test_text_s2_does_not_drift_on_extraneous_append(self):
        vocab=['bicycle','car','tree'];words=torch.nn.functional.normalize(torch.randn(3,512),dim=-1)
        scorer=TextScorer(TinyTeacher(),vocab,words,clip_null(words))
        old,_=scorer.score('A bicycle next to a tree','S2')
        new,_=scorer.score('A bicycle next to a tree; also visible: car.','S2')
        self.assertEqual(old[0],new[0]);self.assertEqual(old[2],new[2]);self.assertEqual(new[1],1.)

if __name__=='__main__':unittest.main()
