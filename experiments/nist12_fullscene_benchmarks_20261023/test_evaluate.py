"""Synthetic checks for the audit evaluator; no simulator labels are opened."""
import unittest
import numpy as np
from evaluate import bbox_iou, box_edges, mask_overlaps, select


class EvaluationTests(unittest.TestCase):
    def test_inclusive_boxes_and_disjoint_regions(self):
        self.assertEqual(bbox_iou([0,0,1,1],[0,0,1,1]),1.)
        self.assertEqual(bbox_iou([0,0,10,10],[10,0,20,10]),0.)
        self.assertAlmostEqual(bbox_iou([0,0,10,10],[0,0,20,10]),.5)
        self.assertEqual(box_edges([0,0,9,9],'audit'),[0.,0.,10.,10.])
        self.assertEqual(box_edges([0,0,10,10],'nemo'),[0.,0.,10.,10.])

    def test_select_only_requested_identity(self):
        rows=[{'object_id':1,'score':.2,'proposal_id':3},
              {'object_id':2,'score':.9,'proposal_id':4},
              {'object_id':1,'score':.4,'proposal_id':5}]
        row,status=select(rows,1)
        self.assertEqual((row['proposal_id'],status),(5,'ok'))
        self.assertEqual(select(rows,3),(None,'no_retained_detection'))

    def test_ties_stay_unresolved(self):
        rows=[{'object_id':1,'score':.5,'proposal_id':2},
              {'object_id':1,'score':.5,'proposal_id':3}]
        self.assertEqual(select(rows,1),(None,'tie'))

    def test_counted_mask_iou_equals_full_pixel_union(self):
        rng=np.random.default_rng(47)
        segmentation=rng.choice([-1,2,8,11],size=(70,90))
        counts={str(i):int(np.count_nonzero(segmentation==i)) for i in (2,8,11)}
        ids={str(i):i for i in (2,8,11)}
        for mask in (rng.random(segmentation.shape)>.8,segmentation==8,np.zeros(segmentation.shape,bool)):
            actual=dict((name,value) for value,name in mask_overlaps(mask,segmentation+1,counts,ids))
            for name,i in ids.items():
                expected=np.count_nonzero(mask & (segmentation==i))/np.count_nonzero(mask | (segmentation==i))
                self.assertEqual(actual[name],expected)


if __name__=='__main__':
    unittest.main()
