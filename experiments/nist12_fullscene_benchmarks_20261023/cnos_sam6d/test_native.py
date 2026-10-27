"""CPU checks of native source calls, proposal coordinates and batching."""
import ast
from pathlib import Path
import types
import unittest

import run_native as trial
trial.setup()
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


def function_from_source(filename, name):
    tree=ast.parse(Path(filename).read_text())
    functions=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(functions)==1
    module=ast.Module(body=functions,type_ignores=[])
    namespace={'torch':torch,'F':F}
    exec(compile(ast.fix_missing_locations(module),str(filename),'exec'),namespace)
    return namespace[name]


class NativeIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cnos=trial.official('cnos')
        cls.sam=trial.official('sam6d')

    def test_cpu_upsampling_matches_native_full_batch_exactly(self):
        chunked=function_from_source(trial.__file__,'upsample_cpu')
        rng=np.random.default_rng(21)
        masks=torch.from_numpy((rng.random((5,6,8))>.6))
        boxes=torch.tensor([[0,0,7,5],[1,2,6,4],[0,1,5,5],[3,0,7,4],[0,0,8,6]])
        for native in [self.cnos,self.sam]:
            original=function_from_source(native.source/('src/model/sam.py' if native is self.cnos else 'model/sam.py'),
                                          'postprocess_resize')
            fake=types.SimpleNamespace(segmentor_width_size=8)
            a=original(fake,{'masks':masks.clone(),'boxes':boxes.clone()},(24,32))
            b=chunked(fake,{'masks':masks.clone(),'boxes':boxes.clone()},(24,32))
            torch.testing.assert_close(a['masks'],b['masks'],rtol=0,atol=0)
            torch.testing.assert_close(a['boxes'],b['boxes'],rtol=0,atol=0)
            self.assertEqual(a['masks'].shape,(5,24,32))
            self.assertTrue(torch.equal(b['masks'].half().float(),b['masks']))

    def test_native_small_object_filter_preserves_proposal_ids(self):
        for native in [self.cnos,self.sam]:
            masks=torch.zeros((3,100,100))
            masks[0,:10,:10]=1;masks[1,:10,:10]=1;masks[2,0,:2]=1
            detection=native.Detections({'masks':masks,
                'boxes':torch.tensor([[0.,0,10,10],[0,0,4,4],[0,0,20,20]]),
                'proposal_ids':torch.tensor([17,18,19])})
            detection.remove_very_small_detections(OmegaConf.create({'min_box_size':.05,'min_mask_size':.0003}))
            self.assertEqual(detection.proposal_ids.tolist(),[17])

    def test_native_class_matching_and_threshold(self):
        metric=self.cnos.PairwiseSimilarity(metric='cosine',chunk_size=16)
        matcher=types.SimpleNamespace(matching_config=types.SimpleNamespace(metric=metric,
            aggregation_function='avg_5',confidence_thresh=.15,max_num_instances=100),
            ref_data={'descriptors':torch.eye(3)[:,None,:].repeat(1,42,1)})
        queries=torch.tensor([[1.,0,0],[0,1,0],[-1,-1,-1.]])
        selected,objects,scores,distribution=self.cnos.Detector.find_matched_proposals(matcher,queries)
        self.assertEqual(selected.tolist(),[0,1]);self.assertEqual(objects.tolist(),[0,1])
        torch.testing.assert_close(scores,torch.ones(2))
        self.assertEqual(distribution.shape,(2,3))

    def test_official_reference_mapping_and_preprocessing(self):
        metadata=trial.read(trial.ROOT/'templates42/metadata.json')
        self.assertEqual(len(metadata['index_templates']),42)
        for native in [self.cnos,self.sam]:
            dataset=native.BOPTemplate(template_dir=str(trial.ROOT/'templates42'),
                obj_ids=[1],processing_config=OmegaConf.create({'image_size':224}),
                level_templates=0,pose_distribution='all')
            np.testing.assert_array_equal(dataset.index_templates,metadata['index_templates'])
            item=dataset[0]
            self.assertEqual(tuple(item['templates'].shape),(42,3,224,224))
            self.assertEqual(tuple(item['template_masks'].shape),(42,224,224))
            self.assertTrue(torch.isfinite(item['templates']).all())
            self.assertTrue((item['template_masks'].sum((1,2))>0).all())
            # Raw black reference padding is normalized after cropping; this
            # distinguishes the native loader from the former white inputs.
            torch.testing.assert_close(item['templates'][0,:,0,0],
                -torch.tensor([.485,.456,.406])/torch.tensor([.229,.224,.225]))


if __name__=='__main__':
    unittest.main()
