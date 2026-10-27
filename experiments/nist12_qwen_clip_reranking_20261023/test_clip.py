"""Focused label-free checks for retrieval direction, mappings, pooling and ties."""
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np


SOURCE = Path(__file__).with_name('run_clip.py')
SPEC = importlib.util.spec_from_file_location('trial_clip', SOURCE)
trial = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trial)


class TinyTokenizer:
    def encode(self, text, **kwargs):
        assert kwargs == dict(add_special_tokens=False, truncation=False)
        return list(range(len(text)))

    def build_inputs_with_special_tokens(self, tokens):
        return [1000] + tokens + [1001]


def fixture():
    inputs = [dict(input_id='CAD_' + str(i), kind='CAD', target='part', view_index=i)
              for i in (1, 2)]
    inputs += [dict(input_id='crop_' + str(i), kind='crop', state='S00', proposal_id=i)
               for i in range(6)]
    direction = np.arange(1., 7.)
    direction /= np.linalg.norm(direction)
    images = np.vstack([-direction, direction, np.eye(6)])
    arrays = dict(image_normalized=images, text_full=images, text_shape=images,
                  text_full_available=np.ones(8, dtype=bool),
                  text_shape_available=np.ones(8, dtype=bool))
    dino = [dict(state='S00', target='part', proposal_id=i, score=float(6 - i),
                 global_score=float(6 - i), patch_score=float(6 - i),
                 selected_CAD_view=1) for i in range(6)]
    return inputs, arrays, dino


class ClipProtocolTests(unittest.TestCase):
    def test_content_tokens_retained_without_truncation(self):
        ids = list(range(151))
        parts = trial.split_content_ids(ids)
        self.assertEqual(list(map(len, parts)), [75, 75, 1])
        self.assertEqual(sum(parts, []), ids)
        self.assertEqual(trial.split_content_ids([]), [])
        descriptions = [dict(input_id='test', mode=mode, status='ok',
            fields=dict(COLOR='blue', SHAPE='x' * 151, MATERIAL='metal'))
            for mode in trial.MODES]
        fields, chunks = trial.text_plan([dict(input_id='test')], descriptions, TinyTokenizer(), 75)
        self.assertEqual([len(row['chunk_indices']) for row in fields], [1, 3, 1, 1, 3, 1])
        self.assertEqual(len(chunks), 10)
        self.assertTrue(all(len(row['input_ids']) <= 77 for row in chunks))
        for field in fields:
            self.assertEqual(sum([chunks[i]['content_token_ids'] for i in field['chunk_indices']], []),
                             field['content_token_ids'])

    def test_missing_native_description_is_explicit(self):
        descriptions = [dict(input_id='test', mode='keywords', status='ok',
            fields=dict(COLOR='blue', SHAPE='cylinder', MATERIAL='metal'))]
        fields, _ = trial.text_plan([dict(input_id='test')], descriptions, TinyTokenizer(), 75)
        self.assertEqual(sum(row['status'] != 'ok' for row in fields), 3)
        self.assertEqual([row['raw_text'] for row in fields[:3]], ['blue', 'cylinder', 'metal'])

    def test_equal_normalized_pooling_and_degenerate_pool(self):
        np.testing.assert_allclose(trial.pool_unit([np.array([2., 0.]), np.array([0., 9.])]),
                                   np.array([1., 1.]) / np.sqrt(2))
        self.assertIsNone(trial.pool_unit([]))
        self.assertIsNone(trial.pool_unit([None]))
        self.assertIsNone(trial.pool_unit([np.array([1., 0.]), np.array([-1., 0.])]))

    def test_cutoff_tie_is_unavailable_not_label_broken(self):
        rows = [dict(proposal_id=i, score=float(6 - i), score_available=True) for i in range(6)]
        self.assertEqual(trial.make_shortlist(rows)['candidate_ids'], [0, 1, 2, 3, 4])
        rows[5]['score'] = rows[4]['score'] - 5e-13
        result = trial.make_shortlist(rows)
        self.assertEqual(result['status'], 'unavailable_ambiguous_top5')
        self.assertEqual(result['candidate_ids'], [])
        self.assertEqual(result['cutoff_tie_ids'], [4, 5])
        rows[5]['score'] = 1.
        rows[1]['score'] = rows[0]['score']
        self.assertEqual(trial.make_shortlist(rows)['status'], 'ok')

    def test_missing_competitor_not_dropped(self):
        rows = [dict(proposal_id=i, score=float(6 - i), score_available=True) for i in range(6)]
        rows[-1].update(score=None, score_available=False)
        shortlist = trial.make_shortlist(rows)
        self.assertEqual(shortlist['status'], 'unavailable_incomplete_CLIP_evidence')
        self.assertFalse(shortlist['candidate_ids'])
        ranked = trial.rank_scores(rows)
        self.assertEqual(ranked['top_ids'], [])
        self.assertEqual(ranked['provisional_top_ids'], [0])
        self.assertEqual(ranked['unavailable_ids'], [5])

    def test_final_ties_remain_unresolved(self):
        rows = [dict(proposal_id=i, score=1., score_available=True) for i in range(5)]
        result = trial.rank_scores(rows)
        self.assertEqual(result['status'], 'tie')
        self.assertEqual(result['top_ids'], list(range(5)))
        self.assertEqual(result['gap'], 0.)

    def test_clip_selects_scene_crops_before_unchanged_dino(self):
        inputs, arrays, dino = fixture()
        view_rows, pairs, shortlists, rankings = trial.score_all(inputs, arrays, dino,
                                                               require_trial_size=False)
        methods = {row['method']: row for row in rankings}
        self.assertEqual(len(view_rows), 6 * 2 * 2)
        self.assertEqual(len(pairs), 6)
        self.assertEqual(methods['dino']['top_ids'], [0])
        self.assertEqual(methods['clip_full_only']['top_ids'], [5])
        self.assertEqual(methods['clip_full_dino']['shortlist_ids'], [5, 4, 3, 2, 1])
        self.assertEqual(methods['clip_full_dino']['top_ids'], [1])
        self.assertEqual(methods['clip_full_dino']['top_score'], dino[1]['score'])
        self.assertEqual(methods['clip_full_dino']['selected_DINO_CAD_views'], {'1': 1})
        self.assertEqual(methods['clip_full_dino']['selected_CLIP_CAD_views'], {'1': 2})
        for pair in pairs:
            self.assertEqual(pair['dino_score'], dino[pair['proposal_id']]['score'])
            self.assertEqual(pair['clip_full']['selected_CLIP_CAD_view'], 2)
            component = pair['clip_full']['components']
            self.assertAlmostEqual(pair['clip_full']['score'], np.mean(list(component.values())))
        flipped = copy.deepcopy(dino)
        for row in flipped:
            row['score'] = -row['score']
        second = trial.score_all(inputs, arrays, flipped, require_trial_size=False)
        self.assertEqual(shortlists, second[2], 'CLIP retrieval cannot depend on DINO rankings.')

    def test_strict_view_completeness_independent_per_text_arm(self):
        inputs, arrays, dino = fixture()
        arrays['text_full_available'][0] = False
        _, pairs, shortlists, rankings = trial.score_all(inputs, arrays, dino,
                                                        require_trial_size=False)
        self.assertTrue(all(pair['clip_full']['available_views'] == 1 for pair in pairs))
        self.assertTrue(all(not pair['clip_full']['score_available'] for pair in pairs))
        self.assertEqual(shortlists[0]['status'], 'unavailable_incomplete_CLIP_evidence')
        methods = {row['method']: row for row in rankings}
        self.assertFalse(methods['clip_full_dino']['top_ids'])
        self.assertEqual(methods['clip_shape_dino']['status'], 'ok')
        self.assertEqual(methods['dino']['status'], 'ok')

    def test_same_dino_winner_survives_if_kept_in_shortlist(self):
        inputs, arrays, dino = fixture()
        # Candidate four is kept by CLIP and strictly wins the independently scored DINO baseline.
        dino[4]['score'] = 99.123456789
        _, _, _, rankings = trial.score_all(inputs, arrays, dino, require_trial_size=False)
        methods = {row['method']: row for row in rankings}
        for method in ('dino', 'clip_full_dino', 'clip_shape_dino'):
            self.assertEqual(methods[method]['top_ids'], [4])
            self.assertEqual(methods[method]['top_score'], dino[4]['score'])


if __name__ == '__main__':
    unittest.main()
