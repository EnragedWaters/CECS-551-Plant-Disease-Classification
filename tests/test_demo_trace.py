"""Verify that teaching visuals are genuine observations, not model changes."""
import unittest
import torch
from plant_disease.demo_trace import observe, describe_pair, choose_batch, color_image, learning_values
from plant_disease.model import state_fingerprint


class TinyFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3,4,1),
            *[torch.nn.ReLU() for _ in range(11)], torch.nn.Conv2d(4,576,1))
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = torch.nn.Linear(576,2)

    def forward(self, image):
        return self.classifier(self.avgpool(self.features(image)).flatten(1))


class TraceTests(unittest.TestCase):
    def test_learning_sums_reconstruct_real_pooling_and_logits(self):
        torch.manual_seed(551)
        model = TinyFeatures().eval()
        before = state_fingerprint(model.state_dict())
        logits, captures = observe(model, torch.rand(1,3,7,7))
        result = learning_values(model, captures, logits, 0)
        self.assertTrue(result['all_576_averages_verified'])
        self.assertTrue(result['all_logits_verified'])
        self.assertEqual(len(result['classifier']), 2)  # Repeated roles share one category.
        for entry in result['pooling']:
            self.assertAlmostEqual(sum(map(sum, entry['grid']))/49, entry['observed'], places=5)
        for entry in result['classifier']:
            for term in entry['terms']:
                self.assertAlmostEqual(term['value']*term['weight'], term['product'])
            total = sum(t['product'] for t in entry['terms']) + entry['remaining_sum'] + entry['bias']
            self.assertAlmostEqual(total, entry['actual_logit'], places=5)
            self.assertEqual(entry['remaining_count'], 571)
        self.assertEqual(before, state_fingerprint(model.state_dict()))

    def test_learning_rejects_mismatched_captures_and_scores(self):
        model = TinyFeatures().eval()
        logits, captures = observe(model, torch.rand(1,3,7,7))
        with self.assertRaisesRegex(ValueError, 'Classifier reconstruction'):
            learning_values(model, captures, logits+1, 0)
        captures['pooled'] = captures['pooled'].clone()
        captures['pooled'][0,0] += 1
        with self.assertRaisesRegex(ValueError, 'Pooling reconstruction'):
            learning_values(model, captures, logits, 0)

    def test_observation_preserves_logits_state_and_cleans_hooks(self):
        model = TinyFeatures().eval()
        image = torch.rand(1,3,8,8)
        before = state_fingerprint(model.state_dict())
        with torch.inference_mode():
            expected = model(image)
            expected_pool = model.avgpool(model.features(image))
        logits, features = observe(model,image)
        self.assertTrue(torch.equal(expected,logits))
        self.assertTrue(torch.equal(expected_pool,features['pooled']))
        self.assertEqual(features['early'].shape,(1,4,8,8))
        self.assertEqual(before,state_fingerprint(model.state_dict()))
        self.assertTrue(all(not layer._forward_hooks for layer in model.modules()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_cleanup_when_observed_forward_fails(self):
        model = TinyFeatures().eval()
        # Fail only after the plain reference pass has completed.
        def fail(module,args,output):
            if model.avgpool._forward_hooks:
                raise RuntimeError('deliberate failure during observation')
        handle = model.classifier.register_forward_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError,'deliberate'):
                observe(model,torch.ones(1,3,4,4))
            self.assertFalse(model.avgpool._forward_hooks)
            self.assertTrue(all(not layer._forward_hooks for layer in model.features))
        finally:
            handle.remove()

    def test_both_models_use_identical_color_scales_and_channel_numbers(self):
        model = TinyFeatures().eval()
        _,features = observe(model,torch.ones(1,3,4,4))
        other = {k:v*2 for k,v in features.items()}
        display = describe_pair({'frozen':features,'finetune':other})
        for a,b in zip(display['frozen'],display['finetune']):
            self.assertEqual(a['scale_limit'],b['scale_limit'])
            self.assertEqual(a['shape'],b['shape'])
            if 'channels' in a:self.assertEqual(a['channels'],[0,1,2,3])
        self.assertEqual(len(display['frozen'][-1]['values']),576)
        self.assertEqual(color_image([[0]],1),color_image([[0]],10))
        self.assertNotEqual(color_image([[1]],1),color_image([[-1]],1))
        with self.assertRaises(ValueError):color_image([[float('nan')]],1)

    def test_batch_selection_ignores_prediction_outcomes_and_input_order(self):
        rows={split:{} for split in ['source_test','target_test']}
        for split in rows:
            for label in range(2):
                for hash_value in ['c','a','b']:
                    key=f'{split}-{label}-{hash_value}'
                    rows[split][key]={'class_id':str(label),'sha256':hash_value,'relative_path':key,'correct':hash_value=='c'}
        sample=choose_batch(rows,2)
        self.assertEqual(len(sample),2)
        self.assertTrue(all(row['sha256']=='a' for pair in sample for row in pair.values()))
        reverse={s:dict(reversed(list(items.items()))) for s,items in rows.items()}
        self.assertEqual(sample,choose_batch(reverse,2))
        with self.assertRaises(ValueError):choose_batch(rows,3)


if __name__=='__main__':unittest.main()
