import math
import unittest
import torch
from model import AiraLM, Config


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def config(self, **changes):
        return Config(**dict(dict(vocab_size=8, block_size=8, n_embd=16, n_head=2, n_layer=1), **changes))

    def test_invalid_dimensions(self):
        for changes in ({'vocab_size': 0}, {'n_embd': 15}, {'dropout': 1}, {'dropout': math.nan}, {'n_head': 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.config(**changes)

    def test_forward_backward(self):
        model = AiraLM(self.config())
        x = torch.randint(8, (2, 8))
        logits, loss = model(x, x)
        self.assertEqual(tuple(logits.shape), (2, 8, 8))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.tok.weight.grad)

    def test_overlong_context(self):
        with self.assertRaisesRegex(ValueError, 'sequence length'):
            AiraLM(self.config())(torch.zeros((1, 9), dtype=torch.long))

    def test_sampling_checks(self):
        model = AiraLM(self.config())
        x = torch.zeros((1, 1), dtype=torch.long)
        for kwargs in ({'temperature': 0}, {'temperature': math.nan}, {'top_k': -1}, {'max_new_tokens': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                list(model.stream(x, **dict({'max_new_tokens': 2}, **kwargs)))

    def test_stream_count(self):
        tokens = list(AiraLM(self.config()).stream(torch.zeros((1, 1), dtype=torch.long), 3))
        self.assertEqual(len(tokens), 3)
        self.assertTrue(all(0 <= token < 8 for token in tokens))


if __name__ == '__main__':
    unittest.main()
