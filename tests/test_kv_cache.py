import unittest

import torch

from model import GPT, GPTConfig


class KVCacheTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = GPT(GPTConfig(
            block_size=4, vocab_size=19, n_layer=2, n_head=2, n_embd=8, dropout=0,
        )).eval()

    def test_cached_logits_match_cropped_full_context(self):
        prompt = torch.tensor([[1, 2, 3, 4, 5, 6]])
        continuation = [7, 8, 9, 10, 11, 12]

        for manual in (False, True):
            with self.subTest(manual_attention=manual):
                for block in self.model.transformer.h:
                    block.attn.flash = not manual
                    if manual:
                        block.attn.register_buffer(
                            "bias", torch.tril(torch.ones(4, 4)).view(1, 1, 4, 4),
                            persistent=False,
                        )

                tokens = prompt.clone()
                cache = [None] * len(self.model.transformer.h)
                for token in continuation:
                    if cache[0] is not None and tokens.size(1) > self.model.config.block_size:
                        cache = [None] * len(self.model.transformer.h)
                    cached_input = tokens[:, -1:] if cache[0] is not None else tokens[:, -4:]
                    cached_logits, _ = self.model(cached_input, kv_cache=cache)
                    full_logits, _ = self.model(tokens[:, -4:])
                    torch.testing.assert_close(cached_logits[:, -1], full_logits[:, -1], rtol=1e-5, atol=1e-6)
                    tokens = torch.cat((tokens, torch.tensor([[token]])), dim=1)

    def test_generate_processes_incrementally_and_resets_each_request(self):
        calls = []
        original_forward = self.model.forward

        def recording_forward(idx, targets=None, kv_cache=None):
            cache_state = None if kv_cache is None else tuple(
                None if item is None else item[0].size(-2) for item in kv_cache
            )
            calls.append((idx.size(1), kv_cache, cache_state))
            return original_forward(idx, targets, kv_cache)

        self.model.forward = recording_forward
        prompt = torch.tensor([[1, 2, 3]])
        self.model.generate(prompt, 6, use_cache=True)
        first_request = calls[:]
        self.assertEqual([call[0] for call in first_request], [3, 1, 4, 4, 4, 4])
        self.assertTrue(all(call[2] is not None for call in first_request))
        self.assertIsNot(first_request[0][1], first_request[2][1])
        self.assertIsNot(first_request[2][1], first_request[3][1])
        self.assertTrue(all(state == (None, None) for _, _, state in first_request[2:]))

        calls.clear()
        self.model.generate(prompt, 2, use_cache=True)
        self.assertEqual([call[0] for call in calls], [3, 1])
        self.assertIsNot(calls[0][1], first_request[0][1])


if __name__ == "__main__":
    unittest.main()
