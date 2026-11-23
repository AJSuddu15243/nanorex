import os
from pathlib import Path
import subprocess
import sys
import unittest


class BenchTest(unittest.TestCase):
    def run_bench(self, *options):
        return subprocess.run(
            [sys.executable, 'bench.py', '--device=cpu', '--compile=False',
             '--batch_size=2', '--block_size=8', '--n_layer=1', '--n_head=2',
             '--n_embd=8', '--vocab_size=19', *options],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
            capture_output=True, text=True, timeout=60,
        )

    def test_inference_compares_cache_paths_without_dataset(self):
        # real_data retains its default True: inference must not open train.bin.
        result = self.run_bench('--inference=True', '--prompt_length=5',
                                '--decode_tokens=3', '--warmup=1', '--iterations=2')
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in ('greedy tokens identical', 'KV cache off:', 'KV cache on:',
                         'prompt tokens/s', 'generated tokens/s', 'ms/step',
                         'off time / on time', 'prompt=5, decode_steps=3'):
            self.assertIn(expected, result.stdout)

    def test_inference_rejects_context_overflow(self):
        result = self.run_bench('--inference=True', '--prompt_length=6', '--decode_tokens=3')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('must fit block_size', result.stderr)

    def test_training_usage_still_runs(self):
        result = self.run_bench('--real_data=False')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('time per iteration:', result.stdout)
        self.assertNotIn('KV-cache speedup', result.stdout)


if __name__ == '__main__':
    unittest.main()
