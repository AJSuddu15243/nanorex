"""
A much shorter version of train.py for benchmarking.

Training usage is unchanged. For synthetic inference (no dataset required):
python bench.py --inference=True --device=cpu --compile=False --batch_size=1
Inference compares the existing KV-cache paths using the same random weights.
"""
import os
from contextlib import nullcontext
import numpy as np
import time
import torch
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
batch_size = 12
block_size = 1024
bias = False
real_data = True
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
profile = False # use pytorch profiler, or just simple benchmarking?
inference = False # prefill and greedy decode, with and without KV cache
prompt_length = 128
decode_tokens = 32 # single-token forwards AFTER prefill (which predicts the first token)
warmup = 5
iterations = 20
n_layer = 12
n_head = 12
n_embd = 768
vocab_size = 50304
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

def synchronize():
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    elif torch.device(device).type == 'mps':
        torch.mps.synchronize()


def benchmark_inference():
    if profile:
        raise ValueError('profile=True is only supported by the training benchmark')
    if min(batch_size, prompt_length, decode_tokens, iterations, n_layer, n_head, n_embd, vocab_size) <= 0 or warmup < 1:
        raise ValueError('inference dimensions, iterations and warmup must be positive')
    if prompt_length + decode_tokens > block_size:
        raise ValueError('prompt_length + decode_tokens must fit block_size (no cache rebuilds)')
    if n_embd % n_head:
        raise ValueError('n_embd must be divisible by n_head')

    model = GPT(GPTConfig(block_size=block_size, vocab_size=vocab_size,
                          n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                          dropout=0, bias=bias)).to(device).eval()
    if compile:
        print('Compiling model...')
        model = torch.compile(model)
    # Generate on CPU so the prompt is identical across device choices.
    prompt = torch.randint(vocab_size, (batch_size, prompt_length),
                           generator=torch.Generator().manual_seed(seed)).to(device)
    precision = dtype if device_type == 'cuda' else 'float32'
    print(f'Inference: device={device}, precision={precision}, compile={compile}, seed={seed}, '
          f'batch={batch_size}, prompt={prompt_length}, decode_steps={decode_tokens}, '
          f'block_size={block_size}, layers={n_layer}, heads={n_head}, embd={n_embd}, '
          f'vocab={vocab_size}, warmup={warmup}, iterations={iterations}')
    print('Random weights; greedy argmax; prefill returns last-position logits. '
          'Decode excludes prefill and includes argmax/concatenation; '
          'prefill predicts token 1, decode predicts the next decode_steps tokens.')

    def prefill(use_cache):
        cache = [None] * n_layer if use_cache else None
        logits, _ = model(prompt, kv_cache=cache)
        return logits, cache

    def decode(logits, cache):
        tokens = torch.cat((prompt, logits[:, -1].argmax(dim=-1, keepdim=True)), dim=1)
        for _ in range(decode_tokens):
            idx = tokens if cache is None else tokens[:, -1:]
            logits, _ = model(idx, kv_cache=cache)
            tokens = torch.cat((tokens, logits[:, -1].argmax(dim=-1, keepdim=True)), dim=1)
        return tokens

    with torch.inference_mode(), ctx:
        # Check every step on one shared greedy trajectory before timing either path.
        full, _ = prefill(False)
        cached, cache = prefill(True)
        tokens = prompt
        rtol, atol = (1e-4, 1e-5) if precision == 'float32' else (1e-2, 1e-2)
        for step in range(decode_tokens + 1):
            torch.testing.assert_close(cached, full, rtol=rtol, atol=atol)
            next_token = full[:, -1].argmax(dim=-1, keepdim=True)
            torch.testing.assert_close(cached[:, -1].argmax(dim=-1, keepdim=True), next_token)
            tokens = torch.cat((tokens, next_token), dim=1)
            if step < decode_tokens:
                full, _ = model(tokens)
                cached, _ = model(tokens[:, -1:], kv_cache=cache)
        expected = tokens
        print(f'Correctness: per-step logits close (rtol={rtol}, atol={atol}); greedy tokens identical')

        timings = {}
        for use_cache in (False, True):
            # Warm up both prefill and all growing decode shapes, including compilation.
            for _ in range(warmup):
                logits, cache = prefill(use_cache)
                decode(logits, cache)
            prefill_seconds = decode_seconds = 0.0
            for _ in range(iterations):
                synchronize()
                start = time.perf_counter()
                logits, cache = prefill(use_cache)
                synchronize()
                prefill_seconds += time.perf_counter() - start
                start = time.perf_counter()
                result = decode(logits, cache)
                synchronize()
                decode_seconds += time.perf_counter() - start
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
            prefill_seconds /= iterations
            decode_seconds /= iterations
            timings[use_cache] = (prefill_seconds, decode_seconds)
            print(f'KV cache {"on" if use_cache else "off"}: '
                  f'prefill {prefill_seconds * 1000:.3f} ms/request, '
                  f'{batch_size * prompt_length / prefill_seconds:.1f} prompt tokens/s; '
                  f'decode {decode_seconds * 1000:.3f} ms/request, '
                  f'{decode_seconds * 1000 / decode_tokens:.3f} ms/step, '
                  f'{batch_size * decode_tokens / decode_seconds:.1f} generated tokens/s')
        off, on = timings[False], timings[True]
        print(f'KV-cache speedup (off time / on time): prefill {off[0] / on[0]:.3f}x, '
              f'decode {off[1] / on[1]:.3f}x, '
              f'prefill+decode {sum(off) / sum(on):.3f}x')


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if inference:
    benchmark_inference()
    raise SystemExit(0)

# data loading init
if real_data:
    dataset = 'openwebtext'
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    def get_batch(split):
        data = train_data # note ignore split in benchmarking script
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y
else:
    # alternatively, if fixed data is desired to not care about data loading
    x = torch.randint(vocab_size, (batch_size, block_size), device=device)
    y = torch.randint(vocab_size, (batch_size, block_size), device=device)
    get_batch = lambda split: (x, y)

# model init
gptconf = GPTConfig(
    block_size = block_size, # how far back does the model look? i.e. context size
    n_layer = n_layer, n_head = n_head, n_embd = n_embd, # size of the model
    vocab_size = vocab_size,
    dropout = 0, # for determinism
    bias = bias,
)
model = GPT(gptconf)
model.to(device)

optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)

if compile:
    print("Compiling model...")
    model = torch.compile(model) # pytorch 2.0

if profile:
    # useful docs on pytorch profiler:
    # - tutorial https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    # - api https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
    wait, warmup, active = 5, 5, 5
    num_steps = wait + warmup + active
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
        record_shapes=False,
        profile_memory=False,
        with_stack=False, # incurs an additional overhead, disable if not needed
        with_flops=True,
        with_modules=False, # only for torchscript models atm
    ) as prof:

        X, Y = get_batch('train')
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")

            prof.step() # notify the profiler at end of each step

else:

    # simple benchmarking
    synchronize()
    for stage, num_steps in enumerate([10, 20]): # burnin, then benchmark
        t0 = time.time()
        X, Y = get_batch('train')
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")
        synchronize()
        t1 = time.time()
        dt = t1-t0
        mfu = model.estimate_mfu(batch_size * 1 * num_steps, dt)
        if stage == 1:
            print(f"time per iteration: {dt/num_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")
