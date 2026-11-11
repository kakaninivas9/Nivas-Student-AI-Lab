# Nivas Student AI Lab

A learning-focused adaptation of [Aira-Ai-for-students](https://github.com/adameehanairagroup/Aira-Ai-for-students), with original credit and MIT license retained. Most architecture, training and inference code is upstream work. Improvements were prepared with Codex assistance. See [UPSTREAM.md](UPSTREAM.md).

## Added in this adaptation

- Save evaluated checkpoints after training updates, including one-step runs.

- Validate model dimensions, attention-head divisibility and dropout before model construction.
- Reject empty or overlong contexts and invalid sampling settings with readable errors.
- Check training arguments and dataset split lengths before batch sampling.
- Report missing resume checkpoints and reject vocabulary mismatches rather than silently deleting characters.
- Load training resume checkpoints with `weights_only=True`.
- CPU model regression tests, CI and a synthetic local sample corpus.
- Training console messages compatible with legacy Windows encodings.

## Run a small experiment

Use Python 3.11+ in a virtual environment:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py --data_path examples/corpus.txt --max_iters 1 --batch_size 2 --block_size 8 --n_embd 16 --n_head 2 --n_layer 1 --eval_iters 1 --eval_interval 1 --warmup 0
```

This command uses a local synthetic corpus. The upstream default command without `--data_path` can download TinyStories; dataset terms and suitability require separate review. The one-step example checks the pipeline and does not produce a useful assistant.

## Validation and limits

Tests cover invalid configurations, finite forward/backward loss, context limits, sampling checks and output token counts. CPU smoke training is checked separately. No pretrained weights, accuracy scores or claims of original model architecture are included.

The inherited NumPy inference and story interfaces are retained for study and have not received comprehensive validation. `--resume` restores model weights and vocabulary, not optimizer/RNG progress; it is not exact resumption. Only load checkpoints you trust. Generated text can be inaccurate or inappropriate; inherited word filters are not a child-safety guarantee.

[Original documentation](UPSTREAM-README.md) records the upstream project and is not a claim that this adaptation was independently authored.
