import argparse
import os
import random

import numpy as np
import torch
import torchaudio

from diffwave.inference import predict

DEFAULT_MODEL_PATH = "diffwave/weights.pt"
DEFAULT_SOURCE_DIR = "dataset"
DEFAULT_OUTPUT_DIR = "synthetic"


def find_spectrograms(source_dir):
    """Pairs every <name>.wav in source_dir with its precomputed <name>.wav.spec.npy."""
    pairs = []
    for fname in sorted(os.listdir(source_dir)):
        if not fname.endswith('.wav'):
            continue
        spec_path = os.path.join(source_dir, f'{fname}.spec.npy')
        if os.path.exists(spec_path):
            pairs.append((os.path.splitext(fname)[0], spec_path))
    return pairs


def generate(num_samples, source_dir, model_path, output_dir, device, fast_sampling, seed=None):
    pairs = find_spectrograms(source_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No '<name>.wav.spec.npy' spectrogram files found in {source_dir}. "
            f"Run diffwave/preprocess.py on that directory first."
        )

    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)
    counts = {stem: 0 for stem, _ in pairs}

    for i in range(num_samples):
        stem, spec_path = rng.choice(pairs)
        spectrogram = torch.from_numpy(np.load(spec_path))

        audio, sample_rate = predict(
            spectrogram, model_dir=model_path, device=device, fast_sampling=fast_sampling,
        )

        counts[stem] += 1
        out_path = os.path.join(output_dir, f'{stem}_synth{counts[stem]}.wav')
        torchaudio.save(out_path, audio.cpu(), sample_rate=sample_rate)
        print(f'[{i + 1}/{num_samples}] wrote {out_path}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate new synthetic audio recordings with a trained DiffWave model.'
    )
    parser.add_argument('num_samples', type=int, help='number of new recordings to generate')
    parser.add_argument('--source_dir', default=DEFAULT_SOURCE_DIR,
                         help='directory of source .wav files with precomputed <name>.wav.spec.npy '
                              'spectrograms to condition generation on')
    parser.add_argument('--model_path', default=DEFAULT_MODEL_PATH,
                         help='path to the DiffWave checkpoint (.pt)')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR,
                         help='directory to write generated .wav files to')
    parser.add_argument('--fast', action='store_true', help='use fast sampling (fewer diffusion steps)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducible sampling')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate(
        num_samples=args.num_samples,
        source_dir=args.source_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=torch.device(args.device),
        fast_sampling=args.fast,
        seed=args.seed,
    )
