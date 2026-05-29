"""FASTA export/import for AEGIS genomes.

Encodes population genomes as DNA-like sequences for input to read simulators
(Badread, etc.) and provides a lossless round-trip back to the original bit
array via a sidecar mapping file.

Encoding: 4-letter packing of 2 bits per base — 00→A, 01→C, 10→G, 11→T.
A global uniform-random XOR mask is applied to every individual's flattened
genome before packing, to debias base composition (otherwise fit individuals
with 1-heavy bitstrings would produce T-heavy sequences). XOR preserves
Hamming distance between every pair of individuals exactly, so all population
structure relevant to introgression detection is retained.

The mapping JSON stores the XOR mask, the original genome shape, and the
encoding table — everything needed to invert the transform.
"""

import json
import pathlib
from typing import Tuple, List

import numpy as np

BASES = np.array(["A", "C", "G", "T"])
BASE_TO_PAIR = {"A": (0, 0), "C": (0, 1), "G": (1, 0), "T": (1, 1)}
ENCODING_NAME = "4-letter-v1"


def encode_population_to_fasta(
    population,
    output_dir: pathlib.Path,
    name: str = "dump",
    mask_seed: int = 0,
    line_width: int = 80,
) -> Tuple[pathlib.Path, pathlib.Path]:
    """Write population genomes to <output_dir>/<name>.genome.fasta + <name>.mapping.json.

    Returns (fasta_path, mapping_path).
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    genome_array = population.genomes.array
    n_individuals = len(genome_array)
    original_shape = tuple(int(x) for x in genome_array.shape)

    flat = genome_array.reshape(n_individuals, -1).astype(np.uint8)
    bits_per_individual = int(flat.shape[1])

    pad = bits_per_individual % 2
    if pad:
        flat = np.concatenate([flat, np.zeros((n_individuals, 1), dtype=np.uint8)], axis=1)
    padded_length = bits_per_individual + pad

    mask_rng = np.random.default_rng(mask_seed)
    mask = mask_rng.integers(0, 2, size=padded_length, dtype=np.uint8)
    masked = flat ^ mask[np.newaxis, :]

    pairs = masked.reshape(n_individuals, padded_length // 2, 2)
    base_indices = pairs[:, :, 0] * 2 + pairs[:, :, 1]
    sequences = BASES[base_indices]

    birthdays = getattr(population, "birthdays", None)
    ages = getattr(population, "ages", None)

    fasta_path = output_dir / f"{name}.genome.fasta"
    with open(fasta_path, "w") as fh:
        for i in range(n_individuals):
            birthday = int(birthdays[i]) if birthdays is not None else -1
            age = int(ages[i]) if ages is not None else -1
            header = f">ind_{i}|birthday={birthday}|age={age}"
            fh.write(header + "\n")
            seq = "".join(sequences[i].tolist())
            for j in range(0, len(seq), line_width):
                fh.write(seq[j : j + line_width] + "\n")

    mapping = {
        "encoding": ENCODING_NAME,
        "bit_pair_to_base": {"00": "A", "01": "C", "10": "G", "11": "T"},
        "original_shape": list(original_shape),
        "bits_per_individual": bits_per_individual,
        "padding_bits": pad,
        "n_individuals": int(n_individuals),
        "xor_mask_hex": mask.tobytes().hex(),
        "mask_seed": int(mask_seed),
    }
    mapping_path = output_dir / f"{name}.mapping.json"
    with open(mapping_path, "w") as fh:
        json.dump(mapping, fh, indent=2)

    return fasta_path, mapping_path


def decode_fasta_to_genomes(
    fasta_path: pathlib.Path,
    mapping_path: pathlib.Path,
) -> Tuple[np.ndarray, List[str]]:
    """Read FASTA + mapping JSON, return (genome_array, record_ids).

    genome_array has the same shape and dtype (bool) as the original
    population.genomes.array.
    """
    fasta_path = pathlib.Path(fasta_path)
    mapping_path = pathlib.Path(mapping_path)

    with open(mapping_path) as fh:
        mapping = json.load(fh)

    if mapping["encoding"] != ENCODING_NAME:
        raise ValueError(f"Unsupported encoding {mapping['encoding']!r}; expected {ENCODING_NAME!r}")

    mask = np.frombuffer(bytes.fromhex(mapping["xor_mask_hex"]), dtype=np.uint8).copy()
    original_shape = tuple(mapping["original_shape"])
    bits_per_individual = int(mapping["bits_per_individual"])
    pad = int(mapping["padding_bits"])
    padded_length = bits_per_individual + pad
    n_expected = int(mapping["n_individuals"])

    assert mask.shape == (padded_length,), f"mask length {mask.shape} != padded length {padded_length}"

    record_ids: List[str] = []
    sequences: List[str] = []
    current_id = None
    current_chunks: List[str] = []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences.append("".join(current_chunks))
                    record_ids.append(current_id)
                current_id = line[1:]
                current_chunks = []
            else:
                current_chunks.append(line)
        if current_id is not None:
            sequences.append("".join(current_chunks))
            record_ids.append(current_id)

    n_records = len(sequences)
    assert n_records == n_expected, f"FASTA has {n_records} records but mapping says {n_expected}"

    seq_len = padded_length // 2
    flat = np.zeros((n_records, padded_length), dtype=np.uint8)
    for i, seq in enumerate(sequences):
        if len(seq) != seq_len:
            raise ValueError(f"Record {record_ids[i]!r} has length {len(seq)}; expected {seq_len}")
        for j, base in enumerate(seq):
            b0, b1 = BASE_TO_PAIR[base]
            flat[i, 2 * j] = b0
            flat[i, 2 * j + 1] = b1

    flat ^= mask[np.newaxis, :]

    if pad:
        flat = flat[:, :bits_per_individual]

    decoded = flat.reshape((n_records,) + original_shape[1:]).astype(np.bool_)
    return decoded, record_ids
