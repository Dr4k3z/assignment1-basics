from __future__ import annotations

import os
import pickle
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import regex as re

from .globals import PAT, SPECIAL_TOKENS

type PreToken = tuple[bytes, ...]
type PreTokenCounter = Counter[PreToken]
type Pair = tuple[bytes, bytes]
type PairCounter = Counter[Pair]

type Token = bytes
type Vocab = dict[int, Token]
type Merges = list[Pair]


def _count_adjacent_pairs(pretoken_counts: PreTokenCounter) -> PairCounter:
    pairs_count = Counter()

    for pre_token, freq in pretoken_counts.items():
        for w1, w2 in zip(pre_token, pre_token[1:], strict=False):
            pairs_count[(w1, w2)] += freq

    return pairs_count


def _select_best_pair(pair_counts: PairCounter) -> Pair:
    return max(pair_counts, key=lambda pair: (pair_counts[pair], pair))


def _merge_pair_in_pretoken(pretoken: PreToken, pair: Pair) -> PreToken:
    merged = []
    i = 0

    while i < len(pretoken):
        if i < len(pretoken) - 1 and pretoken[i] == pair[0] and pretoken[i + 1] == pair[1]:
            merged.append(pretoken[i] + pretoken[i + 1])
            i += 2
        else:
            merged.append(pretoken[i])
            i += 1

    return tuple(merged)


def _merge_pair(pretoken_counts: PreTokenCounter, pair: Pair) -> PreTokenCounter:
    new_counts = Counter()

    for pre_token, frequency in pretoken_counts.items():
        merged_pre_token = _merge_pair_in_pretoken(pre_token, pair)
        new_counts[merged_pre_token] += frequency

    return new_counts


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


@dataclass
class PreTokenizer:
    pretoken_counter: PreTokenCounter = field(default_factory=Counter)
    special_tokens: list[str] = field(default_factory=list)

    @staticmethod
    def _word_to_pretoken(word: str) -> PreToken:
        return tuple(bytes([b]) for b in word.encode("utf-8"))

    def _split_on_special_tokens(self, text: str) -> list[str]:
        if not self.special_tokens:
            return [text]

        pattern = "|".join(re.escape(token) for token in self.special_tokens)
        return re.split(pattern, text)

    def fit(self, text: str) -> PreTokenizer:
        for segment in self._split_on_special_tokens(text):
            for match in re.finditer(PAT, segment):
                pretoken_str = match.group()
                pretoken = self._word_to_pretoken(pretoken_str)
                self.pretoken_counter[pretoken] += 1

        return self


class BPETrainer:
    def __init__(
        self, input_path: str | Path, vocab_size: int, special_tokens: list[str] | None = None
    ) -> None:
        self.input_path = Path(input_path)
        self.vocab_size = vocab_size
        self.vocab: Vocab = {x: bytes([x]) for x in range(256)}
        self.merges: Merges = []

        self.special_tokens = special_tokens if special_tokens is not None else SPECIAL_TOKENS

        for special_token in self.special_tokens:
            token = special_token.encode("utf-8")

            if token not in self.vocab.values():
                self.vocab[len(self.vocab)] = token

        self.pretoken_counts = self._build_pretoken_counts()

    @staticmethod
    def save(vocab: Vocab, merges: Merges, output_dir: Path | str) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with (output_dir / "vocab.pkl").open("wb") as file:
            pickle.dump(vocab, file)

        with (output_dir / "merges.pkl").open("wb") as file:
            pickle.dump(merges, file)

    def _build_pretoken_counts(self) -> PreTokenCounter:
        total_counts: PreTokenCounter = Counter()

        # For now, just use the first special token as the chunk delimiter.
        # In TinyStories this is usually "<|endoftext|>".
        split_special_token = self.special_tokens[0].encode("utf-8")

        with self.input_path.open("rb") as file:
            boundaries = _find_chunk_boundaries(
                file=file,
                desired_num_chunks=4,
                split_special_token=split_special_token,
            )

            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
                file.seek(start)
                chunk = file.read(end - start).decode("utf-8", errors="ignore")
                pretokenizer = PreTokenizer(special_tokens=self.special_tokens).fit(chunk)
                total_counts.update(pretokenizer.pretoken_counter)

        return total_counts

    def _add_merge_to_vocab(self, pair: Pair) -> None:
        new_token = pair[0] + pair[1]
        self.vocab[len(self.vocab)] = new_token
        self.merges.append(pair)

    def train(self) -> tuple[Vocab, Merges]:
        pretoken_counts = self.pretoken_counts

        while len(self.vocab) < self.vocab_size:
            pair_counts = _count_adjacent_pairs(pretoken_counts)

            if not pair_counts:
                break

            best_pair = _select_best_pair(pair_counts)
            self._add_merge_to_vocab(best_pair)
            pretoken_counts = _merge_pair(pretoken_counts, best_pair)

        return self.vocab, self.merges


def _get_pairs(pretoken: PreToken) -> list[Pair]:
    return list(zip(pretoken, pretoken[1:], strict=False))


def _bpe_encode_pretoken(
    pretoken: PreToken,
    merge_ranks: dict[Pair, int],
    token_to_id: dict[bytes, int],
) -> list[int]:
    while True:
        pairs = _get_pairs(pretoken)

        ranked_pairs = [pair for pair in pairs if pair in merge_ranks]

        if not ranked_pairs:
            break

        best_pair = min(ranked_pairs, key=lambda pair: merge_ranks[pair])
        pretoken = _merge_pair_in_pretoken(pretoken, best_pair)

    return [token_to_id[token] for token in pretoken]


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.token_to_id = {token: idx for idx, token in vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}

        self.special_tokens = special_tokens if special_tokens is not None else SPECIAL_TOKENS
        self.special_token_bytes = {token: token.encode("utf-8") for token in self.special_tokens}
        self.special_token_ids = {
            token: self.token_to_id[token.encode("utf-8")] for token in self.special_tokens
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath: Path | str,
        merges_filepath: Path | str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        vocab_filepath = Path(vocab_filepath)
        merges_filepath = Path(merges_filepath)

        with vocab_filepath.open("rb") as file:
            vocab = pickle.load(file)

        with merges_filepath.open("rb") as file:
            merges = pickle.load(file)

        return cls(vocab, merges, special_tokens)

    @staticmethod
    def _str_to_pretoken(text: str) -> tuple[bytes, ...]:
        return tuple(bytes([b]) for b in text.encode("utf-8"))

    def _encode_pretoken(self, pretoken: tuple[bytes, ...]) -> list[int]:
        while True:
            pairs = list(zip(pretoken, pretoken[1:], strict=False))
            ranked_pairs = [pair for pair in pairs if pair in self.merge_ranks]

            if not ranked_pairs:
                break

            best_pair = min(ranked_pairs, key=lambda pair: self.merge_ranks[pair])
            pretoken = _merge_pair_in_pretoken(pretoken, best_pair)

        return [self.token_to_id[token] for token in pretoken]

    def _split_text_on_special_tokens(self, text: str) -> list[str]:
        if not self.special_tokens:
            return [text]

        special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(token) for token in special_tokens) + ")"
        return re.split(pattern, text)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []

        for chunk in self._split_text_on_special_tokens(text):
            if chunk == "":
                continue

            if chunk in self.special_token_ids:
                ids.append(self.special_token_ids[chunk])
                continue

            for match in re.finditer(PAT, chunk):
                pretoken_str = match.group()
                pretoken = self._str_to_pretoken(pretoken_str)
                ids.extend(self._encode_pretoken(pretoken))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[idx] for idx in ids)
        return raw.decode("utf-8", errors="replace")
