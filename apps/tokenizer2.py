from collections import Counter
import math
import json
import heapq
import re
from pathlib import Path
from typing import List, Union, Iterable, Iterator
from config import CONFIG

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


class HybridTokenizerLLM:
    """Hybrid tokenizer that blends unigram vocab, WordPiece-style scoring, and BPE-like merges."""

    def __init__(
        self,
        max_sub_len=8,
        min_freq=2,
        pad_token="<PAD>",
        unk_token="<UNK>",
        bos_token="<BOS>",
        eos_token="<EOS>",
        word_start="\u2581",
        alpha=0.5,
        beta=2.0,
        gamma=1.0,
        min_pair_freq=5,
        max_token_length=20,
        length_bonus=0.15,
        merges_per_round=8,
        normalize_whitespace=True,
    ):
        self.max_sub_len = int(max_sub_len)
        self.min_freq = int(min_freq)
        self.vocab = {}
        self.token2id = {}
        self.id2token = {}
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.word_start = word_start

        # Hybrid scoring weights
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.min_pair_freq = int(min_pair_freq)
        # Decoding-time preference toward longer known tokens to reduce char fallback.
        self.length_bonus = float(length_bonus)
        self.merges_per_round = max(1, int(merges_per_round))
        self.normalize_whitespace = bool(normalize_whitespace)

        # Allow merged tokens to grow beyond initial substrings.
        self.max_token_length = int(max_token_length) if max_token_length is not None else max(8, self.max_sub_len * 4)
        self._encode_max_len = self.max_sub_len

        self.vocab[pad_token] = 1.0
        self.vocab[unk_token] = 1.0
        self.vocab[bos_token] = 1.0
        self.vocab[eos_token] = 1.0
        self.vocab[word_start] = 1.0

        # ✅ WAJIB: semua karakter dasar masuk vocab
        base_chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        base_chars += list(" .,!?;:-_/\\'\"()[]{}@#%&*+=<>|`~°²\n\t")
        for ch in base_chars:
            self.vocab[ch] = 1.0


        self._update_token_ids()

    @property
    def special_tokens(self) -> List[str]:
        return [self.pad_token, self.unk_token, self.bos_token, self.eos_token]

    @property
    def bos_token_id(self):
        return self.token2id.get(self.bos_token)

    @property
    def eos_token_id(self):
        return self.token2id.get(self.eos_token)

    def _normalize_text(self, text: str) -> str:
        text = str(text)
        if self.normalize_whitespace:
            text = re.sub(r"\s+", " ", text).strip()
        return text

    def _mark_text(self, text: str) -> str:
        normalized = self._normalize_text(text)
        if not normalized:
            return self.word_start
        # SentencePiece-like boundaries: each word starts with word_start marker.
        return f"{self.word_start}{normalized.replace(' ', self.word_start)}"

    def _iter_corpus(self, corpus: Union[Iterable[str], str]) -> Iterator[str]:
        if isinstance(corpus, str):
            total = sum(1 for _ in open(corpus, "r", encoding="utf-8"))
            with open(corpus, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f, 1):
                    text = self._normalize_text(line)
                    if text:
                        yield text

                    pct = (idx / total) * 100
                    msg = f"[Phase 1: Reading Corpus] {idx}/{total} ({pct:.1f}%)"
                    print(msg, end="\r", flush=True)

            return

        try:
            total = len(corpus)
        except TypeError:
            total = None

        for idx, item in enumerate(corpus, 1):
            text = self._normalize_text(item)
            if text:
                yield text

            if total:
                pct = (idx / total) * 100
                msg = f"[Phase 1: Reading Corpus] {idx}/{total} ({pct:.1f}%)"
            else:
                msg = f"[Phase 1: Reading Corpus] {idx} items"
            print(msg, end="\r", flush=True)

    def _ordered_vocab_items(self):
        specials = self.special_tokens
        items = []
        for tok in specials:
            if tok in self.vocab:
                items.append((tok, float(self.vocab[tok])))

        rest = [(tok, float(score)) for tok, score in self.vocab.items() if tok not in specials]
        rest.sort(key=lambda x: (-x[1], x[0]))
        items.extend(rest)
        return items

    def _normalize_vocab(self):
        cleaned = {}
        for tok, score in self.vocab.items():
            value = float(score)
            if not math.isfinite(value) or value <= 0:
                value = 1e-12
            cleaned[tok] = value

        total = sum(cleaned.values())
        if total <= 0:
            cleaned = {tok: 1.0 for tok in self.special_tokens}
            total = float(len(cleaned))

        self.vocab = {tok: (score / total) for tok, score in cleaned.items()}

    def _trim_vocab(self, target_size: int):
        """Trim vocab while keeping special tokens and bounded char fallback."""
        target_size = max(len(self.special_tokens), int(target_size))
        specials = {tok: self.vocab.get(tok, 1.0) for tok in self.special_tokens}
        all_char_tokens = [
            (tok, score)
            for tok, score in self.vocab.items()
            if tok not in specials and len(tok) == 1
        ]
        all_char_tokens.sort(key=lambda x: x[1], reverse=True)
        # Keep core latin + digit + common punctuation chars for robust OOV fallback.
        essential_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        essential_chars |= set(" .,!?;:-_/\\'\"()[]{}@#%&*+=<>|`~\n\t")
        essential_chars.add(self.word_start)

        char_tokens = {
            tok: score
            for tok, score in all_char_tokens
            if tok in essential_chars
        }
        # Keep some additional frequent single chars, but bound total to protect merge capacity.
        char_budget = min(len(all_char_tokens), max(32, int(target_size * 0.2)))
        char_tokens = dict(
            sorted(char_tokens.items(), key=lambda x: x[1], reverse=True)[:char_budget]
        )        
        # if len(char_tokens) < char_budget:
        #     for tok, score in all_char_tokens:
        #         if tok in char_tokens:
        #             continue
        #         char_tokens[tok] = score
        #         if len(char_tokens) >= char_budget:
        #             break
        keepers = dict(specials)
        keepers.update(char_tokens)

        budget = max(0, target_size - len(keepers))
        ranked = sorted(
            (
                (tok, score)
                for tok, score in self.vocab.items()
                if tok not in keepers
            ),
            key=lambda x: x[1],
            reverse=True,
        )

        trimmed = dict(keepers)
        if len(trimmed) > target_size:
            essentials = list(specials.items())
            chars_ranked = sorted(
                ((tok, score) for tok, score in char_tokens.items()),
                key=lambda x: x[1],
                reverse=True,
            )
            trimmed = dict(essentials)
            remaining = max(0, target_size - len(trimmed))
            trimmed.update(chars_ranked[:remaining])
            self.vocab = trimmed
            return

        trimmed.update(ranked[:budget])
        self.vocab = trimmed

    def _set_training_weights(self, alpha=None, beta=None, gamma=None, min_pair_freq=None):
        if alpha is not None:
            self.alpha = float(alpha)
        if beta is not None:
            self.beta = float(beta)
        if gamma is not None:
            self.gamma = float(gamma)
        if min_pair_freq is not None:
            self.min_pair_freq = int(min_pair_freq)

    def build_initial_vocab(self, corpus: Iterable[str], target_vocab_size: int = None):
        counter = Counter()
        for text in self._iter_corpus(corpus):
            # print(f"\rProcessing corpus... (current vocab size: {len(self.vocab)})", end="", flush=True)
            marked = self._mark_text(text)
            # print("marked:", marked[:50], end="\r", flush=True)
            for i in range(len(marked)):
                # Ensure char fallback coverage.
                # counter[marked[i]] += 1
                # hanya kalau belum ada
                if marked[i] not in self.vocab:
                    counter[marked[i]] += 1

                upper = min(len(marked), i + self.max_sub_len)
                for j in range(i + 2, upper + 1):
                    sub = marked[i:j]
                    counter[sub] += 1

        vocab = {token: freq for token, freq in counter.items() if freq >= self.min_freq}
        if not vocab:
            self._update_token_ids()
            return

        self.vocab.update(vocab)
        if target_vocab_size is not None:
            self._trim_vocab(max(int(target_vocab_size), int(target_vocab_size * 2)))

        self._normalize_vocab()
        self._update_token_ids()
    # def build_initial_vocab(
    #     self,
    #     corpus: Iterable[str],
    #     target_vocab_size: int = None,
    #     progress_callback=None,
    #     show_progress=False,
    # ):

    #     counter = Counter()

    #     corpus_list = list(self._iter_corpus(corpus))
    #     total = len(corpus_list)

    #     for idx, text in enumerate(corpus_list, 1):
    #         marked = self._mark_text(text)

    #         for i in range(len(marked)):
    #             if marked[i] not in self.vocab:
    #                 counter[marked[i]] += 1

    #             upper = min(len(marked), i + self.max_sub_len)

    #             for j in range(i + 2, upper + 1):
    #                 sub = marked[i:j]
    #                 if len(sub) > 6:
    #                     continue
    #                 counter[sub] += 1

    #         # 🔥 PROGRESS REPORT
    #         step = max(1, total // 100)
    #         if idx % step == 0 or idx == total:
    #         # if idx % 10 == 0 or idx == total:
    #             payload = {
    #                 "phase": "build_vocab",
    #                 "current": idx,
    #                 "total": total,
    #                 "progress_pct": (idx / total) * 100,
    #                 "vocab_size": len(self.vocab) + len(counter),
    #             }

    #             if progress_callback:
    #                 progress_callback(payload)
    #             elif show_progress:
    #                 print(
    #                     f"\rPhase2 [build_vocab] {idx}/{total} "
    #                     f"({payload['progress_pct']:.1f}%) "
    #                     f"vocab~{payload['vocab_size']}",
    #                     end="",
    #                     flush=True,
    #                 )

    #     if show_progress:
    #         print()

    #     vocab = {token: freq for token, freq in counter.items() if freq >= self.min_freq}

    #     if not vocab:
    #         self._update_token_ids()
    #         return

    #     self.vocab.update(vocab)

    #     if target_vocab_size is not None:
    #         self._trim_vocab(max(int(target_vocab_size), int(target_vocab_size * 2)))

    #     self._normalize_vocab()
    #     self._update_token_ids()

    def compute_pair_scores(self, corpus: Iterable[str], alpha=None, beta=None, gamma=None, min_pair_freq=None):
        alpha = self.alpha if alpha is None else float(alpha)
        beta = self.beta if beta is None else float(beta)
        gamma = self.gamma if gamma is None else float(gamma)
        min_pair_freq = self.min_pair_freq if min_pair_freq is None else int(min_pair_freq)

        pair_freq = Counter()
        token_freq = Counter()

        total_pairs = 0
        for text in self._iter_corpus(corpus):
            tokens = self.encode_tokens(text)
            if not tokens:
                continue

            token_freq.update(tokens)
            for i in range(len(tokens) - 1):
                a = tokens[i]
                b = tokens[i + 1]
                merged = a + b
                if len(merged) <= self.max_token_length:
                    pair_freq[(a, b)] += 1
                    total_pairs += 1

        if total_pairs == 0:
            return {}

        total_tokens = max(1, sum(token_freq.values()))
        scores = {}
        for (a, b), freq in pair_freq.items():
            if freq < min_pair_freq:
                continue

            merged = a + b
            if merged in self.vocab:
                continue

            p_pair = freq / total_pairs
            p_a = token_freq.get(a, 0) / total_tokens
            p_b = token_freq.get(b, 0) / total_tokens
            # WordPiece-style association signal (PMI-like).
            association = math.log((p_pair + 1e-12) / (p_a * p_b + 1e-12))
            # Unigram-style stability bonus.
            unigram_bonus = math.log(freq + 1.0)
            # BPE-style frequency preference.
            score = alpha * freq + beta * association + gamma * unigram_bonus
            scores[merged] = score

        return scores

    def _add_best_merges(self, scores: dict, remaining_budget: int) -> int:
        if not scores or remaining_budget <= 0:
            return 0

        take_n = min(self.merges_per_round, remaining_budget, len(scores))
        candidates = heapq.nlargest(take_n, scores.items(), key=lambda x: x[1])
        added = 0
        for token, score in candidates:
            if score <= 0 or token in self.vocab:
                continue
            self.vocab[token] = max(score, 1e-12)
            added += 1
        return added

    def _build_progress_payload(self, phase: str, merges: int, max_merges: int, vocab_size: int, target_vocab_size: int) -> dict:
        return {
            "phase": phase,
            "merges": int(merges),
            "max_merges": int(max_merges),
            "vocab_size": int(vocab_size),
            "target_vocab_size": int(target_vocab_size),
        }

    def _default_progress_message(self, payload: dict) -> str:
        vocab_size = int(payload.get("vocab_size", 0))
        target = max(1, int(payload.get("target_vocab_size", 1)))
        merges = int(payload.get("merges", 0))
        max_merges = max(1, int(payload.get("max_merges", 1)))
        vocab_pct = min(100.0, (vocab_size / target) * 100.0)
        merge_pct = min(100.0, (merges / max_merges) * 100.0)
        return (
            f"phase={payload.get('phase')} "
            f"vocab={vocab_size}/{target} ({vocab_pct:5.1f}%) "
            f"merges={merges}/{max_merges} ({merge_pct:5.1f}%)"
        )

    def train(
        self,
        corpus: Union[List[str], str, Iterable[str]],
        vocab_size=500,
        alpha=None,
        beta=None,
        gamma=None,
        min_pair_freq=None,
        max_merges=None,
        prune_every=50,
        progress_callback=None,
        progress_every=25,
        show_progress=False,
        progress_desc="Tokenizer train",
    ):
        self._set_training_weights(alpha=alpha, beta=beta, gamma=gamma, min_pair_freq=min_pair_freq)

        if isinstance(corpus, str):
            return self.train_from_file(
                corpus,
                vocab_size=vocab_size,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                min_pair_freq=self.min_pair_freq,
                max_merges=max_merges,
                prune_every=prune_every,
                progress_callback=progress_callback,
                progress_every=progress_every,
                show_progress=show_progress,
                progress_desc=progress_desc,
            )

        reusable_corpus = list(self._iter_corpus(corpus))
        self.build_initial_vocab(reusable_corpus, target_vocab_size=vocab_size)

        if max_merges is None:
            max_merges = max(0, int(vocab_size) * 2)
        progress_every = max(1, int(progress_every))

        merges = 0
        stagnant_steps = 0
        next_progress_at = progress_every
        last_reported_merges = 0
        pbar = tqdm(total=max_merges, desc=progress_desc, unit="merge", leave=False) if show_progress and tqdm else None

        def report_progress(phase: str):
            nonlocal last_reported_merges
            payload = self._build_progress_payload(
                phase=phase,
                merges=merges,
                max_merges=max_merges,
                vocab_size=len(self.vocab),
                target_vocab_size=int(vocab_size),
            )
            if progress_callback:
                progress_callback(payload)
            if pbar is not None:
                delta = max(0, merges - last_reported_merges)
                if delta:
                    pbar.update(delta)
                pbar.set_postfix_str(f"vocab={len(self.vocab)}/{int(vocab_size)}", refresh=False)
                last_reported_merges = merges
            elif show_progress:
                print("\r" + self._default_progress_message(payload), end="", flush=True)
                last_reported_merges = merges

        while len(self.vocab) < vocab_size and merges < max_merges:
            scores = self.compute_pair_scores(reusable_corpus)
            if not scores:
                break

            remaining = min(vocab_size - len(self.vocab), max_merges - merges)
            added = self._add_best_merges(scores, remaining_budget=remaining)
            if added == 0:
                stagnant_steps += 1
                if stagnant_steps >= 5:
                    break
            else:
                stagnant_steps = 0
            merges += added

            if prune_every and merges % int(prune_every) == 0:
                self._trim_vocab(max(vocab_size, int(vocab_size * 1.5)))
                self._normalize_vocab()
                self._update_token_ids()
            if merges >= next_progress_at or len(self.vocab) >= vocab_size:
                report_progress("train")
                while next_progress_at <= merges:
                    next_progress_at += progress_every

        self._trim_vocab(vocab_size)
        self._normalize_vocab()
        self._update_token_ids()
        report_progress("done")
        if pbar is not None:
            pbar.close()
        elif show_progress:
            print()

    def train_from_file(
        self,
        file_path: str,
        vocab_size=500,
        alpha=None,
        beta=None,
        gamma=None,
        min_pair_freq=None,
        max_merges=None,
        prune_every=50,
        progress_callback=None,
        progress_every=25,
        show_progress=False,
        progress_desc="Tokenizer train(file)",
    ):
        self._set_training_weights(alpha=alpha, beta=beta, gamma=gamma, min_pair_freq=min_pair_freq)
        print("Building initial vocab...")
        self.build_initial_vocab(file_path, target_vocab_size=vocab_size)

        if max_merges is None:
            max_merges = max(0, int(vocab_size) * 2)
        progress_every = max(1, int(progress_every))

        merges = 0
        stagnant_steps = 0
        next_progress_at = progress_every
        last_reported_merges = 0
        pbar = tqdm(total=max_merges, desc=progress_desc, unit="merge", leave=False) if show_progress and tqdm else None

        def report_progress(phase: str):
            nonlocal last_reported_merges
            payload = self._build_progress_payload(
                phase=phase,
                merges=merges,
                max_merges=max_merges,
                vocab_size=len(self.vocab),
                target_vocab_size=int(vocab_size),
            )
            if progress_callback:
                progress_callback(payload)
            if pbar is not None:
                delta = max(0, merges - last_reported_merges)
                if delta:
                    pbar.update(delta)
                pbar.set_postfix_str(f"vocab={len(self.vocab)}/{int(vocab_size)}", refresh=False)
                last_reported_merges = merges
            elif show_progress:
                print("\r" + self._default_progress_message(payload), end="", flush=True)
                last_reported_merges = merges

        print("Training tokenizer...")
        while len(self.vocab) < vocab_size and merges < max_merges:
            scores = self.compute_pair_scores(file_path)
            if not scores:
                break

            remaining = min(vocab_size - len(self.vocab), max_merges - merges)
            added = self._add_best_merges(scores, remaining_budget=remaining)
            if added == 0:
                stagnant_steps += 1
                if stagnant_steps >= 5:
                    break
            else:
                stagnant_steps = 0
            merges += added

            if prune_every and merges % int(prune_every) == 0:
                self._trim_vocab(max(vocab_size, int(vocab_size * 1.5)))
                self._normalize_vocab()
                self._update_token_ids()
            if merges >= next_progress_at or len(self.vocab) >= vocab_size:
                report_progress("train_from_file")
                while next_progress_at <= merges:
                    next_progress_at += progress_every

        self._trim_vocab(vocab_size)
        self._normalize_vocab()
        self._update_token_ids()
        report_progress("done")
        if pbar is not None:
            pbar.close()
        elif show_progress:
            print()

    def _update_token_ids(self):
        ordered_items = self._ordered_vocab_items()
        self.vocab = dict(ordered_items)
        # self._encode_max_len = max(1, min(self.max_token_length, max((len(tok) for tok in self.vocab.keys()), default=self.max_sub_len)))
        self._encode_max_len = self.max_token_length

        self.token2id = {t: i for i, (t, _) in enumerate(ordered_items)}
        self.id2token = {i: t for t, i in self.token2id.items()}

    def encode_tokens(self, text: str) -> List[str]:
        marked = self._mark_text(text)
        n = len(marked)
        dp = [(float("-inf"), None)] * (n + 1)
        dp[0] = (0.0, None)

        for i in range(n):
            if dp[i][0] == float("-inf"):
                continue

            upper = min(n, i + self._encode_max_len)
            for j in range(i + 1, upper + 1):
                token = marked[i:j]
                if token in self.vocab:
                    score = math.log(max(self.vocab[token], 1e-12)) + self.length_bonus * (len(token) - 1)
                    candidate = dp[i][0] + score
                    if candidate > dp[j][0]:
                        dp[j] = (candidate, i)

        tokens = []
        idx = n
        while idx > 0:
            prev = dp[idx][1]
            if prev is None:
                # Character fallback guarantees progress on unseen spans.
                tokens.append(marked[idx - 1])
                idx -= 1
            else:
                tokens.append(marked[prev:idx])
                idx = prev

        return list(reversed(tokens))

    def encode(
        self,
        text: str,
        max_length: int = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> dict:
        tokens = self.encode_tokens(text)
        unk_id = self.token2id.get(self.unk_token, 1)
        ids = [self.token2id.get(t, unk_id) for t in tokens]
        if add_bos:
            if self.bos_token_id is None:
                raise KeyError(f"Missing BOS token in token2id: {self.bos_token}")
            ids.insert(0, self.bos_token_id)
        if add_eos:
            if self.eos_token_id is None:
                raise KeyError(f"Missing EOS token in token2id: {self.eos_token}")
            ids.append(self.eos_token_id)
        # ids = []
        # for t in tokens:
        #     if t in self.token2id:
        #         ids.append(self.token2id[t])
        #     else:
        #         # fallback per karakter
        #         for ch in t:
        #             ids.append(self.token2id.get(ch, unk_id))

        if max_length is None:
            max_length = len(ids)
        attention_mask = [1] * len(ids)

        if len(ids) < max_length:
            pad_len = max_length - len(ids)
            pad_id = self.token2id[self.pad_token]
            ids.extend([pad_id] * pad_len)
            attention_mask.extend([0] * pad_len)
        else:
            ids = ids[:max_length]
            if add_eos and ids:
                ids[-1] = self.eos_token_id
            attention_mask = attention_mask[:max_length]

        return {"input_ids": ids, "attention_mask": attention_mask}

    def encode_ids(
        self,
        text: str,
        max_length: int = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        return self.encode(
            text,
            max_length=max_length,
            add_bos=add_bos,
            add_eos=add_eos,
        )["input_ids"]

    def encode_batch(
        self,
        texts: List[str],
        max_length: int = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> dict:
        batch_input_ids, batch_attention_mask = [], []
        for text in texts:
            encoded = self.encode(
                text,
                max_length=max_length,
                add_bos=add_bos,
                add_eos=add_eos,
            )
            batch_input_ids.append(encoded["input_ids"])
            batch_attention_mask.append(encoded["attention_mask"])
        return {"input_ids": batch_input_ids, "attention_mask": batch_attention_mask}

    def decode_tokens(self, tokens: List[str], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            skip = {self.pad_token, self.bos_token, self.eos_token}
            tokens = [tok for tok in tokens if tok not in skip]
        decoded = "".join(tokens).replace(self.word_start, " ")
        return re.sub(r"\s+", " ", decoded).strip()

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        pad_id = self.token2id[self.pad_token]
        tokens = [
            self.id2token.get(i, self.unk_token)
            for i in ids
            if not skip_special_tokens or i != pad_id
        ]
        return self.decode_tokens(tokens, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, batch_ids: List[List[int]]) -> List[str]:
        return [self.decode(ids) for ids in batch_ids]

    def save_vocab(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "2.1",
            "word_start": self.word_start,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "max_sub_len": self.max_sub_len,
            "max_token_length": self.max_token_length,
            "min_freq": self.min_freq,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "min_pair_freq": self.min_pair_freq,
            "length_bonus": self.length_bonus,
            "merges_per_round": self.merges_per_round,
            "normalize_whitespace": self.normalize_whitespace,
            "vocab": self.vocab,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        if isinstance(loaded, dict) and "vocab" in loaded:
            self.word_start = loaded.get("word_start", self.word_start)
            self.pad_token = loaded.get("pad_token", self.pad_token)
            self.unk_token = loaded.get("unk_token", self.unk_token)
            self.bos_token = loaded.get("bos_token", self.bos_token)
            self.eos_token = loaded.get("eos_token", self.eos_token)
            self.max_sub_len = int(loaded.get("max_sub_len", self.max_sub_len))
            self.max_token_length = int(loaded.get("max_token_length", self.max_token_length))
            self.min_freq = int(loaded.get("min_freq", self.min_freq))
            self.alpha = float(loaded.get("alpha", self.alpha))
            self.beta = float(loaded.get("beta", self.beta))
            self.gamma = float(loaded.get("gamma", self.gamma))
            self.min_pair_freq = int(loaded.get("min_pair_freq", self.min_pair_freq))
            self.length_bonus = float(loaded.get("length_bonus", self.length_bonus))
            self.merges_per_round = int(loaded.get("merges_per_round", self.merges_per_round))
            self.normalize_whitespace = bool(loaded.get("normalize_whitespace", self.normalize_whitespace))
            self.vocab = loaded["vocab"]
        else:
            # Backward compatible path (legacy file only contains vocab mapping).
            self.vocab = loaded

        for tok in self.special_tokens:
            if tok not in self.vocab:
                self.vocab[tok] = 1.0

        self._normalize_vocab()
        self._update_token_ids()

    def save_token_ids(self, token2id_path: str, id2token_path: str = None):
        Path(token2id_path).parent.mkdir(parents=True, exist_ok=True)
        with open(token2id_path, "w", encoding="utf-8") as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)

        if id2token_path:
            Path(id2token_path).parent.mkdir(parents=True, exist_ok=True)
            id2token_str = {str(k): v for k, v in self.id2token.items()}
            with open(id2token_path, "w", encoding="utf-8") as f:
                json.dump(id2token_str, f, ensure_ascii=False, indent=2)

    def load_token_ids(self, token2id_path: str, id2token_path: str = None):
        with open(token2id_path, "r", encoding="utf-8") as f:
            self.token2id = json.load(f)

        if id2token_path:
            with open(id2token_path, "r", encoding="utf-8") as f:
                id2token_str = json.load(f)
                self.id2token = {int(k): v for k, v in id2token_str.items()}
        else:
            self.id2token = {v: k for k, v in self.token2id.items()}

        next_id = max(self.id2token.keys(), default=-1) + 1
        for tok in self.special_tokens:
            if tok in self.token2id:
                continue
            self.token2id[tok] = next_id
            self.id2token[next_id] = tok
            next_id += 1

    def save_all(self, vocab_path: str, token2id_path: str, id2token_path: str = None):
        self.save_vocab(vocab_path)
        self.save_token_ids(token2id_path, id2token_path)

    def load_all(self, vocab_path: str, token2id_path: str, id2token_path: str = None):
        self.load_vocab(vocab_path)
        self.load_token_ids(token2id_path, id2token_path)


def build_tokenizer_from_config(cfg=CONFIG) -> HybridTokenizerLLM:
    """
    Build tokenizer instance using values from config object/class.
    """
    return HybridTokenizerLLM(
        max_sub_len=cfg.MAX_SUB_LEN,
        min_freq=cfg.MIN_FREQ,
        pad_token=cfg.PAD_TOKEN,
        unk_token=cfg.UNK_TOKEN,
        bos_token=getattr(cfg, "BOS_TOKEN", "<BOS>"),
        eos_token=getattr(cfg, "EOS_TOKEN", "<EOS>"),
        alpha=cfg.ALPHA,
        beta=cfg.BETA,
        gamma=cfg.GAMMA,
        min_pair_freq=getattr(cfg, "MIN_PAIR_FREQ", 2),
        max_token_length=getattr(cfg, "MAX_TOKEN_LENGTH", None),
        length_bonus=getattr(cfg, "LENGTH_BONUS", 0.15),
        merges_per_round=getattr(cfg, "MERGES_PER_ROUND", 8),
        normalize_whitespace=getattr(cfg, "NORMALIZE_WHITESPACE", True),
    )


def load_tokenizer_from_config(cfg=CONFIG) -> HybridTokenizerLLM:
    """
    Build and load tokenizer artifacts (vocab/token ids) from config paths.
    """
    tokenizer = build_tokenizer_from_config(cfg)
    tokenizer.load_all(
        str(cfg.VOCAB_PATH),
        str(cfg.TOKEN2ID_PATH),
        str(cfg.ID2TOKEN_PATH),
    )
    return tokenizer


if __name__ == "__main__":
    print("=" * 60)
    print("HybridTokenizerLLM with Config Integration")
    print("=" * 60)

    corpus = CONFIG.get_corpus()
    print(f"\nCorpus loaded: {len(corpus)} lines")
    print(f"First line: {corpus[0][:50]}...")

    tokenizer = build_tokenizer_from_config(CONFIG)

    print(f"\nTraining tokenizer with vocab_size={CONFIG.VOCAB_SIZE}...")
    tokenizer.train(
        corpus,
        vocab_size=CONFIG.VOCAB_SIZE,
        alpha=CONFIG.ALPHA,
        beta=CONFIG.BETA,
        gamma=CONFIG.GAMMA,
        min_pair_freq=getattr(CONFIG, "MIN_PAIR_FREQ", 2),
        max_merges=getattr(CONFIG, "MAX_MERGES", None),
        prune_every=getattr(CONFIG, "PRUNE_EVERY", 50),
        progress_every=max(1, int(getattr(CONFIG, "PROGRESS_EVERY", getattr(CONFIG, "PRUNE_EVERY", 25)))),
        show_progress=True,
        progress_desc="Tokenizer train",
    )
    print(f"Vocab size: {len(tokenizer.vocab)}")

    test_text = "mengembangkan tokenizer"
    encoded = tokenizer.encode(test_text, max_length=CONFIG.MAX_LENGTH or len(test_text))
    print(f"\nInput IDs: {encoded['input_ids'][:10]}...")
    print(f"Attention mask: {encoded['attention_mask'][:10]}...")

    tokenizer.save_all(
        str(CONFIG.VOCAB_PATH),
        str(CONFIG.TOKEN2ID_PATH),
        str(CONFIG.ID2TOKEN_PATH),
    )
    print(f"\nSaved to {CONFIG.VOCAB_PATH}")

    tokenizer2 = build_tokenizer_from_config(CONFIG)
    tokenizer2.load_all(
        str(CONFIG.VOCAB_PATH),
        str(CONFIG.TOKEN2ID_PATH),
        str(CONFIG.ID2TOKEN_PATH),
    )
    print(f"Loaded vocab size: {len(tokenizer2.vocab)}")
