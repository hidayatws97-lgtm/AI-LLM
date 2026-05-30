# Dokumentasi Modul: Tokenizer, Dataset, Embedding, Attention, Transformer

Dokumen ini menjelaskan tujuan, fungsi utama, parameter penting, dan alur input-output dari modul:
- `tokenizer2.py`
- `dataset2.py`
- `embedding.py`
- `attention.py`
- `transformerBlock.py`
- `transformerStack.py`
- `gptModel.py`
- `finetune2.py`
- `tools/generate_gpt_finetune2.py`
- `config.py` (sebagai pusat konfigurasi)

## Quickstart Pipeline (5 Langkah)

Jalankan dari root project:
- `c:\Projects\Python\AI\LLM\hai06`

Urutan praktis yang disarankan:
- langkah 1-3 untuk workflow GPT dasar
- langkah 4-5 untuk workflow instruction fine-tuning dan chatbot

### 1) Train GPT Dasar

```powershell
.\venv\Scripts\python.exe tools\train_gpt.py --env production --max-steps 10000 --save-every 500 --save-dir reports\checkpoints
```

Output utama:
- checkpoint periodik: `reports\checkpoints\gpt_step_*.pt`
- checkpoint terakhir: `reports\checkpoints\gpt_last.pt`

### 2) Evaluasi Checkpoint Dasar

```powershell
.\venv\Scripts\python.exe tools\eval_model.py --env production --checkpoint reports\checkpoints\gpt_last.pt --steps 500 --log-every 25
```

Metrik utama:
- `loss`, `ppl`, `top1_acc`, `top5_acc`, `score`, `status`

### 3) Generate dari Checkpoint Dasar

```powershell
.\venv\Scripts\python.exe tools\generate_gpt.py --env production --checkpoint reports\checkpoints\gpt_last.pt --prompt "djarum foundation" --max-new-tokens 20 --temperature 0.9 --top-k 40 --top-p 0.95
```

Catatan:
- Untuk mode interaktif, jalankan `generate_gpt.py` tanpa `--prompt`.
- Jika belum ada tokenizer/token file, siapkan dulu melalui pipeline tokenizer/dataset di bagian tools.

### 4) Fine-tune dari File Teks

```powershell
.\venv\Scripts\python.exe .\finetune2.py --env production --data-path data\finetune.txt --lr 5e-6 --epochs 2 --steps-per-epoch 1000 --save-every 200
```

Output utama:
- checkpoint periodik: `reports\checkpoints\gpt_finetuned_v2_step_*.pt`
- checkpoint akhir: `reports\checkpoints\gpt_finetuned_v2.pt`

### 5) Chat dari Model Fine-tuned

```powershell
.\venv\Scripts\python.exe tools\generate_gpt_finetune2.py --env production --checkpoint reports\checkpoints\gpt_finetuned_v2.pt
```

Mode:
- interaktif chat-style dengan format `### Instruction:` / `### Response:`

## 1) Gambaran Arsitektur

Alur data end-to-end:

```text
Raw text (corpus.txt / input user)
        |
        v
HybridTokenizerLLM (tokenizer2.py)
  - encode_tokens -> token string
  - token2id      -> integer ids
        |
        v
Pretokenize (dataset2.py -> tokens_*.bin)
        |
        v
Packed/Sliding Dataset (dataset2.py)
  - input_ids, labels (shifted)
        |
        v
EmbeddingLayer (embedding.py)
  - token embedding + positional embedding
  - output tensor (B, T, D)
        |
        v
MultiHeadSelfAttention (attention.py)
  - causal self-attention
  - output tensor (B, T, D)
        |
        v
TransformerStack (transformerStack.py)
  - stack N transformer block (decoder-only)
  - output tensor (B, T, D)
        |
        v
GPTModel (gptModel.py)
  - embedding + transformer stack + lm head
  - helper build_training_components() untuk model+tokenizer+dataloader
        |
        v
Fine-tuning text stream (`finetune2.py`)
  - tokenizer.encode() -> token ids
  - batch dict {"input_ids", "labels"}
  - model.forward_batch(batch)
        |
        v
Instruction-chat inference (`tools/generate_gpt_finetune2.py`)
  - format prompt `### Instruction:` / `### Response:`
  - autoregressive decoding
  - extract response for chatbot output
```

`config.py` mengontrol hampir semua parameter agar tiap modul konsisten.

## 2) Modul `config.py`

Tujuan:
- Menyediakan konfigurasi terpusat untuk `development`, `production`, `testing`.
- Menyimpan path artefak tokenizer + dataset.
- Menyediakan parameter tokenizer, dataset, dan embedding.

Objek penting:
- `get_config(env)` -> pilih kelas config berdasarkan env.
- `CONFIG` -> default config berdasarkan env var `TOKENIZER_ENV`.

Parameter kunci:
- Tokenizer:
  - `VOCAB_SIZE`, `MIN_FREQ`, `MAX_SUB_LEN`
  - `ALPHA`, `BETA`, `GAMMA`, `MIN_PAIR_FREQ`
  - `MAX_TOKEN_LENGTH`, `PRUNE_EVERY`, `MERGES_PER_ROUND`
- Dataset:
  - `DATASET_TOKEN_FILE`, `DATASET_BLOCK_SIZE`, `DATASET_BATCH_SIZE`
  - `DATASET_TYPE` (`packed`/`sliding`), `SLIDING_STRIDE`
  - `DATASET_DTYPE`, `DATASET_NUM_WORKERS`, dsb.
- Embedding:
  - `EMBEDDING_DIM`
  - `EMBEDDING_DROPOUT`
  - `POSITIONAL_EMBEDDING_TYPE` (`learned`/`sinusoidal`)
  - `EMBEDDING_MAX_SEQ_LEN`
- Attention:
  - `ATTENTION_NUM_HEADS`
  - `ATTENTION_DROPOUT`
  - `ATTENTION_USE_BIAS`
  - `ATTENTION_MAX_SEQ_LEN`
- Transformer:
  - `TRANSFORMER_NUM_LAYERS`
  - `TRANSFORMER_MLP_RATIO`
  - `TRANSFORMER_DROPOUT`
  - `TRANSFORMER_USE_BIAS`

## 3) Modul `tokenizer2.py`

### 3.1 Tujuan

Menyediakan tokenizer hybrid (gabungan ide unigram + WordPiece-like + BPE-like merge) untuk membentuk vocabulary dan mengubah teks menjadi token/id.

### 3.2 Kelas Utama: `HybridTokenizerLLM`

Fungsi utama:
- `train(...)` / `train_from_file(...)`:
  - Melatih vocabulary dari corpus.
  - Menghasilkan `vocab`, `token2id`, `id2token`.
- `encode_tokens(text)`:
  - Mengubah teks menjadi list token.
- `encode(text, max_length=None)`:
  - Menghasilkan dict:
    - `input_ids`
    - `attention_mask`
- `decode(ids)`:
  - Mengembalikan teks dari ID.
- `save_all(...)` / `load_all(...)`:
  - Simpan dan muat artefak tokenizer.

Parameter inti konstruktor:
- `max_sub_len`, `min_freq`
- `pad_token`, `unk_token`
- `alpha`, `beta`, `gamma`
- `min_pair_freq`, `max_token_length`
- `length_bonus`, `merges_per_round`
- `normalize_whitespace`

### 3.3 Factory helper (integrasi config)

- `build_tokenizer_from_config(cfg=CONFIG)`:
  - Build tokenizer dengan parameter dari config.
- `load_tokenizer_from_config(cfg=CONFIG)`:
  - Build + load artefak (`VOCAB_PATH`, `TOKEN2ID_PATH`, `ID2TOKEN_PATH`).

### 3.4 Ilustrasi Input -> Output Tokenizer

Contoh:

```text
Input text:
"mengembangkan tokenizer"

Output encode_tokens:
["▁meng", "embangkan", "▁token", "izer"]   # ilustratif, token aktual tergantung vocab

Output encode:
{
  "input_ids": [53, 981, 77, 412],
  "attention_mask": [1, 1, 1, 1]
}
```

## 4) Modul `dataset2.py`

### 4.1 Tujuan

Menyiapkan data training LLM dari token IDs secara efisien:
- pretokenization corpus ke file biner
- dataset chunking (packed/sliding)
- dataloader factory

### 4.2 Fungsi Utama

#### `pretokenize_corpus(corpus_path, tokenizer, ...)`

Tujuan:
- Membaca corpus line-by-line
- Menjalankan tokenization
- Konversi token -> id
- Simpan ke file biner `tokens_*.bin` (dtype sesuai config)

Parameter penting:
- `output_path` (default `CONFIG.PRETOKENIZE_OUTPUT_PATH`)
- `dtype` (default `CONFIG.PRETOKENIZE_DTYPE`)
- `log_interval` (default `CONFIG.PRETOKENIZE_LOG_INTERVAL`)

#### `PackedDataset`

Karakteristik:
- Memakai `np.memmap` dari token file
- Membuat blok urutan tetap
- `labels` adalah `input_ids` yang digeser 1 langkah

Output `__getitem__`:
- `{"input_ids": x, "labels": y}`
- Bentuk:
  - `x`: `(block_size,)`
  - `y`: `(block_size,)`

#### `SlidingWindowDataset`

Karakteristik:
- Jendela geser dengan `stride`
- Lebih fleksibel, sample lebih banyak

Output sama:
- `{"input_ids": x, "labels": y}`

#### `create_dataloader(...)`

Factory DataLoader:
- Pilih dataset via `dataset_type`
- Terapkan parameter batch/worker/memory dari config atau override

### 4.3 Ilustrasi Input -> Output Dataset

Misal token stream:

```text
[10, 11, 12, 13, 14, 15, 16]
block_size = 4
```

Sample pertama (`PackedDataset`):

```text
chunk       = [10, 11, 12, 13, 14]
input_ids   = [10, 11, 12, 13]
labels      = [11, 12, 13, 14]
```

Tujuan label shifting:
- model belajar menebak token berikutnya.

## 5) Modul `embedding.py`

### 5.1 Tujuan

Mengubah token IDs menjadi representasi vektor:
- token embedding
- positional embedding
- dropout

### 5.2 Komponen

#### `SinusoidalPositionalEncoding`
- Positional encoding fixed (non-trainable parameter, disimpan sebagai buffer).

#### `EmbeddingLayer`

Proses `forward(input_ids)`:
1. Lookup token embedding -> `(B, T, D)`
2. Buat positional embedding -> `(B, T, D)` (learned/sinusoidal)
3. Jumlahkan token + positional
4. Dropout
5. Return tensor `(B, T, D)`

Validasi:
- Jika `T > max_seq_len`, akan raise error.

Parameter utama:
- `vocab_size`
- `d_model` (`EMBEDDING_DIM`)
- `max_seq_len`
- `dropout`
- `pos_type` (`learned`/`sinusoidal`)

### 5.3 Helper Integrasi

- `EmbeddingLayer.from_config(cfg, vocab_size=None, max_seq_len=None)`
- `build_embedding_from_tokenizer(tokenizer, cfg=CONFIG, ...)`
- `build_tokenizer_and_embedding(cfg=CONFIG, load_tokenizer=True)`
- `forward_batch(batch)`:
  - memakai `batch["input_ids"]` langsung dari output dataset.

### 5.4 Ilustrasi Input -> Output Embedding

Contoh:

```text
input_ids shape: (B=2, T=5)
[[ 10,  11,  12,  13,  14],
 [101, 102, 103, 104, 105]]

d_model = 768

output embedding shape:
(2, 5, 768)
```

Interpretasi:
- Tiap token sekarang direpresentasikan menjadi vektor 768 dimensi.

## 6) Modul `attention.py`

### 6.1 Tujuan

Menyediakan komponen `MultiHeadSelfAttention` yang bisa langsung menerima output embedding `(B, T, D)`.

### 6.2 Komponen penting

- `MultiHeadSelfAttention(..., max_seq_len, n_head, d_model, ...)`
- `forward(x, attention_mask=None)`:
  - input: hidden states dari embedding
  - output: hidden states setelah attention
- `from_config(cfg=CONFIG, ...)`
- `forward_batch(batch, embedding_layer)`:
  - langsung dari batch dataset (`input_ids`) -> embedding -> attention
- `build_attention_from_config(cfg=CONFIG, ...)`
- `build_tokenizer_embedding_attention(cfg=CONFIG, load_tokenizer=True)`

### 6.3 Ilustrasi Input -> Output Attention

```text
input hidden (dari embedding): (B, T, D)
contoh: (2, 128, 768)

output hidden (setelah attention): (B, T, D)
contoh: (2, 128, 768)
```

## 7) Contoh Penggunaan End-to-End

```python
from config import CONFIG
from dataset2 import create_dataloader
from embedding import build_tokenizer_and_embedding
from attention import build_attention_from_config
from transformerStack import build_transformer_stack_from_config

tokenizer, embedding = build_tokenizer_and_embedding(CONFIG, load_tokenizer=True)
attention = build_attention_from_config(CONFIG)
transformer = build_transformer_stack_from_config(CONFIG)
dataloader = create_dataloader(
    token_file=str(CONFIG.DATASET_TOKEN_FILE),
    block_size=CONFIG.DATASET_BLOCK_SIZE,
    batch_size=CONFIG.DATASET_BATCH_SIZE,
    shuffle=False,
)

batch = next(iter(dataloader))
hidden = embedding.forward_batch(batch)
context = attention(hidden)
stack_out = transformer(context)

print(batch["input_ids"].shape)  # (B, T)
print(hidden.shape)              # (B, T, D)
print(context.shape)             # (B, T, D)
print(stack_out.shape)           # (B, T, D)
```

## 7.1) Integrasi `gptModel.py` (Factory Utama Training)

`GPTModel` sudah terintegrasi dengan:
- tokenizer (`build_tokenizer_and_embedding`)
- embedding (`EmbeddingLayer`)
- transformer stack (`TransformerStack`)
- dataloader (`create_dataloader`)
- config environment (`get_config(env)`)

Helper utama:
- `GPTModel.from_config(cfg)`:
  - build model dari config
  - `max_seq_len` resolved aman via fallback (attention -> embedding -> dataset block size)
- `GPTModel.from_config_with_tokenizer(cfg, load_tokenizer=True)`:
  - build model + load tokenizer artifacts
- `GPTModel.build_training_components(cfg, ...)`:
  - return dict: `{"model", "tokenizer", "dataloader"}`
  - default dataloader diambil dari `cfg` yang sama (mencegah mismatch env model vs data)

## 8) Kapan Perlu Retrain Tokenizer / Pretokenize Ulang

Retrain tokenizer jika berubah:
- corpus
- parameter tokenizer
- target vocab

Pretokenize ulang jika berubah:
- tokenizer artifact (`vocab/token2id/id2token`)
- corpus
- file token bin belum ada

Jika hanya refactor kode integrasi tanpa mengubah artefak tokenizer/data, biasanya tidak perlu ulang.

## 8.1) Modul `finetune2.py`

### 8.1.1 Tujuan

`finetune2.py` adalah script fine-tuning GPT yang mengikuti pola repo ini semaksimal mungkin:
- memakai `get_config(env)` dari `config.py`
- membangun model dan tokenizer lewat `GPTModel.from_config_with_tokenizer(...)`
- memakai `tokenizer.encode()` dari `HybridTokenizerLLM`
- membentuk batch dict seperti `dataset2.py`
- menjalankan training melalui `model.forward_batch(batch)`
- menyimpan checkpoint dengan format yang konsisten dengan trainer utama

Perbedaannya dengan `tools/train_gpt.py`/`tools/train_gptV2.py`:
- `finetune2.py` tidak membaca `tokens_*.bin` lewat `create_dataloader(...)`
- script ini mengambil sumber data dari satu file teks fine-tune, lalu membangun mini-batch random langsung dari token stream hasil `tokenizer.encode()`

Script ini cocok ketika:
- Anda sudah punya checkpoint dasar GPT
- Anda ingin adaptasi model ke domain tertentu dari file teks tambahan
- Anda ingin workflow fine-tune yang tetap konsisten dengan kontrak model/batch repo

### 8.1.2 Alur Kerja

```text
File teks fine-tune
        |
        v
HybridTokenizerLLM.encode(text)
        |
        v
input_ids (1 stream token)
        |
        v
Random chunk sampler
  - ambil window sepanjang block_size
  - labels = input_ids shifted by 1
        |
        v
Batch dict
{
  "input_ids": ...,
  "labels": ...
}
        |
        v
GPTModel.forward_batch(batch)
        |
        v
loss + logits
        |
        v
backward / grad accumulation / AMP / checkpoint save
```

### 8.1.3 Komponen Internal Penting

`_resolve_device(requested)`:
- menyelesaikan device string `auto|cpu|cuda`

`_to_device(batch, device)`:
- memindahkan semua tensor dalam batch dict ke device target
- polanya sama dengan helper di `tools/train_gpt.py` dan `tools/train_gptV2.py`

`_save_checkpoint(...)`:
- menyimpan checkpoint fine-tune
- payload mencakup:
  - `epoch`
  - `step`
  - `env`
  - `data_path`
  - `model_state_dict`
  - `optimizer_state_dict`

`_load_resume_checkpoint(...)`:
- memuat bobot model dan optimizer dari checkpoint
- kompatibel dengan format modern `model_state_dict`
- punya fallback ke format lama `model`

`_encode_text_with_repo_api(tokenizer, text)`:
- memakai API repo yang sudah ada: `tokenizer.encode(text)`
- membuang trailing pad token jika ada
- hasil akhirnya adalah tensor 1D berisi stream token IDs

`_build_batch(tokens, batch_size, block_size)`:
- memilih posisi start acak
- membentuk `input_ids` dan `labels`
- mengembalikan batch dict dengan format yang sama seperti output `dataset2.py`

### 8.1.4 Loop Training

Loop training `finetune2.py` mengikuti pola trainer utama repo:
- model di-set `train()`
- batch dibentuk lalu dipindahkan ke device
- forward pass dilakukan lewat `model.forward_batch(batch)`
- loss dicek agar finite
- backward mendukung:
  - gradient accumulation
  - mixed precision (`--amp`)
- gradient clipping opsional via `--grad-clip`
- logging menampilkan:
  - `loss`
  - `ma_loss` (moving-average loss)
  - `lr`
  - estimasi `ppl`
  - elapsed time
- heartbeat 30 detik memastikan run panjang tetap terlihat hidup
- checkpoint periodik bisa disimpan via `--save-every`

### 8.1.5 Arti Parameter CLI

Parameter utama:
- `--env development|production|testing`
- `--data-path <path>`
- `--checkpoint <path>`
- `--save-path <path>`
- `--resume <path>`
- `--device auto|cpu|cuda`

Hyperparameter training:
- `--epochs <int>`
- `--steps-per-epoch <int>`
- `--batch-size <int>`
- `--block-size <int>`
- `--lr <float>`
- `--weight-decay <float>`
- `--grad-clip <float>`

Fitur trainer lanjutan:
- `--grad-accum <int>`
  - effective batch size = `batch_size * grad_accum`
- `--amp`
  - mixed precision pada CUDA
- `--log-every <int>`
- `--loss-window <int>`
- `--save-every <int>`
  - `0` berarti nonaktif

### 8.1.6 Perilaku `--checkpoint` vs `--resume`

`--checkpoint`:
- dipakai sebagai base model awal
- biasanya menunjuk ke checkpoint hasil pretraining, misalnya `reports\checkpoints\gpt_last.pt`

`--resume`:
- opsional
- jika diisi, state fine-tune dari checkpoint resume akan menimpa state dari `--checkpoint`
- dipakai untuk melanjutkan fine-tune yang belum selesai

Urutannya:
1. load model dasar dari `--checkpoint`
2. jika `--resume` ada, load ulang model + optimizer dari checkpoint resume

### 8.1.7 Moving-Average Loss

`moving-average loss` dihitung dari `deque(maxlen=loss_window)`.

Tujuan:
- membuat monitoring lebih stabil
- mengurangi noise dari loss per-step yang fluktuatif

Interpretasi:
- `loss` = loss batch saat ini
- `ma_loss` = rata-rata loss dari beberapa step terakhir

### 8.1.8 Mixed Precision (`--amp`)

Saat `--amp` aktif dan device adalah CUDA:
- forward pass dibungkus `torch.amp.autocast(...)`
- backward/update memakai `GradScaler`

Manfaat:
- penggunaan VRAM lebih efisien
- training di GPU sering lebih cepat

### 8.1.9 Gradient Accumulation (`--grad-accum`)

Tujuan:
- mensimulasikan batch lebih besar tanpa menaikkan kebutuhan memori secara drastis

Contoh:
- `batch_size=4`
- `grad_accum=8`

Maka effective batch size menjadi `32`.

Catatan:
- loss dibagi `grad_accum` sebelum `backward()`
- `global_step` mengacu ke optimizer step, bukan micro-step

### 8.1.10 Checkpoint yang Dihasilkan

Checkpoint periodik:
- nama file mengikuti pola:
  - `<save_path stem>_step_<global_step><suffix>`

Contoh:
- `reports\checkpoints\gpt_finetuned_v2_step_200.pt`

Checkpoint final:
- lokasi mengikuti `--save-path`

Isi checkpoint:

```python
{
    "epoch": ...,
    "step": ...,
    "env": ...,
    "data_path": ...,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
}
```

### 8.1.11 Contoh Penggunaan

Fine-tune dasar:

```powershell
.\venv\Scripts\python.exe .\finetune2.py --env production --data-path data\finetune.txt
```

Fine-tune dengan effective batch lebih besar:

```powershell
.\venv\Scripts\python.exe .\finetune2.py --env production --data-path data\finetune.txt --batch-size 4 --grad-accum 8 --steps-per-epoch 2000
```

Fine-tune GPU dengan AMP:

```powershell
.\venv\Scripts\python.exe .\finetune2.py --env production --device cuda --amp --data-path data\finetune.txt --lr 3e-6
```

Simpan checkpoint periodik:

```powershell
.\venv\Scripts\python.exe .\finetune2.py --env production --data-path data\finetune.txt --save-every 100 --save-path reports\checkpoints\gpt_domain_adapt.pt
```

Lanjutkan fine-tune sebelumnya:

```powershell
.\venv\Scripts\python.exe .\finetune2.py --env production --checkpoint reports\checkpoints\gpt_last.pt --resume reports\checkpoints\gpt_finetuned_v2_step_200.pt --data-path data\finetune.txt --save-every 100
```

### 8.1.12 Kapan Memakai `finetune2.py` vs `tools/train_gpt.py`

Gunakan `finetune2.py` jika:
- Anda punya file teks tambahan khusus domain
- Anda ingin adaptasi model tanpa membangun ulang `tokens_*.bin`
- Anda ingin workflow cepat untuk domain adaptation / continued pretraining skala kecil-menengah

Gunakan `tools/train_gpt.py` atau `tools/train_gptV2.py` jika:
- Anda ingin training penuh dari token file terstruktur
- Anda ingin pipeline dataset yang lebih dekat ke pretraining utama
- Anda ingin seluruh proses bergantung pada `create_dataloader(...)`

### 8.1.13 Risiko dan Catatan Praktis

Hal yang perlu diperhatikan:
- file fine-tune harus cukup panjang
  - minimal lebih besar dari `block_size + 1`
- jika teks terlalu pendek, `_build_batch(...)` akan gagal
- karena sampling dilakukan dari satu stream token:
  - distribusi batch bergantung langsung pada isi file fine-tune
  - data sangat repetitif bisa membuat model cepat overfit
- `--resume` melanjutkan optimizer state juga
  - ini bagus untuk kontinuitas training
  - jika ingin start ulang optimizer dari model hasil fine-tune, gunakan checkpoint itu sebagai `--checkpoint`, bukan `--resume`

Rekomendasi praktis:
- mulai dari LR kecil, misalnya `5e-6` sampai `3e-5`
- gunakan `--loss-window` agar monitoring lebih stabil
- gunakan `--save-every` saat run panjang
- untuk GPU, aktifkan `--amp`
- untuk VRAM terbatas, naikkan `--grad-accum` daripada langsung membesarkan `--batch-size`

## 9) Script Tools Terkait

Catatan:
- Semua command di bawah diasumsikan dijalankan dari root project:
  - `c:\Projects\Python\AI\LLM\hai06`
- Untuk Windows PowerShell, gunakan interpreter:
  - `.\venv\Scripts\python.exe`

### 9.1 `tools/train_tokenizer_progress.py`

Tujuan:
- Training tokenizer dengan progress live.

Contoh:

```powershell
.\venv\Scripts\python.exe tools\train_tokenizer_progress.py --env production --save
```

Dengan preview dataloader setelah training:

```powershell
.\venv\Scripts\python.exe tools\train_tokenizer_progress.py --env development --preview-batches 2 --save
```

Opsi umum:
- `--env development|production|testing`
- `--corpus <path>`
- `--vocab-size <int>`
- `--progress-every <int>`
- `--preview-batches <int>`
- `--save`

### 9.2 `tools/demo_dataset2.py`

Tujuan:
- Demo interaktif: text -> tokens -> token ids -> `input_ids` dan `labels` (shifted).

Contoh:

```powershell
.\venv\Scripts\python.exe tools\demo_dataset2.py --env production
```

Jika tidak ingin auto pretokenize:

```powershell
.\venv\Scripts\python.exe tools\demo_dataset2.py --env production --no-pretokenize
```

Opsi umum:
- `--env development|production|testing`
- `--block-size <int>`
- `--no-pretokenize`

### 9.3 `tools/check_embedding_pipeline.py`

Tujuan:
- Sanity check 1 batch dataloader -> forward embedding.
- Validasi token id tidak melebihi `vocab_size`.

Contoh:

```powershell
.\venv\Scripts\python.exe tools\check_embedding_pipeline.py --env production
```

Jika token file belum ada, auto-generate:

```powershell
.\venv\Scripts\python.exe tools\check_embedding_pipeline.py --env production --auto-pretokenize
```

Opsi umum:
- `--env development|production|testing`
- `--batch-size <int>`
- `--block-size <int>`
- `--dataset-type packed|sliding`
- `--stride <int>`
- `--auto-pretokenize`

### 9.4 `tools/demo_embedding.py`

Tujuan:
- Demo interaktif embedding:
  - text -> tokens -> token ids -> tensor embedding.
- Bisa ekspor plot 3D (`scatter`/`surface`).

Contoh interaktif dasar:

```powershell
.\venv\Scripts\python.exe tools\demo_embedding.py --env production
```

Dengan ekspor plot 3D:

```powershell
.\venv\Scripts\python.exe tools\demo_embedding.py --env production --save-plots --plot-3d-mode scatter
```

Custom output folder:

```powershell
.\venv\Scripts\python.exe tools\demo_embedding.py --env production --save-plots --plot-3d-mode surface --output-dir reports\embedding_custom
```

Opsi umum:
- `--env development|production|testing`
- `--max-length <int>`
- `--show-dims <int>`
- `--save-plots`
- `--plot-3d-mode surface|scatter`
- `--output-dir <path>`

### 9.5 `tools/vocab_analysis.py`

Tujuan:
- Analisis distribusi penggunaan token pada corpus.
- Menyimpan top-k token ke JSON.
- Membantu evaluasi apakah vocabulary tokenizer sudah efektif terhadap corpus target.

Fungsi utama di dalam script:
- `load_tokenizer(...)`
  - Build tokenizer dari config aktif.
  - Load artefak `vocab`, `token2id`, dan `id2token`.
  - Bisa memakai path default dari config atau path manual via argumen CLI.
- `analyze_corpus(corpus_path, tokenizer, max_lines=None)`
  - Membaca corpus baris per baris.
  - Setiap baris non-kosong di-tokenisasi memakai `tokenizer.encode_tokens(...)`.
  - Menghitung:
    - `token_counter`: frekuensi tiap token yang benar-benar dipakai di corpus
    - `total_tokens`: total semua token hasil tokenisasi
    - `total_sequences`: jumlah baris non-kosong yang dianalisis
    - `avg_seq_length`: rata-rata panjang token per baris
    - `unk_percentage`: persentase token `<UNK>` terhadap seluruh token
- `long_tail_analysis(counter)`
  - Mengukur distribusi head-mid-tail dari frekuensi token.
  - Meringkas apakah corpus didominasi sedikit token sangat sering atau lebih merata.
- `save_top_tokens(counter, output_path, top_k=1000)`
  - Menyimpan token paling sering muncul ke file JSON.
  - Format output berupa list pasangan `[token, frekuensi]`.

Alur kerja script:
1. Ambil environment config dari `--env` atau default `production`.
2. Resolve path corpus dan artefak tokenizer.
3. Load tokenizer dari artefak yang tersedia.
4. Tokenisasi corpus untuk menghitung statistik penggunaan token.
5. Hitung metrik long-tail dari distribusi frekuensi token.
6. Cetak ringkasan hasil ke terminal.
7. Simpan top-k token ke file JSON.

Contoh:

```powershell
.\venv\Scripts\python.exe tools\vocab_analysis.py --env production --top_k 1000
```

Analisis subset data:

```powershell
.\venv\Scripts\python.exe tools\vocab_analysis.py --env testing --max_lines 5000 --top_k 200
```

Opsi umum:
- `--env development|production|testing`
- `--corpus <path>`
- `--vocab_path <path>`
- `--token2id_path <path>`
- `--id2token_path <path>`
- `--max_lines <int>`
- `--top_k <int>`
- `--output <path>`

Penjelasan argumen:
- `--env`
  - Memilih konfigurasi default path dari `config.py`.
  - Berguna bila Anda punya artefak tokenizer berbeda untuk `development`, `testing`, dan `production`.
- `--corpus`
  - Override file corpus yang mau dianalisis.
  - Jika tidak diisi, script memakai `cfg.CORPUS_FILE`.
- `--vocab_path`, `--token2id_path`, `--id2token_path`
  - Override artefak tokenizer secara manual.
  - Berguna saat ingin membandingkan tokenizer tertentu terhadap corpus tertentu tanpa mengganti config utama.
- `--max_lines`
  - Membatasi jumlah baris yang diproses.
  - Cocok untuk inspeksi cepat atau sanity check sebelum analisis penuh.
- `--top_k`
  - Menentukan berapa banyak token teratas yang disimpan ke JSON.
- `--output`
  - Lokasi file JSON hasil simpan top tokens.
  - Jika tidak diisi, default ke folder yang sama dengan vocab aktif dengan nama `top_tokens.json`.

Arti output terminal:
- `Vocab Used`
  - Jumlah token unik yang benar-benar muncul di corpus analisis.
  - Ini bukan selalu ukuran seluruh vocab tokenizer, melainkan ukuran vocab yang terpakai.
- `Total Tokens`
  - Jumlah semua token hasil tokenisasi corpus.
- `Total Sequences`
  - Jumlah baris non-kosong yang ikut dianalisis.
- `Avg Seq Length`
  - Rata-rata jumlah token per baris.
  - Nilai terlalu besar bisa menandakan segmentasi token kurang efisien.
- `% UNK Usage`
  - Proporsi token `<UNK>`.
  - Idealnya kecil; makin tinggi biasanya berarti coverage vocab terhadap corpus makin buruk.

Arti metrik long-tail:
- `Top 1% Share`
  - Persentase total kemunculan token yang disumbang oleh 1% token paling sering.
- `Top 10% Share`
  - Persentase total kemunculan token yang disumbang oleh 10% token paling sering.
- `Head (50%) Share`
  - Kontribusi separuh token teratas terhadap total frekuensi token.
- `Mid (40%) Share`
  - Kontribusi kelompok tengah token terhadap total frekuensi token.
- `Tail (10%) Share`
  - Kontribusi 10% token terbawah terhadap total frekuensi token.
- `Hapax Count`
  - Jumlah token yang hanya muncul satu kali.
- `Hapax Ratio`
  - Persentase token unik yang hanya muncul satu kali.

Cara membaca hasil secara praktis:
- `% UNK Usage` tinggi:
  - kemungkinan vocab kurang cocok dengan domain corpus
  - bisa jadi perlu retrain tokenizer atau menambah data domain
- `Top 10% Share` sangat tinggi:
  - corpus sangat didominasi token umum
  - normal untuk data natural language, tetapi jika terlalu ekstrem bisa menandakan vocab belum cukup kaya
- `Hapax Ratio` sangat tinggi:
  - banyak token langka
  - bisa berarti corpus sangat beragam, noisy, atau vocab terlalu memecah kata menjadi token-token kecil
- `Avg Seq Length` terlalu panjang untuk kalimat yang relatif pendek:
  - sering menjadi sinyal bahwa tokenisasi belum efisien dan masih terlalu banyak fallback ke token kecil/karakter

Contoh penggunaan praktis:
- Mengecek kualitas vocab production terhadap corpus production:

```powershell
.\venv\Scripts\python.exe tools\vocab_analysis.py --env production --top_k 1000
```

- Membandingkan vocab production terhadap corpus fine-tune khusus:

```powershell
.\venv\Scripts\python.exe tools\vocab_analysis.py --env production --corpus data\corpus-ft.txt --output reports\top_tokens_ft.json
```

- Melakukan inspeksi cepat 5000 baris pertama:

```powershell
.\venv\Scripts\python.exe tools\vocab_analysis.py --env testing --max_lines 5000 --top_k 200
```

Kapan tool ini berguna:
- Setelah training tokenizer selesai.
- Sebelum pretokenize dataset skala besar.
- Saat hasil encode terlihat terlalu panjang atau banyak token kecil.
- Saat model fine-tune terasa lemah pada domain tertentu dan ingin cek coverage vocabulary.

Catatan:
- Script ini menganalisis penggunaan token aktual pada corpus, bukan kualitas semantik token.
- Jika corpus sangat besar, proses bisa memakan waktu karena seluruh baris ditokenisasi ulang.
- Progress bar ditampilkan memakai `tqdm` selama proses pembacaan corpus.

### 9.6 `tools/demo_attention.py`

Tujuan:
- Demo interaktif attention:
  - text -> tokens -> token ids -> embedding output -> attention internals -> attention output.
- Menampilkan ringkasan top-k bobot attention per query token untuk beberapa head.
- Dapat load checkpoint embedding+attention dan export heatmap attention per-head.

Contoh interaktif dasar:

```powershell
.\venv\Scripts\python.exe tools\demo_attention.py --env production
```

Batasi panjang token dan tampilkan lebih banyak head:

```powershell
.\venv\Scripts\python.exe tools\demo_attention.py --env development --max-length 32 --show-heads 4 --topk 5
```

Load checkpoint model terlatih:

```powershell
.\venv\Scripts\python.exe tools\demo_attention.py --env production --checkpoint reports\checkpoints\model_latest.pt
```

Simpan heatmap attention per-head:

```powershell
.\venv\Scripts\python.exe tools\demo_attention.py --env production --checkpoint reports\checkpoints\model_latest.pt --save-heatmaps --heatmap-heads 8
```

Opsi umum:
- `--env development|production|testing`
- `--max-length <int>`
- `--show-dims <int>`
- `--show-heads <int>`
- `--topk <int>`
- `--checkpoint <path>`
- `--strict-load`
- `--save-heatmaps`
- `--heatmap-dir <path>`
- `--heatmap-heads <int>`
- `--heatmap-cmap <name>`

### 9.7 `tools/demo_transformer.py`

Tujuan:
- Demo interaktif transformer stack:
  - text -> tokens -> token ids -> embedding -> output tiap layer transformer -> logits.
- Menampilkan top-k prediksi token berikutnya per posisi token.
- Fokus untuk inspeksi/debug forward pass, bukan untuk generate autoregressive multi-token.
- Dapat load checkpoint untuk `embedding` + `transformer`.
- Kompatibel dengan checkpoint hasil `tools/train_gpt.py` (`gpt_last.pt`), termasuk bobot `lm_head`.

Contoh interaktif dasar:

```powershell
.\venv\Scripts\python.exe tools\demo_transformer.py --env production
```

Batasi panjang token dan tampilkan lebih banyak prediksi:

```powershell
.\venv\Scripts\python.exe tools\demo_transformer.py --env development --max-length 32 --topk 8 --show-layers 4
```

Load checkpoint model:

```powershell
.\venv\Scripts\python.exe tools\demo_transformer.py --env production --checkpoint reports\checkpoints\model_latest.pt
```

Load checkpoint dari trainer GPT terintegrasi:

```powershell
.\venv\Scripts\python.exe tools\demo_transformer.py --env development --checkpoint reports\checkpoints\gpt_last.pt
```

Opsi umum:
- `--env development|production|testing`
- `--max-length <int>`
- `--show-dims <int>`
- `--show-layers <int>`
- `--topk <int>`
- `--checkpoint <path>`
- `--strict-load`

### 9.8 `tools/train_gpt.py`

Tujuan:
- Script training GPT end-to-end dengan pipeline terintegrasi `GPTModel.build_training_components()`.
- Build model + tokenizer + dataloader dari satu `cfg` environment.
- Menyimpan checkpoint periodik/final (`gpt_step_*.pt`, `gpt_last.pt`).
- Default hyperparameter training dapat diatur dari `config.py` lewat `TRAIN_*`
  (CLI arg tetap bisa override).

Contoh minimal:

```powershell
.\venv\Scripts\python.exe tools\train_gpt.py --env development --epochs 1 --max-steps 200
```

Dengan checkpoint periodik:

```powershell
.\venv\Scripts\python.exe tools\train_gpt.py --env development --epochs 1 --max-steps 500 --save-every 100 --save-dir reports\checkpoints
```

Contoh run production (custom hyperparameter):

```powershell
.\venv\Scripts\python.exe tools\train_gpt.py --env production --device cpu --max-steps 10000 --lr 3e-4 --batch-size 16 --block-size 96 --loss-window 100 --plateau --plateau-patience 4 --plateau-factor 0.5 --early-stop --early-stop-patience 5 --early-stop-min-delta 0.001 --log-every 20 --save-every 500 --grad-clip 1.0  --save-dir reports\checkpoints
```

Lanjutkan training dari checkpoint terakhir:

```powershell
.\venv\Scripts\python.exe tools\train_gpt.py --env production --resume reports\checkpoints\gpt_last.pt --max-steps 1000 --early-stop --plateau --log-every 1
```

Opsi umum:
- `--env development|production|testing`
- `--device auto|cpu|cuda`
- `--epochs <int>`
- `--max-steps <int>`
- `--lr <float>`
- `--weight-decay <float>`
- `--grad-clip <float>`
- `--log-every <int>`
- `--save-every <int>`
- `--save-dir <path>`
- `--token-file <path>`
- `--loss-window <int>`
- `--plateau` / `--no-plateau`
- `--plateau-factor <float>`
- `--plateau-patience <int>`
- `--plateau-threshold <float>`
- `--plateau-min-lr <float>`
- `--plateau-cooldown <int>`
- `--early-stop` / `--no-early-stop`
- `--early-stop-patience <int>`
- `--early-stop-min-delta <float>`
- `--resume <path>`
- Dataloader override:
  - `--batch-size <int>`
  - `--block-size <int>`
  - `--dataset-type packed|sliding`
  - `--stride <int>`
  - `--shuffle` / `--no-shuffle`
  - `--num-workers <int>`
  - `--pin-memory` / `--no-pin-memory`
  - `--persistent-workers` / `--no-persistent-workers`

Parameter config terkait trainer:
- `TRAIN_DEVICE`, `TRAIN_EPOCHS`, `TRAIN_MAX_STEPS`
- `TRAIN_LR`, `TRAIN_WEIGHT_DECAY`, `TRAIN_GRAD_CLIP`
- `TRAIN_LOG_EVERY`, `TRAIN_SAVE_DIR`, `TRAIN_SAVE_EVERY`
- `TRAIN_LOSS_WINDOW`
- `TRAIN_PLATEAU_ENABLED`, `TRAIN_PLATEAU_FACTOR`, `TRAIN_PLATEAU_PATIENCE`
- `TRAIN_PLATEAU_THRESHOLD`, `TRAIN_PLATEAU_MIN_LR`, `TRAIN_PLATEAU_COOLDOWN`
- `TRAIN_EARLY_STOP_ENABLED`, `TRAIN_EARLY_STOP_PATIENCE`, `TRAIN_EARLY_STOP_MIN_DELTA`

Catatan stabilitas loss:
- `train_gpt.py` memiliki guard non-finite loss (`nan/inf`):
  step akan di-skip dan warning dicetak, training tetap lanjut.
- Scheduler `ReduceLROnPlateau` memakai moving-average loss (window `TRAIN_LOSS_WINDOW`/`--loss-window`).
- Saat `--resume` dipakai:
  - model + optimizer dimuat dari checkpoint,
  - `global_step` dilanjutkan dari checkpoint,
  - `--max-steps` dihitung sebagai jumlah step tambahan untuk run saat ini.

### 9.9 `tools/eval_model.py`

Tujuan:
- Evaluasi checkpoint GPT (`gpt_last.pt` atau checkpoint lain) menggunakan pipeline yang sama dengan training:
  - `get_config(env)` dari `config.py`
  - `GPTModel.from_config(cfg)` dari `gptModel.py`
  - `create_dataloader(...)` dari `dataset2.py`
- Menghitung metrik:
  - `loss`
  - `ppl` (perplexity)
  - `top1_acc`
  - `top5_acc`
  - skor ringkas (`score`) + status readiness
- Menampilkan progress evaluasi per-N step (`--log-every`) agar run panjang tetap terlihat berjalan.

Contoh evaluasi dasar:

```powershell
.\venv\Scripts\python.exe tools\eval_model.py --env production --checkpoint reports\checkpoints\gpt_last.pt --steps 200 --device auto
```

Evaluasi dengan progress lebih rapat:

```powershell
.\venv\Scripts\python.exe tools\eval_model.py --env production --checkpoint reports\checkpoints\gpt_last.pt --steps 1000 --log-every 25
```

Override dataset eval (misal token file/batch/block khusus):

```powershell
.\venv\Scripts\python.exe tools\eval_model.py --env production --checkpoint reports\checkpoints\gpt_last.pt --token-file data\prod\tokens_prod.bin --batch-size 8 --block-size 256 --dataset-type packed
```

Evaluasi sliding window:

```powershell
.\venv\Scripts\python.exe tools\eval_model.py --env development --checkpoint reports\checkpoints\gpt_last.pt --dataset-type sliding --stride 16 --steps 300
```

Opsi umum:
- `--checkpoint <path>` (wajib)
- `--env development|production|testing`
- `--device auto|cpu|cuda`
- `--steps <int>`
- `--log-every <int>`
- `--token-file <path>`
- `--batch-size <int>`
- `--block-size <int>`
- `--dataset-type packed|sliding`
- `--stride <int>`

Catatan:
- `--log-every` default mengikuti `TRAIN_LOG_EVERY` di `config.py` (fallback `20`).
- Evaluasi memakai `shuffle=False` agar konsisten/deterministik.
- Kompatibel dengan checkpoint `tools/train_gpt.py` (`model_state_dict`) dan format fallback lama (`model` / direct state-dict).

### 9.10 `tools/generate_gpt.py`

Tujuan:
- Generate teks autoregressive multi-token dari checkpoint GPT (`gpt_last.pt` atau checkpoint lain).
- Mendukung one-shot prompt dan mode interaktif REPL.
- Mendukung sampling parameter umum untuk inference.

Contoh one-shot:

```powershell
.\venv\Scripts\python.exe tools\generate_gpt.py --env production --checkpoint reports\checkpoints\gpt_last.pt --prompt "djarum foundation" --max-new-tokens 20 --temperature 0.9 --top-k 40 --top-p 0.95
```

Contoh greedy (deterministik):

```powershell
.\venv\Scripts\python.exe tools\generate_gpt.py --env production --prompt "djarum foundation" --max-new-tokens 20 --temperature 0 --top-k 0 --top-p 1 --seed 42
```

Mode interaktif:

```powershell
.\venv\Scripts\python.exe tools\generate_gpt.py --env production --checkpoint reports\checkpoints\gpt_last.pt
```

Opsi umum:
- `--env development|production|testing`
- `--checkpoint <path>` (default: `<TRAIN_SAVE_DIR>\gpt_last.pt`)
- `--prompt "<text>"` (jika kosong -> mode interaktif)
- `--max-new-tokens <int>`
- `--temperature <float>` (`<=0` berarti greedy)
- `--top-k <int>` (`0` menonaktifkan top-k)
- `--top-p <float>` (nucleus sampling, rentang `(0, 1]`)
- `--repetition-penalty <float>` (`>=1.0`)
- `--seed <int>`
- `--device auto|cpu|cuda`

### 9.11 `tools/generate_gpt_finetune2.py`

Tujuan:
- Menjalankan inference/chat dari checkpoint hasil fine-tuning instruction-style.
- Memakai prompt template yang konsisten dengan dataset fine-tune:
  - `### Instruction:`
  - `### Response:`
- Menyediakan mode interaktif multi-turn sederhana dengan history terbatas.
- Kompatibel dengan checkpoint fine-tune modern dan beberapa format checkpoint fallback.

Kapan dipakai:
- Saat model sudah di-fine-tune memakai `finetune2.py`
- Saat Anda ingin mencoba perilaku chatbot dari data `corpus-ft.txt` atau dataset serupa
- Saat ingin menguji kualitas jawaban dengan format prompt yang sama seperti saat instruction tuning

Alur kerja:

```text
User input
    |
    v
History chat terakhir (maks 6 turn)
    |
    v
Prompt formatter
  ### Instruction:
  <user_input>

  ### Response:
    |
    v
Tokenizer.encode(...)
    |
    v
GPTModel forward autoregressive
    |
    v
Sampling (temperature/top-k/top-p)
    |
    v
Decode hasil
    |
    v
Extract hanya bagian Response
```

Komponen penting:

`_resolve_device(requested)`:
- memilih `cuda` jika `auto` dan GPU tersedia
- fallback ke `cpu`

`_resolve_model_state(payload)`:
- mengambil state model dari:
  - `model_state_dict`
  - `state_dict`
  - direct state-dict

`_load_model(cfg, checkpoint, device)`:
- build model + tokenizer dari config
- load checkpoint ke model
- set `eval()` untuk inference

`_format_prompt(user_input, history)`:
- membangun prompt instruction-style yang konsisten dengan corpus fine-tune
- mempertahankan maksimal 6 turn history terakhir

`_extract_response(full_text)`:
- mengambil hanya bagian jawaban setelah `### Response:`
- memotong jika model mulai menghasilkan `### Instruction:` baru

`_sample_next_token(logits, temperature, top_k, top_p)`:
- greedy jika `temperature <= 0`
- mendukung:
  - temperature sampling
  - top-k
  - top-p / nucleus sampling

`generate(...)`:
- encode prompt
- generate token demi token hingga `max_new_tokens`
- stop jika model mulai menghasilkan marker `### Instruction:` baru pada bagian generasi

Perbedaan dengan `tools/generate_gpt.py`:
- `generate_gpt.py` lebih umum untuk text generation biasa
- `generate_gpt_finetune2.py` khusus untuk model yang sudah diarahkan ke format instruction/response
- prompt dan parsing output di sini disesuaikan untuk chatbot instruction tuning

Argumen CLI:
- `--env development|production|testing`
  - memilih konfigurasi model/tokenizer
- `--checkpoint <path>`
  - checkpoint fine-tune yang akan dipakai
  - default: `reports/checkpoints/gpt_finetuned_v2.pt`
- `--max-new-tokens <int>`
  - jumlah token maksimum yang digenerate
- `--temperature <float>`
  - suhu sampling
  - `<= 0` berarti greedy
- `--top-k <int>`
  - batasi sampling ke top-k token terbaik
  - `0` berarti nonaktif
- `--top-p <float>`
  - nucleus sampling
  - rentang valid `(0, 1]`
- `--device auto|cpu|cuda`
  - device inference
- `--seed <int>`
  - seed untuk sampling yang lebih reproduktif

Default penting:
- checkpoint default:
  - `reports/checkpoints/gpt_finetuned_v2.pt`
- history yang dipakai:
  - 6 turn terakhir
- mode:
  - interaktif REPL/chat

Contoh command dasar:

Jalankan chatbot dengan checkpoint fine-tuned default:

```powershell
.\venv\Scripts\python.exe tools\generate_gpt_finetune2.py --env production
```

Pakai checkpoint tertentu:

```powershell
.\venv\Scripts\python.exe tools\generate_gpt_finetune2.py --env production --checkpoint reports\checkpoints\gpt_finetuned_v2.pt
```

Mode CPU eksplisit:

```powershell
.\venv\Scripts\python.exe tools\generate_gpt_finetune2.py --env production --device cpu
```

Mode GPU eksplisit:

```powershell
.\venv\Scripts\python.exe tools\generate_gpt_finetune2.py --env production --device cuda
```

Sampling lebih kreatif:

```powershell
.\venv\Scripts\python.exe tools\generate_gpt_finetune2.py --env production --temperature 0.9 --top-k 50 --top-p 0.95
```

Sampling lebih stabil/konservatif:

```powershell
.\venv\Scripts\python.exe tools\generate_gpt_finetune2.py --env production --temperature 0.4 --top-k 20 --top-p 0.8
```

Greedy decoding:

```powershell
.\venv\Scripts\python.exe tools\generate_gpt_finetune2.py --env production --temperature 0 --top-k 0 --top-p 1
```

Jalankan dengan seed tetap:

```powershell
.\venv\Scripts\python.exe tools\generate_gpt_finetune2.py --env production --seed 42
```

Pakai checkpoint hasil fine-tune lain:

```powershell
.\venv\Scripts\python.exe tools\generate_gpt_finetune2.py --env production --checkpoint reports\checkpoints\gpt_domain_adapt.pt --temperature 0.7 --top-k 40 --top-p 0.9
```

Contoh sesi interaktif:

```text
User > Halo
AI > Halo! Ada yang ingin kamu bahas atau lanjutkan dari project kamu?

User > Tolong jelaskan tokenizer
AI > Tokenizer adalah komponen yang mengubah teks menjadi token agar model bisa memproses input.
```

Perilaku history:
- setiap jawaban yang dihasilkan akan disimpan dalam format:

```text
### Instruction:
<user_input>

### Response:
<response>
```

- saat user mengirim pesan baru, history terakhir digabung ke prompt
- hanya 6 turn terakhir yang dipertahankan untuk membatasi panjang konteks

Catatan tentang stop condition:
- script berhenti generate jika pada bagian output baru mulai muncul `### Instruction:`
- ini mencegah model terus membuat pasangan instruksi-respons berikutnya sendiri

Validasi dan error handling:
- `--max-new-tokens` harus `>= 1`
- `--top-k` harus `>= 0`
- `--top-p` harus berada di rentang `(0, 1]`
- checkpoint diverifikasi ada sebelum dimuat
- mode interaktif aman untuk:
  - input kosong
  - `Ctrl+C`
  - `Ctrl+D`
- jika inference gagal pada satu prompt, program tetap hidup dan menampilkan pesan error

Troubleshooting umum:

Jika muncul `Checkpoint not found`:
- pastikan path checkpoint benar
- cek apakah hasil fine-tune sudah benar-benar tersimpan di `reports\checkpoints`

Jika muncul error tokenizer artifacts not found:
- pastikan file berikut ada sesuai env:
  - `VOCAB_PATH`
  - `TOKEN2ID_PATH`
  - `ID2TOKEN_PATH`

Jika jawaban kosong atau terlalu pendek:
- naikkan `--max-new-tokens`
- coba `--temperature 0.7 --top-k 40 --top-p 0.9`
- cek kualitas dataset fine-tune

Jika jawaban terlalu acak:
- turunkan `--temperature`
- turunkan `--top-k`
- turunkan `--top-p`

Jika jawaban terlalu repetitif:
- coba sampling lebih agresif:
  - `--temperature 0.8`
  - `--top-k 50`
  - `--top-p 0.95`

Rekomendasi penggunaan:
- untuk evaluasi stabil:
  - `--temperature 0.4 --top-k 20 --top-p 0.8`
- untuk eksplorasi respons:
  - `--temperature 0.7 --top-k 40 --top-p 0.9`
- untuk debugging cepat:
  - `--device cpu --seed 42`

### 9.12 `generate_finetuneV3.py`

Tujuan:
- Menjalankan inference dari checkpoint hasil `finetuneV3.py`.
- Memakai format prompt instruction V3 yang sesuai dengan data fine-tune:
  - `.user.`
  - `.assistant.`
- Mendukung one-shot prompt lewat `--prompt`.
- Mendukung mode interaktif jika `--prompt` tidak diisi.
- Kompatibel dengan checkpoint yang menyimpan state model dalam:
  - `model_state_dict`
  - `state_dict`
  - direct state-dict

Kapan dipakai:
- Setelah model selesai di-fine-tune memakai `finetuneV3.py`.
- Saat ingin menguji checkpoint default `reports\checkpoints\gpt_finetuneV3.pt`.
- Saat ingin mencoba respons model dengan format `.user.` / `.assistant.` yang sama seperti data `data\corpus-ft.txt`.
- Saat ingin membandingkan output sampling stabil, kreatif, dan greedy.

Alur kerja:

```text
User prompt
    |
    v
Prompt formatter
  .user. <prompt>
  .assistant.
    |
    v
Tokenizer.encode(...)
    |
    v
GPTModel forward autoregressive
    |
    v
Sampling
  - temperature
  - top-k
  - top-p
  - repetition penalty
    |
    v
Decode token IDs
    |
    v
Extract teks setelah .assistant.
```

Default penting:
- environment:
  - `TOKENIZER_ENV`, fallback ke `production`
- checkpoint:
  - `<TRAIN_SAVE_DIR>\gpt_finetuneV3.pt`
  - jika `TRAIN_SAVE_DIR` tidak ada di config, fallback ke `reports\checkpoints\gpt_finetuneV3.pt`
- prompt template:
  - `.user. <isi prompt>`
  - `.assistant. `
- mode:
  - one-shot jika `--prompt` diisi
  - interaktif jika `--prompt` tidak diisi

Contoh one-shot dasar:

```powershell
.\venv\Scripts\python.exe .\generate_finetuneV3.py --env production --prompt "Halo, jelaskan tokenizer secara singkat"
```

Pakai checkpoint eksplisit:

```powershell
.\venv\Scripts\python.exe .\generate_finetuneV3.py --env production --checkpoint reports\checkpoints\gpt_finetuneV3.pt --prompt "Apa fungsi fine-tuning?"
```

Mode interaktif:

```powershell
.\venv\Scripts\python.exe .\generate_finetuneV3.py --env production --checkpoint reports\checkpoints\gpt_finetuneV3.pt
```

Di mode interaktif:
- ketik prompt biasa pada `Prompt >`
- ketik `:q` untuk keluar
- input kosong akan dilewati

Mode CPU eksplisit:

```powershell
.\venv\Scripts\python.exe .\generate_finetuneV3.py --env production --device cpu --prompt "Halo"
```

Mode GPU eksplisit:

```powershell
.\venv\Scripts\python.exe .\generate_finetuneV3.py --env production --device cuda --prompt "Halo"
```

Sampling stabil/konservatif:

```powershell
.\venv\Scripts\python.exe .\generate_finetuneV3.py --env production --prompt "Jelaskan fungsi attention" --temperature 0.4 --top-k 20 --top-p 0.8 --repetition-penalty 1.1
```

Sampling lebih kreatif:

```powershell
.\venv\Scripts\python.exe .\generate_finetuneV3.py --env production --prompt "Buat contoh jawaban chatbot" --temperature 0.9 --top-k 50 --top-p 0.95 --repetition-penalty 1.05
```

Greedy decoding/deterministik:

```powershell
.\venv\Scripts\python.exe .\generate_finetuneV3.py --env production --prompt "Apa itu tokenizer?" --temperature 0 --top-k 0 --top-p 1 --seed 42
```

Menggunakan raw prompt:

```powershell
.\venv\Scripts\python.exe .\generate_finetuneV3.py --env production --raw-prompt --prompt ".user. Halo`n.assistant. "
```

Catatan `--raw-prompt`:
- prompt tidak akan otomatis dibungkus dengan `.user.` dan `.assistant.`
- output tidak akan diekstrak hanya bagian assistant-nya
- berguna untuk debugging format prompt, melihat output mentah, atau eksperimen marker khusus

Mengganti marker:

```powershell
.\venv\Scripts\python.exe .\generate_finetuneV3.py --env production --user-marker ".user." --assistant-marker ".assistant." --prompt "Halo"
```

Argumen CLI:
- `--env development|production|testing`
  - memilih config model/tokenizer
- `--checkpoint <path>`
  - checkpoint fine-tune yang akan dimuat
- `--prompt "<text>"`
  - prompt user; jika tidak diisi masuk mode interaktif
- `--raw-prompt`
  - memakai prompt apa adanya tanpa template otomatis
- `--user-marker <text>`
  - marker user, default `.user.`
- `--assistant-marker <text>`
  - marker assistant, default `.assistant.`
- `--max-new-tokens <int>`
  - jumlah token baru maksimum; harus `>= 1`
- `--temperature <float>`
  - `<= 0` memakai greedy decoding
- `--top-k <int>`
  - top-k sampling; `0` berarti nonaktif
- `--top-p <float>`
  - nucleus sampling; rentang valid `(0, 1]`
- `--repetition-penalty <float>`
  - penalti repetisi; harus `>= 1.0`
- `--seed <int>`
  - seed opsional untuk sampling lebih reproduktif
- `--device auto|cpu|cuda`
  - device inference; default mengikuti `TRAIN_DEVICE` dari config

Komponen internal penting:

`_load_model(checkpoint, cfg, device)`:
- build `GPTModel` dan tokenizer dari config
- load bobot checkpoint
- pindahkan model ke device
- set `model.eval()`

`generate_text(...)`:
- encode prompt
- potong konteks input ke `model.embedding.max_seq_len`
- generate token autoregressive sampai `max_new_tokens`
- decode seluruh sequence menjadi teks

`format_instruction_prompt(user_input, user_marker, assistant_marker)`:
- membuat prompt standar:

```text
.user. <user_input>
.assistant.
```

`extract_assistant_text(output, assistant_marker, stop_tokens)`:
- mengambil teks setelah marker assistant
- memotong output jika model mulai menghasilkan marker `.user.` atau `.assistant.` berikutnya

Validasi dan error handling:
- `--max-new-tokens` harus `>= 1`
- `--top-k` harus `>= 0`
- `--top-p` harus berada dalam rentang `(0, 1]`
- `--repetition-penalty` harus `>= 1.0`
- checkpoint dicek sebelum model dimuat
- prompt kosong akan ditolak pada proses generate
- jika artefak tokenizer tidak ada, script menampilkan path `VOCAB_PATH`, `TOKEN2ID_PATH`, dan `ID2TOKEN_PATH` yang diharapkan

Troubleshooting umum:

Jika muncul `Checkpoint not found`:
- pastikan checkpoint `finetuneV3.py` sudah dibuat
- cek path default `reports\checkpoints\gpt_finetuneV3.pt`
- atau isi path manual lewat `--checkpoint`

Jika muncul error tokenizer artifacts not found:
- pastikan tokenizer sudah pernah dilatih dan disimpan untuk env yang dipakai
- cek kesesuaian `--env` dengan artefak tokenizer di config

Jika jawaban kosong:
- naikkan `--max-new-tokens`
- coba nonaktifkan extraction sementara dengan `--raw-prompt`
- cek apakah data fine-tune benar-benar memakai marker `.user.` dan `.assistant.`

Jika jawaban terlalu repetitif:
- naikkan `--repetition-penalty`, misalnya `1.15`
- turunkan `--temperature`
- turunkan `--top-k` atau `--top-p`

Jika jawaban terlalu acak:
- gunakan `--temperature 0.4 --top-k 20 --top-p 0.8`
- untuk hasil paling stabil, gunakan `--temperature 0 --top-k 0 --top-p 1`

Rekomendasi penggunaan:
- evaluasi stabil:
  - `--temperature 0.4 --top-k 20 --top-p 0.8 --repetition-penalty 1.1`
- eksplorasi respons:
  - `--temperature 0.7 --top-k 40 --top-p 0.95 --repetition-penalty 1.1`
- debugging deterministik:
  - `--device cpu --temperature 0 --top-k 0 --top-p 1 --seed 42`
