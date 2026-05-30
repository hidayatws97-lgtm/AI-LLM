arsitektur end-to-end LLM yang telah dibuat, terbagi dalam 5 fase utama:
① Data Preparation — dimulai dari raw text corpus, lalu diproses oleh HybridTokenizerLLM (encode ke token string + integer ids), kemudian disimpan sebagai file tokens_*.bin melalui pretokenize, dan dikemas menjadi Packed/Sliding Dataset dengan input_ids dan labels (shifted).
② Model Architecture — token melewati Embedding Layer (token + positional embedding → tensor B,T,D), lalu masuk ke Multi-Head Self-Attention (Q·K·V causal attention), dan ditumpuk dalam Transformer Stack sebanyak N blok decoder-only, diakhiri LM Head yang menghasilkan probabilitas token berikutnya.
③ Pre-training (GPT Model) — seluruh komponen arsitektur dirakit menjadi satu GPT Model lengkap dengan build_training_components(), dilatih menggunakan Cross-Entropy Loss dan Adam optimizer + backpropagation, menghasilkan pretrained weights.
④ Fine-tuning — weights pre-trained di-fine-tune menggunakan data instruksi: tokenizer encode input, dibentuk batch {input_ids, labels}, lalu model.forward_batch() dijalankan untuk menghasilkan fine-tuned weights.
⑤ Instruction-Chat Inference — prompt diformat dengan template .user. / .assistant., kemudian decoding autoregresif dilakukan token demi token hingga menghasilkan output chatbot yang siap digunakan.

