# harpia-train

Fine-tuning on candle, CPU-capable:

- `train_classifier`: full fine-tuning of BERT/RoBERTa/XLM-R sequence classifiers and NLI
  cross-encoders (head reuse with label remapping, class-weighted loss, AdamW with decoupled
  weight decay, warmup + linear decay, gradient clipping, per-epoch macro-F1, best checkpoint in
  Hugging Face format, `train.json`);
- `train_lora_sft`: LoRA supervised fine-tuning of Qwen2/Llama decoders on assistant-masked
  samples, exported as a PEFT adapter and optionally as merged weights;
- `train_lora_dpo` (experimental): DPO with a frozen reference (the base model).

Dropout is not applied during training (candle's encoder implementations are inference-only).
