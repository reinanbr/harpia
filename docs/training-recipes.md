# Training recipes (`harpia-train`)

All recipes run on CPU; GPU support follows candle's `cuda`/`metal` features.

## NLI cross-encoder (the errei-cita recipe)

Fine-tune a multilingual NLI model on claim/evidence pairs, reusing its head
(`entailment/neutral/contradiction` → your labels), with errei's hyperparameters:

```bash
harpia train classifier \
  --base models/multilingual-MiniLMv2-L6-mnli-xnli \
  --train pairs_train.jsonl --val pairs_val.jsonl --out models/my-nli \
  --premise b --hypothesis a --label label \
  --labels sustenta,neutro,contradiz \
  --label-map sustenta=entailment,neutro=neutral,contradiz=contradiction \
  --epochs 3 --batch-size 16 --lr 2e-5 --max-len 192
```

The best epoch by validation macro-F1 is saved in Hugging Face format (it loads in
`harpia-local` and in Python `transformers`) with `train.json` (settings, history, timing).
Recipe details: AdamW with decoupled weight decay 0.01 (not on biases/norms), 10% linear warmup
then linear decay, global gradient-norm clip 1.0, class-weighted cross-entropy `n/(k·count)`.

## LoRA SFT of a small chat model

```bash
harpia train sft --base models/qwen2.5-0.5b-instruct --data chat.jsonl --out adapters/my-sft \
  --r 16 --alpha 32 --epochs 2 --lr 2e-4 --max-len 2048 --merge
harpia chat --provider local --model adapters/my-sft/merged "Olá!"
```

Rows may be OpenAI messages, ShareGPT, Alpaca or prompt/completion; they are rendered with the
model's chat template and only assistant tokens are trained. The adapter is written in PEFT
format (`adapter_config.json`, `adapter_model.safetensors`); `--merge` also writes full weights.

## DPO (experimental)

```bash
harpia train dpo --base models/qwen2.5-0.5b-instruct --data prefs.jsonl --out adapters/my-dpo --beta 0.1
```

The frozen base model is the reference policy.

## Known differences and limits

- Dropout follows the model's `config.json` (`hidden_dropout_prob`, `attention_probs_dropout_prob`,
  `classifier_dropout`) as in transformers; override with `--dropout` (0 disables). LoRA dropout
  (`--lora-dropout`) applies to the adapter input, as in PEFT. Masks are seeded per epoch.
- Every epoch writes `<out>/checkpoint/` (trainable weights, AdamW moments, RNG, history);
  `--resume` continues a run exactly (tests check bit-identical results) and `--stop-after N`
  time-boxes a run.
- LoRA targets the attention and MLP projections by default.
- Training uses Harpia's own differentiable forward passes (`harpia_train::nn`), because
  candle-nn's fused kernels (softmax, layer/RMS norm, rotary embeddings) have no backward pass.
  Tests check that these forwards match `harpia-local` inference within 1e-4 and that gradients
  reach every layer.
- Embedding fine-tuning for RAG (contrastive loss) is not implemented yet.
