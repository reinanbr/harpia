//! # harpia-train
//!
//! Fine-tuning in pure Rust on [candle](https://github.com/huggingface/candle):
//!
//! | function | what | status |
//! |---|---|---|
//! | [`train_classifier`] | full fine-tuning of BERT/RoBERTa/XLM-R classifiers and NLI cross-encoders | stable |
//! | [`train_lora_sft`] | LoRA supervised fine-tuning of Qwen2/Llama decoders, PEFT export | stable |
//! | [`train_lora_dpo`] | LoRA DPO against the frozen base model | experimental |
//!
//! Building blocks: [`backend::TrainStore`] (frozen / trainable / LoRA weights behind a candle
//! `VarBuilder`), [`optim`] (AdamW groups, schedules, clipping, losses, macro F1).
//!
//! Differences from typical PyTorch recipes, by design or by current limits: no dropout during
//! training; optimizer moments are not checkpointed (a restarted run starts a fresh optimizer);
//! LoRA dropout is not supported.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod backend;
mod classifier;
mod lora;
pub mod nn;
pub mod optim;

pub use classifier::{ClassifierConfig, EpochReport, Example, Progress, TrainReport, train_classifier};
pub use lora::{LoraConfig, LoraReport, LoraTrainConfig, PreferenceIds, train_lora_dpo, train_lora_sft};
