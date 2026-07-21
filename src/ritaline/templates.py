"""Starter YAML templates used by `ritaline init`."""

ENDPOINT_YAML = """# OpenAI-compatible endpoint used to GENERATE the Q&A data.
# Keep secrets in environment variables, never in this file.
base_url: https://api.deepseek.com
model: deepseek-v4-flash
api_key_env: DEEPSEEK_API_KEY
chat_completions_path: /chat/completions
timeout_seconds: 120
max_retries: 5
retry_backoff_seconds: 1.5
max_concurrency: 4
temperature: 0.2
max_tokens: 800
response_format: json_object
headers: {}
# Provider-specific fields are merged into the JSON request body.
extra_body:
  thinking:
    type: disabled
"""

JOB_YAML = """generation:
  qa_count: 100
  chunk_size_chars: 7000
  chunk_overlap_chars: 700
  min_chunk_chars: 300
  deduplicate_questions: true
  max_attempts_per_pair: 6
  seed: 42
  output_path: outputs/qa_pairs.jsonl
  training_dataset_path: outputs/training_dataset.jsonl

  system_prompt: |
    You generate high-quality training data from supplied source text.
    Every answer must be fully supported by the source. Do not use outside knowledge.
    Questions must be clear, self-contained, and useful for supervised fine-tuning.

  styles:
    - name: factual
      instruction: Ask for a concrete fact explicitly stated in the source.
    - name: explanatory
      instruction: Ask for an explanation of a process, relationship, or rationale in the source.
    - name: comparative
      instruction: Ask the learner to compare two items only when both appear in the source.
    - name: applied
      instruction: Ask how a source principle applies to a realistic grounded case.

  # You can override user_prompt_template here. It must contain:
  # {style_name}, {style_instruction}, {source_name}, {page_range}, and {chunk}.

training:
  enabled: true
  # This must be downloadable model weights or a local model directory.
  # An API-only endpoint name cannot be locally fine-tuned by TRL.
  model_name_or_path: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  output_dir: outputs/ritaline-model
  method: lora  # lora, qlora, or full
  trust_remote_code: false
  max_seq_length: 2048
  num_train_epochs: 1
  max_steps: -1
  learning_rate: 0.0002
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 8
  warmup_ratio: 0.03
  weight_decay: 0.0
  logging_steps: 10
  save_steps: 100
  eval_steps: 100
  save_total_limit: 2
  eval_ratio: 0.05
  gradient_checkpointing: true
  packing: false
  seed: 42
  bf16: null
  fp16: null
  report_to: []
  resume_from_checkpoint: null
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_target_modules: all-linear
  qlora_compute_dtype: auto
"""
