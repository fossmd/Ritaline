# Ritaline

Ritaline turns a PDF or TXT document into a grounded question-and-answer dataset, then optionally fine-tunes an open-weight causal language model on that data.

It is designed as a normal Python package with both a CLI and a Python API, so it can later be published to PyPI.

## What it does

1. Reads a `.pdf` or `.txt` file.
2. Extracts and normalizes text.
3. Splits the text into overlapping chunks while preserving page provenance.
4. Assigns Q&A styles in strict round-robin order.
5. Calls an OpenAI-compatible Chat Completions endpoint once per requested pair.
6. Validates JSON output, retries failures, rejects empty pairs, and optionally removes duplicate questions.
7. Writes resumable raw JSONL and TRL-compatible prompt-completion JSONL.
8. Optionally fine-tunes a Hugging Face/local model with full SFT, LoRA, or QLoRA.

## Important model distinction

Ritaline uses two model settings:

- **Generation endpoint model** in `endpoint.yaml`: the remote or locally served model that creates Q&A pairs.
- **Training model** in `job.yaml`: downloadable Hugging Face weights or a local model directory that TRL can fine-tune.

An API-only model name cannot be locally fine-tuned unless its provider exposes a separate fine-tuning API. For example, you can use a hosted DeepSeek endpoint to generate data and a smaller open-weight DeepSeek distillation model as the trainable base.

## Project layout

```text
ritaline/
├── pyproject.toml
├── README.md
├── LICENSE
├── examples/
│   ├── endpoint.yaml
│   ├── job.yaml
│   └── sample.txt
├── src/ritaline/
│   ├── __init__.py
│   ├── __main__.py
│   ├── chunking.py
│   ├── cli.py
│   ├── config.py
│   ├── dataset.py
│   ├── documents.py
│   ├── exceptions.py
│   ├── generation.py
│   ├── llm.py
│   ├── models.py
│   ├── pipeline.py
│   ├── prompts.py
│   ├── templates.py
│   └── training.py
└── tests/
    ├── test_chunking.py
    ├── test_config.py
    ├── test_documents.py
    ├── test_generation.py
    └── test_llm.py
```

## Installation

For Q&A generation only:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For LoRA or full fine-tuning:

```bash
pip install -e ".[train]"
```

For QLoRA with bitsandbytes:

```bash
pip install -e ".[qlora]"
```

Development tools:

```bash
pip install -e ".[dev]"
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Quick start

Create starter configs:

```bash
ritaline init configs
```

Set the API key named in `configs/endpoint.yaml`:

```bash
export DEEPSEEK_API_KEY="your-key"
```

Validate the configuration:

```bash
ritaline validate \
  --endpoint configs/endpoint.yaml \
  --job configs/job.yaml
```

Inspect prompts without making API calls:

```bash
ritaline preview document.pdf \
  --endpoint configs/endpoint.yaml \
  --job configs/job.yaml \
  --count 3
```

Generate only the Q&A data:

```bash
ritaline generate document.pdf \
  --endpoint configs/endpoint.yaml \
  --job configs/job.yaml
```

Generate data and fine-tune the configured model:

```bash
ritaline run document.pdf \
  --endpoint configs/endpoint.yaml \
  --job configs/job.yaml
```

Fine-tune later from an existing raw Ritaline dataset:

```bash
ritaline train \
  --job configs/job.yaml \
  --dataset outputs/qa_pairs.jsonl
```

## Endpoint configuration

`endpoint.yaml` configures an OpenAI-compatible `/chat/completions` endpoint.

```yaml
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
extra_body:
  thinking:
    type: disabled
```

### Endpoint fields

| Field | Purpose |
|---|---|
| `base_url` | Provider root URL, without `/chat/completions` unless the provider requires it. |
| `model` | Model used to generate Q&A pairs. |
| `api_key_env` | Environment variable containing the API key. Set to `null` for an unauthenticated local endpoint. |
| `chat_completions_path` | Chat Completions route. |
| `timeout_seconds` | Per-request timeout. |
| `max_retries` | HTTP/network retries inside each generation attempt. |
| `retry_backoff_seconds` | Initial exponential retry delay. |
| `max_concurrency` | Maximum simultaneous endpoint requests. |
| `temperature` | Sampling temperature. Low values are normally better for grounded Q&A. |
| `max_tokens` | Maximum completion tokens for each Q&A pair. |
| `response_format` | `json_object` or `none`, for endpoints that do not support JSON mode. |
| `headers` | Additional HTTP headers. |
| `extra_body` | Provider-specific request fields merged into the JSON body. |

Secrets are read from environment variables and should not be committed to YAML.

## Generation configuration

The `generation` section of `job.yaml` controls chunking, prompt behavior, style assignment, and output paths.

```yaml
generation:
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

  styles:
    - name: factual
      instruction: Ask for a concrete fact explicitly stated in the source.
    - name: explanatory
      instruction: Ask for an explanation of a process, relationship, or rationale.
    - name: comparative
      instruction: Compare two items only when both occur in the source.
    - name: applied
      instruction: Ask how a source principle applies to a source-grounded case.
```

For four styles and ten pairs, the style order is:

```text
factual, explanatory, comparative, applied,
factual, explanatory, comparative, applied,
factual, explanatory
```

The style is tied to the output slot, so retries do not disturb round-robin distribution.

### Custom user prompt

`user_prompt_template` is optional. When supplied, it must contain all five placeholders:

```text
{style_name}
{style_instruction}
{source_name}
{page_range}
{chunk}
```

The default template asks for exactly one JSON object:

```json
{"question": "...", "answer": "..."}
```

## Training configuration

```yaml
training:
  enabled: true
  model_name_or_path: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  output_dir: outputs/ritaline-model
  method: lora

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
```

### Training methods

- `lora`: trains parameter-efficient LoRA adapters. This is the recommended default.
- `qlora`: loads the base model in 4-bit NF4 and trains LoRA adapters. Install the `qlora` extra.
- `full`: updates the entire model and generally needs substantially more GPU memory.

Ritaline uses TRL conversational prompt-completion records and enables completion-only loss, so the question tokens are treated as the prompt and loss is applied to the answer completion.

### Precision

With both `bf16` and `fp16` set to `null`, Ritaline chooses:

- BF16 when a CUDA device reports BF16 support.
- FP16 on other CUDA devices.
- FP32 on CPU.

Explicitly setting unsupported precision raises an error instead of silently continuing.

## Output files

### Raw resumable dataset

`outputs/qa_pairs.jsonl` contains one record per pair:

```json
{
  "slot_index": 0,
  "question": "What does the source state about ...?",
  "answer": "The source states ...",
  "style": "factual",
  "source_name": "document.pdf",
  "source_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "chunk_index": 3,
  "page_start": 8,
  "page_end": 9,
  "generation_model": "deepseek-v4-flash",
  "metadata": {
    "attempt": 1,
    "char_start": 18900,
    "char_end": 25900
  }
}
```

This file is used for resume and auditing.

### Training export

`outputs/training_dataset.jsonl` contains TRL-compatible prompt-completion records:

```json
{
  "prompt": [{"role": "user", "content": "Question text"}],
  "completion": [{"role": "assistant", "content": "Answer text"}],
  "style": "factual",
  "source_name": "document.pdf",
  "source_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "chunk_index": 3,
  "page_start": 8,
  "page_end": 9
}
```

The training command reads the raw file because it validates all Ritaline metadata before converting records in memory.

## Resume behavior

Generation is resumable by default. A saved pair is reused only when:

- Its slot is still within the requested `qa_count`.
- Its source filename matches the current document.
- Its normalized source-content SHA-256 fingerprint matches.
- Its generation model matches the endpoint config.
- Its style matches the round-robin style expected for that slot.

Successful pairs are appended immediately. At the end, the file is atomically rewritten in slot order. If some slots fail, successful output remains on disk and the command reports a partial result. Running the same command again retries only missing slots.

Start from scratch with:

```bash
ritaline generate document.pdf \
  --endpoint configs/endpoint.yaml \
  --job configs/job.yaml \
  --no-resume
```

## Python API

### End-to-end

```python
from ritaline import load_document, QAGenerator
from ritaline.config import load_endpoint_config, load_job_config
import asyncio

endpoint = load_endpoint_config("configs/endpoint.yaml")
job = load_job_config("configs/job.yaml")
document = load_document("document.pdf")

generator = QAGenerator(endpoint, job.generation)
pairs = asyncio.run(generator.generate(document, resume=True))

print(len(pairs))
print(pairs[0].question)
```

### Full pipeline

```python
from ritaline import run_pipeline
from ritaline.config import load_endpoint_config, load_job_config

endpoint = load_endpoint_config("configs/endpoint.yaml")
job = load_job_config("configs/job.yaml")

pairs, model_path = run_pipeline(
    "document.pdf",
    endpoint,
    job,
    resume=True,
    skip_training=False,
)
```

### Build configs in Python

```python
from pathlib import Path
from ritaline import EndpointConfig, GenerationConfig, JobConfig, QAStyle, TrainingConfig

endpoint = EndpointConfig(
    base_url="http://localhost:8000/v1",
    model="my-served-model",
    api_key_env=None,
)

job = JobConfig(
    generation=GenerationConfig(
        qa_count=20,
        styles=[
            QAStyle(name="factual", instruction="Ask a directly stated fact."),
            QAStyle(name="why", instruction="Ask for a source-supported reason."),
        ],
        output_path=Path("outputs/raw.jsonl"),
        training_dataset_path=Path("outputs/train.jsonl"),
    ),
    training=TrainingConfig(
        model_name_or_path="Qwen/Qwen3-0.6B",
        output_dir=Path("outputs/model"),
        method="lora",
    ),
)
```

## Local endpoints

Any server that implements an OpenAI-compatible Chat Completions route can be used, including vLLM, SGLang, llama.cpp servers, or compatible hosted providers.

Example:

```yaml
base_url: http://localhost:8000/v1
model: local-model-name
api_key_env: null
chat_completions_path: /chat/completions
response_format: none
extra_body: {}
```

Set `response_format: none` when the server rejects OpenAI JSON mode. The prompt still instructs the model to return JSON, and Ritaline validates the response.

## PDF behavior

Ritaline uses PyMuPDF text extraction with natural-position sorting. It does not run OCR. A scanned image-only PDF will raise a clear error and should be OCR-processed before use.

PDF text order depends on how the PDF was produced. Always use `ritaline preview` and inspect generated data before training.

## Data quality guidance

Synthetic data can contain subtle unsupported claims even with grounding instructions. Before expensive training:

1. Generate a small pilot set.
2. Review samples from every Q&A style.
3. Check answers against source pages.
4. Remove low-quality or repetitive records.
5. Keep a held-out evaluation set that was not used for training.
6. Confirm that you have the right to process the source document and train on its content.

The package validates structure, not semantic truth. A future extension could add a second-pass verifier model or retrieval-based citation checks.

## Testing

```bash
pytest
```

The included tests cover:

- PDF/TXT-independent text loading behavior.
- Page-aware chunking.
- Configuration validation.
- JSON extraction.
- Exact Q&A count and round-robin style assignment with a fake endpoint.

## Build and publish to PyPI

Update these values before publishing:

- Package version in `pyproject.toml` and `src/ritaline/__init__.py`.
- Author information.
- Repository URLs.
- License and project metadata.

Build:

```bash
python -m build
```

Validate distributions:

```bash
python -m twine check dist/*
```

Upload to TestPyPI first:

```bash
python -m twine upload --repository testpypi dist/*
```

Then publish to PyPI:

```bash
python -m twine upload dist/*
```

Use a PyPI API token and trusted publishing where possible. Do not store credentials in the repository.

## Current limitations

- Only PDF and TXT input are accepted.
- OCR is not included.
- Each endpoint request creates one Q&A pair; this favors validation and exact style assignment over minimum API cost.
- Local training supports causal language models through Hugging Face TRL.
- Provider-managed fine-tuning APIs are not implemented.
- Distributed training configuration is delegated to Accelerate/Transformers rather than wrapped by additional Ritaline settings.
- Semantic hallucination detection is not yet included.

## License

MIT
