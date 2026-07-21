"""Optional local supervised fine-tuning with Hugging Face TRL and PEFT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import TrainingConfig
from .dataset import load_qa_pairs
from .exceptions import TrainingError


def _training_imports() -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise TrainingError(
            "Training dependencies are missing. Install `ritaline[train]` for LoRA/full "
            "training or `ritaline[qlora]` for QLoRA."
        ) from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


def _resolve_precision(config: TrainingConfig, torch: Any) -> tuple[bool, bool, Any]:
    cuda_available = bool(torch.cuda.is_available())
    bf16_supported = cuda_available and bool(torch.cuda.is_bf16_supported())

    bf16 = config.bf16 if config.bf16 is not None else bf16_supported
    fp16 = config.fp16 if config.fp16 is not None else cuda_available and not bf16
    if bf16 and fp16:
        raise TrainingError("bf16 and fp16 cannot both be enabled")
    if bf16 and not bf16_supported:
        raise TrainingError("bf16 was requested but is not supported by the current CUDA device")
    if fp16 and not cuda_available:
        raise TrainingError("fp16 was requested but CUDA is not available")

    dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32
    return bool(bf16), bool(fp16), dtype


def fine_tune(dataset_path: str | Path, config: TrainingConfig) -> Path:
    """Fine-tune a local/Hugging Face causal LM using generated Q&A pairs."""
    if not config.enabled:
        raise TrainingError("Training is disabled in the job configuration")

    pairs = load_qa_pairs(dataset_path)
    if not pairs:
        raise TrainingError(f"No Q&A pairs found in {dataset_path}")

    imports = _training_imports()
    torch = imports["torch"]
    Dataset = imports["Dataset"]
    LoraConfig = imports["LoraConfig"]
    BitsAndBytesConfig = imports["BitsAndBytesConfig"]
    SFTConfig = imports["SFTConfig"]
    SFTTrainer = imports["SFTTrainer"]

    bf16, fp16, dtype = _resolve_precision(config, torch)
    dataset = Dataset.from_list([pair.training_record() for pair in pairs])

    eval_dataset = None
    train_dataset = dataset
    if config.eval_ratio > 0 and len(dataset) >= 20:
        eval_size = max(1, round(len(dataset) * config.eval_ratio))
        split = dataset.train_test_split(test_size=eval_size, seed=config.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        gradient_checkpointing=config.gradient_checkpointing,
        packing=config.packing,
        max_length=config.max_seq_length,
        completion_only_loss=True,
        seed=config.seed,
        data_seed=config.seed,
        bf16=bf16,
        fp16=fp16,
        report_to=config.report_to or "none",
        model_init_kwargs={
            "trust_remote_code": config.trust_remote_code,
            "dtype": dtype,
        },
    )

    peft_config = None
    quantization_config = None
    if config.method in {"lora", "qlora"}:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config.lora_target_modules,
        )

    if config.method == "qlora":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise TrainingError(
                "QLoRA requires bitsandbytes. Install `ritaline[qlora]`."
            ) from exc
        compute_dtype = dtype
        if config.qlora_compute_dtype == "bfloat16":
            compute_dtype = torch.bfloat16
        elif config.qlora_compute_dtype == "float16":
            compute_dtype = torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    trainer = SFTTrainer(
        model=config.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        quantization_config=quantization_config,
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    processing_class = getattr(trainer, "processing_class", None)
    if processing_class is not None and hasattr(processing_class, "save_pretrained"):
        processing_class.save_pretrained(str(output_dir))
    return output_dir
