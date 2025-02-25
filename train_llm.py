import os
import torch
import math
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments, set_seed
)
import data_reading_utils
from cmd_line_arguments import ScriptArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers.utils import logging
import text2sql_dataset



if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # logging.set_verbosity_error()
    logger = logging.get_logger("transformers")


    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # read data
    training_sft_dataset, testing_sft_dataset = [], []
    if args.sql_dataset_name == "pauq":
        training_sft_dataset = data_reading_utils.create_pauq_sft_dataset(args.path_to_training_file,
                                                                          args.tables_info_path,
                                                                            tokenizer,
                                                                          phase="train",
                                                                          try_one_batch=args.try_one_batch,
                                                                          batch_size=args.per_device_train_batch_size)

        testing_sft_dataset = data_reading_utils.create_pauq_sft_dataset(args.path_to_testing_file,
                                                                          args.tables_info_path,
                                                                          tokenizer,
                                                                          phase="train",
                                                                          try_one_batch=args.try_one_batch,
                                                                          batch_size=args.per_device_train_batch_size)

    elif args.sql_dataset_name == "ehrsql":
        training_sft_dataset = data_reading_utils.create_ehrsql_sft_dataset(args.path_to_training_file,
                                                                          args.tables_info_path,
                                                                          tokenizer,
                                                                          phase="train",
                                                                          try_one_batch=args.try_one_batch,
                                                                          batch_size=args.per_device_train_batch_size)

        testing_sft_dataset = data_reading_utils.create_ehrsql_sft_dataset(args.path_to_testing_file,
                                                                         args.tables_info_path,
                                                                         tokenizer,
                                                                         phase="train",
                                                                         try_one_batch=args.try_one_batch,
                                                                         batch_size=args.per_device_train_batch_size)

    tokenized_train_sft_dataset = text2sql_dataset.Text2SQLDataset(sft_dataset=training_sft_dataset,
                                                                   tokenizer=tokenizer, max_length=args.max_seq_length,
                                                                   device=device)
    tokenized_test_sft_dataset = text2sql_dataset.Text2SQLDataset(sft_dataset=testing_sft_dataset,
                                                                  tokenizer=tokenizer, max_length=args.max_seq_length,
                                                                  device=device)

    print('Training samples total size: ', len(tokenized_train_sft_dataset))
    # model

    model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map = 'cuda')
    print('Loaded model!')

    # following https://arxiv.org/pdf/2305.14314
    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=16,
            bias="none",
            target_modules=['q_proj', 'v_proj',
                            'k_proj', 'o_proj',
                            'gate_proj',
                            'up_proj', 'down_proj'],
            task_type="CAUSAL_LM",
        )

    batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    num_update_steps_per_epoch = len(training_sft_dataset) // batch_size
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    total_train_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    print('My total train steps: ', total_train_steps)

    num_warmup_steps = int(0.03 * total_train_steps)

    #approx log every 10 steps
    logging_steps = 10
    eval_steps = 100
    #approx save on every epoch
    save_steps=num_update_steps_per_epoch

    # training setup
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.gradient_accumulation_steps,
        optim="adamw_torch",
        save_steps=save_steps,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        learning_rate=args.learning_rate,
        bf16=True,
        max_grad_norm=0.3,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=num_warmup_steps,
        lr_scheduler_type="cosine",
        report_to=args.report_to,
        overwrite_output_dir=args.overwrite_output_dir,
        logging_dir=args.logging_dir,
        logging_strategy='steps',
        run_name=args.run_name,
        save_total_limit=1
    )

    response_template = f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    # The assistant answer is ignored during loss calculation
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)


    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_train_sft_dataset,
        eval_dataset=tokenized_test_sft_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        data_collator=collator,
    )
    print('Begin training!')
    trainer.train()
    print(f'Training finished, saving to {args.output_dir}')

    output_dir = os.path.join(args.output_dir, "final_checkpoints")
    trainer.model.save_pretrained(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)






