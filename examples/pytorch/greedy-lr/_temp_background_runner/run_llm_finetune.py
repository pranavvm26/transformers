import os
import sys
import shutil
import torch
from accelerate import Accelerator
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training, 
    LoraConfig, 
    get_peft_model, 
    AutoPeftModelForCausalLM
)
from random import randrange
from random import randint
from datasets import load_dataset
from random import randint
from itertools import chain
from functools import partial
import sagemaker
from sagemaker.experiments.run import Run
from smexperiments_callback import SageMakerExperimentsCallback


print("transformers", transformers.__version__)


dataset_name = "w601sxs/simpleCoT"

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
    
def format_simple_cot(sample):
    instruction = f"### Instruction\n{sample['source']}"
    answer = f"### Answer\n{sample['target']}" 
    response = f"### Rationale\n{sample['rationale']}"
    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, answer, response] if i is not None])
    return prompt


def run(scheduler):
    
    model_id = "tiiuae/falcon-7b"

    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # The code is provided by the model authors in the repo.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        # quantization_config=bnb_config, 
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model.gradient_checkpointing_enable()
    
    model = prepare_model_for_kbit_training(model)
    
    config = LoraConfig(
        r=1536,
        lora_alpha=3584,
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    
    train_dataset = load_dataset(dataset_name, split="train[:11000]")
    validation_dataset = load_dataset(dataset_name, split="train[11000:11100]")
    test_dataset = load_dataset(dataset_name, split="train[11100:11200]")
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # template dataset to add prompt to each sample
    def template_dataset(sample):
        sample["text"] = f"{format_simple_cot(sample)}{tokenizer.eos_token}"
        return sample

    # apply prompt template per sample
    # train
    train_dataset = train_dataset.map(template_dataset, remove_columns=list(train_dataset.features))
    # validation
    validation_dataset = validation_dataset.map(template_dataset, remove_columns=list(validation_dataset.features))
    # test
    test_dataset = test_dataset.map(template_dataset, remove_columns=list(test_dataset.features))

    # print random sample
    print(validation_dataset[randint(0, len(validation_dataset))]["text"])
    
    # empty list to save remainder from batches to use in next batch
    remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}

    def chunk(sample, remainder, chunk_length=2048):
        
        # Concatenate all texts and add remainder from previous batch
        concatenated_examples = {k: list(chain(*sample[k])) for k in sample.keys()}
        concatenated_examples = {k: remainder[k] + concatenated_examples[k] for k in concatenated_examples.keys()}
        # get total number of tokens for batch
        batch_total_length = len(concatenated_examples[list(sample.keys())[0]])

        # get max number of chunks for batch
        if batch_total_length >= chunk_length:
            batch_chunk_length = (batch_total_length // chunk_length) * chunk_length

        # Split by chunks of max_len.
        result = {
            k: [t[i : i + chunk_length] for i in range(0, batch_chunk_length, chunk_length)]
            for k, t in concatenated_examples.items()
        }
        # add remainder to global variable for next batch
        remainder = {k: concatenated_examples[k][batch_chunk_length:] for k in concatenated_examples.keys()}
        # prepare labels
        result["labels"] = result["input_ids"].copy()
        return result
    
    # training
    lm_train_dataset = train_dataset.map(
        lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(train_dataset.features)
    ).map(
        partial(chunk, remainder=remainder, chunk_length=2048),
        batched=True,
    )

    # validation
    lm_valid_dataset = validation_dataset.map(
        lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(validation_dataset.features)
    ).map(
        partial(chunk, remainder=remainder, chunk_length=2048),
        batched=True,
    )

    # validation
    lm_test_dataset = test_dataset.map(
        lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(test_dataset.features)
    ).map(
        partial(chunk, remainder=remainder, chunk_length=2048),
        batched=True,
    )

    # Print total number of samples
    print(f"Train Length : {len(lm_train_dataset)} || Val Length : {len(lm_valid_dataset)} || Test Length : {len(lm_test_dataset)}")
    
    lr_schedulers = ['cosine', 'constant', 'greedy', 'linear']
    chosen_scheduler = lr_schedulers.index(scheduler)
    
    with Run(
        experiment_name="GreedyLr-Experimentation", 
        run_name=f"run-{scheduler}", 
        sagemaker_session=sagemaker.Session()
    ) as run:
    
        logging_dir = f"./model-outputs/script/{lr_schedulers[chosen_scheduler]}/tensorboard"
        output_dir = f"./model-outputs/script/{lr_schedulers[chosen_scheduler]}/output"
        temp_dir = f"./model-outputs/script/{lr_schedulers[chosen_scheduler]}/peft-model-dir"
        # model_dir = f"./model-outputs/{lr_schedulers[chosen_scheduler]}/fine-tuned-model-dir"

        run.log_parameters(
            {
                "exp_current_scheduler": lr_schedulers[chosen_scheduler],
                "exp_logging_dir": logging_dir,
                "exp_output_dir": output_dir,
                "exp_peft_dir": temp_dir
            }
        ) 
        
        train_args = transformers.TrainingArguments(
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            logging_dir=logging_dir,
            logging_steps=2,
            num_train_epochs=3,
            learning_rate=1e-4,
            bf16=False,
            save_strategy="no",
            output_dir=output_dir,
            report_to="tensorboard",
            lr_scheduler_type=lr_schedulers[chosen_scheduler],
            # # greedy
            # min_lr=1e-6,
            # smooth=True,
            # factor=0.9
            # cosine
            warmup_ratio=0.05
        )
        
        
        trainer = transformers.Trainer(
            model=model,
            train_dataset=lm_train_dataset,
            eval_dataset=lm_valid_dataset,
            args=train_args,
            callbacks=[SageMakerExperimentsCallback(region="us-east-1")],
            data_collator=transformers.DataCollatorForLanguageModeling(
                tokenizer, 
                mlm=False
            )
        )
        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

        trainer.train()

        trainer.model.save_pretrained(
            temp_dir, 
            safe_serialization=False
        )

        del model
        del trainer
        torch.cuda.empty_cache()
    
    
    
    #     model = AutoPeftModelForCausalLM.from_pretrained(
    #         temp_dir,
    #         # low_cpu_mem_usage=True,
    #         device_map="auto",
    #         torch_dtype=torch.float16,
    #     )
    #     model = model.merge_and_unload()

    #     model.save_pretrained(
    #         model_dir, 
    #         safe_serialization=True, 
    #         max_shard_size="9GB"
    #     )
    #     tokenizer.save_pretrained(
    #         save_directory=model_dir, 
    #         from_pt=True
    #     )

    #     shutil.rmtree(temp_dir)
    
    return 0



if __name__ == "__main__":
    _scheduler = sys.argv[1]
    print("************************************")
    print(f"Running training with {_scheduler}")
    print("************************************")
    print("\n")
    run(scheduler=_scheduler)