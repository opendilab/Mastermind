import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from metrics import compute_doudizhu_metrics
from utils import load_dataset
from train_utils import InstructDataset, main


def compute_metrics(pred):
    labels_ids = pred.label_ids[..., 1:]
    pred_ids = pred.predictions[0][..., :-1]
    for id, pred in enumerate(pred_ids):
        pred_ids[id][labels_ids[id] == -100] = 2
        pred_ids[id][pred_ids[id] == -100] = 2
        labels_ids[id][labels_ids[id] == -100] = 2

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    return compute_doudizhu_metrics(pred_str, label_str)


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


def prepare_dataset(tokenizer):
    test_data = load_dataset("doudizhu", split='test')
    train_data = load_dataset("doudizhu", split='train')
    return InstructDataset(train_data, tokenizer), InstructDataset(test_data, tokenizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str, default='NousResearch/Llama-2-7b-hf')
    args = parser.parse_args()
    # Initialize base model.
    base_model = args.pretrained_path
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, revision='main', device_map='auto',
                                                 torch_dtype=torch.float16, load_in_8bit=True)

    # Initialize lora model.
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset, test_dataset = prepare_dataset(tokenizer)

    # Train the model.
    main(train_dataset=train_dataset, test_dataset=test_dataset, model=model,
         tokenizer=tokenizer, compute_metrics=compute_metrics, output_dir='./output_doudizhu')
