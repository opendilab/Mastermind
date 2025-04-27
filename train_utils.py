import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments, TrainerCallback, Trainer


class InstructDataset(Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def create_inputs_and_labels(self, question, answer):
        eop = self.tokenizer.eos_token_id
        prompt = self.tokenizer.encode(question, max_length=2048, truncation=True, add_special_tokens=True)
        completion = self.tokenizer.encode(answer, max_length=2048, truncation=True, add_special_tokens=False)
        inputs = prompt + completion + [eop]
        labels = [-100] * len(prompt) + completion + [eop]
        inputs, labels = torch.tensor(inputs), torch.tensor(labels)
        return inputs, labels

    def __getitem__(self, index):
        item_data = self.data[index]
        input_ids, labels = self.create_inputs_and_labels(question=item_data['sentence'], answer=item_data['answer'])
        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch):
    # Sort the batch in the descending order
    sorted_batch = sorted(batch, key=lambda x: x['input_ids'].shape[0], reverse=True)
    # Get each sequence and pad it
    sequences = [x['input_ids'] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    # Don't forget to grab the labels of the *sorted* batch
    labels = [x['labels'] for x in sorted_batch]
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return {'input_ids': sequences_padded, 'labels': labels_padded}


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if return_outputs:
            outputs = model(input_ids=inputs["input_ids"], labels=inputs["labels"])
            return outputs.loss, outputs
        return model(input_ids=inputs["input_ids"], labels=inputs["labels"]).loss


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


def main(train_dataset, test_dataset, model, tokenizer, compute_metrics, output_dir):
    # Training arguments.
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        evaluation_strategy="steps",
        eval_steps=400,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        report_to=None,
        remove_unused_columns=False,
        per_device_train_batch_size=5,
        per_device_eval_batch_size=10,
        group_by_length=False,
        dataloader_pin_memory=False,
        warmup_steps=3000,
        weight_decay=0.01,
        bf16=True,
        tf32=True,
    )
    trainer = ModifiedTrainer(
        model=model,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    # Resume from the checkpoint
    trainer.add_callback(EvaluateFirstStepCallback())
    trainer.train()
