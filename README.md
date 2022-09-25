# QAwithAutoLabel


## Components

1. qa_trainer

Extend from transformer.trainer class in order to make it track the training accuracy of each epoch

### In the original training flow (transformer.trainer):
- Calling train function
```python
trainer = QATrainer(
    model=model,
    args=training_args,
    train_dataset=subtrain,
    eval_dataset=valid,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.train()
```
- After preprocess, it calls inner_training_loop for running all epoches. 

- For each epoch, it calls training_step to return the training loss (tr_loss_step)

### In the new training flow (qa_trainer):
- The training_step to return the training loss (tr_loss_step) and model's output for the current batch data.

- In inner_training_loop, we count all the correct start and end answers. And log it into system (wandb)
```python
metrics = dict()
metrics["start_acc"] = start_cor / n
metrics["end_acc"] = end_cor / n
self.log(metrics)
```
