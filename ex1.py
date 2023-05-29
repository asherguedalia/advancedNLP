import torch
import sys
import time
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, \
    RobertaTokenizer, RobertaForSequenceClassification, ElectraTokenizer, ElectraForSequenceClassification
from transformers import get_linear_schedule_with_warmup

import random
import numpy as np
from ex1_helper import encode_sentences, predict_on_test_data


import pandas as pd
# Set the seed value for reproducibility


def train_model(seed_value, train_size, val_size, model_to_use):

    # Set the seed for Python's random module
    random.seed(seed_value)

    # Set the seed for NumPy
    np.random.seed(seed_value)

    # Set the seed for PyTorch (both CPU and GPU)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    train_df = pd.read_csv('SST-2/train.tsv', sep='\t')

    if 0 < train_size < len(train_df):
        train_df = train_df[:prediction_size]

    # val_df = pd.read_csv('SST-2/dev.tsv', sep='\t')[:512]

    if 'robert' in model_to_use:
        tokenizer = RobertaTokenizer.from_pretrained(model_to_use)
        model = RobertaForSequenceClassification.from_pretrained(
            model_to_use,
            num_labels=2,  # Positive and negative sentiment
            output_attentions=False,
            output_hidden_states=False
        )
    elif 'electra' in model_to_use:
        tokenizer = ElectraTokenizer.from_pretrained(model_to_use)
        model = ElectraForSequenceClassification.from_pretrained(
            model_to_use,
            num_labels=2,  # Positive and negative sentiment
            output_attentions=False,
            output_hidden_states=False
        )
    else:
        tokenizer = BertTokenizer.from_pretrained(model_to_use)
        model = BertForSequenceClassification.from_pretrained(
            model_to_use,
            num_labels=2,  # Positive and negative sentiment
            output_attentions=False,
            output_hidden_states=False
        )

    train_inputs, train_masks = encode_sentences(train_df['sentence'], tokenizer)
    # val_inputs, val_masks = encode_sentences(val_df['sentence'], tokenizer)


    # Step 6: Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_labels = torch.tensor(train_df['label'])
    val_labels = torch.tensor(train_df['label'])

    batch_size = 256
    train_data = torch.utils.data.TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = torch.utils.data.RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    num_epochs = 1
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )


    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            inputs, masks, labels = batch

            model.zero_grad()
            outputs = model(inputs, attention_mask=masks, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            print(f"Epoch {epoch + 1}/{num_epochs} | Step {step}/{len(train_dataloader)} | Loss: {loss.item()}")

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} - Average training loss: {avg_train_loss}")
    #
    # # Step 7: Validation
    # model.eval()
    #
    # val_data = torch.utils.data.TensorDataset(val_inputs, val_masks, val_labels)
    # val_dataloader = DataLoader(val_data, batch_size=batch_size)
    #
    # total_val_loss = 0
    # predictions = []
    # total_examples = 0
    # correct_predictions = 0
    # with torch.no_grad():
    #     for batch in val_dataloader:
    #         batch = tuple(t.to(device) for t in batch)
    #         inputs, masks, labels = batch
    #
    #         outputs = model(inputs, attention_mask=masks, labels=labels)
    #         loss = outputs.loss
    #         logits = outputs.logits
    #
    #         predicted_labels = torch.argmax(logits, dim=1)
    #         total_examples += labels.size(0)
    #         correct_predictions += (predicted_labels == labels).sum().item()
    #


    # Specify the path where you want to save the model
    output_dir = "./{}_finetuned_model".format(model_to_use)

    # Save the model using the 'save_pretrained' method
    model.save_pretrained(output_dir)

    # Save the tokenizer as well if needed
    tokenizer.save_pretrained(output_dir)

    print("Trained model saved to:", output_dir)


if __name__ == '__main__':

    seed_number = int(sys.argv[1])
    training_size = int(sys.argv[2])
    val_size = int(sys.argv[3])
    prediction_size = int(sys.argv[4])

    print('args are {0}, {1}, {2}, {3}'.format(seed_number, training_size, val_size, prediction_size))
    model_names = ['roberta-base', 'bert-base-uncased', 'google/electra-base-generator']

    results = []
    time_dict = {k: 0 for k in model_names}

    for model_name in model_names:

        accs = []
        train_times = []
        predict_times = []
        for i in range(int(seed_number)):
            train_start = time.time()
            train_model(i, training_size, val_size, model_name)
            train_end = time.time() - train_start

            predict_start = time.time()
            acc = predict_on_test_data("{}_finetuned_model".format(model_name), prediction_size=val_size)
            predict_end = time.time() - predict_start

            train_times.append(train_end)
            predict_times.append(predict_end)
            accs.append(acc)

        arr = np.array(accs)
        std = arr.std()
        mean = arr.mean()

        cur_model_res = [model_name, mean, std]
        results.append(cur_model_res)

        time_dict[model_name] = (np.array(train_times).mean(), np.array(predict_times).mean())

    with open("res.txt", "w") as file:
        for result in results:
            file.write(f"{result[0]},{result[1]} +- {result[2]}\n")
        file.write("----\n")
        file.write(f"train time, predict time,{time_dict}\n")

    # now predict on best
    winner_acc_idx = 0
    winner_acc = 0
    for i in range(3):
        if results[i][1] > winner_acc:
            winner_acc = results[i][1]
            winner_acc_idx = i
    predict_on_test_data("{}_finetuned_model".format(model_names[winner_acc_idx]), prediction_size=prediction_size)