import torch
import pandas as pd
import os
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, \
    RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification, ElectraTokenizer, \
    ElectraForSequenceClassification, ElectraConfig
from torch.utils.data import TensorDataset, DataLoader


def encode_sentences(sentences, tokenizer):
    input_ids = []
    attention_masks = []
    for sentence in sentences:
        encoded = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'].squeeze())
        attention_masks.append(encoded['attention_mask'].squeeze())
    return torch.stack(input_ids), torch.stack(attention_masks)


def predict_on_test_data(saved_model_path, prediction_size=-1, final_test=False):

    if final_test:
        test_df = pd.read_csv('SST-2/test.tsv', sep='\t')
    else:
        test_df = pd.read_csv('SST-2/dev.tsv', sep='\t')

    if 0 < prediction_size < len(test_df):
        test_df = test_df[:prediction_size]

    model_directory = saved_model_path

    if 'robert' in saved_model_path:
        tokenizer = RobertaTokenizer.from_pretrained(saved_model_path)
        # Load the model configuration
        model_config = RobertaConfig.from_pretrained(model_directory)
        # Load the fine-tuned model
        model = RobertaForSequenceClassification(model_config)

    elif 'electra' in saved_model_path:
        tokenizer = ElectraTokenizer.from_pretrained(saved_model_path)
        # Load the model configuration
        model_config = ElectraConfig.from_pretrained(model_directory)
        # Load the fine-tuned model
        model = ElectraForSequenceClassification(model_config)

    else:
        tokenizer = BertTokenizer.from_pretrained(saved_model_path)
        # Load the model configuration
        model_config = BertConfig.from_pretrained(model_directory)
        # Load the fine-tuned model
        model = BertForSequenceClassification(model_config)

    test_inputs, test_masks = encode_sentences(test_df['sentence'], tokenizer)



    model.load_state_dict(torch.load(os.path.join(model_directory, "pytorch_model.bin")))

    # Load the saved model

    test_data = TensorDataset(test_inputs, test_masks)
    test_dataloader = DataLoader(test_data, batch_size=128)

    predictions = []
    with torch.no_grad():
        for batch in test_dataloader:
            batch = tuple(t.to(torch.device("cpu")) for t in batch)
            inputs, masks = batch

            outputs = model(inputs, attention_mask=masks)
            logits = outputs.logits

            # Apply softmax to obtain probabilities
            probabilities = torch.softmax(logits, dim=1)

            # Get the predicted labels or perform any other post-processing
            predicted_labels = torch.argmax(probabilities, dim=1)

            # Append the predicted labels to the list of predictions
            predictions.extend(predicted_labels.cpu().numpy().tolist())

    test_df['predictions'] = predictions

    if final_test:
        # Convert DataFrame to text file
        test_df[['sentence', 'predictions']].to_csv('predictions.txt', sep='###', header=False, index=False)
        return

    else:
        acc = len(test_df.loc[test_df.predictions == test_df.label]) / len(test_df)
        print('{}% acc'.format(round(acc * 1000) / 1000))

        return acc


if __name__ == '__main__':
    path = 'google/electra-base-generator_finetuned_model'
    predict_on_test_data(path)
    # predict_on_test_data("roberta-base_finetuned_model")
    # predict_on_test_data("bert-base-uncased_finetuned_model")

    # got 0.553% acc on when i did one epoch on 512 samples and tested on validation data -- this is for bert-based-uncased
    # 0.735% acc when on 2048 one epoch same data!
    # 0.87% acc on 2048*4 one epoch
    # 89.4 for roberta
    # 51.7 electra, 82.3% for elecra after training on like almost all the data if not all
