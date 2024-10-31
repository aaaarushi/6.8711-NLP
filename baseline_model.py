from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup

# CHECK FOR GPU!!
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"RUNNING ON DEVICE {device}...")


# reading the data
path = "./1000_trimmed_lyrics.csv"
print(f"USING DATASET AT {path}...")
df = pd.read_csv(path)
df['decade_label'] = df['year'].apply(lambda x: str(x)[:3] + '0s').astype('category').cat.codes
labels = torch.tensor(df['decade_label'].values)

# BERT tokenization: add [CLS] and [SEP], pad/truncate, create attention masks
MAX_LEN = 128
print("TOKENIZING LYRICS DATA...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenized_data = tokenizer.batch_encode_plus(
    df["lyrics"].tolist(),                    
    add_special_tokens = True,  
    max_length = MAX_LEN,
    padding = "max_length",
    truncation = True,      
    return_attention_mask = True,  
    return_tensors = 'pt',     
)

input_ids = tokenized_data['input_ids']
attention_masks = tokenized_data['attention_mask']


# split into train/validation/test and create DataLoaders
BATCH_SIZE = 32
print("SPLITTING AND CREATING DATALOADERS...")

train_inputs, temp_inputs, train_labels, temp_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.4)
validation_inputs, test_inputs, validation_labels, test_labels = train_test_split(temp_inputs, temp_labels, random_state=2018, test_size=0.5)
train_masks, temp_masks, _, _ = train_test_split(attention_masks, labels, random_state=2018, test_size=0.4)
validation_masks, test_masks, _, _ = train_test_split(temp_masks, temp_labels, random_state=2018, test_size=0.5)

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=32)


# TRAINING LOOP
def train(model, optimizer, name, scheduler=None, print_every=100, epochs=10):
    loss_log = []

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
            b_labels = b_labels.to(torch.long)

            model.zero_grad()
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

            loss = loss_fct(outputs.logits.view(-1, 5), b_labels.view(-1))
            total_loss += loss.item()
            if step % print_every == 0:
                print("  loss: ", loss.item())
                loss_log.append(loss.item())
          
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if scheduler:
                scheduler.step()

        print(f"EPOCH {epoch} Training Loss: {total_loss / len(train_dataloader)}")
        print(f"EPOCH {epoch} Training Accuracy: {calc_accuracy(model, train_dataloader)}")
        print(f"EPOCH {epoch} Validation Accuracy: {calc_accuracy(model, validation_dataloader)}")


    # PLOT AND SAVE
    print(f"\nFINAL Test Accuracy: {calc_accuracy(model, test_dataloader)}")

    plt.plot(range(0, len(loss_log)*print_every, print_every), loss_log)
    plt.title(f"Loss Curve for {name}")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(f'./loss_curve_{name}.png')
    plt.close()

    model.save_pretrained(f'./model_{name}')
    tokenizer.save_pretrained(f'./tokenizer_{name}')


# ACCURACY CALC
def calc_accuracy(model, dataloader):
    model.eval()
    total_accuracy = 0

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        b_labels = b_labels.to(torch.long)

        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).flatten()
        correct_preds = torch.sum(preds == b_labels).item()
        total_accuracy += correct_preds / b_labels.size(0)

    return total_accuracy / len(dataloader)



# run training loop with set hyperparameters
LR = 5e-5
DROPOUT = 0.4
DECAY = 0.1
EPOCHS = 10

# load pretrained BERT with single classification layer on top
config = BertConfig.from_pretrained("bert-base-uncased", 
                                    num_labels=5,
                                    hidden_dropout_prob=DROPOUT,
                                    attention_probs_dropout_prob=DROPOUT)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
model.to(device)
print("LOADING MODEL WITH PARAMETERS...")
print(model.config)

loss_fct = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=LR,
                              eps=1e-8,
                              weight_decay=DECAY)

# scheduler = get_linear_schedule_with_warmup(optimizer,
#                                             num_warmup_steps = 0,  # default value
#                                             num_training_steps = len(train_dataloader) * EPOCHS)

print(f"TRAINING WITH ADAMW AND LEARNING RATE {LR} AND DROPOUT {DROPOUT} AND WEIGHT DECAY {DECAY}....")
train(model, optimizer, "medium_trimmed", print_every=50, epochs=EPOCHS)
