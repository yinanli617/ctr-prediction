import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
from torch.optim.lr_scheduler import StepLR
import argparse
from google.cloud import storage
import logging


WIDE_DIM = 453
COLUMNS = ['label',
           'device_modelSEP8251SEP256',
           'app_idSEP8552SEP256',
           'site_idSEP4737SEP256',
           'site_domainSEP7745SEP256',
           'app_domainSEP559SEP128',
           ]
COLUMNS = ['wide_feature_' + str(i) for i in range(WIDE_DIM)] + COLUMNS
EMBEDDING_INPUTS = COLUMNS[-5:]
GOOGLE_APPLICATION_CREDENTIALS = 'kfp-yli-2b9eae382b6c.json'


# This helper function retrieves the urls of training and validation data stored on GCS
# In step 1 - spark job, the features are saved in 10,000 partitions => thus 10,000 csv files
def list_blobs(bucket_name):
    """Lists all the blobs in the bucket."""
    # bucket_name = "your-bucket-name"

    storage_client = storage.Client(project='kfp-yli')

    # Note: Client.list_blobs requires at least package version 1.17.0.
    train_blobs = storage_client.list_blobs(bucket_name, prefix='avazu-ctr-prediction/training_csv/')
    val_blobs = storage_client.list_blobs(bucket_name, prefix='avazu-ctr-prediction/validation_csv/')

    base_url = 'https://storage.googleapis.com/kfp-yli/'
    train_urls = [base_url + b.name for b in train_blobs if b.name.endswith('.csv')]
    val_urls = [base_url + b.name for b in val_blobs if b.name.endswith('.csv')]

    return train_urls, val_urls


# The wide and deep model architecture
class WideAndDeep(nn.Module):
    def __init__(self, wide_dim, embedding_inputs, hidden_layers, dropout_p=0.5):
        super().__init__()
        self.wide_dim = wide_dim
        self.embedding_inputs = embedding_inputs
        self.deep_feature_dim = 0
        self.hidden_layers = hidden_layers

        # For each deep feature, create an embedding layer to convert them to embeddings
        for embedding_input in self.embedding_inputs:
            col_name, vocab_size, embed_dim = embedding_input.split('SEP')
            setattr(self, col_name + '_emb_layer', nn.Embedding(int(vocab_size), int(embed_dim)))
            self.deep_feature_dim += int(embed_dim)

        # A series of hidden layers that take the embeddings as input
        self.linear_layer_1 = nn.Linear(self.deep_feature_dim, self.hidden_layers[0])
        self.bn_1 = nn.BatchNorm1d(self.hidden_layers[0])
        for i, hidden_layer in enumerate(self.hidden_layers[1:]):
            setattr(self, f'linear_layer_{i + 2}', nn.Linear(self.hidden_layers[i], hidden_layer))

        self.dropout = nn.Dropout(p=dropout_p)

        # Final dense layer that combine the wide features and the deep features and generate output
        self.fc = nn.Linear(self.wide_dim + self.hidden_layers[-1], 1)

    def forward(self, X_w, X_d):
        embeddings = [getattr(self, col_name + '_emb_layer')(X_d[:, i].long())
                      for i, embedding_input in enumerate(self.embedding_inputs)
                      for col_name in embedding_input.split('SEP')
                      if not col_name.isdigit()
                      ]

        deep_out = torch.cat(embeddings, dim=-1)  # concatenate the embeddings of all deep features

        for i, _ in enumerate(self.hidden_layers):
            deep_out = F.relu(getattr(self, f'linear_layer_{i + 1}')(deep_out))

        X_w = self.dropout(X_w)  # Apply a dropout layer to the wide features for regularization purposes
        fc_input = torch.cat([X_w, deep_out], dim=-1)  # concatenate the wide and processed deep features
        out = self.fc(fc_input)

        return out


# The dataset
# Split the dataset into <num_workers> parts, so that each worker get a unique copy of a part of the dataset
# https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset

class CtrDataset(Dataset):
    def __init__(self, urls):
        self.urls = urls

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, item):
        df = pd.read_csv(self.urls[item], header=None)
        df.columns = COLUMNS
        X_w = df.iloc[:, :WIDE_DIM].values.astype(np.float32).squeeze()
        X_d = df[EMBEDDING_INPUTS].values.astype(np.float32).squeeze()
        label = df['label'].values.astype(np.float32).squeeze()

        return X_w, X_d, label


# Training function
def training(args, model, device, train_dl, scaler, optimizer, loss_fn):
    model.train()
    total_loss = 0
    n = 0
    for batch_i, (X_w, X_d, label) in enumerate(train_dl):
        X_w = X_w.squeeze().to(device, non_blocking=True)
        X_d = X_d.squeeze().to(device, non_blocking=True)
        label = label.squeeze().unsqueeze(1).to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(X_w, X_d)
            loss = loss_fn(outputs, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * label.shape[0]
        n += label.shape[0]
        running_loss = total_loss / n

        if (batch_i + 1) % args.batch_interval == 0:
            msg = f"Training batch {batch_i+1}/{len(train_dl)}\trunning loss: {running_loss:.5f}"
            logging.info(msg)

    return running_loss


# Validation function
def validation(args, model, device, val_dl, loss_fn):
    model.eval()
    total_loss = 0
    n = 0
    for batch_i, (X_w, X_d, label) in enumerate(val_dl):
        X_w = X_w.squeeze().to(device, non_blocking=True)
        X_d = X_d.squeeze().to(device, non_blocking=True)
        label = label.squeeze().unsqueeze(1).to(device, non_blocking=True)

        with torch.no_grad() and torch.cuda.amp.autocast():
            outputs = model(X_w, X_d)
            loss = loss_fn(outputs, label)

        total_loss += loss.item() * label.shape[0]
        n += label.shape[0]
        running_loss = total_loss / n

        if (batch_i + 1) % args.batch_interval == 0:
            msg = f"Validating batch {batch_i+1}/{len(val_dl)}\trunning loss: {running_loss:.5f}"
            logging.info(msg)

    return running_loss


def main():
    parser = argparse.ArgumentParser(description="CTR prediction with wide and deep neural network")
    parser.add_argument("--epochs", type=int, default=5, metavar="N",
                        help="number of epochs to train (default: 5)")
    # parser.add_argument("--batch-size", type=int, default=2048, metavar="N",
    #                     help="batch size (default: 2048)")
    parser.add_argument("--num-workers", type=int, default=2, metavar="N",
                        help="number of workers (default: 2)")
    parser.add_argument("--batch-interval", type=int, default=500, metavar="N",
                        help="number of batches to wait between each logging")
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR",
                        help="learning rate (default: 0.001)")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=42, metavar="S",
                        help="random seed (default: 42)")
    parser.add_argument("--save-model", action="store_true", default=False,
                        help="For Saving the current Model")
    parser.add_argument("--log-path", type=str, default="",
                        help="Path to save logs. Print to StdOut if log-path is not set")
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")

    # Use this format (%Y-%m-%dT%H:%M:%SZ) to record timestamp of the metrics.
    # If log_path is empty print log to StdOut, otherwise print log to the file.
    if args.log_path == "":
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            level=logging.INFO)
    else:
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            level=logging.INFO,
            filename=args.log_path)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_urls, val_urls = list_blobs('kfp-yli')
    train_dset = CtrDataset(train_urls)
    val_dset = CtrDataset(val_urls)
    train_dl = DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=args.num_workers)
    val_dl = DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = WideAndDeep(wide_dim=WIDE_DIM, embedding_inputs=EMBEDDING_INPUTS, hidden_layers=[512, 256, 128],
                        dropout_p=0.7,
                        )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 #                              weight_decay=0.01,
                                 )

    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

    lr_scheduler = StepLR(optimizer,
                          step_size=5,
                          gamma=0.1,
                          last_epoch=-1,
                          verbose=True,
                          )

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(1, args.epochs + 1):
        print(f'\n=============== Epoch {epoch} ===============\n')
        train_losses.append(training(args, model, device, train_dl, scaler, optimizer, loss_fn))
        val_losses.append(validation(args, model, device, val_dl, loss_fn))

        lr_scheduler.step()
        print('\n')
        if val_losses[-1] < best_val_loss and args.save_model:
            best_val_loss = val_losses[-1]
            print('Best model saved.\n')
            torch.save(model.state_dict(), './saved_model.pt')


if __name__ == "__main__":
    main()