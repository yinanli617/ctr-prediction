import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
from torch.optim.lr_scheduler import StepLR
import argparse

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

class CtrDataset(IterableDataset):
    def __init__(self, chunksize=10000, train=True):
        super().__init__()
        self.train = train
        if self.train:
            self.num_lines = 28303473  # wc -l ./train_full.csv
            self.path = './train_full.csv'
        else:
            self.num_lines = 12125494  # wc -l ./validation_full.csv
            self.path = './validation_full.csv'
        self.chunksize = chunksize
        self.start = 0
        self.end = self.num_lines + self.start - 1

    def process_data(self, data):
        for i, chunk in enumerate(data):
            if self.start + i * chunk.shape[0] >= self.end:
                break
            else:
                chunk.columns = COLUMNS

                # Don't repeat at the end of each partition
                size = min(self.chunksize, self.end - (self.start + i * chunk.shape[0]))

                X_w = chunk.iloc[:size, :WIDE_DIM].values.astype(np.float32).squeeze()
                X_d = chunk.iloc[:size][EMBEDDING_INPUTS].values.astype(np.float32).squeeze()
                label = chunk.iloc[:size]['label'].values.astype(np.float32).squeeze()
                yield X_w, X_d, label

    def __iter__(self):
        self.df = pd.read_csv(self.path,
                              header=None,
                              chunksize=self.chunksize,
                              skiprows=self.start,
                              )
        return self.process_data(self.df)


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)


# A helper function to get the total number of batches
def get_total(dset, dl):
    temp = int(math.ceil((dset.end - dset.start) / float(dl.num_workers)))
    total = int(math.ceil(temp / dset.chunksize)) * dl.num_workers

    return total


# Training function
def training(model, device, train_dl, scaler, optimizer, loss_fn, train_total):
    pbar = tqdm(train_dl, total=train_total)
    model.train()
    total_loss = 0
    n = 0
    for batch_i, (X_w, X_d, label) in enumerate(pbar):
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
        pbar.set_description(f'Training Loss: {running_loss}')

    return running_loss


# Validation function
def validation(model, device, val_dl, loss_fn, val_total):
    global running_loss
    pbar = tqdm(val_dl, total=val_total)
    model.eval()
    total_loss = 0
    n = 0
    for batch_i, (X_w, X_d, label) in enumerate(pbar):
        X_w = X_w.squeeze().to(device, non_blocking=True)
        X_d = X_d.squeeze().to(device, non_blocking=True)
        label = label.squeeze().unsqueeze(1).to(device, non_blocking=True)

        with torch.no_grad() and torch.cuda.amp.autocast():
            outputs = model(X_w, X_d)
            loss = loss_fn(outputs, label)

        total_loss += loss.item() * label.shape[0]
        n += label.shape[0]

        running_loss = total_loss / n
        pbar.set_description(f'Validation Loss: {running_loss}')

    return running_loss


def main():
    parser = argparse.ArgumentParser(description="CTR prediction with wide and deep neural network")
    parser.add_argument("--epochs", type=int, default=5, metavar="N",
                        help="number of epochs to train (default: 5)")
    parser.add_argument("--batch-size", type=int, default=2048, metavar="N",
                        help="batch size (default: 2048)")
    parser.add_argument("--num-workers", type=int, default=2, metavar="N",
                        help="number of workers (default: 2)")
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR",
                        help="learning rate (default: 0.001)")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=42, metavar="S",
                        help="random seed (default: 42)")
    parser.add_argument("--save-model", action="store_true", default=False,
                        help="For Saving the current Model")
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    model = WideAndDeep(wide_dim=WIDE_DIM, embedding_inputs=EMBEDDING_INPUTS, hidden_layers=[512, 256, 128],
                        dropout_p=0.7,
                        )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-3,
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
        train_dset = CtrDataset(train=True, chunksize=args.batch_size)
        train_dl = DataLoader(train_dset,
                              batch_size=1,
                              num_workers=args.num_workers,
                              worker_init_fn=worker_init_fn,
                              )
        val_dset = CtrDataset(train=False, chunksize=4 * args.batch_size)
        val_dl = DataLoader(val_dset,
                            batch_size=1,
                            num_workers=args.num_workers,
                            worker_init_fn=worker_init_fn,
                            )
        train_total = get_total(train_dset, train_dl)
        val_total = get_total(val_dset, val_dl)

        print(f'=============== Epoch {epoch} ===============')
        train_losses.append(training(model, device, train_dl, scaler, optimizer, loss_fn, train_total))
        val_losses.append(validation(model, device, val_dl, loss_fn, val_total))

        lr_scheduler.step()
        print('\n')
        if val_losses[-1] < best_val_loss and args.save_model:
            best_val_loss = val_losses[-1]
            print('Best model saved.\n')
            torch.save(model.state_dict(), './saved_model.pt')


if __name__ == "__main__":
    main()