from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Model_Chadal_Marion_Savini_Thomas import Model
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
import time
import os
import pandas as pd
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CE = torch.nn.CrossEntropyLoss()
def contrastive_loss(v1, v2):
  logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
  labels = torch.arange(logits.shape[0], device=v1.device)
  return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nb_epochs = 50
batch_size = 32
learning_rate = 4e-5

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = Model(model_name=model_name, num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
model.to(device)


epoch = 0
loss = 0
losses = []
val_losses = []
count_iter = 0
time1 = time.time()
printEvery = 50
best_validation_loss = 1000000

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.01)

scheduler = ReduceLROnPlateau(optimizer, 'min')


for epoch in range(nb_epochs):
    print(f'-----EPOCH {epoch + 1}-----')
    model.train()
    epoch_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        graph_batch = batch.to(device)
        
        x_graph, x_text = model(graph_batch, input_ids.to(device), attention_mask.to(device))
        current_loss = contrastive_loss(x_graph, x_text)
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
        epoch_loss += current_loss.item()
        
        if (batch_idx + 1) % printEvery == 0:
            print(f"Iteration: {batch_idx + 1}, Training loss: {epoch_loss / (batch_idx + 1):.4f}")
    
    avg_epoch_loss = epoch_loss / len(train_loader)
    losses.append(avg_epoch_loss)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch.input_ids
            attention_mask = batch.attention_mask
            graph_batch = batch.to(device)
            
            x_graph, x_text = model(graph_batch, input_ids.to(device), attention_mask.to(device))
            current_loss = contrastive_loss(x_graph, x_text)
            val_loss += current_loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f'Validation loss: {avg_val_loss:.4f}')
    scheduler.step(avg_val_loss)
    
    if avg_val_loss < best_validation_loss:
        best_validation_loss = avg_val_loss
        print('Validation loss improved, saving checkpoint...')
        save_path = os.path.join('./PositionalEncodingv4/', f'model_epoch_{epoch + 1}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validation_loss': avg_val_loss,
        }, save_path)
        print(f'Checkpoint saved to: {save_path}')


print('loading best model...')
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

graph_model = model.get_graph_encoder()
text_model = model.get_text_encoder()

test_cids_dataset = GraphDataset(root='./data/', gt=gt, split='test_cids')
test_text_dataset = TextDataset(file_path='./data/test_text.txt', tokenizer=tokenizer)

idx_to_cid = test_cids_dataset.get_idx_to_cid()

test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)

graph_embeddings = []
for batch in test_loader:
    for output in graph_model(batch.to(device)):
        graph_embeddings.append(output.tolist())

test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)
text_embeddings = []
for batch in test_text_loader:
    for output in text_model(batch['input_ids'].to(device), 
                             attention_mask=batch['attention_mask'].to(device)):
        text_embeddings.append(output.tolist())


from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(text_embeddings, graph_embeddings)

solution = pd.DataFrame(similarity)
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
solution.to_csv('./PositionalEncodingv4/submission.csv', index=False)

plt.figure(figsize=(10, 6))
plt.plot(range(1, nb_epochs + 1), losses, label='Training Loss', color='blue')
plt.plot(range(1, nb_epochs + 1), val_losses, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid(True)
plot_save_path = './PositionalEncodingv4/loss_plot.png'
plt.savefig(plot_save_path)
print(f'Loss plot saved to {plot_save_path}')
