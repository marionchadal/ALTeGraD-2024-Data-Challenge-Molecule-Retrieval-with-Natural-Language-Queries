import torch
from torch import nn
from torch_geometric.nn import GINConv, global_mean_pool

from transformers import AutoModel

class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        super(GraphEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(nout)
        self.dropout = nn.Dropout(p=0.1)
        
        self.conv1 = GINConv(
            nn.Sequential(nn.Linear(num_node_features, graph_hidden_channels),
                          nn.BatchNorm1d(graph_hidden_channels), self.relu,
                          nn.Linear(graph_hidden_channels, graph_hidden_channels))
        )
        self.conv2 = GINConv(
            nn.Sequential(nn.Linear(graph_hidden_channels, graph_hidden_channels),
                          nn.BatchNorm1d(graph_hidden_channels), self.relu,
                          nn.Linear(graph_hidden_channels, graph_hidden_channels), nn.Tanh())
        )
        self.conv3 = GINConv(
            nn.Sequential(nn.Linear(graph_hidden_channels, graph_hidden_channels),
                          nn.BatchNorm1d(graph_hidden_channels), self.relu,
                          nn.Linear(graph_hidden_channels, graph_hidden_channels), nn.Tanh())
        )
        
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)


    def forward(self, graph_batch):
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.dropout(x)
        x = self.mol_hidden2(x)
        
        return x

class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        return encoded_text.last_hidden_state[:, 0, :]

class Model(nn.Module):
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels):
        super(Model, self).__init__()
        self.graph_encoder = GraphEncoder(num_node_features, nout, nhid, graph_hidden_channels)
        self.text_encoder = TextEncoder(model_name)
        
    def get_graph_encoder(self):
        return self.graph_encoder
    
    def get_text_encoder(self):
        return self.text_encoder
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
