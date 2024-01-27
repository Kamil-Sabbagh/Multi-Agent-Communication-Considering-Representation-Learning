import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class ControlTransformerDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as file:
            next(file)  # Skip the first line
            for line in file:
                entry = json.loads(line)
                self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_observation = self.data[idx]['current_observation']
        action = self.data[idx]['action']
        next_observation = self.data[idx]['next_observation']
        return torch.tensor(current_observation, dtype=torch.float), torch.tensor(action, dtype=torch.float), torch.tensor(next_observation, dtype=torch.float)


file_path = 'log_file.json'
dataset = ControlTransformerDataset(file_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class ControlTransformer(nn.Module):
    def __init__(self, obs_size, action_size, d_model, nhead, num_layers, dim_feedforward):
        super(ControlTransformer, self).__init__()
        self.obs_embedding = nn.Linear(obs_size, d_model)
        self.action_embedding = nn.Linear(action_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, obs_size)  # New output layer

    def forward(self, observations, actions):
        obs_emb = self.obs_embedding(observations)
        action_emb = self.action_embedding(actions.unsqueeze(-1))
        x = obs_emb + action_emb
        x = self.positional_encoding(x)
        transformer_output = self.transformer_encoder(x)
        output = self.output_layer(transformer_output)  # Apply the output layer
        return output




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Reshape pe to match the input tensor x
        x = x + self.pe[:x.size(1), :].unsqueeze(0).repeat(x.size(0), 1, 1)
        return x




# Ensure that obs_size matches the actual size of your observation vectors
obs_size = 61  # Update this based on the actual size of your observation vectors
action_size = 1
d_model = 256
nhead = 4
num_layers = 3
dim_feedforward = 512

model = ControlTransformer(obs_size, action_size, d_model, nhead, num_layers, dim_feedforward)




# Define your loss function and optimizer
criterion = torch.nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5000  
for epoch in range(num_epochs):
    for current_obs, action, next_obs in dataloader:
        # Forward pass
        predictions = model(current_obs, action)
        
        # Compute loss (assuming your task is to predict the next observation)
        loss = criterion(predictions, next_obs)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


#### Spam ###
    

# Training loop
num_epochs = 5000  
for epoch in range(num_epochs):
    for current_obs, action, next_obs in dataloader:
        # Forward pass
        forward_dynamics_pred, inverse_dynamics_pred, _ = model(current_obs, action, next_obs)
        
        # Compute loss (assuming your task is to predict the next observation)
        loss = criterion(forward_dynamics_pred, next_obs)
        loss = criterion(inverse_dynamics_pred, inverse_dynamics_pred)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')