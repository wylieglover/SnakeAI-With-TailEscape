import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, game_over):
        state_array = np.array(state)
        next_state_array = np.array(next_state)
        action_array = np.array(action)
        reward_array = np.array(reward)
        
        state = torch.tensor(state_array, dtype=torch.float)
        next_state = torch.tensor(next_state_array, dtype=torch.float)
        action = torch.tensor(action_array, dtype=torch.long)
        reward = torch.tensor(reward_array, dtype=torch.float)
        
        # (n, x) n = 4
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0) 
            next_state = torch.unsqueeze(next_state, 0) 
            action = torch.unsqueeze(action, 0) 
            reward = torch.unsqueeze(reward, 0) 
            game_over = (game_over, )
            
        # 1: predicted Q values with current state
        pred = self.model(state)
        
        target = pred.clone()
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state))
            
            target[idx][torch.argmax(action).item()] = Q_new
        # 2: Q_new = r + y(gamma) * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_neww
        
        self.optimizer.zero_grad() # do this in PyTorch
        loss = self.criterion(target, pred)
        loss.backward()
        
        self.optimizer.step()
