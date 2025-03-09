import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

pd.options.display.precision = 17

class FBFETLSTM(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, output_dim):
        super(FBFETLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.regressor = self.make_regressor()
        
        # Initialize weights
        self.initialize_weights()

    def forward(self, x):
        # x shape: (batch_size, time_steps, input_dim)
        if len(x.shape) == 2:
            # Add batch dimension if it's missing
            x = x.unsqueeze(0)
            
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, time_steps, hidden_dim)
        
        # Apply regressor to all time steps at once
        batch_size, time_steps, _ = lstm_out.shape
        lstm_out_reshaped = lstm_out.reshape(-1, self.hidden_dim)  # (batch_size * time_steps, hidden_dim)
        
        # Pass through regressor
        outputs = self.regressor(lstm_out_reshaped)  # (batch_size * time_steps, output_dim)
        outputs = torch.relu(outputs)  # ReLU를 사용하여 음수 제거
        #outputs = torch.abs(outputs)
        #outputs = torch.clamp(outputs, min=1e-11) # <- veilog-A 무산 기념 원래는 못쓰던 기능 추가
        # Reshape back to sequence form
        outputs = outputs.reshape(batch_size, time_steps, self.output_dim)

        return outputs
    
    def make_regressor(self):
        layers = []

        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim // 2))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(self.hidden_dim // 2, self.output_dim))
        regressor = nn.Sequential(*layers)
        return regressor
    
    def initialize_weights(self):
        """Initialize weights for LSTM and Linear layers"""
        # LSTM 가중치 초기화
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # 입력 게이트 가중치 초기화
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # 은닉 상태 가중치 초기화
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # 편향 초기화
                param.data.fill_(0)
                # 망각 게이트 편향을 1로 초기화
                if 'bias_ih' in name:
                    n = param.size(0)
                    start, end = n//4, n//2
                    param.data[start:end].fill_(1.)
        
        # Linear 레이어 가중치 초기화
        for layer in self.regressor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

class FBFETGRU(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, output_dim):
        super(FBFETGRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.regressor = self.make_regressor()
        self.reset_parameters()


    def forward(self, x):
        # x shape: (batch_size, time_steps, input_dim)
        if len(x.shape) == 2:
            # Add batch dimension if it's missing
            x = x.unsqueeze(0)
            
        gru_out, _ = self.gru(x)
        # lstm_out shape: (batch_size, time_steps, hidden_dim)
        
        # Apply regressor to all time steps at once
        batch_size, time_steps, _ = gru_out.shape
        gru_out_reshaped = gru_out.reshape(-1, self.hidden_dim)  # (batch_size * time_steps, hidden_dim)
        
        # Pass through regressor
        outputs = self.regressor(gru_out_reshaped)  # (batch_size * time_steps, output_dim)
    
        #gru_out_reshaped = torch.relu(gru_out_reshaped)  # ReLU를 사용하여 음수 제거 <- 절대값을 이용하여 음수를 제거하는 방안으로 변경
        #outputs = self.fc(gru_out_reshaped)
        outputs = torch.relu(outputs)
        # Reshape back to sequence form
        outputs = outputs.reshape(batch_size, time_steps, self.output_dim)

        return outputs

    def make_regressor(self):
        layers = []

        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        regressor = nn.Sequential(*layers)
        return regressor

    def reset_parameters(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

        for layer in self.regressor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)