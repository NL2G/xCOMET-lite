import torch
import torch.nn as nn

class MQMHead(nn.Module):
    N_SEVERITY_CLASSES = 4
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.hidden2tag = nn.Linear(hidden_dim, self.N_SEVERITY_CLASSES)
    
    def forward(self, hidden_states):
        mqm_score_prediction = self.estimator(hidden_states[:, 0])
        logits = self.hidden2tag(hidden_states[:, 1:])
    
        return mqm_score_prediction, logits

class MQMModel(nn.Module):
    def __init__(self, ...)
