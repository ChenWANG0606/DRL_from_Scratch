import torch

env_name = 'CartPole-v1'
gamma = 0.99
lr = 0.001
goal_score = 200
log_interval = 10
device = torch.device("mps" if torch.mps.is_available() else "cpu")