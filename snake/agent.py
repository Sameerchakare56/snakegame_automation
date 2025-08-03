import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGame, Direction
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython import display
import os
import pickle
import pandas as pd

# Live plot setup

scores = []
mean_scores = []

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.01

class LinearQNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearQNet, self).__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 16)
        self.linear5 = nn.Linear(16, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        return self.linear5(x)

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(11, 3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

    def get_state(self, game):
        head = game.snake[0]
        point_l = head._replace(x=head.x - 20)
        point_r = head._replace(x=head.x + 20)
        point_u = head._replace(y=head.y - 20)
        point_d = head._replace(y=head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game._is_collision(point_r)) or
            (dir_l and game._is_collision(point_l)) or
            (dir_u and game._is_collision(point_u)) or
            (dir_d and game._is_collision(point_d)),

            (dir_u and game._is_collision(point_r)) or
            (dir_d and game._is_collision(point_l)) or
            (dir_l and game._is_collision(point_u)) or
            (dir_r and game._is_collision(point_d)),

            (dir_d and game._is_collision(point_r)) or
            (dir_u and game._is_collision(point_l)) or
            (dir_r and game._is_collision(point_u)) or
            (dir_l and game._is_collision(point_d)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self._train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        for state, action, reward, next_state, done in mini_sample:
            self._train_step(state, action, reward, next_state, done)

    def _train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        pred = self.model(state)
        target = pred.clone().detach()

        Q_new = reward
        if not done:
            Q_new = reward + self.gamma * torch.max(self.model(next_state)).item()

        target[action.argmax().item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def save_model(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        torch.save(self.model.state_dict(), os.path.join(model_folder_path, file_name))

    def load_model(self, file_name='model.pth'):
        file_path = os.path.join('./model', file_name)
        if os.path.exists(file_path):
            self.model.load_state_dict(torch.load(file_path))
            self.model.eval()
        else:
            print(f"Model file {file_name} not found.")

    def save_training_data(self, file_name='training_data.pkl'):
        data = {
            'memory': list(self.memory),
            'n_games': self.n_games,
            'scores': scores,
            'mean_scores': mean_scores
        }
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)
        print(f"Training data saved to {file_name}")

    def load_training_data(self, file_name='training_data.pkl'):
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                data = pickle.load(f)
                self.memory = deque(data['memory'], maxlen=MAX_MEMORY)
                self.n_games = data['n_games']
                global scores, mean_scores
                scores = data['scores']
                mean_scores = data['mean_scores']
            print(f"Training data loaded from {file_name}")
        else:
            print(f"Training data file {file_name} not found.")

    def export_scores_to_csv(self, filename='scores.csv'):
        df = pd.DataFrame({'scores': scores, 'mean_scores': mean_scores})
        df.to_csv(filename, index=False)
        print(f"Scores exported to {filename}")
