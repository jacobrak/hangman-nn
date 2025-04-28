import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import requests

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Word Setup ===
def cache(total_words=500_000):
    url = f"https://random-word-api.herokuapp.com/word?number={total_words}&length=5"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    print(f"Warning: Word API failed with status {response.status_code}.")
    return []

Word_set = cache()

def pick_random():
    return random.choice(Word_set)

# === Hangman Game ===
class Hangman:
    def __init__(self):
        self.word = pick_random()
        self.guessed_word = ["_"] * len(self.word)
        self.lives = 5
        self.guessed_letters = []

    def __str__(self):
        return " ".join(self.guessed_word)

    def guess(self, letter):
        if len(letter) != 1 or not letter.isalpha():
            return "Invalid input"

        letter = letter.lower()
        if letter in self.guessed_letters:
            return "Already guessed"

        self.guessed_letters.append(letter)
        if letter in self.word:
            for i, char in enumerate(self.word):
                if char == letter:
                    self.guessed_word[i] = letter
            return "Correct"
        else:
            self.lives -= 1
            return "Incorrect"

    def is_game_over(self):
        if "_" not in self.guessed_word:
            return "win"
        elif self.lives <= 0:
            return "lose"
        return None

    def get_state(self):
        return self.guessed_word.copy(), self.guessed_letters.copy()

# === DQN Agent ===
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, lr=0.001, gamma=0.9, epsilon=0.2):
        self.gamma = gamma
        self.epsilon = epsilon
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.letter_to_index = {c: i for i, c in enumerate(self.alphabet)}
        self.model = DQN(input_size=157, output_size=26).to(device=device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def state_to_tensor(self, guessed_word, guessed_letters, lives):
        word_vec = []
        for ch in guessed_word:
            letter_vec = [0] * 26
            if ch != "_":
                letter_vec[self.letter_to_index[ch]] = 1
            word_vec.extend(letter_vec)

        guessed_vec = [1 if c in guessed_letters else 0 for c in self.alphabet]
        lives_vec = [lives / 5.0]

        state = word_vec + guessed_vec + lives_vec
        return torch.tensor(state, dtype=torch.float32, device=device)


    def choose_action(self, state_tensor, guessed_letters):
        if random.random() < self.epsilon:
            available = [c for c in self.alphabet if c not in guessed_letters]
            return random.choice(available) if available else None

        with torch.no_grad():
            q_values = self.model(state_tensor)
        mask = torch.tensor([0 if c in guessed_letters else 1 for c in self.alphabet], dtype=torch.bool, device=device)
        masked_q = q_values.masked_fill(~mask, float('-inf'))
        best_action_index = torch.argmax(masked_q).item()
        return self.alphabet[best_action_index]

    def train_step(self, state, action, reward, next_state, done):
        state_tensor = state
        target = self.model(state_tensor).clone().detach()
        action_index = self.letter_to_index[action]

        if done:
            target[action_index] = reward
        else:
            next_q = self.model(next_state)
            max_next_q = torch.max(next_q).item()
            target[action_index] = reward + self.gamma * max_next_q

        prediction = self.model(state_tensor)
        loss = self.loss_fn(prediction, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# === Training Loop ===
def train_dqn_agent(agent, episodes=5000):
    for episode in range(episodes):
        game = Hangman()
        while True:
            gw, gl = game.get_state()
            state_tensor = agent.state_to_tensor(gw, gl, game.lives)
            action = agent.choose_action(state_tensor, gl)
            if not action:
                break

            feedback = game.guess(action)
            status = game.is_game_over()

            # Reward logic
            if status == "win":
                reward = 500
                done = True
                next_state_tensor = None
            elif status == "lose":
                correct = sum(c != '_' for c in game.guessed_word)
                reward = (correct ** 2) * 50 if correct > 0 else -100
                done = True
                next_state_tensor = None
            else:
                reward = 10 if feedback == "Correct" else -1
                next_gw, next_gl = game.get_state()
                next_state_tensor = agent.state_to_tensor(next_gw, next_gl, game.lives)
                done = False

            agent.train_step(state_tensor, action, reward, next_state_tensor, done)

            if done:
                break

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1} completed.")

    print("Training complete.")

if __name__ == "__main__":
    agent = DQNAgent()
    train_dqn_agent(agent, episodes=10000)
    torch.save(agent.model.state_dict(), "dqn_hangman.pt")