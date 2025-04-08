from hangman import *
import random

class Hangman:
    def __init__(self):
        self.word = get_random_word() 
        self.guessed_word = ["_"] * len(self.word)
        self.lives = 6
        self.guessed_letters = []
    def __str__(self):
        return " ".join(self.guessed_word)

    def guess(self, letter):
        if len(letter) != 1 or not letter.isalpha():
            return "Please enter a valid single letter."

        letter = letter.lower()

        if letter in self.guessed_letters:
            return "You already guessed this letter"
        
        self.guessed_letters.append(letter)

        if letter in self.word:
            for i in range(len(self.word)):
                if self.word[i] == letter:
                    self.guessed_word[i] = letter
            return "Correct!"
        else:
            self.lives -= 1
            return f"Incorrect! Attempts left: {self.lives}"
        
    def is_game_over(self):
        # Check if the game is over (either word is guessed or attempts run out)
        if "_" not in self.guessed_word:
            return "Congratulations! You've guessed the word: " + self.word
        elif self.lives <= 0:
            return f"Game over! The word was: {self.word}"
        else:
            return None
        
    def get_state(self):
        return (self.guessed_word, self.guessed_letters)

class Agent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.word_length = 5
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.q_table = {}

    def get_state(self):
        return Hangman.get_state()
     
    def normalize_state(self, state):
        guessed_word, guessed_letters = state
        return (tuple(guessed_word), tuple(sorted(guessed_letters)))

    def update_q_table(self, state, action, reward, next_state, guessed_letters):
        # Normalize states
        state = self.normalize_state(state)
        next_state = self.normalize_state(next_state)

        # Get old Q-value
        old_q = self.q_table.get((state, action), 0.0)

        # Get max future Q-value from next state
        available_actions = [a for a in self.alphabet if a not in guessed_letters]
        future_qs = [self.q_table.get((next_state, a), 0.0) for a in available_actions]
        max_future_q = max(future_qs) if future_qs else 0.0

        # Q-learning update
        new_q = old_q + self.learning_rate * (reward + self.discount_factor * max_future_q - old_q)

        # Store updated value
        self.q_table[(state, action)] = new_q

    def choose_action(self, state, guessed_letters):
        available_actions = [a for a in self.alphabet if a not in guessed_letters]

        if random.random() < self.exploration_rate:
                return random.choice(available_actions)

        max_q = float('-inf')
        best_action = None
        for action in available_actions:
            q = self.q_table.get((state, action), 0.0)
            if q > max_q:
                max_q = q
                best_action = action

        return best_action if best_action else random.choice(available_actions)
    

import pickle

def train_agent(agent, episodes=1000, save_path="q_table.pkl", verbose=True):
    for episode in range(episodes):
        hangman = Hangman()

        while True:
            # Get and normalize current state
            state = hangman.get_state()
            normalized_state = agent.normalize_state(state)

            # Choose action
            action = agent.choose_action(normalized_state, hangman.guessed_letters)

            # Perform action in game
            feedback = hangman.guess(action)

            # Determine reward
            if "Correct" in feedback:
                reward = 1
            elif "Incorrect" in feedback:
                reward = -1
            else:
                reward = 0

            # Get next state
            next_state = hangman.get_state()

            # Update Q-table
            agent.update_q_table(state, action, reward, next_state, hangman.guessed_letters)

            # Check if game ended
            game_status = hangman.is_game_over()
            if game_status:
                # Bonus/final reward
                final_reward = 10 if "Congratulations" in game_status else -10
                agent.update_q_table(state, action, final_reward, next_state, hangman.guessed_letters)
                break

        # Log progress
        if verbose and (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1} complete")

    # Save Q-table
    with open(save_path, "wb") as f:
        pickle.dump(agent.q_table, f)

    if verbose:
        print(f"Training complete. Q-table saved to '{save_path}'")

if __name__ == "__main__":
    train_agent(Agent())