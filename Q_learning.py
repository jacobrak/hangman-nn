from hangman import *
import random
import pickle


def cache(total_words=500_000):
    url  = f"https://random-word-api.herokuapp.com/word?number={total_words}&length=5"

    response = requests.get(url)
    if response.status_code == 200:
        word_list = response.json()
    else:
        print(f"Warning: Word API failed with status {response.status_code}.")
        return
    return (word_list)

Word_set = cache()

def pick_random():
    word = random.choice(Word_set)
    return word

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
        return (self.guessed_word.copy(), self.guessed_letters.copy())

class Agent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.q_table = {}

    def normalize_state(self, state):
        guessed_word, guessed_letters = state
        return (tuple(guessed_word), tuple(sorted(guessed_letters)))

    def update_q_table(self, state, action, reward, next_state, guessed_letters):
        state = self.normalize_state(state)
        next_state_norm = self.normalize_state(next_state) if next_state else None

        old_q = self.q_table.get((state, action), 0.0)
        
        # Calculate max future Q for non-terminal states
        if next_state_norm:
            available_actions = [a for a in self.alphabet if a not in guessed_letters]
            future_qs = [self.q_table.get((next_state_norm, a), 0.0) for a in available_actions]
            max_future_q = max(future_qs) if future_qs else 0.0
        else:
            max_future_q = 0.0  # Terminal state

        new_q = old_q + self.learning_rate * (reward + self.discount_factor * max_future_q - old_q)
        self.q_table[(state, action)] = new_q

    def choose_action(self, state, guessed_letters):
        available_actions = [a for a in self.alphabet if a not in guessed_letters]
        if random.random() < self.exploration_rate or not available_actions:
            return random.choice(available_actions) if available_actions else None

        # Exploit best known action
        state_norm = self.normalize_state(state)
        q_values = [(a, self.q_table.get((state_norm, a), 0.0)) for a in available_actions]
        return max(q_values, key=lambda x: x[1])[0]

def train_agent(agent, episodes=10000, save_path="q_table.pkl"):
    for episode in range(episodes):
        hangman = Hangman()
        while True:
            state = hangman.get_state()
            action = agent.choose_action(state, hangman.guessed_letters)
            if action is None:
                break

            feedback = hangman.guess(action)
            game_status = hangman.is_game_over()

            # Determine reward
            if game_status == "win":
                reward = 500
                next_state = None  # Terminal state
            elif game_status == "lose":
                correct = sum(c != '_' for c in hangman.guessed_word)
                reward = (correct ** 2) * 50 if correct > 0 else -100
                next_state = None
            else:
                if "Correct" in feedback:
                    reward = 10
                else:
                    reward = -1
                next_state = hangman.get_state()

            # Update Q-table
            agent.update_q_table(state, action, reward, next_state, hangman.guessed_letters)

            if game_status:
                break

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Q-table size: {len(agent.q_table)}")

    with open(save_path, "wb") as f:
        pickle.dump(agent.q_table, f)
    print(f"Training complete. Q-table saved to {save_path}")

if __name__ == "__main__":
    train_mode = False
    if train_mode == True:
        agent = Agent()

        with open("q_table.pkl", "rb") as f:
            agent.q_table = pickle.load(f)
    
    train_agent(Agent(), episodes=15000)