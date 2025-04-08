# evaluation.py
import time
import random
from Q_learning import *
import pickle

def evaluate_agent(agent, games=10, delay=0.5, verbose=True):
    wins = 0
    total_lives_left = 0

    for game in range(games):
        hangman = Hangman(letters=3)
        guesses_this_game = []

        print(f"\n🔠 Game {game + 1}")
        print("Word:", " ".join(hangman.guessed_word))

        while True:
            state = hangman.get_state()
            normalized_state = agent.normalize_state(state)

            # Choose best known action
            available_actions = [a for a in agent.alphabet if a not in hangman.guessed_letters]
            best_action = None
            best_q = float('-inf')

            for action in available_actions:
                q = agent.q_table.get((normalized_state, action), 0.0)
                if q > best_q:
                    best_q = q
                    best_action = action

            action = best_action if best_action else random.choice(available_actions)
            guesses_this_game.append(action)

            feedback = hangman.guess(action)
            print(f"Guess: {action} ➤ {feedback}")
            print("Current word:", " ".join(hangman.guessed_word))
            time.sleep(delay)

            if hangman.is_game_over():
                result = hangman.is_game_over()
                print(result)

                if "Congratulations" in result:
                    wins += 1
                    total_lives_left += hangman.lives
                break

        if verbose:
            print("Guessed letters:", guesses_this_game)
            print("-" * 40)

    # Summary
    win_rate = wins / games
    avg_lives = total_lives_left / wins if wins > 0 else 0

    print("\n🎯 Evaluation Results:")
    print(f"Games played: {games}")
    print(f"Games won: {wins}")
    print(f"Win rate: {win_rate * 100:.2f}%")
    print(f"Average lives left (wins only): {avg_lives:.2f}")

agent = Agent()

with open("q_table.pkl", "rb") as f:
    agent.q_table = pickle.load(f)

# Run evaluation
evaluate_agent(agent, games=10, delay=0.8)

def print_q_table(agent):
    print("Q-table contents:")
    for state_action, q_value in agent.q_table.items():
        print(f"State-Action: {state_action} Q-value: {q_value}")


#print_q_table(agent)