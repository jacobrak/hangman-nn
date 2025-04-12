# evaluation.py
import time
import random
from Q_learning import *
import pickle

def evaluate_agent(agent, games=10, delay=0.5, verbose=True):
    wins = 0
    total_lives_left = 0

    # Disable exploration for evaluation
    original_exploration = agent.exploration_rate
    agent.exploration_rate = 0

    # Verify Q-table is loaded
    if not agent.q_table:
        print("⚠️ WARNING: Q-table is empty!")
        return

    for game in range(games):
        hangman = Hangman(letters=5)
        guesses_this_game = []

        print(f"\n🔠 Game {game + 1}")
        print("Word:", " ".join(hangman.guessed_word))

        while True:
            state = hangman.get_state()
            normalized_state = agent.normalize_state(state)
            
            # Use agent's decision logic
            action = agent.choose_action(normalized_state, hangman.guessed_letters)
            
            if action is None:
                break  # No valid actions left

            guesses_this_game.append(action)
            feedback = hangman.guess(action)
            
            print(f"Guess: {action} ➤ {feedback}")
            print("Current word:", " ".join(hangman.guessed_word))
            time.sleep(delay)

            game_status = hangman.is_game_over()
            if game_status:
                print(game_status)
                print(hangman.word)
                if "Congratulations" in game_status:
                    wins += 1
                    total_lives_left += hangman.lives
                break

        if verbose:
            print("Guessed letters:", guesses_this_game)
            print("-" * 40)

    # Restore original exploration rate
    agent.exploration_rate = original_exploration

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


#print(f"Loaded Q-table: {agent.q_table}")


def print_q_table(agent):
    print("Q-table contents:")

    sorted_q_table = sorted(agent.q_table.items(), key=lambda x: x[1], reverse=False)

    for state_action, q_value in sorted_q_table:
        print(f"State-Action: {state_action} Q-value: {q_value}")


#print_q_table(agent)

