from hangman import *

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

    def state(self):
        return Hangman.get_state() 

    def update_q_table(self):
        pass


