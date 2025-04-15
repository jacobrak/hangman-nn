import requests

class Hangman:
    def __init__(self):
        self.word = get_random_word() 
        self.guessed_word = ["_"] * len(self.word)
        self.lives = 5
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


def get_random_word(letters=5):
    url = f"https://random-word-api.herokuapp.com/word?length={letters}"

    response = requests.get(url)
    if response.status_code == 200:
        word = response.json()[0] 
    return word

if __name__ == "__main__":
    hangman_game = Hangman() 
    print("Welcome to Hangman!")
    
    while True:
        print(f"Word: {hangman_game}")
        guess = input("Guess a letter: ")
        result = hangman_game.guess(guess)
        print()
        print(result)
        
        # Check if the game is over
        game_status = hangman_game.is_game_over()
        if game_status:
            print(game_status)
            break