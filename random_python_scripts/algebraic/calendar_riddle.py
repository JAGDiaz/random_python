import numpy as np
from itertools import permutations

def letters_to_number(word, key):
    number = ""
    for letter in word:
        number += key[letter]
    return int(number)

if __name__ == "__main__":

    letters = list(set("zeroesonesbinary"))
    words = ["zeroes", "ones", "binary"]

    digits = [str(char) for char in range(10)]

    for perm in permutations(digits):
        key = {letter: digit for letter, digit in zip(letters, perm)}

        if "0" in (key["s"], key["z"], key["o"], key["b"]):
            continue

        numbers = [letters_to_number(word, key) for word in words]

        if sum(numbers[:-1]) == numbers[-1]:
            print(f"{words[0]} + {words[1]} = {words[-1]}")
            print(f"{numbers[0]} + {numbers[1]} = {numbers[-1]}")