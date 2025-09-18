import random
def play():
    print("Welcome to Rock, Paper, Scissors!")
    print("Enter your choice: rock, paper, or scissors")
    
    pl = input("Your choice: ").lower()
    choices = ["rock", "paper", "scissors"]
    
    if pl not in choices:
        return "Invalid choice! Please choose rock, paper, or scissors."
    
    computer = random.choice(choices)
    print(f"Computer chose: {computer}")

    if pl == computer:
        return "It's a tie!"

    if (pl == "rock" and computer == "scissors") or \
       (pl == "scissors" and computer == "paper") or \
       (pl== "paper" and computer == "rock"):
        return "You win!"
    else:
        return "Computer wins!"
print(play())