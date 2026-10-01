# Q-Learning Mini Project - FrozenLake

This project demonstrates a basic Reinforcement Learning agent using the Q-Learning algorithm. 
The agent learns to navigate a grid (the FrozenLake environment) from a start position to a goal without falling into holes.

## Setup Instructions

1. **Create a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the script:**
   ```bash
   python q_learning.py
   ```

## What it does:
- **Trains** a Q-Learning agent over 500 episodes to solve the puzzle.
- **Plots** the learning curve (how the rewards improved over time) and saves it as `learning_curve.png`.
- **Demonstrates** the trained agent by opening a graphical window where you can watch the agent effortlessly find its way to the goal!
