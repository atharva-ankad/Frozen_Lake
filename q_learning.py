import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def main():
    # 1. Initialize the environments
    # is_slippery=False makes it deterministic, easier to learn for a basic demo
    # train_env is used for training without rendering to speed things up
    train_env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
    
    # 2. Initialize Q-table
    state_size = train_env.observation_space.n
    action_size = train_env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # Hyperparameters
    total_episodes = 2000        # Total episodes for training (increased to ensure it finds the goal)
    learning_rate = 0.8          # Learning rate (alpha)
    max_steps = 99               # Max steps per episode
    gamma = 0.95                 # Discounting rate

    # Exploration parameters
    epsilon = 1.0                 # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 0.01            # Minimum exploration probability 
    decay_rate = 0.005            # Exponential decay rate (slowed down for more exploration)

    # List of rewards
    rewards = []

    print("Training the agent...")
    
    # 3. Train the Agent
    for episode in range(total_episodes):
        state, info = train_env.reset()
        step = 0
        done = False
        total_rewards = 0
        
        for step in range(max_steps):
            # Choose an action (a) in the current world state (s)
            exp_exp_tradeoff = np.random.uniform(0, 1)
            
            # If greater than epsilon, exploit (choose action with biggest Q value)
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(qtable[state,:])
            # Else, explore (choose random action)
            else:
                action = train_env.action_space.sample()

            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
            
            total_rewards += reward
            state = new_state
            
            if done:
                break
                
        # Reduce epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        rewards.append(total_rewards)

    print("Training finished!\n")
    print("Final Q-table:\n", np.round(qtable, 3))

    # 4. Plot the learning progress
    # Calculate moving average for a smoother curve
    window = 20
    moving_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(total_episodes), moving_avg, color='blue', label='Moving Average of Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Agent Learning Progress (Q-Learning)')
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join(os.path.dirname(__file__), 'learning_curve.png')
    plt.savefig(plot_path)
    print(f"\nSaved learning curve plot to '{plot_path}'")
    plt.close()

    # 5. Watch the trained agent play
    print("\nStarting demonstration...")
    print("If a PyGame window doesn't appear, ensure you are running this in an environment with display support.")
    time.sleep(2)
    
    # env is used for testing and visualizes the agent's actions
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="human")
    
    state, info = env.reset()
    done = False
    
    print("\n----- Demonstrating Trained Agent -----")
    for step in range(max_steps):
        # If the Q-table is all zeros for a state, take a random action instead of getting stuck going 'Left' (action 0)
        if np.max(qtable[state,:]) == 0 and np.min(qtable[state,:]) == 0:
            action = env.action_space.sample()
        else:
            action = np.argmax(qtable[state,:])
        
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        time.sleep(0.2) # slightly faster so it doesn't drag on
        state = new_state
        
        if done:
            if reward == 1:
                print("Goal Reached! Agent wins! 🎉")
            else:
                print("Agent fell into a hole! Game Over. 😭")
            break
            
    env.close()

if __name__ == "__main__":
    main()
