#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

ENV_SIZE = 5

class GridWorld():

    def __init__(self, env_size):
        self.env_size = env_size
        # Initialize the value function and set terminal state value to 0
        self.V = np.zeros((env_size, env_size))
        self.terminal_state = (4, 4)
        self.V[self.terminal_state] = 0

        # Define the transition probabilities and rewards
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
        self.action_description = ["Right", "Left", "Down", "Up"]
        self.gamma = 1.0  # Discount factor
        self.reward = -1  # Reward for non-terminal states
        self.pi_greedy = np.zeros((self.env_size, self.env_size), dtype=int)

       # Initialize terminal and grey state rewards
        self.terminal_reward = 10  # Reward for terminal state (4,4)
        self.grey_reward = -5     # Reward for grey states (2,2), (3,0), (0,4)
        self.default_reward = -1  # Reward for regular states
        
        # List of grey states
        self.grey_states = [(2, 2), (3, 0), (0, 4)]
        
        # Define the transition probabilities and rewards
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
        self.action_description = ["Right", "Left", "Down", "Up"]
        self.gamma = 1.0  # Discount factor
        self.pi_greedy = np.zeros((self.env_size, self.env_size), dtype=int)
    
    def get_reward(self, i, j):
        if (i, j) == self.terminal_state:
            return self.terminal_reward
        elif (i, j) in self.grey_states:

            return self.grey_reward
        else:
            return self.default_reward

    '''@brief Checks if there is the change in V is less than preset threshold
    '''
    def is_done(self, i, j):
        pass
    
    '''@brief Returns True if the state is a terminal state
    '''
    def is_terminal_state(self, i, j):
        return (i, j) == self. terminal_state
    
    '''
    @brief Overwrites the current state-value function with a new one
    '''
    def update_value_function(self, V):
        self.V = np.copy(V)

    '''
    @brief Returns the full state-value function V_pi
    '''
    def get_value_function(self):
        return self.V

    '''@brief Returns the stored greedy policy
    '''
    def get_policy(self):
        return self.pi_greedy
    
    '''@brief Prints the policy using the action descriptions
    '''
    def print_policy(self):
        action_symbols = ["→", "←", "↓", "↑"]  # Right, Left, Down, Up
        for i in range(self.env_size):
            row = ""
            for j in range(self.env_size):
                if self.is_terminal_state(i, j):
                    row += " T  "  # Terminal state
                else:
                    action_index = self.pi_greedy[i, j]
                    row += f" {action_symbols[action_index]}  "
            print(row)

    '''@brief Calculate the maximim value by following a greedy policy
    '''
    def calculate_max_value(self, i, j):
        # Start with a very low value as the maximum
        max_value = float('-inf')
        best_action = None
        best_actions_str = ""

        # Loop through all actions
        for action_index in range(len(self.actions)):
            next_i, next_j = self.step(action_index, i, j)
            if self.is_valid_state(next_i, next_j):
                # Get the reward for the next state using get_reward
                reward = self.get_reward(next_i, next_j)
                
                # Update max_value if we find a higher value
                value = reward + self.gamma * self.V[next_i, next_j]
                if value > max_value:
                    max_value = value
                    best_action = action_index
                    best_actions_str = self.action_description[action_index]

        return max_value, best_action, best_actions_str
    
    '''@brief Returns the next state given the chosen action and current state
    '''
    def step(self, action_index, i, j):
        # We are assuming a Transition Probability Matrix where
        # P(s'|s) = 1.0 for a single state and 0 otherwise
        action = self.actions[action_index]
        return i + action[0], j + action[1]
    
    '''@brief Checks if a state is within the acceptable bounds of the environment
    '''
    def is_valid_state(self, i, j):
        valid = 0 <= i < self.env_size and 0 <= j < self.env_size
        return valid
    
    def update_greedy_policy(self):
        for i in range(ENV_SIZE):
            for j in range(ENV_SIZE):
                # TODO: calculate the greedy policy and populate self.pi_greedy

                # TODO: Optional - Add the optimal action description to self.pi_str to be able to print it
                pass
        
    def value_iteration(self, threshold=0.01, num_iterations=1000):
        for iteration in range(num_iterations):
            V_copy = np.copy(self.V)
            
            # Iterate over all states
            for i in range(self.env_size):
                for j in range(self.env_size):
                    if self.is_terminal_state(i, j):
                        continue  # Skip terminal states
                    
                    # Compute max value for this state
                    max_value, best_action, _ = self.calculate_max_value(i, j)
                    V_copy[i, j] = max_value
                    
                    # Update the greedy policy
                    self.pi_greedy[i, j] = best_action

            # Check for convergence (change in value function)
            if np.max(np.abs(V_copy - self.V)) < threshold:
                print(f"Converged after {iteration} iterations.")
                break

            self.update_value_function(V_copy)  # Update the value function

    def value_iterationOptimized(self, threshold=1e-4, num_iterations=1000):
        for _ in range(num_iterations):
            delta = 0  # Track the maximum change
            
            for i in range(self.env_size):
                for j in range(self.env_size):
                    if self.is_terminal_state(i, j):
                        continue  # Skip terminal state
                    
                    # Compute max value for this state
                    max_value, best_action, _ = self.calculate_max_value(i, j)

                    # Update the value function in-place
                    old_value = self.V[i, j]
                    self.V[i, j] = max_value
                    
                    # Track the max change across states
                    delta = max(delta, abs(old_value - max_value))

                    # Store best action in greedy policy
                    self.pi_greedy[i, j] = best_action

            # Stop if values have converged
            if delta < threshold:
                break

# Visualization of the value function and policy
def plot_grid(gridworld):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks(np.arange(0, gridworld.env_size, 1))
    ax.set_yticks(np.arange(0, gridworld.env_size, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_xlim([0, gridworld.env_size])
    ax.set_ylim([0, gridworld.env_size])
    ax.grid(which='both')

    # Plot the value function
    V = gridworld.get_value_function()
    for i in range(gridworld.env_size):
        for j in range(gridworld.env_size):
            ax.text(j + 0.5, gridworld.env_size - i - 0.5, f'{V[i, j]:.2f}', ha='center', va='center', color='blue')

    # Plot the policy (action descriptions)
    pi = gridworld.get_policy()
    for i in range(gridworld.env_size):
        for j in range(gridworld.env_size):
            action = pi[i, j]
            if action == 0:
                ax.text(j + 0.5, gridworld.env_size - i - 0.5, '→', ha='center', va='center', color='green')
            elif action == 1:
                ax.text(j + 0.5, gridworld.env_size - i - 0.5, '←', ha='center', va='center', color='green')
            elif action == 2:
                ax.text(j + 0.5, gridworld.env_size - i - 0.5, '↓', ha='center', va='center', color='green')
            elif action == 3:
                ax.text(j + 0.5, gridworld.env_size - i - 0.5, '↑', ha='center', va='center', color='green')

    plt.show()