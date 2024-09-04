import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.95, epsilon=0.05):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # Initialize Q-table with zeros
        self.q_table = np.zeros(state_size + [action_size])

    def _state_index(self, state):
        # Flatten the state to a single index
        state_index = tuple(state)
        return state_index

    def choose_action(self, state):
        state_index = self._state_index(state)

        if np.random.rand() < self.epsilon:
            # Explore: random action
            action = np.random.randint(self.action_size, size=len(state))
            return action

        # Exploit: best action
        q_values = self.q_table[state_index]
        best_action = np.argmax(q_values)
        action = [best_action] * len(state)  # Assuming same best action for each dimension
        return action

    def learn(self, state, action, reward, next_state):
        state_index = self._state_index(state)
        next_state_index = self._state_index(next_state)

        # Q-learning formula
        current_q = self.q_table[state_index][action[0]]  # Using the first action for simplicity
        next_max_q = np.max(self.q_table[next_state_index])
        self.q_table[state_index][action[0]] = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
