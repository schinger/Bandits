import numpy as np
import matplotlib.pyplot as plt

class BanditEnv:
    def __init__(self, true_probs):
        self.true_probs = true_probs
        self.num_arms = len(true_probs)

    def pull_arm(self, arm_index):
        return 1 if np.random.rand() < self.true_probs[arm_index] else 0

class EpsilonGreedyBandit:
    def __init__(self, num_arms, epsilon=0.1):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.Q_values = np.zeros(num_arms)
        self.N_counts = np.zeros(num_arms, dtype=int)

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_arms)
        else:
            return np.argmax(self.Q_values)

    def update(self, arm_index, reward):
        self.N_counts[arm_index] += 1
        self.Q_values[arm_index] += (reward - self.Q_values[arm_index]) / self.N_counts[arm_index]

class SoftmaxBandit:
    def __init__(self, num_arms, learning_rate=0.1):
        self.num_arms = num_arms
        self.learning_rate = learning_rate
        self.preferences = np.zeros(num_arms)
        self.N = 0
        self.avg_reward = 0.0

    def select_arm(self):
        probs = np.exp(self.preferences) / np.sum(np.exp(self.preferences))
        return np.random.choice(self.num_arms, p=probs)

    def update(self, arm_index, reward):
        self.N += 1
        self.avg_reward += (reward - self.avg_reward) / self.N
        probs = np.exp(self.preferences) / np.sum(np.exp(self.preferences))
        for a in range(self.num_arms):
            if a == arm_index:
                self.preferences[a] += self.learning_rate * (reward - self.avg_reward) * (1 - probs[a])
            else:
                self.preferences[a] += self.learning_rate * (reward - self.avg_reward) * (0 - probs[a])


class UCBBandit:
    def __init__(self, num_arms, alpha=1.0):
        self.num_arms = num_arms
        self.alpha = alpha
        self.Q_values = np.zeros(num_arms)
        self.N_counts = np.zeros(num_arms, dtype=int)
        self.total_steps = 0

    def select_arm(self):
        if np.any(self.N_counts == 0):
            return np.argmin(self.N_counts) # Explore arms that haven't been pulled yet
        ucb_values = self.Q_values + np.sqrt((self.alpha * np.log(self.total_steps)) / (2 * self.N_counts))
        return np.argmax(ucb_values)

    def update(self, arm_index, reward):
        self.N_counts[arm_index] += 1
        self.total_steps += 1
        self.Q_values[arm_index] += (reward - self.Q_values[arm_index]) / self.N_counts[arm_index]


class ThompsonSamplingBandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.alpha_beta_params = np.ones((num_arms, 2)) # [alpha, beta] for each arm, initialized to Beta(1,1) - uniform prior

    def select_arm(self):
        samples = [np.random.beta(self.alpha_beta_params[a, 0], self.alpha_beta_params[a, 1]) for a in range(self.num_arms)]
        return np.argmax(samples)

    def update(self, arm_index, reward):
        if reward == 1:
            self.alpha_beta_params[arm_index, 0] += 1 # increment alpha for reward=1
        else:
            self.alpha_beta_params[arm_index, 1] += 1 # increment beta for reward=0





if __name__ == '__main__':
    num_arms = 10
    true_probs = np.random.rand(num_arms)
    print(f"True probabilities of arms: {true_probs}")
    bandit_env = BanditEnv(true_probs)
    num_steps = 1000

    # Initialize bandits
    epsilon_greedy_bandit = EpsilonGreedyBandit(num_arms, epsilon=0.1)
    softmax_bandit = SoftmaxBandit(num_arms, learning_rate=0.2)
    ucb_bandit = UCBBandit(num_arms, alpha=1.0)
    thompson_sampling_bandit = ThompsonSamplingBandit(num_arms)

    bandits = [epsilon_greedy_bandit, softmax_bandit, ucb_bandit, thompson_sampling_bandit]
    bandit_names = ["Epsilon-Greedy", "Softmax", "UCB", "Thompson Sampling"]
    cumulative_rewards = {name: [] for name in bandit_names}
    average_rewards = {name: [] for name in bandit_names} # Store average rewards

    for bandit, name in zip(bandits, bandit_names):
        current_cumulative_reward = 0
        for step in range(num_steps):
            chosen_arm = bandit.select_arm()
            reward = bandit_env.pull_arm(chosen_arm)
            bandit.update(chosen_arm, reward)
            current_cumulative_reward += reward
            cumulative_rewards[name].append(current_cumulative_reward)
            average_rewards[name].append(current_cumulative_reward / (step + 1))


    # Plotting
    plt.figure(figsize=(10, 6))
    for name in bandit_names:
        plt.plot(average_rewards[name], label=name)

    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Performance Comparison of Multi-arm Bandit Algorithms")
    plt.legend()
    plt.grid(True)
    plt.show()
