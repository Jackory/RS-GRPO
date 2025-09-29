import numpy as np
import matplotlib.pyplot as plt

# Set fixed random seed to ensure reproducible results
np.random.seed(42)

class BanditEnv:
    """100-armed bandit environment with fixed bimodal reward distribution."""
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.n_arms = 100
        self.means = self._create_bimodal_rewards()

    def _create_bimodal_rewards(self):
        """Create a smooth, fixed bimodal reward distribution using Gaussian peaks."""
        x = np.arange(self.n_arms)
        
        # Peak 1: High reward
        mu1, sigma1, height1 = 40, 5, 0.9
        # Peak 2: Medium reward
        mu2, sigma2, height2 = 75, 5, 0.5
        
        peak1 = height1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2))
        peak2 = height2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2))
    
        baseline = 0.1
        means = baseline + peak1 + peak2
        return means

    def reset(self):
        return 0

    def step(self, action):
        return self.means[action]

class RiskPGAgent:
    def __init__(self, n_arms, learning_rate=0.1, beta=0.1, buffer_size=16, seed=42, 
                max_steps=200000):
        np.random.seed(seed + 1)
        
        self.n_arms = n_arms
        self.alpha = learning_rate
        self.beta = beta
        self.buffer_size = buffer_size  # Buffer size for batch updates
        self.max_steps = max_steps
        
        # Initialize policy parameters with specific distribution
        self.theta = self._initialize_policy_distribution()
        
        # Experience buffer
        self.buffer = []
        
    def _initialize_policy_distribution(self):
        """Initialize a smooth policy distribution biased towards the second reward peak."""
        x = np.arange(self.n_arms)
        
        # Center the initial policy around the second reward peak
        mu, sigma = 75, 5.0
        
        # Create a Gaussian distribution
        initial_probs = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        
        # Add a small baseline probability to ensure all actions have a non-zero chance
        initial_probs += 0.1
        
        initial_probs /= np.sum(initial_probs)
        
        # Convert probability distribution to logits (theta)
        theta = np.log(initial_probs + 1e-8)
        theta = theta - np.mean(theta)
        
        return theta

    def get_policy(self):
        return self._stable_softmax(self.theta)

    def _stable_softmax(self, x):
        """Numerically stable softmax."""
        z = x - np.max(x)
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        return numerator / denominator

    def choose_action(self):
        policy = self.get_policy()
        return np.random.choice(self.n_arms, p=policy)

    def store_experience(self, action, reward):
        self.buffer.append((action, reward))
        
        # If buffer is full, perform batch update
        if len(self.buffer) >= self.buffer_size:
            self._batch_update()
            self.buffer = []  # Clear buffer after update

    def _batch_update(self):
        if len(self.buffer) == 0:
            return
            
        policy = self.get_policy()
        
        rewards = []
        actions = []
        for action, reward in self.buffer:
            rewards.append(reward)
            actions.append(action)
        
        rewards = np.array(rewards)
        actions = np.array(actions)
        
        mean_Z = np.sum(rewards) / len(rewards)
        mean_weights = rewards - mean_Z
        if self.beta == 0:
            Z = mean_Z
            weights = mean_weights
        else:
            # Use numerically stable computation to prevent overflow
            # Subtract max for numerical stability before exponential
            max_reward = np.max(self.beta * rewards)
            stable_exp = np.exp(self.beta * rewards - max_reward)
            Z = (np.log(np.mean(stable_exp)) + max_reward) / self.beta
            
            weights = rewards - Z
            beta_weights = self.beta * weights
            weights = (np.exp(beta_weights)-1) / self.beta
        
        total_grad = np.zeros(self.n_arms)
        
        # Calculate policy gradient
        for i, (action, weight) in enumerate(zip(actions, weights)):
            grad_log_pi = -policy.copy()
            grad_log_pi[action] += 1
            total_grad += weight * grad_log_pi
        
        self.theta += self.alpha * total_grad / len(self.buffer)

    def update(self, action, reward):
        """Store experience and potentially update policy parameters."""
        self.store_experience(action, reward)
        
    def force_update(self):
        """Force an update even if buffer is not full (useful at end of episode)."""
        if len(self.buffer) > 0:
            self._batch_update()
            self.buffer = []

class ExperimentRunner:
    """Experiment runner responsible for running experiments and visualizing results"""
    
    def __init__(self):
        self.results = {}  # Store all experiment results
        
    def calculate_policy_entropy(self, policy):
        """Calculate entropy of policy distribution"""
        return -np.sum(policy * np.log(policy + 1e-9))
    
    def run_single_experiment(self, beta_value, n_steps=200000, seed=42):
        """Run a single experiment"""
        env = BanditEnv(seed=seed)
        agent = RiskPGAgent(n_arms=env.n_arms, beta=beta_value, seed=seed, max_steps=n_steps)
        
        rewards_history = []
        entropy_history = []
        theta_history = []
        
        for step in range(n_steps):
            theta = agent.theta.copy()
            theta_history.append(theta)
            action = agent.choose_action()
            reward = env.step(action)
            agent.update(action, reward)
            rewards_history.append(reward)
            policy = agent.get_policy()
            entropy_history.append(self.calculate_policy_entropy(policy))
        
        # Force final update if there are remaining experiences in buffer
        agent.force_update()
        
        return {
            'beta': beta_value,
            'rewards': rewards_history,
            'entropy': entropy_history,
            'theta_history': theta_history,  # Add theta history
        }
    
    def run_multiple_experiments(self, beta_values, n_steps=200000, seed=42):
        """Run multiple experiments with different beta values"""
        self.results = {}
        
        for i, beta in enumerate(beta_values):
            print(f"Running experiment: ($\\beta$={beta})")
            experiment_name = f'$\\beta$={beta}'
            experiment_seed = seed
            self.results[experiment_name] = self.run_single_experiment(
                beta, n_steps, seed=experiment_seed)
        
        np.save('bandit_results.npy', self.results)
        return self.results
    
    def plot_moving_average(self, data, window_size=1000):
        """Calculate moving average"""
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    def plot_results(self, save_path='bandit.png', figsize=(16, 6)):
        """Plot experiment results"""
        if not self.results:
            print("No results to plot. Run experiments first.")
            return
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=figsize)
        
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_linewidth(1.5)
        ax1.spines['left'].set_linewidth(1.5)

        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_linewidth(1.5)
        ax2.spines['left'].set_linewidth(1.5)
        
        # Plot reward curves
        for name, result in self.results.items():
            beta = result['beta']
            rewards_smooth = self.plot_moving_average(result['rewards'])
            ax1.plot(rewards_smooth, label=f'$\\beta$={beta}')

        ax1.set_xlabel('Accuracy', fontsize=24)
        ax1.set_ylabel('Proportion', fontsize=24)
        ax1.grid(True, linestyle='--', alpha=0.6, which='both')
        ax1.tick_params(axis='x', labelsize=14)
        ax1.tick_params(axis='y', which='major', labelsize=10)
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Reward')
        ax1.set_title('Reward')
        ax1.legend()
        
        
        # Plot policy entropy curves
        for name, result in self.results.items():
            beta = result['beta']
            ax2.plot(result['entropy'], label=f'$\\beta$={beta}')
        
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Policy Entropy')
        ax2.set_title('Policy Entropy')
        # ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")


if __name__ == "__main__":
    np.random.seed(42)
    runner = ExperimentRunner()
    beta_values = np.array([0, 2, 4, 8, 16, 32, 64])
    runner.run_multiple_experiments(beta_values, n_steps=int(4e5), seed=42)
    runner.plot_results(save_path='100arm_bandit.png')