# from sac_cheetah import batch_size, buffer_size, eval_intervals, eval_iterations, max_iterations, min_buffer_size
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.distributions import Normal
import os


class MLPContinuousQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256, ), activation_fn=F.relu):
        super(MLPContinuousQNetwork, self).__init__()
        self.input_layer = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.activation_fn = activation_fn

    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        x = self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
        x = self.output_layer(x)

        return x


class MLPContinuousDoubleQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims, activation_fn):
        super().__init__()
        self.q1 = MLPContinuousQNetwork(state_dim, action_dim, hidden_dims, activation_fn)
        self.q2 = MLPContinuousQNetwork(state_dim, action_dim, hidden_dims, activation_fn)
        
    def forward(self, s, a):
        return self.q1(s, a), self.q2(s, a)


class MLPGaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(512, ), activation_fn=F.relu):
        super(MLPGaussianPolicy, self).__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.mu_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))

        mu = self.mu_layer(x)
        log_std = torch.tanh(self.log_std_layer(x))

        return mu, log_std.exp()    


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size):
        if np.isscalar(state_dim):
            state_dim = (state_dim, )
        if np.isscalar(action_dim):
            action_dim = (action_dim, )

        self.s = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.a = np.zeros((max_size, *action_dim), dtype=np.float32)
        self.r = np.zeros((max_size, 1), dtype=np.float32)
        self.s_prime = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.uint8)

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def __getitem__(self, idx):
        return (
            self.s[idx],
            self.a[idx],
            self.r[idx],
            self.s_prime[idx],
            self.done[idx]
        )

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.s[idx]),
            torch.FloatTensor(self.a[idx]),
            torch.FloatTensor(self.r[idx]),
            torch.FloatTensor(self.s_prime[idx]),
            torch.FloatTensor(self.done[idx])
        )

    def store(self, s, a, r, s_prime, done):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_prime[self.ptr] = s_prime
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


class SAC:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=(1024, 1024),
        activation_fn=F.relu,
        buffer_size=int(1e6),
        min_buffer_size=5000,
        batch_size=256,
        policy_lr=0.0001,
        critic_lr=0.0001,
        gamma=0.99,
        tau=0.005,
        alpha=0.2
    ):
        self.action_dim = action_dim
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma
        self.min_buffer_size = min_buffer_size
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = MLPGaussianPolicy(state_dim, action_dim, hidden_dims, activation_fn).to(self.device)
        self.critic = MLPContinuousDoubleQNetwork(state_dim, action_dim, hidden_dims, activation_fn).to(self.device)
        self.target_critic = MLPContinuousDoubleQNetwork(state_dim, action_dim, hidden_dims, activation_fn).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.t = 0
        self.buffer = ReplayBuffer(state_dim, action_dim, buffer_size)

    @torch.no_grad()
    def act(self, s, training=True):
        if(self.buffer.size < self.min_buffer_size) and training:
            return np.random.rand(self.action_dim)

        self.policy.train(training)

        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        mu, std = self.policy(s)
        z = torch.normal(mu, std) if training else mu
        action = torch.tanh(z)

        return action.cpu().numpy()
    
    def sample_action(self, state):
        mu, std = self.policy(state)
        m = Normal(mu, std)
        z = m.rsample()
        a = torch.tanh(z)

        log_prob = m.log_prob(z)
        log_prob -= torch.log(1 - a.pow(2) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)

        return a, log_prob
    
    def learn(self):
        self.policy.train()
        self.critic.train()

        s, a, r, s_prime, done = self.buffer.sample(self.batch_size)
        s, a, r, s_prime, done = map(lambda x: x.to(self.device), [s, a, r, s_prime, done])

        with torch.no_grad():
            a_prime, log_prob_prime = self.sample_action(s_prime)
            next_q1, next_q2 = self.target_critic(s_prime, a_prime)
            next_q = torch.min(next_q1, next_q2)
            td_target = r + (1 - done) * self.gamma * (next_q - self.alpha * log_prob_prime)
        
        # Update the critic network
        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Sample a current action for the policy gradient
        a_sampled, log_prob = self.sample_action(s)
        policy_q1, policy_q2 = self.critic(s, a_sampled)
        policy_q = torch.min(policy_q1, policy_q2)

        policy_loss = -(policy_q- self.alpha * log_prob).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        result = {'Step': self.t, 'policy_loss': policy_loss.item(), 'value_loss': critic_loss.item()}

        return result
    
    def step(self, transition):
        result = None
        self.t += 1
        self.buffer.store(*transition)

        if self.buffer.size >= self.min_buffer_size:
            result = self.learn()
            for t_p, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                t_p.data.copy_(self.tau * p.data + (1 - self.tau) * t_p.data)

        return result
    
    def save_model(self, path='saved_models'):
        if not os.path.exists(path):
            os.makedirs(path)
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, os.path.join(path, 'sac_cheetah_model.pth'))
        
        print(f"Model saved to {os.path.join(path, 'sac_cheetah_model.pth')}")
        

def evaluate(env_name, agent, seed, eval_iterations):
    env = gym.make(env_name)
    scores = []
    for i in range(eval_iterations):
        (s, _), terminated, truncated, score = env.reset(seed=seed + 100 + i), False, False, 0
        while not (terminated or truncated):
            a = agent.act(s, training=False)
            s_prime, r, terminated, truncated, _ = env.step(a)
            score += r
            s = s_prime
        scores.append(score)
    env.close()

    return round(np.mean(scores), 4)
            

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    env_name = 'HalfCheetah-v4'

    seed = 0
    seed_all(seed)
    hidden_dims = (256, 256, )
    max_iterations = 100000
    eval_intervals = 10000
    eval_iterations = 10

    buffer_size = int(1e6)
    min_buffer_size = 5000
    batch_size = 256
    gamma = 0.99

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SAC(
        state_dim,
        action_dim,
        hidden_dims=hidden_dims,
        buffer_size=buffer_size,
        min_buffer_size=min_buffer_size,
        batch_size=batch_size,
        gamma=gamma,
    )

    logger = []
    (s, _), terminated, truncated = env.reset(seed=seed), False, False
    for t in tqdm(range(1, max_iterations + 1)):
        a = agent.act(s)
        s_prime, r, terminated, truncated, _ = env.step(a)

        result = agent.step((s, a, r, s_prime, terminated))
        s = s_prime

        if result is not None:
            logger.append([t, 'policy_loss', result['policy_loss']])
            logger.append([t, 'value_loss', result['value_loss']])
        
        if terminated or truncated:
            (s, _), terminated, truncated = env.reset(), False, False
        
        if t % eval_intervals == 0:
            score = evaluate(env_name, agent, seed, eval_iterations)
            logger.append([t, 'Avg return', score])

            # 주기적으로 모델 저장
            agent.save_model(f'saved_models/checkpoint_{t}')
        
    # 최종 모델 저장
    agent.save_model('saved_models/final_model')

    ######여기서부터

    logger = pd.DataFrame(logger)
    logger.columns = ['step', 'key', 'value']

    fig = plt.figure(figsize=(12, 4))

    ax = fig.add_subplot(1, 3, 1)
    key = 'Avg return'
    ax.plot(logger.loc[logger['key'] == key, 'step'], logger.loc[logger['key'] == key, 'value'], 'b-')
    ax.grid(axis='y')
    ax.set_title("Average return over 10 episodes")
    ax.set_xlabel('Steps')
    ax.set_ylabel('Avg return')

    ax = fig.add_subplot(1, 3, 2)
    key = 'policy_loss'
    ax.plot(logger.loc[logger['key'] == key, 'step'], logger.loc[logger['key'] == key, 'value'], 'b-')
    ax.grid(axis='y')
    ax.set_title("Policy loss")
    ax.set_xlabel('Steps')
    ax.set_ylabel('Policy loss')

    ax = fig.add_subplot(1, 3, 3)
    key = 'value_loss'
    ax.plot(logger.loc[logger['key'] == key, 'step'], logger.loc[logger['key'] == key, 'value'], 'b-')
    ax.grid(axis='y')
    ax.set_title("Value loss")
    ax.set_xlabel('Steps')
    ax.set_ylabel('Value loss')

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

    ######여기까지

    # 학습 결과 시각화
    '''
    logger = np.array(logger)

    plt.figure(figsize=(12, 8))

    # Plot policy loss
    policy_loss_data = logger[logger[:, 1] == 'policy_loss']
    plt.subplot(3, 1, 1)
    plt.plot(policy_loss_data[:, 0], policy_loss_data[:, 2])
    plt.title('Policy Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')

    # Plot value loss
    value_loss_data = logger[logger[:, 1] == 'value_loss']
    plt.subplot(3, 1, 2)
    plt.plot(value_loss_data[:, 0], value_loss_data[:, 2])
    plt.title('Value Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')

    # Plot average return
    avg_return_data = logger[logger[:, 1] == 'Avg return']
    plt.subplot(3, 1, 3)
    plt.plot(avg_return_data[:, 0], avg_return_data[:, 2])
    plt.title('Average Return')
    plt.xlabel('Steps')
    plt.ylabel('Return')

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
    '''

if __name__ == '__main__':
    main()
