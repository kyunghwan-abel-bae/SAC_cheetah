import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from einops import asnumpy

class CustomHalfCheetahEnv(gym.Wrapper):
    def __init__(self):
        env = gym.make('HalfCheetah-v4')
        super().__init__(env)
        
        # 치타의 올바른 자세를 정의 (근사값)
        self.target_height = 0.7  # 치타의 목표 높이
        self.target_orientation = 0.0  # 치타의 목표 방향 (0은 수평)
        
        # 환경 정보 출력 (디버깅용)
        print("\nHalfCheetah Environment Info:")
        print(f"Observation Space: {env.observation_space}")
        print(f"Action Space: {env.action_space}")
        
        # 상태 공간 크기 저장
        self.qpos_dim = env.unwrapped.model.nq  # position 차원
        self.qvel_dim = env.unwrapped.model.nv  # velocity 차원
        
        print(f"\nState Space Details:")
        print(f"Position dimension (nq): {self.qpos_dim}")
        print(f"Velocity dimension (nv): {self.qvel_dim}")
        
        # 관절 정보 설정
        # HalfCheetah는 다음과 같은 구조를 가짐:
        # qpos: [x, z, theta, bthigh, bshin, bfoot, fthigh, fshin, ffoot]
        # qvel: [dx, dz, dtheta, dbthigh, dbshin, dbfoot, dfthigh, dfshin, dffoot]
        self.joint_structure = {
            'root': {'pos': slice(0, 3), 'vel': slice(0, 3)},  # x, z, rotation
            'back_leg': {'pos': slice(3, 6), 'vel': slice(3, 6)},  # bthigh, bshin, bfoot
            'front_leg': {'pos': slice(6, 9), 'vel': slice(6, 9)}  # fthigh, fshin, ffoot
        }
    
    def get_joint_info(self):
        """현재 관절 상태 정보를 안전하게 가져옵니다."""
        try:
            qpos = self.env.unwrapped.data.qpos
            qvel = self.env.unwrapped.data.qvel
            
            info = {
                'root': {
                    'pos': qpos[self.joint_structure['root']['pos']],
                    'vel': qvel[self.joint_structure['root']['vel']]
                },
                'back_leg': {
                    'pos': qpos[self.joint_structure['back_leg']['pos']],
                    'vel': qvel[self.joint_structure['back_leg']['vel']]
                },
                'front_leg': {
                    'pos': qpos[self.joint_structure['front_leg']['pos']],
                    'vel': qvel[self.joint_structure['front_leg']['vel']]
                }
            }
            return info
        except Exception as e:
            print(f"Error getting joint info: {e}")
            return None
    
    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        
        # 기존 보상 (전진 속도)
        forward_reward = reward
        
        try:
            # 관절 정보 가져오기
            joint_info = self.get_joint_info()
            if joint_info is None:
                return next_state, forward_reward, terminated, truncated, info
            
            # 자세 보상 계산
            root_pos = joint_info['root']['pos']
            root_vel = joint_info['root']['vel']
            
            # 높이 보상: 치타의 높이가 목표 높이에 가까울수록 보상
            height_reward = -2.0 * abs(root_pos[1] - self.target_height)  # z축이 1번 인덱스
            
            # 방향 보상: 치타가 수평을 유지할수록 보상
            orientation_reward = -1.0 * abs(root_pos[2])  # theta
            
            # 과도한 회전 페널티
            rotation_penalty = -0.1 * (root_vel[2] ** 2)  # dtheta
            
            # 다리 사용 보상
            back_leg_vel = joint_info['back_leg']['vel']
            front_leg_vel = joint_info['front_leg']['vel']
            
            # 앞뒤 다리 균형있게 사용하도록 보상
            back_leg_activity = np.mean(np.square(back_leg_vel))
            front_leg_activity = np.mean(np.square(front_leg_vel))
            leg_balance_reward = -1.0 * abs(front_leg_activity - back_leg_activity)
            
            # 다리 활성도 보상 (모든 다리를 적절히 사용하도록)
            leg_activity_reward = 0.2 * (front_leg_activity + back_leg_activity)
            
            # 에너지 효율성 (과도한 다리 움직임 억제)
            all_joint_vel = np.concatenate([back_leg_vel, front_leg_vel])
            energy_penalty = -0.05 * np.sum(np.square(all_joint_vel))
            
            # 최종 보상 계산
            modified_reward = (
                1.0 * forward_reward +     # 전진 보상
                0.3 * height_reward +      # 높이 보상
                0.3 * orientation_reward + # 방향 보상
                0.1 * rotation_penalty +   # 회전 페널티
                0.4 * leg_balance_reward + # 다리 균형 보상
                0.3 * leg_activity_reward + # 다리 활성도 보상
                0.2 * energy_penalty       # 에너지 효율성
            )
            
            return next_state, modified_reward, terminated, truncated, info
            
        except Exception as e:
            print(f"Error in reward calculation: {e}")
            return next_state, forward_reward, terminated, truncated, info

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ReplayBuffer:
    def __init__(self, max_size, min_size, device):
        self.max_size = max_size
        self.min_size = min_size
        self.buffer = deque(maxlen=max_size)
        self.device = device
    
    def push(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states = np.array([t[0] for t in transitions])
        actions = np.array([t[1] for t in transitions])
        rewards = np.array([t[2] for t in transitions])
        next_states = np.array([t[3] for t in transitions])
        dones = np.array([t[4] for t in transitions])
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).reshape(-1, 1).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def is_ready(self):
        return len(self) >= self.min_size

class MLPContinuousQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256,), activation_fn=F.relu):
        super(MLPContinuousQNetwork, self).__init__()
        self.input_layer = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.activation_fn = activation_fn

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
        x = self.output_layer(x)
        return x

class MLPGaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256,), activation_fn=F.relu):
        super(MLPGaussianPolicy, self).__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.activation_fn = activation_fn
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_scale = torch.FloatTensor([1.0]).to(device)
        self.action_bias = torch.FloatTensor([0.0]).to(device)

    def forward(self, state):
        x = self.activation_fn(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Compute log probability
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        action = action * self.action_scale + self.action_bias
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class SAC:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=(256, 256),
        buffer_size=int(1e6),
        min_buffer_size=5000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr=3e-4,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.critic1 = MLPContinuousQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic2 = MLPContinuousQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic1_target = MLPContinuousQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic2_target = MLPContinuousQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor = MLPGaussianPolicy(state_dim, action_dim, hidden_dims).to(self.device)
        
        # Copy parameters to target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.replay_buffer = ReplayBuffer(buffer_size, min_buffer_size, self.device)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
    
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.actor.sample(state)
        return asnumpy(action.cpu())[0]
    
    def step(self, transition):
        self.replay_buffer.push(transition)
        
        if not self.replay_buffer.is_ready():
            return None
            
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        
        self.critic1_optimizer.zero_grad()
        q1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        q2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        actions_new, log_probs, _ = self.actor.sample(states)
        q1_new = self.critic1(states, actions_new)
        q2_new = self.critic2(states, actions_new)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return {
            'policy_loss': actor_loss.item(),
            'value_loss': (q1_loss.item() + q2_loss.item()) / 2
        }
    
    def save_model(self, path='saved_models'):
        if not os.path.exists(path):
            os.makedirs(path)
        
        torch.save({
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
        }, os.path.join(path, 'sac_cheetah_model.pth'))
        
        print(f"Model saved to {os.path.join(path, 'sac_cheetah_model.pth')}")

def evaluate(env_name, agent, seed, eval_iterations):
    eval_env = gym.make(env_name)
    eval_env.reset(seed=seed + 100)
    eval_scores = []
    
    for _ in range(eval_iterations):
        state, done, truncated = eval_env.reset()[0], False, False
        score = 0
        
        while not (done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = eval_env.step(action)
            score += reward
            state = next_state
            
        eval_scores.append(score)
    
    return np.mean(eval_scores)

if __name__ == "__main__":
    env = CustomHalfCheetahEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    seed = 0
    seed_all(seed)
    hidden_dims = (256, 256,)
    max_iterations = 1000000
    eval_intervals = 10000
    eval_iterations = 10
    
    buffer_size = int(1e6)
    min_buffer_size = 5000
    batch_size = 256
    gamma = 0.99
    
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
    (s, _), terminated, truncated = env.reset(), False, False
    
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
            score = evaluate('HalfCheetah-v4', agent, seed, eval_iterations)
            logger.append([t, 'Avg return', score])
            print(f"Step: {t}, Average Return: {score:.2f}")
            
            # 주기적으로 모델 저장
            agent.save_model(f'saved_models/checkpoint_{t}')
    
    # 최종 모델 저장
    agent.save_model('saved_models/final_model')
    
    # 학습 결과 시각화
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
