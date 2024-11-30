import gymnasium as gym
import torch
import numpy as np
from sac_cheetah import SAC
import time
import matplotlib.pyplot as plt

def visualize_trained_agent(model_path, num_episodes=3):
    """학습된 SAC 에이전트를 시각화합니다."""
    
    print(f"Loading model from {model_path}")
    
    # 환경 생성 (rgb_array 모드 사용)
    env = gym.make('HalfCheetah-v4', render_mode='rgb_array')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # matplotlib 설정
    plt.ion()  # 대화형 모드 활성화
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 에이전트 생성 및 모델 로드
    agent = SAC(state_dim, action_dim)
    try:
        checkpoint = torch.load(model_path)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        print("Successfully loaded model!")
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        plt.close(fig)
        return
    
    total_rewards = []
    
    try:
        # 여러 에피소드 실행
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            truncated = False
            steps = 0
            
            print(f"\nStarting Episode {episode + 1}")
            
            while not (done or truncated):
                # 행동 선택
                with torch.no_grad():
                    action = agent.act(state)
                
                # 환경에서 한 스텝 진행
                next_state, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                state = next_state
                steps += 1
                
                # 현재 상태 렌더링
                frame = env.render()
                if frame is not None:
                    ax.clear()
                    ax.imshow(frame)
                    plt.pause(0.01)  # 화면 업데이트 및 딜레이
            
            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    except Exception as e:
        print(f"Error during visualization: {e}")
    
    finally:
        env.close()
        plt.close(fig)
        plt.ioff()  # 대화형 모드 비활성화
    
    if total_rewards:
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        print(f"\nAverage Reward over {len(total_rewards)} episodes: {avg_reward:.2f} ± {std_reward:.2f}")
    else:
        print("\nNo episodes completed successfully.")

if __name__ == "__main__":
    # 모델 경로 설정
    model_paths = [
        'saved_models/final_model/sac_cheetah_model.pth',  # 최종 모델
        'saved_models/checkpoint_100000/sac_cheetah_model.pth',  # 중간 체크포인트
    ]
    
    # 각 모델 시각화
    for path in model_paths:
        print(f"\nTesting model: {path}")
        try:
            visualize_trained_agent(path)
        except FileNotFoundError:
            print(f"Model file not found: {path}")
            continue
        except Exception as e:
            print(f"Error during visualization: {e}")
            continue
        print("-" * 50)
