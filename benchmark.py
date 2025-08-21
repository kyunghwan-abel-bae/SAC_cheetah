

import time
import sys
from tqdm import tqdm
import numpy as np

def run_benchmark(agent_module, device_name):
    """주어진 에이전트 모듈에 대한 벤치마크를 실행합니다."""
    
    print(f"\n--- {device_name} 버전 벤치마크 실행 ---")
    
    # 실제 환경과 동일한 조건으로 테스트
    env = agent_module.CustomHalfCheetahEnv()
    
    # 현재 하이퍼파라미터 설정 사용
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    batch_size = 1024
    min_buffer_size = 10000
    
    # 벤치마크를 위한 스텝 수 설정
    num_benchmark_steps = 1000
    
    agent_module.seed_all(42)
    
    agent = agent_module.SAC(
        state_dim,
        action_dim,
        hidden_dims=(256, 256),
        buffer_size=int(1e6),
        min_buffer_size=min_buffer_size,
        batch_size=batch_size,
        gamma=0.99,
    )
    
    (s, _), terminated, truncated = env.reset(), False, False
    
    print(f"{num_benchmark_steps} 스텝 동안의 전체 성능을 측정합니다.")
    
    # 준비 단계 (리플레이 버퍼 채우기) - 시간 측정 없음
    print("준비 중... (리플레이 버퍼 채우기)")
    for _ in tqdm(range(min_buffer_size)):
        a = agent.act(s)
        s_prime, r, terminated, truncated, _ = env.step(a)
        agent.replay_buffer.push((s, a, r, s_prime, terminated))
        s = s_prime
        if terminated or truncated:
            (s, _), terminated, truncated = env.reset(), False, False
            
    # 벤치마크 단계 - 시간 측정
    print("시간 측정 시작...")
    start_time = time.time()
    
    for _ in tqdm(range(1, num_benchmark_steps + 1)):
        a = agent.act(s)
        s_prime, r, terminated, truncated, _ = env.step(a)
        agent.step((s, a, r, s_prime, terminated)) # 업데이트 포함
        s = s_prime
        
        if terminated or truncated:
            (s, _), terminated, truncated = env.reset(), False, False
            
    end_time = time.time()
    total_time = end_time - start_time
    steps_per_sec = num_benchmark_steps / total_time
    
    print(f"\n--- 벤치마크 결과 ---")
    print(f"테스트 버전: {device_name}")
    print(f"{num_benchmark_steps} 스텝 총 소요 시간: {total_time:.2f} 초")
    print(f"성능: {steps_per_sec:.2f} steps/second")
    print("------------------------\n")
    
    return steps_per_sec

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ['cpu', 'mps']:
        print("사용법: python benchmark.py [cpu|mps]")
        sys.exit(1)
        
    choice = sys.argv[1]
    
    if choice == 'cpu':
        import sac_cheetah as agent_module
        run_benchmark(agent_module, "CPU")
    else: # mps
        import sac_cheetah_mps as agent_module
        run_benchmark(agent_module, "MPS")

