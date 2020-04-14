import gym
import numpy as np
from Env_wrappers import make_env
from utils import plot_learning_curve
from DDQAgent import DDQAgent

if __name__ == '__main__':
    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf
    test_mode = False
    n_games = 500
    print(f'Input_dims: {env.observation_space.shape}')
    agent = DDQAgent(lr=0.00025, gamma=0.99, 
                       obs_dims=env.observation_space.shape,
                       num_actions=env.action_space.n, 
                       mem_size=50000,
                       mini_batchsize=64, epsilon_dec=(5e-6),
                       env_name='PongNoFrameskip-v4',
                       algo_name='DDQAgent')
    
    if test_mode:
        agent.load_models()

    n_steps = 0
    scores, eps_history, steps_arr = [], [], []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.get_action(observation)
            new_observation, reward, done, _ = env.step(action)
            score += reward
            

            if not test_mode:
                agent.store_memory(observation, action, reward,
                                   new_observation, done)
                agent.learn()
            
            observation = new_observation
            n_steps += 1
        
        scores.append(score)
        eps_history.append(agent.epsilon)
        steps_arr.append(n_steps)

        sample_len = min(100, len(scores))
        avg_score = np.mean(scores[-(sample_len):])
        if avg_score > best_score:
            best_score = avg_score
            if not test_mode:
                agent.save_models()

        print(f'Episode {i}: score={score}, average score={avg_score}, epsilon={agent.epsilon}, steps={n_steps}')
    
    fname = f'{agent.algo_name}_{agent.env_name}_lr{agent.lr}_{str(n_games)}_games.png'
    figure_file = "plots/" + fname
    plot_learning_curve(steps_arr, scores, eps_history, figure_file)

