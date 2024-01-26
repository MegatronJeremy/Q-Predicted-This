import random
import numpy as np
import pandas
import seaborn
from matplotlib import pyplot

import config
from environment import Environment, Action

def linearize_pos(st, m):
    return m * st[0] + st[1]

def get_optimal_action(q_tab, st, m):
    st = linearize_pos(st, m)
    return Action(np.argmax(q_tab[st]))

def get_action_eps_greedy_policy(env: Environment, q_tab, st, eps):
    prob = random.uniform(0, 1)
    # if in exploitation return max index, else return random action if exploring
    return Action(np.argmax(q_tab[st])) if prob > eps else env.get_random_action()

def train(num_episodes, max_steps, lr, gamma, eps_min, eps_max, eps_dec_rate, env: Environment):
    avg_returns = []
    avg_steps = []

    field_map_len = len(env.get_field_map()[0]) * len(env.get_field_map())
    action_space_len = len(env.get_all_actions())
    q_tab = np.zeros((field_map_len, action_space_len))

    m: int = len(env.get_field_map()[0])

    for episode in range(num_episodes):
        avg_returns.append(0.)
        avg_steps.append(0)
        eps = eps_min + (eps_max - eps_min) * np.exp(-eps_dec_rate * episode)

        # reset environment
        env.reset()
        st = env.get_agent_position()
        st = linearize_pos(st, m)

        for step in range(max_steps):
            action = get_action_eps_greedy_policy(env, q_tab, st, eps)
            act_ind = action.value

            new_st, rew, done = env.step(action)
            new_st = linearize_pos(new_st, m)

            q_tab[st][act_ind] = q_tab[st][act_ind] + lr * (rew + gamma * np.max(q_tab[new_st]) - q_tab[st][act_ind])
            if done:
                avg_returns[-1] += rew
                avg_steps[-1] += step + 1
                break
            st = new_st
    return q_tab, avg_returns, avg_steps

def evaluate(num_episodes, max_steps, env: Environment, q_tab):
    ep_rew_lst = []
    steps_lst = []
    m: int = len(env.get_field_map()[0])

    for episode in range(num_episodes):
        # st = env.reset(seed=episode)[0]
        env.reset()
        st = env.get_agent_position()
        st = linearize_pos(st, m)

        step_cnt = 0
        ep_rew = 0
        for step in range(max_steps):
            act = Action(np.argmax(q_tab[st]))
            new_st, rew, done = env.step(act)
            step_cnt += 1
            ep_rew += rew
            if done:
                break
            st = new_st
            st = linearize_pos(st, m)

        ep_rew_lst.append(ep_rew)
        steps_lst.append(step_cnt)

    print(f'TEST Mean reward: {np.mean(ep_rew_lst):.2f}')
    print(f'TEST STD reward: {np.std(ep_rew_lst):.2f}')
    print(f'TEST Mean steps: {np.mean(steps_lst):.2f}')

def line_plot(data, name, show):
    pyplot.figure(f'Average {name} per episode: {np.mean(data):.2f}')

    df = pandas.DataFrame({
        name: [np.mean(data[i * config.chunk:(i + 1) * config.chunk])
               for i in range(config.number_of_episodes // config.chunk)],
        'episode': [config.chunk * i
                    for i in range(config.number_of_episodes // config.chunk)]})

    plot = seaborn.lineplot(data=df, x='episode', y=name, marker='o',
                            markersize=5, markerfacecolor='red')
    plot.get_figure().savefig(f'{name}.png')
    if show:
        pyplot.show()