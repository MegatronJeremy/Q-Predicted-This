import time
from threading import Thread

import config
import qlearning
from environment import Environment, Quit

environment = Environment(f'maps/map2.txt')
running = True


def render_env():
    while running:
        environment.render(config.FPS)


if __name__ == '__main__':
    try:
        q_tab, avg_returns, avg_steps = qlearning.train(7000, 100, 0.05,
                                                        0.95, 0.005, 1, 0.001, environment)
        qlearning.evaluate(100, 100, environment, q_tab)
        qlearning.line_plot(avg_returns, "return", True)
        qlearning.line_plot(avg_steps, "steps", True)

        environment.reset()
        st = environment.get_agent_position()
        m = len(environment.get_field_map()[0])

        Thread(target=render_env).start()

        time.sleep(1)  # always sleep for one second here
        while True:
            action = qlearning.get_optimal_action(q_tab, st, m)
            st, _, done = environment.step(action)
            time.sleep(config.SLEEP_TIME)
            if done:
                break
    except Quit:
        pass
    finally:
        running = False
