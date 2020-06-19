import time
from env.tsp_env import TSPEnv


class Simulator:
    def __init__(self, env: TSPEnv):
        self.env = env

    def start(self, player, rank, episode):
        state = self.env.initial_state()
        total_s_t = time.time()
        while True:
            s_t = time.time()
            move, best_sol = player.get_action()
            state = self.env.next_state(state, move)
            print(
                "Process %2d, episode %d, time %2f ---> %3d, best %4f" % (
                    rank, episode, time.time() - s_t, len(state['tour']), best_sol))
            game_end = self.env.is_done_state(state)
            if game_end:
                print(time.time() - total_s_t)
                return state['tour'], self.env.get_return(state)
