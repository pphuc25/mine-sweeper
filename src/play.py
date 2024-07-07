from src.game_env import *
from DQN_agent import *
import argparse
from tqdm import tqdm
from keras.models import load_model

# intake MinesweeperEnv parameters, beginner mode by default
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DQN to play Minesweeper')
    parser.add_argument('--width', type=int, default=9,
                        help='width of the board')
    parser.add_argument('--height', type=int, default=9,
                        help='height of the board')
    parser.add_argument('--n_mines', type=int, default=10,
                        help='Number of mines on the board')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of episodes to train on')
    parser.add_argument('--model_path', type=str, default='conv64x4_dense512x2_y0.1_minlr0.001',
                        help='Name of model')

    return parser.parse_args()

params = parse_args()



env = MinesweeperEnv(params.width, params.height, params.n_mines)
agent = DQNAgent(env, params.model_path)

# my_model = load_model(f'DQN/models/{params.model}.h5')
model = load_model(f'DQN/models/{params.model_path}.keras')


for episode in tqdm(range(1, params.episodes+1), unit='episode'):
    env.reset()
    episode_reward = 0
    past_n_wins = env.n_wins

    done = False
    while not done:
        current_state = env.state_im

        action = agent.get_action(current_state)

        new_state, reward, done = env.step(action)
        
        # print(action)
        # print(reward)
        # print(new_state)
        if done:
            print(env.grid)
            for row in env.grid:
                gridrow = []
                for cell in row:
                    if cell.mine:
                        gridrow.append('X')
                    elif cell.visible:
                        gridrow.append(cell.value)
                    else:
                        gridrow.append(cell.value)
                print(gridrow)
            break