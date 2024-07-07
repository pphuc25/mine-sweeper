import argparse
import tkinter as tk
from keras.models import load_model

from .agent_ui import MinesweeperUI, MinesweeperAgent

def parse_args():
    parser = argparse.ArgumentParser(description='Play Minesweeper online using a DQN')
    parser.add_argument('--model_name', type=str, default='conv64x4_dense512x2_y0.1_minlr0.001',
                        help='name of model')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to play')

    return parser.parse_args()

params = parse_args()


def main():
    root = tk.Tk()
    rows, cols = 9, 9
    ui = MinesweeperUI(root, rows, cols)

    my_model = load_model(f"models/{params.model_name}.keras")  # Replace with your actual model
    agent = MinesweeperAgent(my_model, ui)

    def run_episode():
        agent.reset()
        done = False
        while not done:
            current_state = agent.state
            action = agent.get_action(current_state)
            new_state, done = agent.step(action)
            root.after(100)  # Add a small delay to visualize changes

        root.after(params.episodes, run_episode)  # Schedule the next episode

    run_episode()
    root.mainloop()

if __name__ == "__main__":
    main()
