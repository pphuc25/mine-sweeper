import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import os
from keras.models import load_model


class MinesweeperUI:
    def __init__(self, master, rows, cols):
        self.master = master
        self.rows = rows
        self.cols = cols
        self.buttons = [[None for _ in range(cols)] for _ in range(rows)]
        self.images = self.load_images()
        self.create_widgets()

    def load_images(self):
        image_dir = 'data/pics'  # Adjust this to your image directory
        image_files = {
            'U': 'unsolved.png',
            '0': 'zero.png',
            '1': 'one.png',
            '2': 'two.png',
            '3': 'three.png',
            '4': 'four.png',
            '5': 'five.png',
            '6': 'six.png',
            '7': 'seven.png',
            '8': 'eight.png',
            'B': 'mine.png'
        }
        images = {}
        for key, filename in image_files.items():
            img = Image.open(os.path.join(image_dir, filename))
            img = img.resize((30, 30), Image.LANCZOS)  # Resize images to fit buttons
            images[key] = ImageTk.PhotoImage(img)
        return images

    def create_widgets(self):
        for r in range(self.rows):
            for c in range(self.cols):
                button = tk.Button(self.master, image=self.images['U'], width=30, height=30)
                button.grid(row=r, column=c)
                self.buttons[r][c] = button

    def update_cell(self, row, col, value):
        self.buttons[row][col].config(image=self.images[value])
        self.master.update_idletasks()

    def reset_board(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.update_cell(r, c, 'U')

class MinesweeperAgent:
    def __init__(self, model, ui):
        self.ui = ui
        
        self.mode = 'beginner'
        self.nrows = 9
        self.ncols = 9
        # self.reset()
        self.ntiles = self.nrows * self.ncols
        self.board = self.get_initial_board()
        self.state = self.get_state(self.board)

        self.epsilon = 0.01
        self.model = model

    def reset(self):
        self.ui.reset_board()
        self.board = self.get_initial_board()
        self.state = self.get_state(self.board)

    def get_initial_board(self):
        return [{'value': 'U', 'index': (y, x)} for x in range(self.nrows) for y in range(self.ncols)]

    def get_state(self, board):
        state_im = [t['value'] for t in board]
        state_im = np.reshape(state_im, (self.nrows, self.ncols, 1)).astype(object)

        state_im[state_im=='U'] = -1
        state_im[state_im=='B'] = -2

        for i in range(9):
            state_im[state_im==str(i)] = i

        state_im = state_im.astype(np.int8) / 8
        state_im = state_im.astype(np.float16)

        return state_im

    def get_action(self, state):
        board = self.state.reshape(1, self.ntiles)
        unsolved = [i for i, x in enumerate(board[0]) if x==-0.125]

        rand = np.random.random()

        if rand < self.epsilon:
            move = np.random.choice(unsolved)
        else:
            moves = self.model.predict(np.reshape(self.state, (1, self.nrows, self.ncols, 1)))
            moves[board!=-0.125] = np.min(moves)
            move = np.argmax(moves)

        return move

    def get_neighbors(self, action_index):
        board_2d = [t['value'] for t in self.board]
        board_2d = np.reshape(board_2d, (self.nrows, self.ncols))

        tile = self.board[action_index]['index']
        x, y = tile[0], tile[1]

        neighbors = []
        for col in range(y-1, y+2):
            for row in range(x-1, x+2):
                if (-1 < x < self.nrows and
                    -1 < y < self.ncols and
                    (x != row or y != col) and
                    (0 <= col < self.ncols) and
                    (0 <= row < self.nrows)):
                    neighbors.append(board_2d[col,row])

        return neighbors

    def step(self, action_index):
        done = False

        row, col = self.board[action_index]['index']
        value = np.random.choice(['0', '1', '2', '3', '4', 'B'])  # Simulate uncovering a cell
        self.board[action_index]['value'] = value
        self.ui.update_cell(row, col, value)

        if value == 'B':
            done = True
        else:
            self.state = self.get_state(self.board)

        if all(t['value'] != 'U' for t in self.board if t['value'] != 'B'):
            done = True

        return self.state, done
