import numpy as np
import random
import copy

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Move:
    start_pos: Tuple[int]
    end_pos: Tuple[int]
    captures: Tuple[int] = None
    double_captures: Tuple[int] = None
    triple_captures: Tuple[int] = None

    def __hash__(self):
        return hash((
            tuple(self.start_pos),
            tuple(self.end_pos),
            tuple(self.captures) if self.captures else (),
            tuple(self.double_captures) if self.double_captures else (),
            tuple(self.triple_captures) if self.triple_captures else ()
        ))

    def __eq__(self, other):
        return (self.start_pos == other.start_pos and
                self.end_pos == other.end_pos and
                self.captures == other.captures and
                self.double_captures == other.double_captures and
                self.triple_captures == other.triple_captures)

class Checkers:
    def __init__(self):
      self.curr_turn = -1
      self.rows = 8
      self.cols = 8
      self.draw_counter = 0
      self.initialize_board()

    def initialize_board(self):
      self.board = np.zeros((self.rows, self.cols), dtype=int)  # rowxcol board initialized to zero (empty cells)
      for i, j in np.ndindex(self.board.shape):
        if (i == 0 or i == 2) and j % 2 == 1:
          self.board[i, j] = 1
        if i == 1 and j % 2 == 0:
          self.board[i, j] = 1
        if (i == 7 or i == 5) and j % 2 == 0:
          self.board[i, j] = -1
        if i == 6 and j % 2 == 1:
          self.board[i, j] = -1

    def display_board(self):
      print(self.board)

    def check_bounds(self, x, y):
      return 0 <= x <= 7 and 0 <= y <= 7

    def get_pawn_directions(self):
      for i, j in np.ndindex(self.board.shape):
        if self.curr_turn == 1 and self.board[i, j] == 1:
          return [[1, -1], [1, 1]]
        elif self.curr_turn == -1 and self.board[i, j] == -1:
          return [[-1, -1], [-1, 1]]

    def get_king_directions(self):
      for i, j in np.ndindex(self.board.shape):
        if (self.curr_turn == 1 and self.board[i, j] == 2) or (self.curr_turn == -1 and self.board[i, j] == -2):
          return [[1, -1], [1, 1], [-1, -1], [-1, 1]]


    def check_capture(self, x, y, dx, dy):
      dest_row = x + 2 * dx
      dest_col = y + 2 * dy
      return self.check_bounds(dest_row, dest_col) and self.board[dest_row, dest_col] == 0

    def check_game_state (self):
      red_pieces = np.sum(self.board == 1)
      black_pieces = np.sum(self.board == -1)
      if red_pieces > 0 and black_pieces == 0:
        #print("The winner is Red!")
        return 1
      elif black_pieces > 0 and red_pieces == 0:
        #print("The winner is Black!")
        return 2
      elif self.is_draw():
        #print("The game is a draw!")
        return 3

    def available_actions(self):
      available = []
      pawn_positions = []
      king_positions = []


      # Get all current player piece positions
      for i, j in np.ndindex(self.board.shape):
        if self.board[i, j] == self.curr_turn:
          pawn_positions.append([i, j])
        if self.board[i, j] == self.curr_turn * 2:
          king_positions.append([i, j])

      double_captures = None
      triple_captures = None

      #For Pawns
      for row, col in pawn_positions:
        for dx, dy in self.get_pawn_directions():
          if not self.check_bounds(row + dx, col + dy):
            continue
          start_pos = [row, col]

          dest = self.board[row + dx, col + dy]
          #Single Captures

          if (dest == -self.curr_turn or dest == -self.curr_turn * 2) and self.check_capture(row, col, dx, dy):
            end_pos_x = row + 2 * dx
            end_pos_y = col + 2 * dy
            end_pos = [end_pos_x, end_pos_y]
            captures = [row + dx, col + dy]
            double_captures = None
            triple_captures = None
            #Double Captures
            #implementing new dx, dy as dxa, dya to consider all possible options, not just the original selected dx, dy
            available.append(Move(start_pos, end_pos, captures, double_captures, triple_captures))

            for dxa, dya in self.get_pawn_directions():
              if not self.check_bounds(end_pos_x + dxa, end_pos_y + dya):
                continue
              double_dest = self.board[end_pos_x + dxa, end_pos_y + dya]
              if (double_dest == -self.curr_turn or double_dest == -self.curr_turn * 2) and self.check_capture(end_pos_x, end_pos_y, dxa, dya):
                end_pos_ax = row + 2 * dx + 2 * dxa
                end_pos_ay = row + 2 * dy + 2 * dya
                end_pos = [end_pos_x + 2 * dxa, end_pos_y + 2 * dya]
                double_captures = [end_pos_x + dxa, end_pos_y + dya]
                triple_captures = None
                #Triple Captures
                #Same process as before
                available.append(Move(start_pos, end_pos, captures, double_captures, triple_captures))
                for dxb, dyb in self.get_pawn_directions():
                  if not self.check_bounds(end_pos_ax + dxb, end_pos_ay + dyb):
                    continue
                  triple_dest = self.board[end_pos_ax + dxb, end_pos_ay + dyb]
                  if (triple_dest == -self.curr_turn or triple_dest == -self.curr_turn * 2) and self.check_capture(end_pos_ax, end_pos_ay, dxb, dyb):
                    end_pos = [end_pos_ax + 2 * dxb, end_pos_ay + 2 * dyb]
                    triple_captures = [end_pos_ax + dxb, end_pos_ay + dyb]
                    available.append(Move(start_pos, end_pos, captures, double_captures, triple_captures))

          elif dest == 0:
            end_pos = [row + dx, col + dy]
            captures = None
            double_captures = None
            triple_captures = None
            available.append(Move(start_pos, end_pos, captures, double_captures, triple_captures))

      #For kings
      for row, col in king_positions:
        for dx, dy in self.get_king_directions():
          if not self.check_bounds(row + dx, col + dy):
            continue
          start_pos = [row, col]

          dest = self.board[row + dx, col + dy]
          #Single Captures

          if (dest == -self.curr_turn or dest == -self.curr_turn * 2) and self.check_capture(row, col, dx, dy):
            end_pos_x = row + 2 * dx
            end_pos_y = col + 2 * dy
            end_pos = [end_pos_x, end_pos_y]
            captures = [row + dx, col + dy]
            double_captures = None
            triple_captures = None
            #Double Captures
            #implementing new dx, dy as dxa, dya to consider all possible options, not just the original selected dx, dy
            available.append(Move(start_pos, end_pos, captures, double_captures, triple_captures))

            for dxa, dya in self.get_king_directions():
              if not self.check_bounds(end_pos_x + dxa, end_pos_y + dya):
                continue
              double_dest = self.board[end_pos_x + dxa, end_pos_y + dya]
              if (double_dest == -self.curr_turn or double_dest == -self.curr_turn * 2) and self.check_capture(end_pos_x, end_pos_y, dxa, dya):
                end_pos_ax = row + 2 * dx + 2 * dxa
                end_pos_ay = row + 2 * dy + 2 * dya
                end_pos = [end_pos_x + 2 * dxa, end_pos_y + 2 * dya]
                double_captures = [end_pos_x + dxa, end_pos_y + dya]
                triple_captures = None
                #Triple Captures
                #Same process as before
                available.append(Move(start_pos, end_pos, captures, double_captures, triple_captures))

                for dxb, dyb in self.get_king_directions():
                  if not self.check_bounds(end_pos_ax + dxb, end_pos_ay + dyb):
                    continue
                  triple_dest = self.board[end_pos_ax + dxb, end_pos_ay + dyb]
                  if (triple_dest == -self.curr_turn or triple_dest == -self.curr_turn * 2) and self.check_capture(end_pos_ax, end_pos_ay, dxb, dyb):
                    end_pos = [end_pos_ax + 2 * dxb, end_pos_ay + 2 * dyb]
                    triple_captures = [end_pos_ax + dxb, end_pos_ay + dyb]
                    available.append(Move(start_pos, end_pos, captures, double_captures, triple_captures))

          elif dest == 0:
            end_pos = [row + dx, col + dy]
            captures = None
            double_captures = None
            triple_captures = None
            available.append(Move(start_pos, end_pos, captures, double_captures, triple_captures))

      return available


    def make_move(self, move):
        # Place player's marker (1 for AI, -1 for human) at the specified place
        row, col = move.end_pos
        start_row, start_col = move.start_pos
        self.board[row, col] = self.board[start_row, start_col]
        self.board[start_row, start_col] = 0


        if move.captures:
          capture_row, capture_col = move.captures
          self.board[capture_row, capture_col] = 0

        if move.double_captures:
          double_capture_row, double_capture_col = move.double_captures
          self.board[double_capture_row, double_capture_col] = 0

        if move.triple_captures:
          triple_capture_row, triple_capture_col = move.triple_captures
          self.board[triple_capture_row, triple_capture_col] = 0

        self.curr_turn *= -1
        self.check_game_state()

    def random_move(self):
      moves = self.available_actions()
      if not moves:
          return None
      move = random.choice(moves)
      self.make_move(move)

    def check_king(self):
      for i, j in np.ndindex(self.board.shape):
        if i == 7 and self.board[i, j] == 1:
          self.board[i, j] = 2
          return 1
        elif i == 0 and self.board[i, j] == -1:
          self.board[i, j] = -2
          return -1

    def is_draw(self):
      if self.draw_counter == 40:
        return True
      if not self.available_actions():
        return True



class Q_Learning:
    def __init__(self, game):
        self.game = game
        self.q_table = {}
        # Parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # Exploration vs Exploitation
        self.episodes = 100
        self.max_steps = 70  # Max steps per episode

        # Rewards
        self.reward_goal = 1000
        self.reward_capture = 20
        self.reward_king = 10
        self.reward_draw = 20
        self.reward_normal = -1
        self.reward_lose = -1000


    def best_move(self, state):
        if state in self.q_table:
            q_values = []
            actions = self.game.available_actions() #I Randomly added Make sure correct
            for action in actions:
              q_values.append(self.q_table[(state, action)])
            best_q_value = max(q_values)  # Find the best Q-value
            best_actions = [action for action in actions if self.q_table[(state, action)] == best_q_value]
            if len(best_actions) > 1:
              return random.choice(best_actions)
            return best_actions
        else:
            actions = self.game.available_actions()
            return random.choice(actions)  # If no Q-value exists, pick randomly

    def get_next_state(self, state, action):
        """Get the next state after taking an action."""
        self.game.make_move(action)
        new_state = tuple(self.game.board.flatten())  # Convert next state to tuple (hashable)
        new_board = self.game.board
        return new_state, new_board # If invalid move, stay in the same state

    def get_reward(self, move):
        """Get the reward for being in a state."""
        game_state = self.game.check_game_state()
        if game_state == 1 and self.game.curr_turn == 1:  # Red wins
            return self.reward_goal
        if game_state == 2 and self.game.curr_turn == -1:  # Black wins
            return self.reward_goal
        if game_state == 1 and self.game.curr_turn == -1:  # Red wins
            return self.reward_lose
        if game_state == 2 and self.game.curr_turn == 1:  # Black wins
            return self.reward_lose
        if game_state == 3:  # Draw
            return self.reward_draw
        if self.game.check_king() == 1 and self.game.curr_turn == 1:
            return self.reward_king
        if self.game.check_king() == -1 and self.game.curr_turn == -1:
            return self.reward_king
        if move.captures or move.double_captures or move.triple_captures:
            return self.reward_capture
        return self.reward_normal

    def epsilon_greedy(self, state):
        """Choose an action using epsilon-greedy strategy."""
        actions = self.game.available_actions()
        if random.uniform(0, 1) < self.epsilon:
          try:
            return random.choice(actions)  # Explore by picking a random action
          except IndexError:
            if self.game.check_game_state() == 3:
              print(self.game.board)
              print("chunkyboi")
              return False

        # For exploitation: choose the best action based on Q-values
        if state in self.q_table:
            q_values = []
            for action in actions:
              q_values.append(self.q_table[(state, action)])
            best_q_value = max(q_values)  # Find the best Q-value
            best_actions = [action for action in actions if self.q_table[(state, action)] == best_q_value]
            return random.choice(best_actions) # If there are multiple, pick randomly from the best ones
        else:
          try:
            return random.choice(actions)
          except IndexError:
            if self.game.check_game_state() == 3:
              print(self.gameboard)
              print("chunky")
              return False

    def learn(self):
        # Q-learning Algorithm (Interactive Learning)
        for episode in range(self.episodes):
            self.game.initialize_board()
            state = tuple(self.game.board.flatten())  # Start state (flattened board)
            for step in range(self.max_steps):
                action = self.epsilon_greedy(state)  # Get the chosen action
                    # if not action:
                    #   if game.check_game_state() == 3:
                    #     print("funky")
                    #     break
                    #print(game.board)
                # Get next state after taking the action
                original_board = copy.deepcopy(self.game.board)
                next_state, new_board = self.get_next_state(state, action)  # Now next_state is a tuple
                reward = self.get_reward(action)
                self.game.board = copy.deepcopy(original_board)

                # Use the (state, action) pair to get and update the Q-value
                current_q_value = self.q_table.get((state, action), 0)  # Default to 0 if no value exists
                # Correct calculation of future_q_value: no need for .values() anymore
                future_q_value = self.q_table.get((next_state, action), 0)  # Default to 0 if no value exists for the next state-action pair

                # Q-value update rule
                self.q_table[(state, action)] = current_q_value + self.learning_rate * (reward + self.discount_factor * future_q_value - current_q_value)

                state = next_state  # Update the state for the next iteration
                self.game.board = new_board

                # If the agent reaches the goal, break the loop
                if self.game.check_game_state() == 1:
                    print("aa")
                    break
                if self.game.check_game_state() == 2:
                    print("ad")
                    break
                if self.game.check_game_state() == 3:
                    print("al")
                    break

def checkers_main():
    print("Welcome to Checkers! You are the white pieces (represented by -1) and the Bot is the black pieces (represented by 1).")
    game = Checkers()

    # Training
    learner = Q_Learning(game)
    learner.learn()

    game.initialize_board()
    game.display_board()
    game.curr_turn = -1

    while True:
        # Human move
        start_row, start_col = map(int, input("Enter piece you want to move (row and column) i.e. 5 0 : ").split())
        row, col = map(int, input("Enter where you want to move your piece (row and column) i.e. 4 1 : ").split())

        start_pos = [start_row, start_col]
        end_pos = [row, col]

        available_actions = game.available_actions()

        move_made = False

        for move in available_actions:
            if move.start_pos == start_pos and move.end_pos == end_pos:
                move_made = True
                game.make_move(move)
                game.check_king()

                if move.captures or move.double_captures or move.triple_captures:
                    game.draw_counter = 0
                else:
                    game.draw_counter += 1
                break

        if not move_made:
            print("Invalid Move!")
            continue
        game.check_king()
        chosen = learner.best_move(tuple(game.board.flatten()))
        bot_move = game.make_move(chosen)
        game.check_king()
        game.display_board()

if __name__ == "__main__":
    checkers_main()

