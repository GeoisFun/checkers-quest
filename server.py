from flask import Flask, render_template, request, jsonify
import checkers  # Importing the Checkers class and Move class
import random
import numpy as np
import copy
import ast

app = Flask(__name__)

# Initialize the checkers game
game = checkers.Checkers()

# debug only
#game.board = np.array(ast.literal_eval("[[0, 1, 0, 0, 0, 0, 0, -2], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, -2, 0, 2], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, -2, 0, 2, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]"))
#print("After debug init", game.board)
# np.loadtxt("debug.txt")

@app.route("/")
def index():
    return open("index.html").read()

@app.route("/get_board")
def get_board():
    return jsonify(game.board.tolist())

@app.route("/move", methods=["POST"])
def move():
    data = request.json
    board_state = data.get("board")
    game.curr_turn = -1
    # Update the board with the provided state (from client)
    game.board = board_state
    # Determine available actions
    real_board = np.zeros((8, 8), dtype=int) 
    for x in range(8):
        for y in range(8):
            if game.board[x][y] == 0:
                real_board[x, y] = 0
            if game.board[x][y] == 1:
                real_board[x, y] = 1
            if game.board[x][y] == -1:
                real_board[x, y] = -1
            if game.board[x][y] == 2:
                real_board[x, y] = 2
            if game.board[x][y] == -2:
                real_board[x, y] = -2   
    print("Before Move: ", real_board.tolist())    
    game.board = real_board
    available_actions = game.available_actions()
    if not available_actions:
        updated_board = game.board.tolist()
        response = {"board": updated_board}
        return jsonify(response)

    # Choose a random move for now (could be replaced with AI logic)
    game.check_king()
    chosen_move = random.choice(available_actions)
    # Apply the move
    game.make_move(chosen_move)
    game.check_king()
    # Send the updated board back to the client
    updated_board = game.board.tolist()  # Convert the numpy array to a list for JSON serialization
    print("After Move: ", updated_board)    

    # Respond with the chosen move and the updated board
    response = {
        "move": {
            "start_pos": chosen_move.start_pos,
            "end_pos": chosen_move.end_pos,
            "captures": chosen_move.captures,
            "double_captures": chosen_move.double_captures,
            "triple_captures": chosen_move.triple_captures
        },
        "board": updated_board
    }

    return jsonify(response)

@app.route('/send-value', methods=['POST'])
def receive_value():
    data = request.json  # Extract JSON data
    print(data, "data")
    move_board = data.get("board") 
    print(type(move_board), "type") # Get the "value" key from JSON
    original_board = copy.deepcopy(game.board)
    game.board = np.array(move_board)
    print(game.board, "game board")
    game.curr_turn = 1
    possible_moves = game.available_actions()
    game.board = copy.deepcopy(original_board)
    all_moves = []
    for move in possible_moves:
        all_moves.append([move.start_pos, move.end_pos, move.captures, move.double_captures, move.triple_captures])
    response = {"moves": all_moves}
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
