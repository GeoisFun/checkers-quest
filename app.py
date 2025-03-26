from flask import Flask, render_template, request, jsonify
import checkers  # Importing the Checkers class and Move class
import random
import numpy as np
import copy

app = Flask("checkers")

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
    #Precursors
    data = request.json
    board_state = data.get("board")
    curr_player = data.get("player")
<<<<<<< HEAD:app.py
    #I think we need to switch current player by -1
    if curr_player == -1:
=======
    if curr_player == -1:
        game.curr_turn = -1
    else:
>>>>>>> 6abc6f9dc31bf4062358d9dfbae43f28ea66ab6b:server.py
        game.curr_turn = 1
    else:
        game.curr_turn = -1
    print("curplayer", curr_player)

    # Update the board from list to Numpy (Since Javascript only recognizes lists)
    game.board = np.array(board_state)

    # Determine available actions
    available_actions = game.available_actions()

    # If no possible moves, send the board right back
    if not available_actions:
        response = {"board": board_state}
        return jsonify(response)

    # Choose a random move for now (could be replaced with AI logic)
    game.check_king()
    chosen_move = random.choice(available_actions)

    # Apply the move
    game.make_move(chosen_move)
    game.check_king()
    print("after move made, to check if move has been switched")
    # Send the updated board back to the client
    updated_board = game.board.tolist()  # Convert the numpy array to a list for JSON serialization
    #DEBUG:  print("After Move: ", updated_board)    

    # Respond with the chosen move and the updated board
    response = {
        "move": {"start_pos": chosen_move.start_pos, "end_pos": chosen_move.end_pos, "captures": chosen_move.captures, "double_captures": chosen_move.double_captures, "triple_captures": chosen_move.triple_captures},
        "board": updated_board
    }

    return jsonify(response)

@app.route('/send-value', methods=['POST'])
def receive_value():
    #Precursors
    data = request.json
    move_board = data.get("board")
    #DEBUG: print(type(move_board), "type")

    #To Find Available Moves without having to code the function into Index.Html
    original_board = copy.deepcopy(game.board)
    game.board = np.array(move_board)
    #DEBUG: print(game.board, "game board")
    game.curr_turn = -1
    available_moves = game.available_actions()
    game.board = copy.deepcopy(original_board)

    all_moves = []
    for move in available_moves:
        all_moves.append([move.start_pos, move.end_pos, move.captures, move.double_captures, move.triple_captures])
    response = {"moves": all_moves}

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)