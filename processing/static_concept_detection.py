import numpy as np
import chess
import traceback
import re

# Things that were considered as part of few-shot prompts/system prompts at one point.
def get_attacks(fen: str) -> list[dict]:
    board = chess.Board(fen)
    moves = []
    for move in board.legal_moves:
        is_capture = board.is_capture(move)
        gives_check = board.gives_check(move)
        if not is_capture and not gives_check:
            continue                       
        san = board.san(move)
        cap_piece = board.piece_at(move.to_square)
        moves.append({
            "san": san,
            "is_capture": is_capture,
            "gives_check": gives_check,
            "captured_piece": (
                chess.piece_symbol(cap_piece.piece_type).upper()
                if cap_piece else "-"
            )
        })
    return moves

def get_potential_opponent_attacks(fen: str) -> list[dict]:

    board = chess.Board(fen)
    opp_turn = not board.turn 
    board.turn = opp_turn
    opp_fen = board.fen()
    opp_attacks = get_attacks(opp_fen)  
    filtered_opp_attacks = [
    attack_dict for attack_dict in opp_attacks
    if not (attack_dict['gives_check'] is True and attack_dict['is_capture'] is False)
    ]

    return filtered_opp_attacks

def move_to_dict(fen, san: str):
    """Return a normalised dictionary that fully describes *san* relative to
    the current *board* position.

    The shape corresponds to the enriched schema agreed for the commentary
    pipeline and is designed so the CoT can perform literal equality checks
    instead of re-parsing SAN every time.
    """
    board = chess.Board(fen)
    try:
        move = board.parse_san(san)
    except ValueError:

        print(f"Warning: Could not parse SAN '{san}' for FEN '{fen}'. Returning None.")
        return None # Or raise an error, depending on desired behavior

    piece_moved_symbol = board.piece_at(move.from_square).symbol()
    piece_moved = piece_moved_symbol.upper()

    if board.is_en_passant(move):
        captured_piece = 'P'
        is_capture = True
    else:
        captured_piece_obj = board.piece_at(move.to_square)
        captured_piece = captured_piece_obj.symbol().upper() if captured_piece_obj else None
        is_capture = captured_piece is not None

    try:
        board.push(move)
        gives_check = board.is_check()
        board.pop() 
    except Exception as e:
         print(f"Error checking check status for move {san} on FEN {fen}: {e}")
         gives_check = False

    return {
        "san": san,
        "piece_moved": piece_moved,
        "is_capture": is_capture,
        "captured_piece": captured_piece,  # e.g. "P", "Q", "N", None
        "gives_check": gives_check,
    }

# --- Helper for Stockfish Evaluation (Normalized) ---
def get_eval_and_pv(stockfish_engine, current_fen: str, current_depth: int, current_turn: chess.Color):
    """
    Gets Stockfish evaluation and principal variation UCI move.
    Normalizes centipawn evaluation: positive means advantage for the player whose turn it is.
    """
    try:
        stockfish_engine.set_fen_position(current_fen)
        stockfish_engine.set_depth(current_depth)
        # Get evaluation in standard UCI format (positive = White advantage)
        evaluation = stockfish_engine.get_evaluation()

        # Normalize CP eval based on whose turn it is
        if current_turn == chess.BLACK and evaluation.get('type') == 'cp':
            evaluation['value'] *= -1 # Invert score if it's Black's turn

        # Mate scores are usually relative already
        top_moves = stockfish_engine.get_top_moves(1)
        pv_uci = top_moves[0]["Move"] if top_moves else None
        return evaluation, pv_uci
    except Exception as e:
        print(f"Stockfish error on FEN {current_fen}: {e}")
        # Return a default normalized evaluation (0 cp) on error
        return ({'type': 'cp', 'value': 0}, None)

def get_structured_engine_evaluation(
        initial_fen: str, # FEN *before* moves[0]
        moves: list[str], # Expects UCI moves: [pre_move, target_move, opponent_reply]
        stockfish_engine,
        depth: int = 20
    ) -> dict:
    """
    Analyzes a three-move sequence starting from the initial fen.
    Returns evaluations normalized to the player whose turn it is at each state.
    Does NOT perform blunder detection or counterfactual analysis.

    Args:
        initial_fen: The initial FEN string before any of the moves in `moves`.
        moves: A list of three moves in UCI format [pre_move, target_move, opponent_reply].
        stockfish_engine: An initialized Stockfish instance.
        depth: Search depth for Stockfish.

    Returns:
        A dictionary with structured analysis data for the played sequence,
        including normalized evaluations. Positive evaluation values always
        favor the player whose turn it is.
    """
    try:
        # --- Setup Boards and FENs ---
        board_state = chess.Board(initial_fen)

        # Apply pre_move to get to State 0
        pre_move_obj = chess.Move.from_uci(moves[0])
        board_state.push(pre_move_obj)
        fen_s0 = board_state.fen()
        turn_s0 = board_state.turn # Turn at state 0

        # Apply target_move to get to State 1
        target_move_uci = moves[1]
        target_move_obj = chess.Move.from_uci(target_move_uci)
        target_move_san = board_state.san(target_move_obj)
        target_move_dict = move_to_dict(fen_s0, target_move_san)
        board_state.push(target_move_obj)
        fen_s1 = board_state.fen()
        turn_s1 = board_state.turn # Turn at state 1

        # Apply opponent_reply to get to State 2
        opponent_reply_uci = moves[2]
        opponent_reply_obj = chess.Move.from_uci(opponent_reply_uci)
        opponent_reply_san = board_state.san(opponent_reply_obj)
        opponent_reply_dict = move_to_dict(fen_s1, opponent_reply_san)
        board_state.push(opponent_reply_obj)
        fen_s2 = board_state.fen()
        turn_s2 = board_state.turn # Turn at state 2

        # --- Evaluate Played Sequence (Using Normalized Evals) ---
        eval_s0, _ = get_eval_and_pv(stockfish_engine, fen_s0, depth, turn_s0) # Eval for player at S0
        eval_s2, best_move_after_s2_uci = get_eval_and_pv(stockfish_engine, fen_s2, depth, turn_s2) # Eval for player at S2

        # Get SAN for best move after S2
        best_move_after_s2_dict = None
        if best_move_after_s2_uci:
             board_temp_s2 = chess.Board(fen_s2)
             try:
                best_move_after_s2_san = board_temp_s2.san(chess.Move.from_uci(best_move_after_s2_uci))
                best_move_after_s2_dict = move_to_dict(fen_s2, best_move_after_s2_san)
             except ValueError as e:
                print(f"Error getting SAN for best move {best_move_after_s2_uci} on FEN {fen_s2}: {e}")

        # Collect mate flags
        mate_flags = {k:v for k,v in (("s0",eval_s0), ("s2",eval_s2))
                      if v.get("type")=="mate"} # Use .get() for safety

        # --- Assemble Final Result ---
        result = {
            # Played Sequence Info
            "target_fen": fen_s0, # State before target move
            "fen_state_1": fen_s1, # State after target move
            "fen_state_2": fen_s2, # State after opponent reply
            "target_move_dict": target_move_dict,
            "opponent_reply_dict": opponent_reply_dict,
            "eval_s0": eval_s0,
            "eval_s2": eval_s2,
            "best_move_after_s2_dict": best_move_after_s2_dict,
            "mate_flags": mate_flags, # Contains mate info for s0, s1, s2 if applicable
        }

        return result

    except chess.InvalidMoveError as e:
        print(f"Invalid move UCI in sequence for initial FEN {initial_fen}: {moves} - {e}")
        traceback.print_exc()
        return {"error": f"Invalid move in sequence: {e}"}
    except ValueError as e:
         print(f"ValueError during processing for FEN {initial_fen}: {e}")
         traceback.print_exc()
         return {"error": f"ValueError: {e}"}
    except Exception as e:
        print(f"Overall error in get_structured_engine_evaluation for FEN {initial_fen}: {e}")
        traceback.print_exc()
        return {"error": "Overall processing error"}
