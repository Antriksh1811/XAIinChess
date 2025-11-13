# --- START OF FILE concept_generation.py ---
import io
import os
import traceback
import uuid # Needed for batched saving
from typing import Tuple, Dict, List, Optional

import chess
import chess.pgn
import numpy as np
import onnxruntime as rt

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset

from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from processing.leela_board import LeelaBoard
from processing.models import Stockfish


# === Global Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using PyTorch device: {device}")

stockfish_8_path =  "Stockfish-8/Windows/stockfish_8_x64.exe"


def process_lichess_puzzles(csv_path: str, target_tag: str, save_path_base: str, limit=5000):

    print(f"Processing Lichess Puzzles for tag: '{target_tag}' (Standard Negative Sampling)")
    pos_boards, pos_fens = [], []
    neg_boards, neg_fens = [], []

    try:
        with open(csv_path, 'r') as f:
            f.readline()
            with tqdm(total=limit*2, desc=f"Finding '{target_tag}' examples") as pbar:
                processed_count = 0
                while processed_count < limit*2 :
                    line = f.readline()
                    if not line: break

                    if len(pos_boards) >= limit and len(neg_boards) >= limit:
                        break
                    try:
                        parts = line.strip().split(',')
                        _, fen, moves_str, _, _, _, _, tags_str, _, _ = parts
                        tags = set(tags_str.split())
                        is_positive = target_tag in tags

                        first_move_uci = moves_str.split()[0]
                        board = chess.Board(fen)

                        try:
                            move = chess.Move.from_uci(first_move_uci)
                            if move not in board.legal_moves: continue
                            board.push(move) 
                        except ValueError:
                            continue 

                        leela_board = LeelaBoard(fen=board.fen())
                        leela_board._lcz_push()
                        features = leela_board.lcz_features()

                        # --- Add to lists ---
                        if is_positive and len(pos_boards) < limit:
                            pos_boards.append(features)
                            pos_fens.append(board.fen())
                            pbar.update(1)
                            processed_count += 1
                        elif not is_positive and len(neg_boards) < limit:
                            neg_boards.append(features)
                            neg_fens.append(board.fen())
                            pbar.update(1)
                            processed_count += 1

                    except KeyboardInterrupt:
                         print("Keyboard interrupt during PGN processing.")
                         raise
                    except Exception:
                        continue 

        print(f"Found {len(pos_boards)} positive and {len(neg_boards)} negative examples.")

        if len(pos_boards) == 0 or len(neg_boards) == 0:
             print(f"Warning: No examples found for at least one class for '{target_tag}'. Skipping saving.")
             return False

        pos_boards = np.array(pos_boards, dtype=np.float16)
        neg_boards = np.array(neg_boards, dtype=np.float16)
        pos_fens_np = np.array(pos_fens)
        neg_fens_np = np.array(neg_fens)

        os.makedirs(os.path.dirname(save_path_base), exist_ok=True)
        pos_save_path = f"{save_path_base}_pos.npz"
        np.savez(pos_save_path, boards=pos_boards, fens=pos_fens_np)
        print(f"Saved {len(pos_boards)} positive examples to {pos_save_path}")
        neg_save_path = f"{save_path_base}_neg.npz"
        np.savez(neg_save_path, boards=neg_boards, fens=neg_fens_np)
        print(f"Saved {len(neg_boards)} negative examples to {neg_save_path}")
        return True

    except FileNotFoundError:
        print(f"Error: Lichess puzzle file not found at {csv_path}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during Lichess processing: {e}")
        traceback.print_exc()
        return False

def extract_and_save_activations_batched(
    model_path: str,
    npz_data_path: str,
    target_layer_indices: List[int],
    output_dir: str,
    output_prefix: str,
    max_samples: Optional[int] = None,
    chunk_size: int = 5000
    ):

    print(f"  Model: {model_path}")
    print(f"  Data: {npz_data_path}")
    print(f"  Target Layers: {target_layer_indices}")
    print(f"  Output Dir: {output_dir}")
    print(f"  Output Prefix: {output_prefix}")
    print(f"  Max Samples: {max_samples}")
    print(f"  Chunk Size: {chunk_size}")

    os.makedirs(output_dir, exist_ok=True)
    temp_chunk_dir = os.path.join(output_dir, f"__temp_chunks_{output_prefix}_{uuid.uuid4().hex[:8]}")
    os.makedirs(temp_chunk_dir, exist_ok=True)
    print(f"  Temporary chunk dir: {temp_chunk_dir}")

    temp_files_per_layer = {idx: [] for idx in target_layer_indices}
    final_shapes = {} 
    success = True

    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        sess_options = rt.SessionOptions()
        sess_options.intra_op_num_threads = max(1, os.cpu_count() // 4) 
        sess_options.inter_op_num_threads = max(1, os.cpu_count() // 4) 
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = rt.InferenceSession(model_path, sess_options, providers=providers)
        actual_provider = sess.get_providers()[0]
        print(f"  Using ONNX provider: {actual_provider}")
        input_name = sess.get_inputs()[0].name
        all_output_names = [o.name for o in sess.get_outputs()]
        num_intermediate_layers_available = len(all_output_names) - 2 

        valid_target_indices = []
        target_output_names_map = {}
        for idx in target_layer_indices:

            onnx_output_idx = idx + 2
            if 0 <= idx < num_intermediate_layers_available:
                valid_target_indices.append(idx)
                target_output_names_map[idx] = all_output_names[onnx_output_idx]
                print(f"  Mapping target layer index {idx} to ONNX output '{all_output_names[onnx_output_idx]}'")
            else:
                print(f"  Warning: Target layer index {idx} out of bounds (0 to {num_intermediate_layers_available-1}). Skipping.")

        if not valid_target_indices:
            print("  Error: No valid target layers selected.")
            return False
        outputs_to_run = list(target_output_names_map.values()) 

        print(f"  Checking data source: {npz_data_path}...")
        try:
            with np.load(npz_data_path) as data:
                 if 'boards' not in data:
                      print(f"  Error: 'boards' key not found in {npz_data_path}")
                      return False
                 num_available = data['boards'].shape[0]

        except Exception as e:
             print(f"  Error loading or checking npz file {npz_data_path}: {e}")
             return False

        if num_available == 0:
            print("  Error: Input NPZ file contains no samples.")
            return False

        points_to_process = min(max_samples, num_available) if max_samples is not None else num_available
        print(f"  Planning to process {points_to_process} samples out of {num_available} available.")

        BATCH_SIZE = 128 # Adjust based on GPU memory
        current_chunk_activations = {idx: [] for idx in valid_target_indices}
        samples_processed_in_chunk = 0
        total_samples_processed = 0

        print("  Running ONNX inference and saving chunks...")
        with tqdm(total=points_to_process, desc="ONNX Batches") as pbar:
             indices_to_process = range(points_to_process)
             for i in range(0, points_to_process, BATCH_SIZE):
                batch_indices = indices_to_process[i:min(i + BATCH_SIZE, points_to_process)]
                if not len(batch_indices): continue

                try:
                    with np.load(npz_data_path) as data:
                        batch_data = data['boards'][batch_indices]
                except Exception as e:
                    print(f"\nError loading batch {i//BATCH_SIZE} from {npz_data_path}: {e}")
                    success = False; break

                batch_data_float16 = batch_data.astype(np.float16)

                try:
                     pred = sess.run(outputs_to_run, {input_name: batch_data_float16})
                except Exception as e:
                     print(f"\nError during ONNX inference for batch {i//BATCH_SIZE}: {e}")
                     success = False; break

                for target_idx, onnx_output_name in target_output_names_map.items():
                    output_index_in_pred = outputs_to_run.index(onnx_output_name)
                    raw_activation_batch = pred[output_index_in_pred] 
                    current_chunk_activations[target_idx].append(raw_activation_batch)

                samples_in_batch = len(batch_indices)
                samples_processed_in_chunk += samples_in_batch
                total_samples_processed += samples_in_batch
                pbar.update(samples_in_batch)

                is_last_iteration = (total_samples_processed >= points_to_process)
                if samples_processed_in_chunk >= chunk_size or is_last_iteration:
                    for idx in valid_target_indices:
                        if not current_chunk_activations[idx]: continue 

                        try:
                            chunk_concat = np.concatenate(current_chunk_activations[idx], axis=0)

                            if chunk_concat.ndim < 3: 
                                 print(f"\nWarning: Layer {idx} activation chunk has unexpected low dimension {chunk_concat.ndim} when saving unflattened. Shape: {chunk_concat.shape}")
                            chunk_to_save = chunk_concat

                            temp_filename = f"chunk_{idx}_{uuid.uuid4().hex}.npy"
                            temp_filepath = os.path.join(temp_chunk_dir, temp_filename)
                            np.save(temp_filepath, chunk_to_save.astype(np.float32))
                            temp_files_per_layer[idx].append(temp_filepath)

                        except Exception as e:
                            print(f"\nError processing or saving chunk for layer {idx}: {e}")
                            success = False

                    # Reset for next chunk
                    current_chunk_activations = {idx: [] for idx in valid_target_indices}
                    samples_processed_in_chunk = 0
                    if not success: break 

        if success:
            print("\n  Combining temporary chunk files...")
            for idx in valid_target_indices:
                chunk_files = temp_files_per_layer[idx]
                if not chunk_files:
                    print(f"    Layer {idx}: No chunk files found to combine.")
                    continue

                print(f"    Layer {idx}: Combining {len(chunk_files)} chunk files...")
                all_chunks_data = []
                try:
                    for chunk_file in chunk_files:
                        all_chunks_data.append(np.load(chunk_file, mmap_mode='r'))

                    final_layer_activations = np.concatenate(all_chunks_data, axis=0)
                    final_shapes[idx] = final_layer_activations.shape
                    final_save_filename = f"{output_prefix}_layer_{idx}_activations.npy"
                    final_save_path = os.path.join(output_dir, final_save_filename)

                    np.save(final_save_path, final_layer_activations.astype(np.float16))
                    print(f"    Layer {idx}: Saved final combined activations to {final_save_path} (Shape: {final_shapes[idx]}, Dtype: float16)")

                except Exception as e:
                    print(f"\nError combining or saving final file for layer {idx}: {e}")
                    success = False
                finally:
                     del all_chunks_data[:]
                     del all_chunks_data

        print(f"\n  Cleaning up temporary directory: {temp_chunk_dir}")
        try:
            for idx in temp_files_per_layer:
                 for temp_file in temp_files_per_layer[idx]:
                     if os.path.exists(temp_file):
                         os.remove(temp_file)
            if os.path.exists(temp_chunk_dir):
                os.rmdir(temp_chunk_dir)
            print("  Cleanup successful.")
        except Exception as e:
            print(f"  Warning: Error during temporary file cleanup: {e}")
            print(f"  Please manually delete directory: {temp_chunk_dir}")


        print(f"--- Activation Extraction Finished (Success: {success}) ---")
        return success, final_shapes

    except Exception as e:
        print(f"An unexpected error occurred during activation setup or processing: {e}")
        traceback.print_exc()
        if os.path.exists(temp_chunk_dir):
             try:
                 for idx in temp_files_per_layer:
                     for temp_file in temp_files_per_layer[idx]:
                         if os.path.exists(temp_file): os.remove(temp_file)
                 os.rmdir(temp_chunk_dir)
             except Exception as cleanup_e:
                 print(f"  Warning: Error during cleanup after setup failure: {cleanup_e}")
        return False, {}

class ConvProbe(nn.Module):

    def __init__(self, input_channels: int, hidden_channels: int = 32, dropout_prob: float = 0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding='same')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding='same')
        self.relu2 = nn.ReLU()
        hidden_channels = hidden_channels * 2 

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten() 
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(hidden_channels, 1) 

    def forward(self, x):
   
        x = x.float() 

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.relu2(self.conv2(x))

        x = self.pool(x)     
        x = self.flatten(x)   
        x = self.dropout(x)
        x = self.linear(x)    
        return x

def binary_accuracy_with_guessing_pytorch(y_pred_logits, y_true):
    if y_pred_logits.numel() == 0 or y_true.numel() == 0:
        return torch.tensor(0.0)
    with torch.no_grad():
        y_true_float = y_true.float()
        y_pred_prob = torch.sigmoid(y_pred_logits)
        y_pred_binary = (y_pred_prob > 0.5).float()
        correct = (y_pred_binary == y_true_float).float()
        accuracy = correct.mean()
        corrected_accuracy = (accuracy - 0.5) * 2.0
        return torch.clamp(corrected_accuracy, min=0.0, max=1.0)

    
def perform_concept_detection_pytorch_probe(
    target_layer_indices: List[int],
    concept_name: str,
    activations_dir: str,
    probe_save_dir: str,
    conv_hidden_channels: int,
    dropout_prob: float,
    test_ratio: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    patience: int,
    l2_weight_decays: List[float],
    random_seed: int 
    ) -> Dict:

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    layer_results = {}
    layer_test_metrics = {}

    regularization_params = l2_weight_decays
    reg_param_name = "L2 weight_decay"

    for layer_idx in target_layer_indices:
        print(f"\n--- Processing Layer Index: {layer_idx} for Concept: {concept_name} ---")
        pos_acts_filename = f"{concept_name}_pos_layer_{layer_idx}_activations.npy"
        neg_acts_filename = f"{concept_name}_neg_layer_{layer_idx}_activations.npy"
        pos_acts_path = os.path.join(activations_dir, pos_acts_filename)
        neg_acts_path = os.path.join(activations_dir, neg_acts_filename)

        try:
            raw_pos_acts = np.load(pos_acts_path).astype(np.float32)
            raw_neg_acts = np.load(neg_acts_path).astype(np.float32)
            num_pos = raw_pos_acts.shape[0]
            num_neg = raw_neg_acts.shape[0]

            if num_pos == 0 or num_neg == 0:
                 print(f"  Skipping layer {layer_idx}: Zero samples.")
                 layer_results[layer_idx] = {'best_avg_val_accuracy': np.nan}
                 layer_test_metrics[layer_idx] = None
                 continue

            activation_shape = raw_pos_acts.shape[1:]
            input_channels = activation_shape[0]
            labels_np = np.array([1] * num_pos + [0] * num_neg, dtype=np.float32)
            combined_acts_np = np.concatenate([raw_pos_acts, raw_neg_acts], axis=0)

            del raw_pos_acts, raw_neg_acts

        except Exception as e:
            print(f"Error preparing data for layer {layer_idx}: {e}")
            layer_results[layer_idx] = {'best_avg_val_accuracy': np.nan}
            layer_test_metrics[layer_idx] = None
            continue

        best_layer_avg_val_acc = -1.0
        best_probe_state_for_layer = None
        best_reg_param_for_layer = None
        best_layer_input_shape = None
        best_layer_input_channels = None
        
        loss_fn = nn.BCEWithLogitsLoss()

        indices = range(len(combined_acts_np))
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_ratio, random_state=random_seed, stratify=labels_np
        )

        for reg_param in regularization_params:
            fold_val_accuracies = []
            mapped_fold_splits = []

            original_train_val_indices = np.array(train_val_indices)

            train_fold_indices_rel, val_fold_indices_rel = train_test_split(
                range(len(train_val_indices)), test_size=test_ratio,
                random_state=random_seed + 1, stratify=labels_np[train_val_indices]
            )
            train_fold_indices = original_train_val_indices[train_fold_indices_rel]
            val_fold_indices = original_train_val_indices[val_fold_indices_rel]
            mapped_fold_splits = [(train_fold_indices, val_fold_indices)]

            current_reg_best_val_acc = -1.0
            current_reg_best_state = None

            for fold_idx, (train_indices_fold, val_indices_fold) in enumerate(mapped_fold_splits):
                x_train_fold_np = combined_acts_np[train_indices_fold]
                y_train_fold_np = labels_np[train_indices_fold]
                x_val_fold_np = combined_acts_np[val_indices_fold]
                y_val_fold_np = labels_np[val_indices_fold]

                try:
                    probe = ConvProbe(input_channels, hidden_channels=conv_hidden_channels, dropout_prob=dropout_prob).to(device)
                    optimizer = optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=reg_param)
                    current_l2_decay = reg_param
                        
                except Exception as model_init_e:
                    print(f"Error initializing probe model: {model_init_e}")
                    break

                x_train_fold = torch.from_numpy(x_train_fold_np).to(device)
                y_train_fold = torch.from_numpy(y_train_fold_np).unsqueeze(1).to(device)
                x_val_fold = torch.from_numpy(x_val_fold_np).to(device)
                y_val_fold = torch.from_numpy(y_val_fold_np).unsqueeze(1).to(device)

                train_dataset_fold = TensorDataset(x_train_fold, y_train_fold)
                val_dataset_fold = TensorDataset(x_val_fold, y_val_fold)
                train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True, drop_last=True)
                val_loader_fold = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False)

                best_fold_val_acc = -1.0
                best_fold_state_dict = None
                patience_counter = 0

                for epoch in range(epochs):
                    probe.train()
                    pbar_desc = f"L{layer_idx} Reg{reg_param:.0e} F{fold_idx+1} E{epoch+1}"
                    pbar_train = tqdm(train_loader_fold, desc=pbar_desc, leave=False)
                    for batch_acts, batch_labels in pbar_train:
                        optimizer.zero_grad()
                        outputs = probe(batch_acts)
                        loss = loss_fn(outputs, batch_labels)
                        total_loss = loss 
                        total_loss.backward()
                        optimizer.step()
                        pbar_train.set_postfix({"Loss": total_loss.item()})

                    probe.eval()
                    current_epoch_val_acc = np.nan
                    all_preds_logits_eval = []
                    all_true_labels_eval = []
                    if len(val_loader_fold) > 0:
                        with torch.no_grad():
                             for batch_acts_eval, batch_labels_eval in val_loader_fold:
                                 outputs_eval = probe(batch_acts_eval)
                                 all_preds_logits_eval.append(outputs_eval.cpu())
                                 all_true_labels_eval.append(batch_labels_eval.cpu())
                        if all_preds_logits_eval:
                             test_preds_eval = torch.cat(all_preds_logits_eval)
                             test_labels_eval = torch.cat(all_true_labels_eval)
                             if test_preds_eval.ndim > 1 and test_preds_eval.shape[1] == 1: test_preds_eval = test_preds_eval.squeeze(1)
                             if test_labels_eval.ndim > 1 and test_labels_eval.shape[1] == 1: test_labels_eval = test_labels_eval.squeeze(1)
                             current_epoch_val_acc = binary_accuracy_with_guessing_pytorch(test_preds_eval, test_labels_eval).item()

                    if not np.isnan(current_epoch_val_acc):
                        if current_epoch_val_acc > best_fold_val_acc:
                            best_fold_val_acc = current_epoch_val_acc
                            best_fold_state_dict = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= patience: break

                if best_fold_state_dict is not None:
                    fold_val_accuracies.append(best_fold_val_acc)
                    if best_fold_val_acc > current_reg_best_val_acc:
                         current_reg_best_val_acc = best_fold_val_acc
                         current_reg_best_state = best_fold_state_dict
                else:
                     fold_val_accuracies.append(np.nan)

                del x_train_fold, y_train_fold, x_val_fold, y_val_fold, train_dataset_fold, val_dataset_fold, train_loader_fold, val_loader_fold
                if 'probe' in locals(): del probe
                if 'optimizer' in locals(): del optimizer
                torch.cuda.empty_cache() if device.type == 'cuda' else None

            valid_fold_accs = [acc for acc in fold_val_accuracies if not np.isnan(acc)]
            avg_val_acc_for_reg = np.mean(valid_fold_accs) if valid_fold_accs else np.nan

            if not np.isnan(avg_val_acc_for_reg) and avg_val_acc_for_reg > best_layer_avg_val_acc:
                 best_layer_avg_val_acc = avg_val_acc_for_reg
                 best_probe_state_for_layer = current_reg_best_state
                 best_reg_param_for_layer = reg_param
                 best_layer_input_shape = activation_shape
                 best_layer_input_channels = input_channels


        layer_results[layer_idx] = {
            'best_avg_val_accuracy': best_layer_avg_val_acc if best_layer_avg_val_acc >= 0 else np.nan,
            'best_reg_param': best_reg_param_for_layer,
            'reg_param_name': reg_param_name
        }

        if best_probe_state_for_layer is not None:
            print(f"  Evaluating Best Probe for Layer {layer_idx} on Test Set...")
            try: 
                probe_final = ConvProbe(best_layer_input_channels, hidden_channels=conv_hidden_channels, dropout_prob=dropout_prob).to(device)

                if probe_final:
                    probe_final.load_state_dict(best_probe_state_for_layer)
                    probe_final.eval()

                    x_test_np = combined_acts_np[test_indices]
                    y_test_np = labels_np[test_indices]
                    x_test = torch.from_numpy(x_test_np).to(device)
                    y_test_true = torch.from_numpy(y_test_np).to(device)

                    test_dataset = TensorDataset(x_test, y_test_true.unsqueeze(1))
                    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False)

                    all_test_preds_logits = []
                    all_test_true_labels = []
                    with torch.no_grad():
                        for batch_acts_test, batch_labels_test in test_loader:
                            outputs_test = probe_final(batch_acts_test)
                            all_test_preds_logits.append(outputs_test.cpu())
                            all_test_true_labels.append(batch_labels_test.cpu())

                    final_test_preds_logits = torch.cat(all_test_preds_logits)
                    final_test_true = torch.cat(all_test_true_labels).squeeze()
                    final_test_preds_binary = (torch.sigmoid(final_test_preds_logits) > 0.5).squeeze().int().numpy()
                    final_test_true_numpy = final_test_true.int().numpy()

                    test_acc = accuracy_score(final_test_true_numpy, final_test_preds_binary)
                    test_prec = precision_score(final_test_true_numpy, final_test_preds_binary, zero_division=0)
                    test_recall = recall_score(final_test_true_numpy, final_test_preds_binary, zero_division=0)
                    test_f1 = f1_score(final_test_true_numpy, final_test_preds_binary, zero_division=0)
                    test_cm = confusion_matrix(final_test_true_numpy, final_test_preds_binary)
                    tn, fp, fn, tp = test_cm.ravel() if test_cm.shape == (2, 2) else (0, 0, 0, 0)
                    test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                    print(f"  Test Acc: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_recall:.4f}, Specificity: {test_specificity:.4f}, F1: {test_f1:.4f}")

                    layer_test_metrics[layer_idx] = {
                        'accuracy': test_acc, 'precision': test_prec, 'recall': test_recall,
                        'specificity': test_specificity, 'f1_score': test_f1,
                        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
                    }

                    os.makedirs(probe_save_dir, exist_ok=True)
                    probe_save_filename = f"{concept_name}_layer_{layer_idx}_probe.pth"
                    probe_save_path = os.path.join(probe_save_dir, probe_save_filename)
                    save_data = {
                        'state_dict': best_probe_state_for_layer,'concept_name': concept_name, 'layer_index': layer_idx, 
                        'activation_shape': best_layer_input_shape,'input_channels': best_layer_input_channels,
                        'conv_hidden_channels': conv_hidden_channels,'dropout_prob': dropout_prob,
                        'best_avg_val_accuracy': best_layer_avg_val_acc,'best_reg_param': best_reg_param_for_layer,
                        'best_reg_param_name': reg_param_name, 'test_metrics': layer_test_metrics[layer_idx]
                    }
                    torch.save(save_data, probe_save_path)
                    print(f"  Saved best probe data to {probe_save_path}")

                    del probe_final, x_test, y_test_true, test_dataset, test_loader
                else:
                     layer_test_metrics[layer_idx] = None

            except Exception as final_eval_e:
                 print(f"Error during final test evaluation for layer {layer_idx}: {final_eval_e}")
                 traceback.print_exc()
                 layer_test_metrics[layer_idx] = None
        else:
             layer_test_metrics[layer_idx] = None

    final_results = {}
    for idx in target_layer_indices:
        final_results[idx] = {
            "validation_results": layer_results.get(idx, {'best_avg_val_accuracy': np.nan}),
            "test_metrics": layer_test_metrics.get(idx, None)
        }

    return final_results

# Function required for commentary generation, anything above this is in the training process.
def analyze_puzzle_concept_importance(
    puzzle_fen: str,
    puzzle_moves_uci: List[str],
    onnx_model_path: str,
    probes_dir: str,
    concept_names: List[str],
    target_layer_index: int,
    activation_threshold: float = 0.85 
    ) -> Optional[Dict]:
    

    print(f"\n--- Analyzing Puzzle Concept Importance ---")
    print(f"  Initial FEN: {puzzle_fen}")
    print(f"  Moves: {puzzle_moves_uci}")
    print(f"  Target Layer: {target_layer_index}")
    print(f"  Probes Dir: {probes_dir}")
    print(f"  Activation Threshold: {activation_threshold}")

    if len(puzzle_moves_uci) < 3:
        print("  Error: Need at least 3 moves in puzzle_moves_uci for S0 and S2 analysis.")
        return None

    probes = {}
    probe_metadata = {} 
    loaded_concepts = []
    print("  Loading probes...")
    for concept_name in concept_names:
        probe_filename = f"{concept_name}_layer_{target_layer_index}_probe.pth"
        probe_path = os.path.join(probes_dir, probe_filename)
        if not os.path.exists(probe_path):
            continue
        try:
            probe_data = torch.load(probe_path, map_location='cpu', weights_only=False)
            input_channels = probe_data.get('input_channels')
            hidden_channels = probe_data.get('conv_hidden_channels')
            dropout_prob = probe_data.get('dropout_prob')
            state_dict = probe_data.get('state_dict')

            if not all([input_channels, hidden_channels is not None, dropout_prob is not None, state_dict]):
                 print(f"    Warning: Incomplete probe data for '{concept_name}', skipping.")
                 continue

            probe = ConvProbe(input_channels, hidden_channels, dropout_prob)
            state_dict_device = {k: v.to(device) for k, v in state_dict.items()}
            probe.load_state_dict(state_dict_device)
            probe.to(device) 
            probe.eval()
            probes[concept_name] = probe
            probe_metadata[concept_name] = probe_data 
            loaded_concepts.append(concept_name)

        except Exception as e:
            print(f"    Error loading probe for concept '{concept_name}': {e}")
            traceback.print_exc()

    if not probes:
        print("  Error: No probes loaded successfully.")
        return None
    print(f"  Successfully loaded probes for concepts: {loaded_concepts}")

    onnx_sess = None
    intermediate_output_name = None
    input_name = None
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        sess_options = rt.SessionOptions()
        sess_options.intra_op_num_threads = 1 # Optimize for single run
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        onnx_sess = rt.InferenceSession(onnx_model_path, sess_options, providers=providers)
        actual_provider = onnx_sess.get_providers()[0]
        print(f"  Using ONNX provider: {actual_provider}")

        input_name = onnx_sess.get_inputs()[0].name
        all_output_nodes = onnx_sess.get_outputs()
        all_output_names = [o.name for o in all_output_nodes]

        # Assuming intermediate layers start after policy (0) and value (1) outputs
        if target_layer_index + 2 >= len(all_output_names):
             print(f"  Error: target_layer_index {target_layer_index} is out of bounds for available outputs.")
             return None
        intermediate_output_name = all_output_names[target_layer_index + 2] # Adjust index offset if needed

        print(f"  Targeting ONNX output node: '{intermediate_output_name}' for layer index {target_layer_index}")

    except Exception as e:
        print(f"  Error setting up ONNX session: {e}")
        traceback.print_exc()
        return None

    def get_activations_for_fen(fen: str) -> Optional[torch.Tensor]:
        nonlocal onnx_sess, input_name, intermediate_output_name
        if not onnx_sess or not input_name or not intermediate_output_name: return None
        try:
            leela_board = LeelaBoard(fen=fen)
            leela_board._lcz_push()
            features_np = leela_board.lcz_features()
            features_batch_np = np.expand_dims(features_np, axis=0)
            input_data = {input_name: features_batch_np.astype(np.float16)}

            onnx_outputs = onnx_sess.run([intermediate_output_name], input_data)
            activations_np = onnx_outputs[0]

            activations_tensor = torch.from_numpy(activations_np).to(device)
            return activations_tensor.float() 

        except Exception as e:
            print(f"    Error getting activations for FEN {fen}: {e}")
            return None

    board = chess.Board(puzzle_fen)
    move0 = chess.Move.from_uci(puzzle_moves_uci[0])
    board.push(move0)
    fen_s0 = board.fen()
    print(f"  Getting activations for State 0 (FEN: {fen_s0})...")
    activations_s0 = get_activations_for_fen(fen_s0)

    move1 = chess.Move.from_uci(puzzle_moves_uci[1])
    board.push(move1)

    move2 = chess.Move.from_uci(puzzle_moves_uci[2])
    board.push(move2) # Apply Opponent's reply
    fen_s2 = board.fen()
    print(f"  Getting activations for State 2 (FEN: {fen_s2})...")
    activations_s2 = get_activations_for_fen(fen_s2)

    concept_scores = {}
    relevant_concepts_s0 = [] 
    relevant_concepts_s2 = [] 
    print("  Running probes and checking threshold...")
    with torch.no_grad():
        for concept_name, probe in probes.items():
            score_s0 = 0.0
            score_s2 = 0.0
            try:
                output_s0_logits = probe(activations_s0)
                score_s0 = torch.sigmoid(output_s0_logits).item()

                output_s2_logits = probe(activations_s2)
                score_s2 = torch.sigmoid(output_s2_logits).item()

                concept_scores[concept_name] = {
                    'score_s0': round(score_s0, 4),
                    'score_s2': round(score_s2, 4),
                }

                if score_s0 >= activation_threshold:
                    relevant_concepts_s0.append(concept_name)

                if score_s2 >= activation_threshold:
                    relevant_concepts_s2.append(concept_name)

            except Exception as e:
                print(f"    Error running probe for concept '{concept_name}': {e}")

    del activations_s0, activations_s2
    if device.type == 'cuda': torch.cuda.empty_cache()

    print(f"  Concepts active in S0 (threshold >= {activation_threshold}): {relevant_concepts_s0}")
    print(f"  Concepts active in S2 (threshold >= {activation_threshold}): {relevant_concepts_s2}")
    print("  Analysis complete.")

    results = {
        'concept_scores': concept_scores,       
        'probes_active_before_player_move': relevant_concepts_s0, 
        'probes_active_before_followup': relevant_concepts_s2   
    }

    return results

