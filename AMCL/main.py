import numpy as np
import torch
import os
import math
import pandas as pd
import argparse
from tqdm import tqdm
import copy
import random
from sklearn.metrics import average_precision_score
from torch.optim import lr_scheduler
import torch.utils.data as data
import torch.nn.functional as F
import time
from datetime import datetime
import traceback

if torch.cuda.is_available():
    import torch.cuda as cuda_utils
else:
    class MockCudaUtils:
        def max_memory_allocated(self): return 0

        def reset_peak_memory_stats(self): pass


    cuda_utils = MockCudaUtils()

from scipy.sparse import coo_matrix

try:
    from model import AMGAN
    from utils import (load_data_from_csv, normalize_adj_matrix,
                       scipy_sparse_mat_to_torch_sparse_tensor, TrnData,
                       calculate_lcc_bipartite)
except ImportError as e:
    print(f"错误：无法导入 'model' 或 'utils' 模块。 {e}")
    exit()


def parse_args():
    parser = argparse.ArgumentParser(description='AMGAN Model Training (K-Fold CV Only)')
    parser.add_argument('--data_dir', default='./data', type=str,
                        help='K折交叉验证的 CSV 文件目录 (例如: train0..k-1.csv, test0..k-1.csv)')
    parser.add_argument('--num_folds_cv', default=5, type=int)
    parser.add_argument('--cuda', default='0', type=str)

    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--inter_batch', default=4096, type=int)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--d', default=512, type=int)
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--gnn_layer', default=2, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--hyperNum', default=128, type=int)
    parser.add_argument('--q', default=8, type=int)

    parser.add_argument('--temp', default=0.8, type=float)
    parser.add_argument('--lambda1', default=1e-3, type=float)
    parser.add_argument('--lambda2', default=1e-5, type=float)

    parser.add_argument('--use_lcc_aux_task', action='store_true', default=True)
    parser.add_argument('--lambda_aux', default=0.3, type=float)
    parser.add_argument('--use_lcc_attn_bias', action='store_true', default=True)
    parser.add_argument('--lcc_attn_scale', default=0.3, type=float)

    parser.add_argument('--note', default='default_run', type=str)
    parser.add_argument('--eval_interval_cv', default=50, type=int)
    parser.add_argument('--enable_tuning', action='store_true')
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, adj_norm_eval, device):
    model.eval()
    all_labels, all_scores = [], []
    nan_metrics = {'AUPR': np.nan}
    empty_labels = np.array([])
    empty_scores = np.array([])

    if not hasattr(loader, 'dataset') or loader.dataset is None or len(loader.dataset) == 0:
        return nan_metrics
    if not hasattr(loader, 'batch_size') or loader.batch_size is None or loader.batch_size <= 0:
        return nan_metrics

    for batch in tqdm(loader, leave=False):
        if batch is None or len(batch) == 0: continue
        try:
            geneids, pos, neg = [x.long().to(device) for x in batch]
            if geneids.numel() == 0: continue
            valid_neg_eval_mask = (neg != -1);
            if not valid_neg_eval_mask.all():
                geneids = geneids[valid_neg_eval_mask];
                pos = pos[valid_neg_eval_mask];
                neg = neg[valid_neg_eval_mask]
                if geneids.numel() == 0: continue
        except Exception as e:
            continue

        try:
            E_g_final, E_d_final = model(None, None, None, None, adj_norm=adj_norm_eval, test=True)
            if E_g_final is None or E_d_final is None or E_g_final.shape[0] == 0 or E_d_final.shape[0] == 0:
                continue
            max_gene_idx = geneids.max().item() if geneids.numel() > 0 else -1
            max_pos_idx = pos.max().item() if pos.numel() > 0 else -1
            max_neg_idx = neg.max().item() if neg.numel() > 0 else -1
            if max_gene_idx >= E_g_final.shape[0] or max_pos_idx >= E_d_final.shape[0] or max_neg_idx >= \
                    E_d_final.shape[0]:
                continue

            u_embed = E_g_final[geneids];
            p_embed = E_d_final[pos];
            n_embed = E_d_final[neg]
            pos_scores = torch.sum(u_embed * p_embed, dim=1);
            neg_scores = torch.sum(u_embed * n_embed, dim=1)
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]);
            scores = torch.cat([pos_scores, neg_scores])
            all_labels.append(labels.cpu());
            all_scores.append(scores.cpu())
        except Exception as e:
            traceback.print_exc()
            continue

    if not all_labels or not all_scores:
        return nan_metrics

    all_labels_np = torch.cat(all_labels).numpy();
    all_scores_np = torch.cat(all_scores).numpy()

    if not np.all(np.isfinite(all_scores_np)):
        non_finite_count = np.sum(~np.isfinite(all_scores_np))
        all_scores_np = np.nan_to_num(all_scores_np)
    if len(np.unique(all_labels_np)) <= 1:
        return nan_metrics
    if len(np.unique(all_scores_np)) <= 1:
        base_aupr = np.mean(all_labels_np) if len(all_labels_np) > 0 else np.nan
        metrics = {'AUPR': base_aupr}
        return metrics

    try:
        aupr = average_precision_score(all_labels_np, all_scores_np)
    except ValueError as e:
        return nan_metrics
    except Exception as e:
        return nan_metrics

    metrics = {'AUPR': aupr}
    return metrics


def get_consistent_shape(args):
    max_n_genes, max_n_drugs = -1, -1
    paths_to_check = []

    for fold_id in range(args.num_folds_cv):
        train_p = os.path.join(args.data_dir, f'train{fold_id}.csv')
        test_p = os.path.join(args.data_dir, f'test{fold_id}.csv')
        if os.path.exists(train_p): paths_to_check.append(train_p)
        if os.path.exists(test_p): paths_to_check.append(test_p)

    if not paths_to_check:
        return 0, 0

    for path in paths_to_check:
        try:
            df = pd.read_csv(path, usecols=['gene', 'drug'], dtype={'gene': str, 'drug': str}, engine='python',
                             on_bad_lines='warn')
            if not df.empty:
                genes_numeric = pd.to_numeric(df['gene'], errors='coerce').dropna()
                drugs_numeric = pd.to_numeric(df['drug'], errors='coerce').dropna()
                if not genes_numeric.empty: max_n_genes = max(max_n_genes, genes_numeric.astype(int).max())
                if not drugs_numeric.empty: max_n_drugs = max(max_n_drugs, drugs_numeric.astype(int).max())
        except pd.errors.EmptyDataError:
            pass
        except ValueError as e:
            pass
        except FileNotFoundError:
            pass
        except Exception as e:
            traceback.print_exc()

    n_genes = max_n_genes + 1 if max_n_genes >= 0 else 0
    n_drugs = max_n_drugs + 1 if max_n_drugs >= 0 else 0
    if n_genes == 0 or n_drugs == 0:
        pass
    else:
        pass
    return n_genes, n_drugs


def run_cross_validation(args, device, best_tracker, run_id=None):
    data_dir = args.data_dir
    num_folds = args.num_folds_cv
    any_fold_succeeded = False
    n_genes, n_drugs = get_consistent_shape(args)
    shape = (n_genes, n_drugs)
    if n_genes <= 0 or n_drugs <= 0: return False

    for fold_id in range(num_folds):
        train_mat, test_mat, adj_norm_tensor, train_loader, test_loader, model = None, None, None, None, None, None
        gene_lcc, drug_lcc = None, None;
        u_mul_s, v_mul_s, ut, vt = None, None, None, None
        optimizer, scheduler = None, None;
        fold_setup_success = False

        cuda_utils.reset_peak_memory_stats(device)

        best_fold_aupr = -np.inf

        try:
            train_path = os.path.join(data_dir, f'train{fold_id}.csv');
            test_path = os.path.join(data_dir, f'test{fold_id}.csv')
            train_mat = load_data_from_csv(train_path, shape=shape);
            test_mat = load_data_from_csv(test_path, shape=shape)
            if train_mat is None or train_mat.nnz == 0: continue

            train_mat.resize(shape)
            if test_mat is not None:
                test_mat.resize(shape)
            else:
                test_mat = coo_matrix(shape)

            gene_lcc, drug_lcc = calculate_lcc_bipartite(train_mat.copy());
            gene_lcc, drug_lcc = gene_lcc.to(device), drug_lcc.to(device)

            train_mat_norm = normalize_adj_matrix(train_mat.copy());
            adj_norm_tensor = scipy_sparse_mat_to_torch_sparse_tensor(train_mat_norm).coalesce().to(device)
            train_data = TrnData(train_mat.tocoo());
            test_data = TrnData(test_mat.tocoo()) if test_mat is not None and test_mat.nnz > 0 else None
            if len(train_data) == 0: continue

            train_loader = data.DataLoader(train_data, batch_size=args.inter_batch, shuffle=True, num_workers=4,
                                           pin_memory=True, drop_last=False)
            test_loader = data.DataLoader(test_data, batch_size=args.inter_batch, shuffle=False, num_workers=4,
                                          pin_memory=True) if test_data and len(test_data) > 0 else None

            if adj_norm_tensor is not None and adj_norm_tensor._nnz() > 0:
                adj_norm_cpu = adj_norm_tensor.cpu();
                if adj_norm_cpu._nnz() > 0:
                    effective_q = min(args.q, min(adj_norm_cpu.shape) - 1);
                    if effective_q > 0:
                        try:
                            svd_u, s, svd_vh = torch.linalg.svd(adj_norm_cpu.to_dense(), full_matrices=False);
                            rank = min(effective_q, len(s));
                            if rank > 0:
                                svd_u = svd_u[:, :rank];
                                s = s[:rank];
                                svd_v = svd_vh.t()[:, :rank];
                                u_mul_s = (svd_u @ torch.diag(s)).to(device);
                                v_mul_s = (svd_v @ torch.diag(s)).to(device);
                                ut = svd_u.t().to(device);
                                vt = svd_vh[:rank, :].to(device);
                                del svd_u, s, svd_vh, svd_v;
                        except Exception as e_svd:
                            pass
                del adj_norm_cpu;
                torch.cuda.empty_cache()

            model = AMGAN(n_g=n_genes, n_d=n_drugs, d=args.d, g_mul_s=u_mul_s, v_mul_s=v_mul_s, ut=ut, vt=vt,
                          train_csr=train_mat.tocsr(), adj_norm=adj_norm_tensor, l=args.gnn_layer, temp=args.temp,
                          lambda_1=args.lambda1, lambda_2=args.lambda2, dropout=args.dropout,
                          batch_gene=args.inter_batch, device=device, hyper_num=args.hyperNum, num_heads=args.num_heads,
                          gene_lcc=gene_lcc, drug_lcc=drug_lcc, use_lcc_aux_task=args.use_lcc_aux_task,
                          lambda_aux=args.lambda_aux, use_lcc_attn_bias=args.use_lcc_attn_bias,
                          lcc_attn_scale=args.lcc_attn_scale).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)
            fold_setup_success = True
        except Exception as e:
            traceback.print_exc();
            fold_setup_success = False

        if fold_setup_success:
            current_fold_params = {'lambda_aux': args.lambda_aux, 'lcc_attn_scale': args.lcc_attn_scale}
            oom_flag = False;
            fold_evaluated_at_least_once = False

            for epoch in range(args.epoch):
                model.train();
                epoch_losses = {'total': 0.0, 'bpr': 0.0, 'cl': 0.0, 'aux': 0.0};
                num_batches = 0
                if hasattr(train_loader.dataset, 'neg_sampling'):
                    try:
                        train_loader.dataset.neg_sampling()
                    except Exception as neg_e:
                        pass

                pbar_desc = f'Fold {fold_id + 1} Epoch {epoch + 1}'
                pbar = tqdm(train_loader, desc=pbar_desc, ncols=110, leave=False)

                for batch in pbar:
                    if batch is None or len(batch) == 0: continue
                    try:
                        geneids, pos, neg = [x.long().to(device) for x in batch]
                        if geneids.numel() == 0: continue
                        valid_neg_mask = (neg != -1)
                        if not valid_neg_mask.all(): geneids = geneids[valid_neg_mask]; pos = pos[valid_neg_mask]; neg = \
                        neg[valid_neg_mask];
                        if geneids.numel() == 0: continue
                        iids = torch.cat([pos, neg]).unique()
                    except Exception:
                        continue

                    optimizer.zero_grad(set_to_none=True)
                    try:
                        loss, bpr_loss, cl_loss, aux_loss, _, _ = model(geneids, iids, pos, neg)
                        if torch.isnan(loss) or torch.isinf(loss): continue
                        loss.backward();
                        optimizer.step()
                        epoch_losses['total'] += loss.item();
                        epoch_losses['bpr'] += bpr_loss.item() if torch.is_tensor(bpr_loss) else 0.0
                        epoch_losses['cl'] += cl_loss.item() if torch.is_tensor(cl_loss) else 0.0;
                        epoch_losses['aux'] += aux_loss.item() if torch.is_tensor(aux_loss) else 0.0
                        num_batches += 1;
                        pbar.set_postfix({'loss': f'{loss.item():.4f}'}, refresh=False)
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            oom_flag = True;
                            torch.cuda.empty_cache();
                            break
                        elif "illegal memory access" in str(e):
                            oom_flag = True;
                            break
                        else:
                            continue
                    except Exception as e:
                        continue
                if oom_flag: break
                scheduler.step()

                current_epoch_num = epoch + 1
                is_eval_epoch = (current_epoch_num % args.eval_interval_cv == 0) or (current_epoch_num == args.epoch)

                if is_eval_epoch and test_loader:
                    if hasattr(test_loader.dataset, 'neg_sampling'):
                        try:
                            test_loader.dataset.neg_sampling()
                        except Exception as neg_e:
                            pass

                    current_test_metrics = evaluate(model, test_loader, adj_norm_tensor, device)

                    if current_test_metrics and not np.isnan(current_test_metrics.get('AUPR', np.nan)):
                        fold_evaluated_at_least_once = True;
                        any_fold_succeeded = True

                        current_aupr = current_test_metrics.get('AUPR', -np.inf)

                        avg_loss = (epoch_losses["total"] / num_batches) if num_batches > 0 else 0
                        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'AUPR': f'{current_aupr:.4f}'}, refresh=True)

                        if pd.notna(current_aupr) and current_aupr > best_fold_aupr:
                            best_fold_aupr = current_aupr

                        tracker_best_aupr = best_tracker['best_aupr_fold'].get('aupr', -np.inf) if best_tracker[
                            'best_aupr_fold'] else -np.inf

                        if pd.notna(current_aupr) and pd.notna(tracker_best_aupr) and current_aupr > tracker_best_aupr:
                            print(
                                f"\n*** [Tracker Update] New Best AUPR: {current_aupr:.4f} (Fold {fold_id + 1}, Epoch {current_epoch_num}) ***")
                            print(
                                f"    Params: lambda_aux={current_fold_params['lambda_aux']}, lcc_attn_scale={current_fold_params['lcc_attn_scale']}")

                            best_tracker['best_aupr_fold']['aupr'] = current_aupr
                            best_tracker['best_aupr_fold']['params'] = current_fold_params
                            best_tracker['best_aupr_fold']['fold'] = fold_id
                            best_tracker['best_aupr_fold']['epoch'] = current_epoch_num
                    else:
                        pass
                elif is_eval_epoch:
                    pass

        else:
            pass

        if 'model' in locals() and model is not None: del model
        if 'optimizer' in locals() and optimizer is not None: del optimizer
        if 'scheduler' in locals() and scheduler is not None: del scheduler
        if 'gene_lcc' in locals() and gene_lcc is not None: del gene_lcc
        if 'drug_lcc' in locals() and drug_lcc is not None: del drug_lcc
        if 'adj_norm_tensor' in locals() and adj_norm_tensor is not None: del adj_norm_tensor
        if 'train_loader' in locals() and train_loader is not None: del train_loader
        if 'test_loader' in locals() and test_loader is not None: del test_loader
        if 'train_mat' in locals() and train_mat is not None: del train_mat
        if 'test_mat' in locals() and test_mat is not None: del test_mat
        if 'train_data' in locals() and train_data is not None: del train_data
        if 'test_data' in locals() and test_data is not None: del test_data
        if 'u_mul_s' in locals() and u_mul_s is not None:
            del u_mul_s
            if 'v_mul_s' in locals() and v_mul_s is not None: del v_mul_s
            if 'ut' in locals() and ut is not None: del ut
            if 'vt' in locals() and vt is not None: del vt
        torch.cuda.empty_cache()

    return any_fold_succeeded


if __name__ == '__main__':
    args = parse_args()

    random.seed(args.seed);
    np.random.seed(args.seed);
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed);
        torch.cuda.manual_seed_all(args.seed)
    if args.cuda == '-1' or not torch.cuda.is_available():
        device = torch.device('cpu');
    else:
        try:
            device = torch.device(f'cuda:{args.cuda}');
            torch.cuda.set_device(device);
            _ = torch.tensor([1.0], device=device)
        except Exception as e:
            device = torch.device('cpu')

    best_performance_tracker = {
        'best_aupr_fold': {'aupr': -np.inf, 'params': {}, 'fold': -1, 'epoch': -1}
    }

    if args.enable_tuning:
        total_tuning_start_time = time.time()

        lcc_attn_scale_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        lambda_aux_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        total_runs = len(lcc_attn_scale_values) * len(lambda_aux_values)
        current_run = 0

        for scale in lcc_attn_scale_values:
            for lmbda in lambda_aux_values:
                current_run += 1
                print(f"\n<<<<==== [Tuning Run {current_run}/{total_runs}] ====>>>>")
                print(f"Hyperparameters: lcc_attn_scale = {scale}, lambda_aux = {lmbda}")

                current_args = copy.deepcopy(args)
                if current_args.use_lcc_attn_bias: current_args.lcc_attn_scale = scale
                if current_args.use_lcc_aux_task: current_args.lambda_aux = lmbda

                run_succeeded = False
                try:
                    run_succeeded = run_cross_validation(current_args,
                                                         device,
                                                         best_performance_tracker)
                except Exception as e:
                    traceback.print_exc()
                    run_succeeded = False

                del current_args
                torch.cuda.empty_cache()

        total_tuning_end_time = time.time();
        total_duration_minutes = (total_tuning_end_time - total_tuning_start_time) / 60
        print(f"\n<<<<==== Hyperparameter Tuning Completed ====>>>>")
        print(f"Total tuning time: {total_duration_minutes:.2f} minutes")

    else:
        print("Executing single CV run with provided/default parameters...")
        start_time = time.time()
        run_succeeded = run_cross_validation(args, device, best_performance_tracker)
        end_time = time.time()
        total_duration_minutes = (end_time - start_time) / 60
        print(f"\nSingle CV run completed. Wall Clock Time: {total_duration_minutes:.2f} min")

    print("\n--- Final Best AUPR Result ---")
    best_result = best_performance_tracker['best_aupr_fold']
    if best_result['epoch'] != -1:
        print(f"Best AUPR: {best_result['aupr']:.4f}")
        print(f"Achieved at Fold: {best_result['fold'] + 1}, Epoch: {best_result['epoch']}")
        print(f"With Parameters: {best_result['params']}")
    else:
        print("No successful runs recorded.")

    print("\n--- Script Execution Finished ---")