import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import random
from scipy.sparse import coo_matrix, load_npz, csr_matrix, csc_matrix
import pandas as pd
from tqdm import tqdm

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_dropout(mat, dropout):
    if dropout == 0.0:
        return mat
    if not mat.is_sparse:
        mat = mat.to_sparse_coo() if hasattr(mat, 'to_sparse_coo') else mat.to_sparse()

    indices = mat.indices()
    values = F.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)

def spmm(sp, emb, device):
    sp = sp.coalesce()
    cols = sp.indices()[1]
    rows = sp.indices()[0]
    col_segs = emb[cols] * torch.unsqueeze(sp.values(), dim=1)
    result = torch.zeros((sp.shape[0], emb.shape[1]), device=torch.device(device))
    result.index_add_(0, rows, col_segs)
    return result

def load_data_from_csv(file_path, shape=None):
    try:
        df = pd.read_csv(file_path)
        if not {'gene', 'drug', 'interaction'}.issubset(df.columns):
             raise ValueError("CSV 文件必须包含 'gene', 'drug', 'interaction' 列")

        rows = df['gene'].values
        cols = df['drug'].values
        values = df['interaction'].values

        if shape is None:
            n_genes = df['gene'].max() + 1 if not df.empty else 0
            n_drugs = df['drug'].max() + 1 if not df.empty else 0
            shape = (n_genes, n_drugs)
        else:
            if not df.empty and (rows.max() >= shape[0] or cols.max() >= shape[1]):
                 n_genes = max(rows.max() + 1, shape[0])
                 n_drugs = max(cols.max() + 1, shape[1])
                 shape = (n_genes, n_drugs)

        valid_indices = (rows < shape[0]) & (cols < shape[1])
        if not np.all(valid_indices):
            rows, cols, values = rows[valid_indices], cols[valid_indices], values[valid_indices]

        return coo_matrix((values, (rows, cols)), shape=shape)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise

def calculate_lcc_bipartite(adj_mat):
    n_genes, n_drugs = adj_mat.shape

    if n_genes == 0 or n_drugs == 0:
        return torch.zeros(n_genes, dtype=torch.float), torch.zeros(n_drugs, dtype=torch.float)

    adj_csr = adj_mat.tocsr() if not isinstance(adj_mat, csr_matrix) else adj_mat
    adj_csc = adj_mat.tocsc() if not isinstance(adj_mat, csc_matrix) else adj_mat

    gene_degrees = np.array(adj_csr.sum(axis=1)).flatten()
    drug_degrees = np.array(adj_csc.sum(axis=0)).flatten()

    gene_lcc_raw = np.zeros(n_genes)
    drug_lcc_raw = np.zeros(n_drugs)

    for g in tqdm(range(n_genes), desc="Gene LCC", ncols=80, leave=False):
        _, drug_indices = adj_csr[g, :].nonzero()
        if len(drug_indices) > 0:
            neighbor_drug_degrees = drug_degrees[drug_indices]
            gene_lcc_raw[g] = np.mean(neighbor_drug_degrees)

    for d in tqdm(range(n_drugs), desc="Drug LCC", ncols=80, leave=False):
        gene_indices, _ = adj_csc[:, d].nonzero()
        if len(gene_indices) > 0:
            neighbor_gene_degrees = gene_degrees[gene_indices]
            drug_lcc_raw[d] = np.mean(neighbor_gene_degrees)

    def normalize(lcc_raw):
        min_val = np.min(lcc_raw)
        max_val = np.max(lcc_raw)
        if max_val == min_val:
            return np.full_like(lcc_raw, 0.5 if max_val != 0 else 0, dtype=np.float32)
        else:
            return (lcc_raw - min_val) / (max_val - min_val + 1e-9)

    gene_lcc_norm = normalize(gene_lcc_raw)
    drug_lcc_norm = normalize(drug_lcc_raw)

    return torch.tensor(gene_lcc_norm, dtype=torch.float), torch.tensor(drug_lcc_norm, dtype=torch.float)

def normalize_adj_matrix(adj_mat):
    adj_mat = adj_mat.tocsr()
    rowD = np.array(adj_mat.sum(1)).flatten()
    colD = np.array(adj_mat.sum(0)).flatten()

    rowD[rowD == 0.] = 1.
    colD[colD == 0.] = 1.

    rowD_inv_sqrt = np.power(rowD, -0.5)
    colD_inv_sqrt = np.power(colD, -0.5)

    from scipy.sparse import diags
    rowD_inv_sqrt_diag = diags(rowD_inv_sqrt)
    colD_inv_sqrt_diag = diags(colD_inv_sqrt)

    normalized_adj = rowD_inv_sqrt_diag @ adj_mat @ colD_inv_sqrt_diag

    return normalized_adj.tocoo()

class TrnData(data.Dataset):
    def __init__(self, coomat):
        if not isinstance(coomat, coo_matrix):
            try:
                coomat = coomat.tocoo()
            except AttributeError:
                 raise TypeError("Input must be convertible to COO format SciPy sparse matrix")

        self.rows = coomat.row
        self.cols = coomat.col
        self.num_genes, self.num_drugs = coomat.shape
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def neg_sampling(self):
        num_sampled = 0
        if self.num_drugs <= 0:
             self.negs.fill(-1)
             return

        for i in range(len(self.rows)):
            u = self.rows[i]
            sample_attempts = 0
            max_attempts = self.num_drugs * 3
            while True:
                i_neg = random.randint(0, self.num_drugs - 1)
                if (u, i_neg) not in self.dokmat:
                    self.negs[i] = i_neg
                    num_sampled += 1
                    break
                sample_attempts += 1
                if sample_attempts > max_attempts:
                     self.negs[i] = -1
                     break

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        neg_idx = self.negs[idx]
        return self.rows[idx], self.cols[idx], neg_idx