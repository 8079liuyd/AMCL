import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from utils import spmm
import warnings

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, x):
        seq_len = x.size(-2)
        position = torch.arange(seq_len, device=x.device, dtype=torch.float).unsqueeze(1)
        pe = torch.zeros(seq_len, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * self.div_term)
        pe[:, 1::2] = torch.cos(position * self.div_term)
        if x.dim() == 3:
             pe = pe.unsqueeze(0)
        elif x.dim() != 2:
             raise ValueError("PositionalEncoding 期望输入维度为 2 或 3")
        return pe


class EnhancedMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1,
                 use_lcc_bias=False, lcc_scale=0.1, lcc_values=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = self.head_dim ** -0.5

        self.qkv_linear = nn.Linear(d_model, 3 * d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.feature_enhance = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model)
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.use_lcc_bias = use_lcc_bias
        self.lcc_scale = lcc_scale
        if use_lcc_bias:
            if lcc_values is None:
                raise ValueError("如果 use_lcc_bias 为 True，则必须提供 lcc_values")
            self.register_buffer('lcc_values', lcc_values)


    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(0)

        seq_len, _ = x.shape

        normed_x = self.layer_norm1(x)
        qkv = self.qkv_linear(normed_x)
        qkv = qkv.reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)

        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)

        q = F.normalize(q, p=2, dim=-1) * self.scaling
        k = F.normalize(k, p=2, dim=-1)

        attn_scores = q @ k.transpose(-2, -1)

        if self.use_lcc_bias and hasattr(self, 'lcc_values'):
            lcc = self.lcc_values
            lcc_row = lcc.view(1, seq_len, 1)
            lcc_col = lcc.view(1, 1, seq_len)
            bias = (lcc_row + lcc_col) / 2.0
            bias = bias * self.lcc_scale
            attn_scores = attn_scores + bias

        attn_scores = torch.clamp(attn_scores, min=-10, max=10)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = attn_probs @ v
        attn_output = attn_output.permute(1, 0, 2).reshape(seq_len, self.d_model)
        attn_output = self.out_linear(attn_output)
        attn_output = self.dropout(attn_output)

        x = x + attn_output

        normed_x2 = self.layer_norm2(x)
        enhanced = self.feature_enhance(normed_x2)
        x = x + enhanced

        return x


class EnhancedCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1,
                 use_lcc_bias=False, lcc_scale=0.1, gene_lcc=None, drug_lcc=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = self.head_dim ** -0.5

        self.gene_q_proj = nn.Linear(d_model, d_model)
        self.gene_k_proj = nn.Linear(d_model, d_model)
        self.drug_q_proj = nn.Linear(d_model, d_model)
        self.drug_k_proj = nn.Linear(d_model, d_model)
        self.gene_v_proj = nn.Linear(d_model, d_model)
        self.drug_v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        self.gene_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.Sigmoid())
        self.drug_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.Sigmoid())

        self.dropout = nn.Dropout(dropout)
        self.layer_norm_gene = nn.LayerNorm(d_model)
        self.layer_norm_drug = nn.LayerNorm(d_model)

        self.use_lcc_bias = use_lcc_bias
        self.lcc_scale = lcc_scale
        if use_lcc_bias:
            if gene_lcc is None or drug_lcc is None:
                raise ValueError("如果 use_lcc_bias is True，则必须提供 gene_lcc 和 drug_lcc")
            self.register_buffer('gene_lcc', gene_lcc)
            self.register_buffer('drug_lcc', drug_lcc)

    def attention(self, q, k, v, seq_len_q, seq_len_kv, lcc_q_vals=None, lcc_k_vals=None):
        q = q.view(seq_len_q, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len_kv, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(seq_len_kv, self.num_heads, self.head_dim).transpose(0, 1)

        q = F.normalize(q, p=2, dim=-1) * self.scaling
        k = F.normalize(k, p=2, dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1))

        if self.use_lcc_bias and lcc_q_vals is not None and lcc_k_vals is not None:
            lcc_q_row = lcc_q_vals.view(1, seq_len_q, 1)
            lcc_k_col = lcc_k_vals.view(1, 1, seq_len_kv)
            bias = (lcc_q_row + lcc_k_col) / 2.0
            bias = bias * self.lcc_scale
            scores = scores + bias

        scores = torch.clamp(scores, min=-10.0, max=10.0)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = context.transpose(0, 1).contiguous().view(seq_len_q, self.d_model)

        return context

    def forward(self, gene_embeds, drug_embeds):
        gene_len, _ = gene_embeds.shape
        drug_len, _ = drug_embeds.shape

        normed_gene = self.layer_norm_gene(gene_embeds)
        normed_drug = self.layer_norm_drug(drug_embeds)

        gene_q, gene_k = self.gene_q_proj(normed_gene), self.gene_k_proj(normed_gene)
        gene_v = self.gene_v_proj(gene_embeds)
        drug_q, drug_k = self.drug_q_proj(normed_drug), self.drug_k_proj(normed_drug)
        drug_v = self.drug_v_proj(drug_embeds)

        gene_cross_context = self.attention(gene_q, drug_k, drug_v, gene_len, drug_len,
                                             lcc_q_vals=getattr(self,'gene_lcc',None),
                                             lcc_k_vals=getattr(self,'drug_lcc',None))

        drug_cross_context = self.attention(drug_q, gene_k, gene_v, drug_len, gene_len,
                                             lcc_q_vals=getattr(self,'drug_lcc',None),
                                             lcc_k_vals=getattr(self,'gene_lcc',None))

        gene_gate_input = torch.cat([gene_cross_context, gene_embeds], dim=-1)
        gene_gate_val = self.gene_gate(gene_gate_input)
        drug_gate_input = torch.cat([drug_cross_context, drug_embeds], dim=-1)
        drug_gate_val = self.drug_gate(drug_gate_input)

        gene_out = gene_embeds + gene_gate_val * self.dropout(self.out(gene_cross_context))
        drug_out = drug_embeds + drug_gate_val * self.dropout(self.out(drug_cross_context))

        return gene_out, drug_out


class AMGAN(nn.Module):
    def __init__(self, n_g, n_d, d, g_mul_s, v_mul_s, ut, vt, train_csr, adj_norm, l, temp,
                 lambda_1, lambda_2, dropout, batch_gene, device, hyper_num, num_heads=4,
                 gene_lcc=None, drug_lcc=None, use_lcc_aux_task=False, lambda_aux=0.1,
                 use_lcc_attn_bias=False, lcc_attn_scale=0.1):
        super().__init__()

        self.n_g, self.n_d, self.d, self.l = n_g, n_d, d, l
        self.temp, self.lambda_1, self.lambda_2 = temp, lambda_1, lambda_2
        self.dropout_rate = dropout
        self.device = device
        self.hyper_num, self.num_heads = hyper_num, num_heads

        self.use_lcc_aux_task = use_lcc_aux_task
        self.lambda_aux = lambda_aux
        self.use_lcc_attn_bias = use_lcc_attn_bias
        self.lcc_attn_scale = lcc_attn_scale

        if use_lcc_aux_task or use_lcc_attn_bias:
             if gene_lcc is None or drug_lcc is None:
                 raise ValueError("启用LCC特性时必须提供 gene_lcc 和 drug_lcc 张量。")
             self.register_buffer('gene_lcc', gene_lcc.to(device))
             self.register_buffer('drug_lcc', drug_lcc.to(device))
        else:
             self.register_buffer('gene_lcc', torch.empty(n_g, device=device))
             self.register_buffer('drug_lcc', torch.empty(n_d, device=device))

        self.E_g_0 = nn.Parameter(torch.empty(n_g, d)); nn.init.xavier_uniform_(self.E_g_0)
        self.E_d_0 = nn.Parameter(torch.empty(n_d, d)); nn.init.xavier_uniform_(self.E_d_0)

        self.H_g = nn.Parameter(torch.empty(d, hyper_num)); nn.init.xavier_uniform_(self.H_g)
        self.H_d = nn.Parameter(torch.empty(d, hyper_num)); nn.init.xavier_uniform_(self.H_d)

        self.pos_encoder = PositionalEncoding(d)
        self.self_attn_gene = EnhancedMultiHeadSelfAttention(
            d, num_heads, dropout, self.use_lcc_attn_bias, self.lcc_attn_scale,
            self.gene_lcc if self.use_lcc_attn_bias else None
        )
        self.self_attn_drug = EnhancedMultiHeadSelfAttention(
            d, num_heads, dropout, self.use_lcc_attn_bias, self.lcc_attn_scale,
            self.drug_lcc if self.use_lcc_attn_bias else None
        )
        self.cross_attn = EnhancedCrossAttention(
            d, num_heads, dropout, self.use_lcc_attn_bias, self.lcc_attn_scale,
            self.gene_lcc if self.use_lcc_attn_bias else None,
            self.drug_lcc if self.use_lcc_attn_bias else None
        )

        self.w_attn = nn.Parameter(torch.empty(d, 1)); nn.init.xavier_uniform_(self.w_attn)
        self.fusion_layer = nn.Sequential(
            nn.Linear(d, d * 2), nn.LayerNorm(d * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d * 2, d), nn.LayerNorm(d)
        )

        if self.use_lcc_aux_task:
            self.lcc_pred_head_gene = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, 1), nn.Sigmoid())
            self.lcc_pred_head_drug = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, 1), nn.Sigmoid())
            self.mse_loss = nn.MSELoss()

        self.register_buffer('g_mul_s', g_mul_s); self.register_buffer('v_mul_s', v_mul_s)
        self.register_buffer('ut', ut); self.register_buffer('vt', vt)
        self.register_buffer('adj_norm', adj_norm)

        self.dropout = nn.Dropout(dropout)

    def compute_hypergraph_attention(self, embeds):
        scores = torch.matmul(embeds, self.w_attn)
        return F.softmax(F.leaky_relu(scores), dim=0)

    def dynamic_hypergraph_message_passing(self, embeds, H):
        attention = self.compute_hypergraph_attention(embeds)
        proj = torch.matmul(embeds, H)
        Z = attention * torch.matmul(proj, H.t())
        return F.normalize(Z + 1e-8, p=2, dim=-1)

    def _layer_forward(self, E_g, E_d):
        E_g_pos = E_g + self.pos_encoder(E_g)
        E_d_pos = E_d + self.pos_encoder(E_d)
        E_g_self = self.self_attn_gene(E_g_pos)
        E_d_self = self.self_attn_drug(E_d_pos)
        E_g_cross, E_d_cross = self.cross_attn(E_g_self, E_d_self)
        E_g_cross = self.dropout(E_g_cross); E_d_cross = self.dropout(E_d_cross)

        Z_g = spmm(self.adj_norm, E_d_cross, self.device)
        Z_d = spmm(self.adj_norm.transpose(0, 1), E_g_cross, self.device)

        vt_ei = torch.matmul(self.vt, E_d_cross); G_g = torch.matmul(self.g_mul_s, vt_ei)
        ut_eu = torch.matmul(self.ut, E_g_cross); G_d = torch.matmul(self.v_mul_s, ut_eu)

        H_g_update = self.dynamic_hypergraph_message_passing(E_g_cross, self.H_g)
        H_d_update = self.dynamic_hypergraph_message_passing(E_d_cross, self.H_d)

        E_g_next = Z_g + G_g + H_g_update; E_d_next = Z_d + G_d + H_d_update
        E_g_next = self.fusion_layer(E_g_next); E_d_next = self.fusion_layer(E_d_next)

        E_g_next = F.normalize(E_g_next + 1e-8, p=2, dim=-1)
        E_d_next = F.normalize(E_d_next + 1e-8, p=2, dim=-1)
        return E_g_next, E_d_next, G_g, G_d

    def forward(self, uids, iids, pos, neg, adj_norm=None, test=False):
        E_g_list, E_d_list = [self.E_g_0], [self.E_d_0]
        G_g_list, G_d_list = [], []
        E_g, E_d = self.E_g_0, self.E_d_0
        for _ in range(self.l):
            if self.training and not test:
                 E_g, E_d, G_g, G_d = checkpoint(self._layer_forward, E_g, E_d, use_reentrant=False)
            else:
                 E_g, E_d, G_g, G_d = self._layer_forward(E_g, E_d)
            E_g_list.append(E_g); E_d_list.append(E_d)
            G_g_list.append(G_g); G_d_list.append(G_d)

        E_g_stack = torch.stack(E_g_list); E_d_stack = torch.stack(E_d_list)
        G_g_stack = torch.stack(G_g_list) if G_g_list else None
        G_d_stack = torch.stack(G_d_list) if G_d_list else None

        attn_weights_g = F.softmax(torch.matmul(E_g_stack, self.w_attn), dim=0)
        attn_weights_d = F.softmax(torch.matmul(E_d_stack, self.w_attn), dim=0)
        E_g_agg = torch.sum(E_g_stack * attn_weights_g, dim=0)
        E_d_agg = torch.sum(E_d_stack * attn_weights_d, dim=0)

        if test:
            E_g_final = F.normalize(E_g_agg + 1e-8, p=2, dim=-1)
            E_d_final = F.normalize(E_d_agg + 1e-8, p=2, dim=-1)
            return E_g_final, E_d_final

        if G_g_stack is not None:
            G_g_final = torch.sum(G_g_stack * attn_weights_g[1:], dim=0)
            G_d_final = torch.sum(G_d_stack * attn_weights_d[1:], dim=0)
            G_g_final = F.normalize(G_g_final + 1e-8, p=2, dim=-1)
            G_d_final = F.normalize(G_d_final + 1e-8, p=2, dim=-1)
        else:
             G_g_final = torch.zeros_like(E_g_agg); G_d_final = torch.zeros_like(E_d_agg)

        E_g_final = F.normalize(E_g_agg + 1e-8, p=2, dim=-1)
        E_d_final = F.normalize(E_d_agg + 1e-8, p=2, dim=-1)

        aux_loss = torch.tensor(0.0, device=self.device)
        if self.training and self.use_lcc_aux_task:
            pred_lcc_g = self.lcc_pred_head_gene(E_g_agg[uids])
            pred_lcc_d = self.lcc_pred_head_drug(E_d_agg[iids])
            target_lcc_g = self.gene_lcc[uids]
            target_lcc_d = self.drug_lcc[iids]
            loss_g = self.mse_loss(pred_lcc_g.squeeze(-1), target_lcc_g)
            loss_d = self.mse_loss(pred_lcc_d.squeeze(-1), target_lcc_d)
            if torch.isnan(loss_g) or torch.isnan(loss_d):
                pass
            else:
                 aux_loss = loss_g + loss_d

        cl_loss = self.compute_enhanced_contrastive_loss(E_g_final, G_g_final, E_d_final, G_d_final, uids, iids)

        u_embed = E_g_final[uids]; p_embed = E_d_final[pos]; n_embed = E_d_final[neg]
        pos_scores = torch.sum(u_embed * p_embed, dim=1)
        neg_scores = torch.sum(u_embed * n_embed, dim=1)
        score_diff = pos_scores - neg_scores
        bpr_loss = F.softplus(-score_diff).mean()
        if torch.isnan(bpr_loss):
             bpr_loss = torch.tensor(0.0, requires_grad=True, device=self.device)

        reg_loss = sum(torch.norm(param, p=2).square() for param in self.parameters()) * self.lambda_2

        loss = bpr_loss + self.lambda_1 * cl_loss + reg_loss + self.lambda_aux * aux_loss

        if torch.isnan(loss):
            loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            bpr_loss, cl_loss, aux_loss = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        return loss, bpr_loss, cl_loss, aux_loss, pos_scores, neg_scores

    def compute_enhanced_contrastive_loss(self, E_g, G_g, E_d, G_d, uids, iids):
        def info_nce_loss(anchor, positive, all_keys, temp):
            pos_sim = torch.exp(torch.sum(anchor * positive, dim=1) / temp)
            neg_sim = torch.exp(torch.matmul(anchor, all_keys.t()) / temp).sum(dim=1)
            loss = -torch.log(pos_sim / (neg_sim + 1e-8))
            return loss.mean()

        all_gene_keys = torch.cat([E_g, G_g], dim=0)
        gene_cl = info_nce_loss(E_g[uids], G_g[uids], all_gene_keys, self.temp)
        all_drug_keys = torch.cat([E_d, G_d], dim=0)
        drug_cl = info_nce_loss(E_d[iids], G_d[iids], all_drug_keys, self.temp)

        if torch.isnan(gene_cl) or torch.isnan(drug_cl):
            return torch.tensor(0.0, device=self.device)
        return (gene_cl + drug_cl) / 2