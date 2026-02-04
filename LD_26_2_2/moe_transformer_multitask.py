# ================= moe_transformer_multitask.py =================
import math, torch, torch.nn as nn, torch.nn.functional as F, logging
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from torch_cluster import radius_graph
from torch_scatter import scatter_min
from torch.cuda.amp import autocast
from torch_geometric.data import Batch

from EGNN import ComENetAutoEncoder, initialize_weights
from utils_debug import set_debug, is_debug
from time_series_module_LSTM import TimeSeriesModule

LOGGER = logging.getLogger(__name__)

# =========================
# 等变 Transformer 专家定义
# =========================
class ETExpert(nn.Module):
    """
    等变 Transformer 专家（不依赖 edge_index/f1/f2）。
    输入:
        x:     (N, C) 节点特征（来自 latent_base 或 x_shared）
        batch: (N,)    图分组 id
    输出:
        x_out: (N, C) 更新后的节点特征
    注: 内部维护隐式坐标 r (N,3)，由 x 通过 MLP 预测并在层内等变更新。
    """
    def __init__(self, dim: int, num_layers: int = 2, nheads: int = 4, ff_mult: int = 2):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.nheads = nheads
        self.head_dim = dim // nheads
        assert dim % nheads == 0, "dim 必须可被 nheads 整除"

        # 用于初始化与更新坐标
        self.pos_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 3)
        )

        layers = []
        for _ in range(num_layers):
            layers.append(_ETBlock(dim, nheads, ff_mult))
        self.layers = nn.ModuleList(layers)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, batch=None):
        # x: (N, C), batch: (N,)
        assert batch is not None, "ETExpert 需要 batch 向量用于按图分组注意力"
        N, C = x.size()
        # 初始化坐标（不读取原图坐标）
        r = self.pos_mlp(x)  # (N,3)

        for blk in self.layers:
            x, r = blk(x, r, batch)

        return self.norm(x)


class _ETBlock(nn.Module):
    """
    单层 等变 Transformer：
      - 基于 pairwise 距离的注意力（按图分组）
      - 向量等变坐标更新（EGNN 风格）
    """
    def __init__(self, dim: int, nheads: int, ff_mult: int):
        super().__init__()
        self.dim = dim
        self.nheads = nheads
        self.head_dim = dim // nheads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        # 距离嵌入到 attention logit 的偏置
        self.dist_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, nheads)  # 每个头一个标量偏置
        )

        # 坐标更新门控
        self.coord_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1)
        )

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, r: torch.Tensor, batch: torch.Tensor):
        # x: (N,C), r: (N,3), batch: (N,)
        N, C = x.size()
        H, D = self.nheads, self.head_dim

        x_in = x
        x = self.norm1(x)

        q = self.q_proj(x).view(N, H, D)      # (N,H,D)
        k = self.k_proj(x).view(N, H, D)
        v = self.v_proj(x).view(N, H, D)

        # 计算按图分组的掩码 (N,N)，True 表示同一图
        same_graph = (batch.unsqueeze(0) == batch.unsqueeze(1))  # (N,N)

        # pairwise 距离 (N,N,1)
        with torch.no_grad():
            diff = r.unsqueeze(1) - r.unsqueeze(0)              # (N,N,3)
            dist2 = (diff * diff).sum(-1, keepdim=True).clamp(min=1e-8)

        # 距离偏置 (N,N,H)
        dist_bias = self.dist_mlp(dist2)                        # (N,N,H)

        # 注意力 logits: (H,N,N)
        # qk = (N,H,D) @ (N,H,D)ᵀ 需要两两组合，先扩展维度
        q_ = q.unsqueeze(1)                                     # (1,N,H,D)
        k_ = k.unsqueeze(0)                                     # (N,1,H,D)
        qk = (q_ * k_).sum(-1) * self.scale                     # (N,N,H)
        logits = qk + dist_bias                                 # (N,N,H)

        # mask 跨图项
        mask = ~same_graph                                      # (N,N)
        logits = logits.permute(2, 0, 1)                        # (H,N,N)
        very_neg = torch.finfo(logits.dtype).min / 2
        logits = logits.masked_fill(mask.unsqueeze(0), very_neg)

        attn = torch.softmax(logits, dim=-1)                    # (H,N,N)

        # 注意力聚合
        vT = v.permute(1, 0, 2)                                 # (H,N,D)
        ctx = torch.einsum('hij,hjd->hid', attn, vT)            # (H,N,D)
        ctx = ctx.permute(1, 0, 2).contiguous().view(N, C)      # (N,C)
        x = x_in + self.o_proj(ctx)

        # FFN
        x = x + self.ff(self.norm2(x))

        # 坐标等变更新：Δr_i = Σ_j α_ij (r_i - r_j) * gate(h_i, h_j)
        with torch.no_grad():
            # 使用同一注意力进行坐标更新（各头求平均）
            attn_mean = attn.mean(0)                            # (N,N)
            # 仅同图
            attn_mean = attn_mean * same_graph.float()
            denom = attn_mean.sum(-1, keepdim=True).clamp(min=1e-6)
            attn_norm = attn_mean / denom                       # (N,N)

        gate = torch.sigmoid(self.coord_gate(x)).clamp(min=0.1, max=1.0)  # (N,1)
        # Δr_i = Σ_j a_ij * (r_i - r_j) * gate_i
        delta = torch.einsum('ij,ijk->ik', attn_norm, (r.unsqueeze(1) - r.unsqueeze(0)))
        r = r + gate * delta

        return x, r
    
# ────────────────────────────── 工具 ──────────────────────────────
def _build_experts(n_experts: int, in_dim: int, expert_layers: int):
    """
    构建专家列表：使用等变 Transformer 专家，接口与原专家一致。
    不使用 edge_index/f1/f2，确保与 backbone 信息隔离。
    """
    experts = nn.ModuleList()
    for _ in range(n_experts):   # 几何专家个数（如果你是按任务分别建，请按任务各自数量建）
        experts.append(ETExpert(dim=in_dim, num_layers=expert_layers))
    return experts

# ----------  Batch 修复工具  ----------
def ensure_batch_attr(data, device=None):
    """
    保证 data.batch 存在且为一维 (N,) LongTensor。
    若缺失则填充全 0（单图）；若为 (N,1) 等形状则 flatten。
    返回修正后的 batch 方便链式调用。
    """
    b = getattr(data, 'batch', None)
    if b is None:                          # 完全缺失
        b = torch.zeros(data.num_nodes, dtype=torch.long,
                        device=device if device is not None else data.x.device)
    elif b.dim() > 1:                      # e.g. (N,1)
        b = b.view(-1).to(device if device is not None else b.device)
    data.batch = b
    return b

class Linear(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True,
                 weight_initializer='glorot',
                 bias_initializer='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels)) if bias else None
        self.reset_parameters()
    def reset_parameters(self):
        initialize_weights(self.weight, self.weight_initializer, self.bias)
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class CoordinateDecoder(torch.nn.Module):
    def __init__(self, atom_embedding_dim, hidden_dim):
        super(CoordinateDecoder, self).__init__()
        self.fc1 = Linear(atom_embedding_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, 3)
        self.reset_parameters()
    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        
    def forward(self, latent):
        # latent: (batch_size, num_atoms, atom_embedding_dim)
        x = F.relu(self.fc1(latent))
        coords = self.fc2(x)
        return coords
    
class LatentBottleneck(nn.Module):
    """KL-regularized latent bottleneck (D -> d -> D) for latent diffusion.

    Parameterize q(z|h)=N(mu,diag(exp(logvar))) and use reparameterization
    during training. This is the common KL-AE used in latent diffusion.
    """
    def __init__(self, in_dim: int, bottleneck_dim: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.to_mu = nn.Linear(self.in_dim, self.bottleneck_dim)
        self.to_logvar = nn.Linear(self.in_dim, self.bottleneck_dim)
        self.to_up = nn.Linear(self.bottleneck_dim, self.in_dim)

    def encode(self, x: torch.Tensor):
        mu = self.to_mu(x)
        logvar = self.to_logvar(x).clamp(min=-30.0, max=20.0)
        return mu, logvar

    def decode(self, z: torch.Tensor):
        return self.to_up(z)

    def forward(self, x: torch.Tensor, sample: bool = True):
        mu, logvar = self.encode(x)
        if sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        x_up = self.decode(z)
        kl = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)
        return x_up, z, kl

# ────────────────────────────── 主模型 ─────────────────────────────
class MultiTaskMoE(nn.Module):
    """
    共享骨干  →  三套『按任务独立』Gate / Expert / Head
    任务 = {energy, charge, geometry}
    """
    def __init__(self,
                 backbone_cfg: dict,
                 n_atom_types: int = 95,
                 # 每个任务的专家 & top-k
                 n_experts_energy: int = 2,
                 n_experts_charge: int = 2,
                 n_experts_geom:   int = 4,
                 n_experts_elem:   int = 1,
                 topk_energy: int = 1,
                 topk_charge: int = 1,
                 topk_geom:   int = 2,
                 topk_elem:   int = 1,
                 expert_layers: int = 2,
                 balance_loss_w: float = 1e-2,
                 latent_bottleneck_dim: int = 0,
                 device: str = 'cpu',
                 debug: bool = False):
        super().__init__()
        set_debug(debug)
        self.dev = torch.device(device)
        self.balance_w = balance_loss_w
        self.n_elem_classes = int(n_atom_types)
        self.pad_elem_index = self.n_elem_classes - 1  # 最后一个索引作为 pad

        # ── 共享骨干 ───────────────────────────────────────────
        self.backbone = ComENetAutoEncoder(**backbone_cfg, debug=debug).to(self.dev)
        middle = backbone_cfg['middle_channels']
        num_radial = backbone_cfg['num_radial']
        num_spherical = backbone_cfg['num_spherical']
        latent_dim = backbone_cfg['atom_embedding_dim']
        self.latent_dim = latent_dim
        self.latent_bottleneck_dim = int(latent_bottleneck_dim)
        if 0 < self.latent_bottleneck_dim < latent_dim:
            self.latent_bottleneck = LatentBottleneck(latent_dim, self.latent_bottleneck_dim)
        else:
            self.latent_bottleneck = None
        # ── 三套专家池 ─────────────────────────────────────────
        self.exp_energy = _build_experts(n_experts_energy, latent_dim, expert_layers)
        self.exp_charge = _build_experts(n_experts_charge, latent_dim, expert_layers)
        self.exp_geom   = _build_experts(n_experts_geom,   latent_dim, expert_layers)
        self.exp_elem   = _build_experts(n_experts_elem,   latent_dim, expert_layers)
        
        # ── 三套 Gate ─────────────────────────────────────────
        def _gate(out_dim):          # 共享结构，参数独立
            return nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(),
                                 nn.Linear(latent_dim, out_dim))
        self.gate_energy = _gate(n_experts_energy)
        self.gate_charge = _gate(n_experts_charge)
        self.gate_geom   = _gate(n_experts_geom)
        self.gate_elem   = _gate(n_experts_elem)
        
        # ── 融合映射 + 任务输出头 ───────────────────────────────
        self.map_energy  = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU())
        self.map_charge  = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU())
        self.map_geom    = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU())
        self.map_elem    = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU())
        
        self.head_energy = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(),
                                         nn.Linear(latent_dim, 1))
        self.head_charge = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(),
                                         nn.Linear(latent_dim, 1))
        self.head_elem   = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(),
                                         nn.Linear(latent_dim, self.n_elem_classes))
        
        # 几何 CoordinateDecoder：直接使用论文版实现
        
        self.coord_dec = CoordinateDecoder(
            atom_embedding_dim = backbone_cfg['atom_embedding_dim'],
            hidden_dim = latent_dim)

        # 可行性判别头（基于 latent0/latent_base 的图级池化；避免依赖 H 空间）
        self.head_feas = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 1))
        # 记录 top-k 参数（便于 forward 内部使用）
        self.k_energy, self.k_charge, self.k_geom, self.k_elem = topk_energy, topk_charge, topk_geom, topk_elem

    # ─────────────────────────── forward ───────────────────────────
    def _route(self, gate_layer, exp_list, top_k, latent_base, batch):
        """单任务路由 + expert 融合，返回 (N,D) latent, 负载均衡项"""
        logits = gate_layer(global_mean_pool(latent_base, batch)).clamp(-10., 10.)       # (G,E) NaN-safe
        
        probs = torch.softmax(logits, -1)
        
        # Top‑k 选择
        top_vals, top_idx = probs.topk(top_k, dim=-1)
        gate_graph = torch.zeros_like(probs)
        gate_graph.scatter_(1, top_idx, top_vals)
        gate_graph = gate_graph / (gate_graph.sum(-1, keepdim=True) + 1e-7)
        # print('gate_graph:',gate_graph)
        # 广播到原子级
        gate_atom = gate_graph[batch]                                    # (N,E)
        # print('gate_atom:',gate_atom)
        # 专家计算
        # 注意：专家现在直接在 D=atom_embedding_dim 空间工作（不再依赖 x_shared/H）
        lat_e = []
        for e, expert in enumerate(exp_list):
            if gate_graph[:, e].max() < 1e-6:
                 emb_e = torch.zeros_like(latent_base)               # (N,D)
            else:
                emb_e_raw = expert(latent_base, batch=batch)         # (N,D)
                if self.latent_bottleneck is not None:
                    emb_e, _, _ = self.latent_bottleneck(emb_e_raw, sample=False)
                else:
                    emb_e = emb_e_raw
            lat_e.append(emb_e)
        lat_e = torch.stack(lat_e, 0)                                # (E,N,D)        fused = (gate_atom.t().unsqueeze(-1) * lat_e).sum(0)             # (N,D)
        
        fused = (gate_atom.t().unsqueeze(-1) * lat_e).sum(0)          # (N,D)
        # 负载均衡：importance / load 参见 Switch-Transformer
        importance = probs.mean(0)
        load = (gate_graph > 0).float().mean(0)
        balance = (importance * load).sum() * gate_graph.size(0)
        return fused, balance

    def forward(self, data, return_latent: bool = False):
        # ① 共享编码
        x_shared = self.backbone._forward(data)                  # (N,H)
        # —— 只算一次，并拿到 edge_index / batch ——
        batch = ensure_batch_attr(data)
        latent0_raw = self.backbone.encoder(x_shared)                 # (N,D)
        if self.latent_bottleneck is not None:
            latent0, latent_low, kl_node = self.latent_bottleneck(latent0_raw, sample=self.training)
            kl_loss = kl_node.mean()
        else:
            latent0 = latent0_raw
            latent_low = latent0_raw
            kl_loss = torch.zeros([], device=latent0_raw.device)
        latent_low_dense, _ = to_dense_batch(latent_low, batch, fill_value=0.0)
        feas_logits = self.head_feas(global_mean_pool(latent0, batch))   # (B,1)

        # ② 三任务路由
        lat_E, bal_E = self._route(self.gate_energy, self.exp_energy,
                                   self.k_energy, latent0, batch)
        lat_Q, bal_Q = self._route(self.gate_charge, self.exp_charge,
                                   self.k_charge, latent0, batch)
        lat_G, bal_G = self._route(self.gate_geom,   self.exp_geom,
                                   self.k_geom, latent0, batch)
        lat_Z, bal_Z = self._route(self.gate_elem,   self.exp_elem,
                                   self.k_elem, latent0, batch)        # ③ 输出
        charge_atom  = self.head_charge(self.map_charge(lat_Q))          # (N,1)
        charge_pred, mask = to_dense_batch(charge_atom, batch, fill_value=0.)  # (B,Nmax,1)
        
        energy_graph = global_mean_pool(
            self.head_energy(self.map_energy(lat_E)), batch)             # (B,1)
        energy_pred  = energy_graph.unsqueeze(1)                         # (B,1,1)

        lat_geom, _  = to_dense_batch(self.map_geom(lat_G), batch)       # (B,Nmax,D)
        coords_pred  = self.coord_dec(lat_geom)                          # (B,Nmax,3)
        
        elem_atom          = self.head_elem(self.map_elem(lat_Z))              # # (N,C) logits
        # 注意 fill_value 填充 pad_idx，避免把 H=0 当 pad
        elem_pred, _ = to_dense_batch(elem_atom, batch, fill_value=0.0)          # logits pad，无需数值标签

        # ④ charge_mask 与负载均衡
        balance_loss = self.balance_w * (bal_E + bal_Q + bal_G + bal_Z)

        # print('coords_pred:',coords_pred.shape,'charge_pred:',charge_pred.shape,'energy_pred:',energy_pred.shape,'mask:',mask.shape)
        out = (coords_pred, charge_pred, elem_pred, energy_pred,
                mask.unsqueeze(-1), balance_loss, feas_logits)
        if return_latent:
            return out + (latent_low_dense, kl_loss)
        return out

    def forward_from_latent(
        self,
        latent: torch.Tensor,
        data_static,
        *,
        node_mask=None,
        return_dense: bool = False,
        latent_is_low=None,
    ):
        """
        用外部提供的 latent 直接做下游多任务推断（coords / charge / elem / energy）。

        支持两种 latent 形状：
        - (N, D) / (N, d) : node-level（与 data_static 的原子数一致）
        - (B, Nmax, D) / (B, Nmax, d) : dense-batch（通常来自 latent diffusion）

        若启用了 latent_bottleneck（D→d→D），并且输入 latent 的最后一维等于 d，
        会先用 bottleneck 的 decode(z) 上采样回 D，再参与 gating / routing。

        参数:
            latent: latent 张量（node-level 或 dense-batch）
            data_static: PyG Data/Batch（提供 batch / edge / x_shared 所需信息）
            node_mask: (B, Nmax, 1) 或 (B, Nmax) 的 padding mask（dense latent 时用于 flatten）
            return_dense: True 时返回与 forward() 一致的 7 元组（含 dense 输出与 mask）
            latent_is_low: 显式指定 latent 是否为 bottleneck 低维 z（None 则自动推断）

        返回:
            - return_dense=False:
                coords_pred_node (N,3), charge_pred_node (N,1), elem_pred_node (N,C),
                energy_pred_graph (B,1), balance_loss (scalar)
            - return_dense=True:
                与 forward() 完全一致的：
                (coords_pred, charge_pred, elem_pred, energy_pred, mask.unsqueeze(-1), balance_loss, feas_logits)
        """
        # ① batch 信息（forward_from_latent 仅依赖 batch/图分组；下游计算完全在 D 空间进行）
        b = ensure_batch_attr(data_static)
        
        # ② 若输入为 dense-batch，则先 flatten 回 node-level
        latent_node = latent
        if latent.dim() == 3:
            if node_mask is None:
                # 从 batch 推导一个 dense mask
                ones = torch.ones((b.size(0), 1), device=latent.device, dtype=latent.dtype)
                _, m = to_dense_batch(ones, b, fill_value=0.0)
                node_mask = m
            m = node_mask.squeeze(-1) if node_mask.dim() == 3 else node_mask
            m = m.to(dtype=torch.bool)
            latent_node = latent[m]                                          # (N, D/d)
            if latent_node.size(0) != b.size(0):
                raise ValueError(
                    f"latent 与 data_static 原子数不匹配：latent_node={latent_node.size(0)} vs N={b.size(0)}"
                )

        # ③ 若输入为 bottleneck 低维 z，则先 decode 回 D
        if self.latent_bottleneck is not None:
            if latent_is_low is None:
                latent_is_low = (latent_node.size(-1) == self.latent_bottleneck_dim)
            if latent_is_low:
                latent_base = self.latent_bottleneck.decode(latent_node)     # (N,D)
            else:
                latent_base = latent_node
        else:
            latent_base = latent_node

        # ④ 每个任务各走一条路由
        lat_E, bal_E = self._route(self.gate_energy, self.exp_energy,
                                   self.k_energy, latent_base, b)
        lat_Q, bal_Q = self._route(self.gate_charge, self.exp_charge,
                                   self.k_charge, latent_base, b)
        lat_G, bal_G = self._route(self.gate_geom,   self.exp_geom,
                                   self.k_geom,   latent_base, b)
        lat_Z, bal_Z = self._route(self.gate_elem,   self.exp_elem,
                                   self.k_elem,   latent_base, b)

        balance_loss = self.balance_w * (bal_E + bal_Q + bal_G + bal_Z)

        # ⑤ Heads（node-level）
        energy_node = self.head_energy(self.map_energy(lat_E))               # (N,1)
        energy_graph = global_mean_pool(energy_node, b)                      # (B,1)

        charge_node = self.head_charge(self.map_charge(lat_Q))               # (N,1)
        elem_node   = self.head_elem  (self.map_elem (lat_Z))                # (N,C)
        geom_node   = self.map_geom(lat_G)                                   # (N,D)

        if not return_dense:
            # coords 也变成 node-level
            geom_dense, mask = to_dense_batch(geom_node, b, fill_value=0.0)  # (B,Nmax,D), (B,Nmax)
            coords_dense = self.coord_dec(geom_dense)                        # (B,Nmax,3)
            coords_node  = coords_dense[mask]                                # (N,3)
            return coords_node, charge_node, elem_node, energy_graph, balance_loss

        # ⑥ dense 输出（与 forward() 对齐）
        charge_pred, mask = to_dense_batch(charge_node, b, fill_value=0.0)   # (B,Nmax,1), (B,Nmax)
        elem_pred, _      = to_dense_batch(elem_node,   b, fill_value=0.0)   # (B,Nmax,C)

        geom_dense, _ = to_dense_batch(geom_node, b, fill_value=0.0)         # (B,Nmax,D)
        coords_pred  = self.coord_dec(geom_dense)                             # (B,Nmax,3)
        energy_pred  = energy_graph.unsqueeze(1)                              # (B,1,1)

        feas_logits  = self.head_feas(global_mean_pool(latent_base, b))          # (B,2)
        return (coords_pred, charge_pred, elem_pred, energy_pred,
                mask.unsqueeze(-1), balance_loss, feas_logits)

# ─────────────────────────────────────────────────────────────
#   2.  扩展子类：MultiTaskMoE_TS  —— 过渡帧时序训练
# ─────────────────────────────────────────────────────────────
class MultiTaskMoE_TS(MultiTaskMoE):
    """
    把 EGNN  →  LSTM-TimeSeries  →  MoE  三段计算一次完成  
    1. 加载并冻结已训练好的 EGNN + MoE + Decoder 参数  
    2. 仅训练 LSTM-TS 模块  
    3. forward(batch) 直接输出整条过渡帧预测结果与 mask
    """
    def __init__(self,
                 ts_hidden      : int  = 256,
                 chunk_size     : int  = 16,
                 n_atom_types   : int  = 95,
                 **cfg):
        """
        参数
        ----
        ckpt_path : str
            之前单帧三任务预训练得到的完整 MoE checkpoint。
        ts_hidden : int
            LSTM hidden size.
        chunk_size : int
            TBPTT 片段长度；直接传给 TimeSeriesModule.forward。
        backbone_kw :
            透传给 MultiTaskMoE 的父级 __init__（若需修改 hidden, num_expert 等）。
        """
        """
        cfg 里必须包括 backbone_config 以及 n_experts_* 等关键词。
        """
        # --- 取出必需的 positional 参数 backbone_config ---
        backbone_cfg = cfg.pop('backbone_config')
        n_atom_types = int(cfg.pop('n_atom_types', n_atom_types))
        allowed = {'n_experts_energy', 'n_experts_charge', 'n_experts_geom',
               'topk_energy', 'topk_charge', 'topk_geom', 'topk_elem',
               'expert_layers', 'device'}
        cfg = {k: v for k, v in cfg.items() if k in allowed}

        # 其余 keyword 参数(专家数量/Top-k 等) 直接传给父类
        super().__init__(backbone_cfg, **cfg)

        self.ts = TimeSeriesModule(
            latent_dim      = self.backbone.atom_embedding_dim,
            hidden_size     = ts_hidden
        )
        self.chunk_size = chunk_size

    # ────────────────────────────
    #   forward：过渡帧批量预测
    # ────────────────────────────
    def forward(self, batch):
        """
        输入:
            batch —— PyG Batch；其 .trajectory[t] 存真值第 t 帧 Data
        返回:
            coords_pred : (B,T,Nmax,3)
            charge_pred : (B,T,Nmax,1)
            elem_pred   : (B,T,Nmax,1)
            energy_pred : (B,T,1,1)
            mask        : (B,T,Nmax,1)
            balance_loss: scalar
        """
        device = batch.x.device
        data_list = batch.to_data_list()                     # 拆成原子数可变 list

        # 1) EGNN 编码整条轨迹 latent_seq : (B,T,D,N)
        latent_seq = self.backbone.forward_trajectory(batch)
        if latent_seq.shape[2] == batch.num_nodes:           # 容错: (B,T,N,D) → 调整
            latent_seq = latent_seq.permute(0, 1, 3, 2)

        # 2) LSTM-TS 预测下一帧 latent_pred : (B,T-1,D,N)
        latent_pred, stop_logit = self.ts(latent_seq, chunk_size=self.chunk_size)
        B, T_pred, D, N_pad = latent_pred.shape              # N_pad == batch.num_nodes

        # 3) 计算 batch 内每个分子的最大原子数
        Nmax = max(d.num_nodes for d in data_list)

        # 4) 预分配 padded 张量
        coords_pred = torch.zeros((B, T_pred, Nmax, 3),  device=device)
        charge_pred = torch.zeros((B, T_pred, Nmax, 1),  device=device)
        elem_pred   = torch.zeros((B, T_pred, Nmax, 1), device=device)
        energy_pred = torch.zeros((B, T_pred, 1,    1),  device=device)
        mask_tensor = torch.zeros((B, T_pred, Nmax, 1),  device=device)
        bal_accum   = torch.zeros([], device=device)

        # 5) 逐分子 × 逐帧 推断
        for b_idx, data_static in enumerate(data_list):
            n_i = data_static.num_nodes

            ensure_batch_attr(data_static, device)
            
            # print('data_static batch:',data_static.batch)
            for t in range(T_pred):
                # (N_i,D)
                latent_i = latent_pred[b_idx, t, :, :n_i].transpose(0,1).contiguous()
                # print('latent_i shape:',latent_i.shape)
                # MoE 路由预测
                c_p, q_p, z_p, e_p, bal_p = self.forward_from_latent(latent_i, data_static)

                coords_pred[b_idx, t, :n_i] = c_p
                charge_pred[b_idx, t, :n_i] = q_p
                elem_pred  [b_idx, t, :n_i] = z_p
                energy_pred[b_idx, t, 0, 0] = e_p.squeeze()
                mask_tensor[b_idx, t, :n_i, 0] = 1.0

                bal_accum += bal_p

        # 平均 balance loss
        balance_loss = bal_accum / (B * T_pred)

        return (coords_pred,          # (B,T,Nmax,3)
                charge_pred,          # (B,T,Nmax,1)
                elem_pred,            # (B,T,Nmax,1)
                energy_pred,          # (B,T,1,1)
                mask_tensor,          # (B,T,Nmax,1)
                balance_loss,
                stop_logit)           # (B,T)