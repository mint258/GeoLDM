import argparse
import os
import torch
import numpy as np
import periodictable
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from typing import Any, Dict, List, Optional, Tuple

# 导入你的模型定义
from moe_transformer_multitask import MultiTaskMoE

# -----------------------------------------------------------------------------
# 1. 辅助函数：加载模型 (逻辑复用自 train_latent_diffusion.py)
# -----------------------------------------------------------------------------
def _get(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in cfg:
            return cfg[k]
    return default

def load_frozen_ae(ckpt_path: str, device: torch.device) -> Tuple[MultiTaskMoE, Dict[str, Any]]:
    sd = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(sd, dict) or "cfg" not in sd:
        raise ValueError(
            f"AE ckpt must be a dict with key 'cfg' (and 'moe_state', 'dec_*_state'). got keys={getattr(sd, 'keys', lambda: [])()}"
        )

    cfg = sd["cfg"]
    if not isinstance(cfg, dict):
        # sometimes saved as Namespace-like
        cfg = dict(cfg)

    backbone_config = _get(cfg, "backbone_config", "backbone_cfg")
    if backbone_config is None:
        raise ValueError("cfg must contain 'backbone_config'")

    model = MultiTaskMoE(
        backbone_config,
        n_atom_types=int(_get(cfg, "n_atom_types", default=95)),
        n_experts_energy=int(_get(cfg, "n_experts_energy", "expert_number_E", default=4)),
        n_experts_charge=int(_get(cfg, "n_experts_charge", "expert_number_Q", default=4)),
        n_experts_geom=int(_get(cfg, "n_experts_geom", "expert_number_G", default=4)),
        topk_energy=int(_get(cfg, "topk_energy", default=1)),
        topk_charge=int(_get(cfg, "topk_charge", default=1)),
        topk_geom=int(_get(cfg, "topk_geom", default=1)),
        topk_elem=int(_get(cfg, "topk_elem", default=1)),
        expert_layers=int(_get(cfg, "expert_layers", default=2)),
        device=str(device),
        latent_bottleneck_dim=int(_get(cfg, "latent_bottleneck_dim", default=0)),
    ).to(device)

    # load shared params
    if "moe_state" in sd:
        model.load_state_dict(sd["moe_state"], strict=False)
    else:
        # fallback: try load directly
        model.load_state_dict(sd, strict=False)

    # load task heads (your ckpt stores them separately)
    def load_dec(dec_key: str, map_attr: str, head_attr: str) -> None:
        if dec_key not in sd:
            return
        dec = sd[dec_key]
        if isinstance(dec, dict):
            if "map" in dec and hasattr(model, map_attr):
                getattr(model, map_attr).load_state_dict(dec["map"], strict=True)
            if "head" in dec and hasattr(model, head_attr):
                getattr(model, head_attr).load_state_dict(dec["head"], strict=True)

    load_dec("dec_energy_state", "map_energy", "head_energy")
    load_dec("dec_charge_state", "map_charge", "head_charge")
    load_dec("dec_geom_state", "map_geom", "coord_dec")   # coord_dec is not exactly "head", but stored under "head" in ckpt
    # For geom, your save_checkpoint stores:
    #   dec_G_state = {'map': map_geom.state_dict(), 'head': coord_dec.state_dict()}
    if "dec_geom_state" in sd and isinstance(sd["dec_geom_state"], dict) and "decoder" in sd["dec_geom_state"]:
        model.coord_dec.load_state_dict(sd["dec_geom_state"]["decoder"], strict=True)

    load_dec("dec_elem_state", "map_elem", "head_elem")
    load_dec("dec_feas_state", "map_feas", "head_feas")   # if you have map_feas; if not, this will no-op

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, cfg

# -----------------------------------------------------------------------------
# 2. 核心逻辑：采样与生成
# -----------------------------------------------------------------------------
def generate_samples(model, args, device):
    os.makedirs(args.out_dir, exist_ok=True)
    
    # (1) 准备 Latent Z
    # 你的 latent_dim 是 32 (根据你的日志)
    # 如果 cfg 里有 bottleneck，优先用 bottleneck dim
    latent_dim = args.latent_dim
    if latent_dim is None:
        if model.latent_bottleneck is not None:
            latent_dim = model.latent_bottleneck_dim
        else:
            latent_dim = model.latent_dim
    
    print(f"Sampling config: Scale={args.scale}, Dim={latent_dim}, MaxAtoms={args.max_atoms}")
    
    # 生成随机噪声：标准正态分布 * 缩放因子
    # Shape: (Batch, Max_Atoms, Latent_Dim)
    # 这里的 Max_Atoms 是我们假设的最大原子数，用于生成 Point Cloud
    z = torch.randn(args.num_samples, args.max_atoms, latent_dim).to(device)
    z_scaled = z * args.scale

    # (2) 伪造 Data Batch 信息
    # MultiTaskMoE 需要 batch 向量来区分图，还需要 global_mean_pool
    # 我们构造一个假的 PyG Batch
    # batch_vec: [0,0,0... (N次), 1,1,1... (N次), ...]
    batch_vec = torch.arange(args.num_samples).view(-1, 1).repeat(1, args.max_atoms).view(-1)
    batch_vec = batch_vec.to(device)
    
    # 构造一个假的 Data 对象来承载 batch
    # 不需要 edge_index，因为你的 ETExpert 是基于 transformer/attention 的，内部自己算距离
    dummy_data = Data()
    dummy_data.batch = batch_vec
    dummy_data.num_nodes = args.num_samples * args.max_atoms
    
    # (3) 解码
    print("Decoding...")
    with torch.no_grad():
        # 调用 forward_from_latent
        # 注意：这里 z_scaled 是 (B, N, D)，我们传入 return_dense=True
        # 我们需要提供 node_mask，这里假设所有生成的原子都是“真实”的（不padding），以此测试最大能力
        # 或者我们可以随机生成一些长度，这里为了看空洞，先全部生成
        full_mask = torch.ones(args.num_samples, args.max_atoms, device=device).bool()
        
        # 你的 forward_from_latent 逻辑会自动把 dense z 转成 node-level z
        # 然后跑 MoE, 然后再转回 dense
        (coords_pred, charge_pred, elem_pred, energy_pred, 
         mask_out, balance_loss, feas_logits) = model.forward_from_latent(
            z_scaled, 
            dummy_data, 
            node_mask=full_mask, 
            return_dense=True
        )

    # (4) 保存结果
    print(f"Saving {args.num_samples} xyz files to {args.out_dir} ...")
    
    # 准备元素表映射: index -> Symbol
    # 数据集通常 padding 在最后，index = 94 (如果 n_atom_types=95)
    # 我们需要一个 index 到 元素符号的映射
    pad_idx = model.pad_elem_index
    
    # 获取原子序数 (Z)
    # elem_pred: (B, N, n_classes) -> argmax -> (B, N)
    atom_types = torch.argmax(elem_pred, dim=-1).cpu().numpy()
    coords = coords_pred.cpu().numpy()
    charges = charge_pred.squeeze(-1).cpu().numpy()
    energies = energy_pred.squeeze().cpu().numpy()

    valid_count = 0
    
    for i in tqdm(range(args.num_samples)):
        filename = os.path.join(args.out_dir, f"sample_{i:04d}_scale{args.scale}.xyz")
        
        mol_coords = coords[i]
        mol_atoms = atom_types[i]
        mol_charges = charges[i]
        
        # 过滤掉 PAD 原子 (如果模型预测出了 PAD)
        # 或者过滤掉靠得太近的原子（物理检查）
        
        content = []
        real_atom_count = 0
        
        for j in range(args.max_atoms):
            idx = mol_atoms[j]
            if idx == pad_idx: 
                continue # 跳过 Pad
            
            # 你的数据集 index 是 Z-1 (0-based) 还是 Z (1-based)? 
            # 代码 pretrain 里的 dataset_without_charge.py 显示: 
            # z = element_dict[atom] -> periodictable.elements 的 enumerate
            # periodictable enumerate 0 是 Neutron(n), 1 是 H.
            # 通常 index 0 对应 Z=0? 或者 index 0 对应 H(Z=1)?
            # 假设 dataset 用的是 periodictable 的默认顺序:
            # list(periodictable.elements)[0] 是 n (Z=0), [1] 是 H, [6] 是 C
            try:
                # 获取元素符号
                el = periodictable.elements[idx]
                symbol = el.symbol
            except:
                symbol = "X"
            
            x, y, z_coord = mol_coords[j]
            q = mol_charges[j]
            
            # 简单的物理检查：如果坐标是 NaN 或 Inf
            if np.isnan(x) or np.isinf(x):
                continue

            content.append(f"{symbol:<4} {x:10.5f} {y:10.5f} {z:10.5f} {q:10.5f}")
            real_atom_count += 1
            
        # 写入文件
        if real_atom_count > 0:
            with open(filename, 'w') as f:
                # 第一行：原子数
                f.write(f"{real_atom_count}\n")
                # 第二行：注释（能量预测值）
                f.write(f"Generated by VAE Scale={args.scale} | Pred Energy={energies[i]:.4f}\n")
                # 原子行
                f.write("\n".join(content))
            valid_count += 1

    print(f"Done. {valid_count}/{args.num_samples} files written (filtered empty).")

# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="训练好的模型路径")
    parser.add_argument("--out_dir", type=str, default="./latent_holes_test", help="输出XYZ的文件夹")
    parser.add_argument("--scale", type=float, default=2.18, help="潜空间采样的缩放因子 (std)")
    parser.add_argument("--num_samples", type=int, default=100, help="采样数量")
    parser.add_argument("--max_atoms", type=int, default=25, help="每个分子生成的最大原子数")
    parser.add_argument("--latent_dim", type=int, default=None, help="强制指定Latent Dim，不指定则从模型读取")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    model, _ = load_frozen_ae(args.ckpt_path, device)
    
    generate_samples(model, args, device)