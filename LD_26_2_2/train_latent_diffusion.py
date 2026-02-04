# train_latent_diffusion.py
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from tqdm import tqdm

from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from latent_diffusion import DiffusionConfig, DenoiserTransformer, LatentDiffusion

# ---- Your AE ----
from moe_transformer_multitask import MultiTaskMoE


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def kabsch_rmsd_batch(P: torch.Tensor, Q: torch.Tensor, mask: torch.Tensor) -> float:
    """Kabsch-aligned RMSD averaged over batch.
    P, Q: (B, N, 3); mask: (B, N) bool
    """
    if mask.dim() == 3:
        mask = mask.squeeze(-1)
    mask = mask.to(dtype=torch.bool)
    B = P.size(0)
    rmsds = []
    for i in range(B):
        m = mask[i]
        if m.sum() < 2:
            continue
        p = P[i, m].to(dtype=torch.float32)
        q = Q[i, m].to(dtype=torch.float32)
        p = p - p.mean(dim=0, keepdim=True)
        q = q - q.mean(dim=0, keepdim=True)

        C = p.transpose(0, 1) @ q
        V, S, Wt = torch.linalg.svd(C, full_matrices=False)
        d = torch.sign(torch.det(V @ Wt))
        D = torch.diag(torch.tensor([1.0, 1.0, float(d.item())], device=P.device, dtype=torch.float32))
        U = V @ D @ Wt
        p_aligned = p @ U
        diff = p_aligned - q
        rmsd = torch.sqrt((diff * diff).sum(dim=-1).mean())
        rmsds.append(rmsd)
    if not rmsds:
        return float("nan")
    return float(torch.stack(rmsds).mean().item())


@torch.no_grad()
def denoise_from_t(
    diffusion: "LatentDiffusion",
    zt: torch.Tensor,
    node_mask: torch.Tensor,
    t_start: int,
) -> torch.Tensor:
    """从给定的 z_t 开始，按 t_start→0 逐步反向采样，得到 z_0。"""
    B = zt.size(0)
    z = zt
    for step in range(int(t_start), -1, -1):
        t = torch.full((B,), step, device=zt.device, dtype=torch.long)
        z = diffusion.p_sample(z, t, node_mask=node_mask)
    return z

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def torch_save_atomic(obj: Any, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# -----------------------------
# Dataset adapters (minimal)
# -----------------------------
class SliceDataset(Dataset):
    def __init__(self, base: Dataset, start: int = 0, end: int = -1) -> None:
        self.base = base
        self.start = max(0, int(start))
        self.end = int(end)

        n = len(base)
        if self.end < 0 or self.end > n:
            self.end = n
        if self.start > self.end:
            self.start = self.end

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx: int):
        return self.base[self.start + idx]


class OnlyStableWrapper(Dataset):
    """
    Best-effort: ensure Data.pos is a stable (single-frame) tensor, and remove any
    trajectory-like attributes that could break batching.
    """
    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        data = self.base[idx]

        # common patterns: trajectory (T,N,3) or list of frames
        if hasattr(data, "trajectory") and data.trajectory is not None:
            traj = data.trajectory
            if torch.is_tensor(traj) and traj.dim() == 3:
                data.pos = traj[-1]
            # remove to avoid Batch issues
            try:
                delattr(data, "trajectory")
            except Exception:
                pass

        # other possible names (best effort)
        for name in ["traj", "pos_traj", "coords_traj", "trajectory_pos", "trajectory_coords"]:
            if hasattr(data, name):
                try:
                    delattr(data, name)
                except Exception:
                    pass

        return data


class SafeResampleDataset(Dataset):
    def __init__(self, base: Dataset, max_retries: int = 8, name: str = "dataset") -> None:
        self.base = base
        self.max_retries = int(max_retries)
        self.name = name

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        last_err: Optional[Exception] = None
        for _ in range(self.max_retries):
            try:
                return self.base[idx]
            except Exception as e:
                last_err = e
                idx = random.randrange(len(self.base))
        raise RuntimeError(f"[{self.name}] failed after {self.max_retries} retries. last_err={last_err}")


def create_dataset(data_root: str, start_index: int = 0, end_index: int = -1) -> Dataset:
    """
    Mirrors your pretrain style:
      - if endswith .lmdb -> LmdbMoleculeDataset
      - else -> MoleculeDataset (no charge)
    """
    if data_root.endswith(".lmdb"):
        from dataset_lmdb import LmdbMoleculeDataset  # your project module
        base = LmdbMoleculeDataset(data_root)
    else:
        from dataset_without_charge import MoleculeDataset  # your project module
        base = MoleculeDataset(data_root)

    if start_index != 0 or end_index != -1:
        base = SliceDataset(base, start=start_index, end=end_index)
    return base


def pyg_collate(data_list: List[Any]) -> Batch:
    return Batch.from_data_list(data_list)


# -----------------------------
# AE checkpoint loader (matches your save_checkpoint format)
# -----------------------------
def _get(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in cfg:
            return cfg[k]
    return default


def load_ae_from_ckpt(ckpt_path: str, device: torch.device) -> Tuple[MultiTaskMoE, Dict[str, Any]]:
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


# -----------------------------
# Train / Eval
# -----------------------------
@torch.no_grad()
def eval_one_epoch(
    ae: MultiTaskMoE,
    diffusion: LatentDiffusion,
    loader,
    device: torch.device,
    amp: bool,
    max_batches: int = 50,
    latent_scale: float = 1.0,
    eval_downstream: bool = False,
    downstream_t: int = 200,
    latent_dim: Optional[int] = None,
):
    diffusion.eval()
    num = 0.0
    denom = 0.0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = batch.to(device)
            with torch.cuda.amp.autocast(enabled=amp):
                (coords_pred, charge_pred, elem_pred, energy_pred,
                 node_mask, bal_loss, feas_logits, latent, kl_loss) = ae(batch, return_latent=True)
                latent = latent.detach() * float(latent_scale)
                node_mask = node_mask.detach()
                loss = diffusion.p_losses(latent, node_mask)

            # 按有效节点数加权（更稳定、更可比）
            eff_nodes = float(node_mask.sum().item())  # node_mask: (B,N,1)
            d = int(latent_dim) if latent_dim is not None else int(latent.size(-1))
            w = eff_nodes * d
            num += float(loss.item()) * w
            denom += w
    diffusion.train()
    return (num / max(denom, 1.0)) if denom > 0 else float("nan")


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)

    ensure_dir(args.save_dir)
    with open(os.path.join(args.save_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # ---- load AE (frozen) ----
    ae, ae_cfg = load_ae_from_ckpt(args.ae_ckpt, device=device)

    # infer latent_dim from AE cfg if possible, else from args
    # infer latent_dim (we diffuse the low-dim bottleneck latent returned by AE: latent_low_dense)
    latent_dim = int(args.latent_dim)
    if latent_dim <= 0:
        bottleneck_dim = int(ae_cfg.get("latent_bottleneck_dim", 0))
        if bottleneck_dim > 0:
            latent_dim = bottleneck_dim
        else:
            bc = ae_cfg.get("backbone_config", {})
            latent_dim = int(bc.get("atom_embedding_dim", 0))
        if latent_dim <= 0:
            raise ValueError(
                "Cannot infer latent_dim. Please set --latent_dim explicitly "
                "(should match AE returned latent_low_dense last dim)."
            )

    # ---- diffusion model ----
    dcfg = DiffusionConfig(
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
        linear_beta_start=args.linear_beta_start,
        linear_beta_end=args.linear_beta_end,
        cosine_s=args.cosine_s,
        clip_denoised=args.clip_denoised,
    )
    denoiser = DenoiserTransformer(
        latent_dim=latent_dim,
        model_dim=args.model_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        time_embed_dim=args.time_embed_dim,
        use_cond=False,
    ).to(device)
    diffusion = LatentDiffusion(denoiser=denoiser, cfg=dcfg).to(device)

    opt = torch.optim.AdamW(diffusion.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 0
    global_step = 0

    # resume
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        diffusion.load_state_dict(ck["diffusion"], strict=True)
        opt.load_state_dict(ck["opt"])
        start_epoch = int(ck.get("epoch", 0))
        global_step = int(ck.get("step", 0))
        if "scaler" in ck and args.amp:
            scaler.load_state_dict(ck["scaler"])

    # ---- data ----
    train_sets: List[Dataset] = []
    for did, root in enumerate(args.data_roots):
        ds = create_dataset(root, start_index=args.start_index, end_index=args.end_index)
        ds = OnlyStableWrapper(ds)
        ds = SafeResampleDataset(ds, max_retries=args.max_retries, name=f"train[{did}]")
        train_sets.append(ds)
    train_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=pyg_collate,
    )

    val_loader: Optional[DataLoader] = None
    if args.val_data_roots:
        val_sets: List[Dataset] = []
        for did, root in enumerate(args.val_data_roots):
            ds = create_dataset(root, start_index=args.val_start_index, end_index=args.val_end_index)
            ds = OnlyStableWrapper(ds)
            ds = SafeResampleDataset(ds, max_retries=args.max_retries, name=f"val[{did}]")
            val_sets.append(ds)
        val_ds = ConcatDataset(val_sets) if len(val_sets) > 1 else val_sets[0]
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=pyg_collate,
        )

    # ---- train loop ----
    best_val = float("inf")
    for epoch in range(start_epoch, args.epochs):
        diffusion.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True)

        # epoch-level loss accumulator (node-weighted)
        epoch_num = 0.0
        epoch_denom = 0.0
        epoch_batches = 0
        
        for batch in pbar:
            batch = batch.to(device)

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.amp):
                # ae forward -> latent & mask
                (coords_pred, charge_pred, elem_pred, energy_pred,
                 node_mask, bal_loss, feas_logits, latent, kl_loss) = ae(batch, return_latent=True)

                latent = latent.detach() * float(args.latent_scale)
                node_mask = node_mask.detach()

            with torch.cuda.amp.autocast(enabled=args.amp):
                loss = diffusion.p_losses(latent, node_mask)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(diffusion.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            global_step += 1
            # update epoch avg (node-weighted)
            eff_nodes = float(node_mask.sum().item())
            w = eff_nodes * float(latent_dim)
            epoch_num += float(loss.item()) * w
            epoch_denom += w
            epoch_batches += 1
            epoch_avg = epoch_num / max(epoch_denom, 1.0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{epoch_avg:.4f}", step=global_step)

            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                ckpt_path = os.path.join(args.save_dir, f"ldm_step{global_step}.pt")
                torch_save_atomic(
                    {
                        "diffusion": diffusion.state_dict(),
                        "opt": opt.state_dict(),
                        "scaler": scaler.state_dict() if args.amp else None,
                        "epoch": epoch,
                        "step": global_step,
                        "diffusion_cfg": asdict(dcfg),
                        "latent_dim": latent_dim,
                    },
                    ckpt_path,
                )

        # epoch end save
        ckpt_path = os.path.join(args.save_dir, f"ldm_epoch{epoch+1}.pt")
        torch_save_atomic(
            {
                "diffusion": diffusion.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if args.amp else None,
                "epoch": epoch + 1,
                "step": global_step,
                "diffusion_cfg": asdict(dcfg),
                "latent_dim": latent_dim,
            },
            ckpt_path,
        )
        
        train_avg = epoch_num / max(epoch_denom, 1.0)
        print(f"[TRAIN] epoch={epoch+1} avg_loss={train_avg:.6f} batches={epoch_batches} "
              f"(node-weighted, latent_scale={args.latent_scale}, latent_dim={latent_dim})")

        # validation
        if val_loader is not None and args.val_batches > 0:
            val_loss = eval_one_epoch(
                ae, diffusion, val_loader,
                device=device, max_batches=args.val_batches,
                amp=args.amp,
                latent_scale=float(args.latent_scale),
                latent_dim=int(latent_dim),
            )
            if val_loss < best_val:
                best_val = val_loss
                best_path = os.path.join(args.save_dir, "ldm_best.pt")
                torch_save_atomic(
                    {
                        "diffusion": diffusion.state_dict(),
                        "opt": opt.state_dict(),
                        "scaler": scaler.state_dict() if args.amp else None,
                        "epoch": epoch + 1,
                        "step": global_step,
                        "diffusion_cfg": asdict(dcfg),
                        "latent_dim": latent_dim,
                        "best_val": best_val,
                    },
                    best_path,
                )
            print(f"[VAL] epoch={epoch+1} val_loss={val_loss:.6f} best={best_val:.6f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Latent Diffusion Training (AE frozen)")

    # AE / data
    p.add_argument("--ae_ckpt", type=str, required=True)
    p.add_argument("--data_roots", type=str, nargs="+", required=True)
    p.add_argument("--val_data_roots", type=str, nargs="*", default=[])

    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--end_index", type=int, default=-1)
    p.add_argument("--val_start_index", type=int, default=0)
    p.add_argument("--val_end_index", type=int, default=-1)

    # diffusion hyper
    p.add_argument("--latent_dim", type=int, default=-1, help="<=0 will try infer from AE cfg")
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="cosine", choices=["linear", "cosine"])
    p.add_argument("--linear_beta_start", type=float, default=1e-4)
    p.add_argument("--linear_beta_end", type=float, default=2e-2)
    p.add_argument("--cosine_s", type=float, default=0.008)
    p.add_argument("--clip_denoised", action="store_true")

    # denoiser net
    p.add_argument("--model_dim", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--ff_mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--time_embed_dim", type=int, default=256)

    # train
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--latent_scale", type=float, default=1.0, help="scale latent before diffusion (training stability)")

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_retries", type=int, default=8)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)

    # io
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--save_every_steps", type=int, default=0)
    p.add_argument("--resume", type=str, default="")

    # val
    p.add_argument("--val_batches", type=int, default=50)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)
