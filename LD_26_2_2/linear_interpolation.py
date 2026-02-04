#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
linear_interpolation_rdkit_mutate.py

目标
----
1) 从数据集中随机取一个分子作为样本A (Data.x=原子序号, Data.pos=坐标)
2) 基于A的坐标用 RDKit 推断键并生成 canonical SMILES
3) 对A做“等原子数、低风险”的基团/原子替换，得到化学上可行的样本B（SMILES可sanitize）
   - 默认：卤素互换 (F/Cl/Br/I) 或 O<->S
   - 这类替换保持原子数不变，便于在 latent 空间做线性插值
4) 编码得到 zA, zB，在二者之间做线性插值，逐点解码得到坐标
5) 对每个插值点统计：
   - 原子最小距离、重叠对数（距离 < overlap_thresh）
   - 用 RDKit 从(原子序号, 坐标)重建分子并 sanitize 的成功率（作为“价键/化学合理性” proxy）
   -（可选）输出每一步的 xyz 与 SMILES

说明
----
- 该脚本用于检查“样本分布区域附近的 latent 空间是否存在大量不合理结构”（空洞）。
- RDKit 的 DetermineBonds 是基于共价半径/距离的启发式重建键，适合作为快速合理性筛查。
- 若你的数据坐标本身与 RDKit 的键推断假设偏离较大，A 的 SMILES 也可能构建失败；脚本会自动重采样。

依赖
----
- torch, rdkit
- torch_geometric（用于 Batch.from_data_list）；若缺失会给出友好报错
- 你的工程模块：train_latent_diffusion.py / dataset_lmdb.py / moe_transformer_multitask.py 等

用法示例
--------
python linear_interpolation_rdkit_mutate.py \
  --ae_ckpt small_scale_AE.pth_epoch_30.pt \
  --data_root ../../dataset/unimol/unimol_1000000_2000000.lmdb \
  --start_index 0 --end_index 50000 \
  --steps 11 \
  --out_dir interp_probe/run1 \
  --overlap_thresh 0.65 \
  --save_xyz --save_smiles
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

# -------- optional dependency guard (torch_geometric) --------
try:
    from torch_geometric.data import Batch, Data
except Exception as e:  # pragma: no cover
    raise ImportError(
        "torch_geometric 未安装或不可用。该脚本需要 torch_geometric.data.Batch/Data。"
    ) from e

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
LOGGER = logging.getLogger("linear_interp")


# ---------------- dataclasses ----------------
@dataclass
class MutateInfo:
    mode: str
    atom_idx: int
    old_z: int
    new_z: int
    smiles_a: str
    smiles_b: str


@dataclass
class StepMetric:
    step: int
    alpha: float
    rdkit_ok: bool
    rdkit_smiles: str
    min_pair_dist: float
    num_overlap_pairs: int


# ---------------- utils ----------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomicnum_to_symbol(z: int) -> str:
    try:
        return Chem.GetPeriodicTable().GetElementSymbol(int(z))
    except Exception:
        return "X"


def write_xyz(path: Path, z: torch.Tensor, pos: torch.Tensor, comment: str = "") -> None:
    z_np = z.view(-1).detach().cpu().numpy()
    pos_np = pos.detach().cpu().numpy()
    lines: List[str] = [str(len(z_np)), comment]
    for zi, (x, y, zz) in zip(z_np, pos_np):
        lines.append(f"{atomicnum_to_symbol(int(zi))} {x:.8f} {y:.8f} {zz:.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def is_node_level_tensor(v: Any, n: int) -> bool:
    return torch.is_tensor(v) and v.dim() >= 1 and int(v.size(0)) == int(n)


def trim_padding_like_model(data: Data, pad_idx: int) -> Tuple[Data, torch.Tensor]:
    """
    依据模型 pad_elem_index 做裁剪，返回 (trimmed_data, mask).
    会尽量对所有“节点维度 = N”的张量做同步裁剪（x/pos/y/charge_mask/...）。
    """
    x = data.x.view(-1).long()
    n = int(x.numel())
    mask = x != int(pad_idx)

    dct = data.to_dict()
    out = Data()
    for k, v in dct.items():
        out[k] = v

    for k, v in list(out.items()):
        if is_node_level_tensor(v, n):
            out[k] = v[mask]

    out.x = out.x.view(-1, 1).long()
    out.pos = out.pos.float()
    return out, mask


def build_rdkit_mol_from_z_pos(z: Sequence[int], pos: np.ndarray) -> Chem.Mol:
    """
    用 (atomic numbers, coords) 构建 RDKit Mol，并用 DetermineBonds 推断键。
    成功返回已 sanitize 的 Mol；失败会抛异常。
    """
    mol = Chem.RWMol()
    for zi in z:
        mol.AddAtom(Chem.Atom(int(zi)))
    mol = mol.GetMol()

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        x, y, zz = pos[i]
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(float(x), float(y), float(zz)))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)

    rdDetermineBonds.DetermineBonds(mol, charge=0)  # may raise
    Chem.SanitizeMol(mol)  # may raise
    return mol


def mol_to_canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)


# ---------------- mutation (keep atom count) ----------------
HALOGENS = (9, 17, 35, 53)  # F, Cl, Br, I


def propose_mutations(mol: Chem.Mol, mode: str) -> List[Tuple[int, int]]:
    """
    返回候选 (atom_idx, new_atomic_num) 列表。
    mode:
      - halogen_swap: 只在卤素之间互换
      - os_swap:      O <-> S
      - mixed:        上述两种混合
    """
    props: List[Tuple[int, int]] = []
    for a in mol.GetAtoms():
        idx = a.GetIdx()
        z = a.GetAtomicNum()

        if mode in ("halogen_swap", "mixed") and z in HALOGENS:
            for nz in HALOGENS:
                if nz != z:
                    props.append((idx, nz))

        if mode in ("os_swap", "mixed") and z in (8, 16):  # O or S
            nz = 16 if z == 8 else 8
            props.append((idx, nz))
    return props


def mutate_mol_keep_atoms(mol_a: Chem.Mol, mode: str, rng: random.Random, max_trials: int = 128) -> Tuple[Chem.Mol, Tuple[int, int, int]]:
    """
    在不改变原子数、不改变原子顺序的前提下做替换，保证 B 可 sanitize。
    返回 (mol_b, (atom_idx, old_z, new_z)).
    """
    cand = propose_mutations(mol_a, mode)
    if not cand:
        raise RuntimeError(f"No mutation candidates found for mode={mode}")

    for _ in range(max_trials):
        atom_idx, new_z = rng.choice(cand)
        old_z = mol_a.GetAtomWithIdx(atom_idx).GetAtomicNum()

        rw = Chem.RWMol(mol_a)
        rw.GetAtomWithIdx(atom_idx).SetAtomicNum(int(new_z))
        mol_b = rw.GetMol()
        try:
            Chem.SanitizeMol(mol_b)
            return mol_b, (atom_idx, int(old_z), int(new_z))
        except Exception:
            continue

    raise RuntimeError(f"Failed to produce a sanitizable mutated mol in {max_trials} trials.")


# ---------------- geometry checks ----------------
def overlap_stats(pos: torch.Tensor, thresh: float) -> Tuple[float, int]:
    """
    返回 (min_pair_dist, num_pairs_with_dist<thresh).
    pos: (N,3)
    """
    if pos.size(0) < 2:
        return float("inf"), 0
    d = torch.cdist(pos, pos)
    iu = torch.triu_indices(d.size(0), d.size(1), offset=1)
    upper = d[iu[0], iu[1]]
    min_d = float(upper.min().item())
    n_bad = int((upper < float(thresh)).sum().item())
    return min_d, n_bad


def rdkit_check_from_tensor(z: torch.Tensor, pos: torch.Tensor) -> Tuple[bool, str, str]:
    """
    从 (z,pos) 尝试构建 RDKit mol 并 sanitize。
    返回 (ok, smiles, err_msg).
    """
    z_list = [int(v + 1) for v in z.view(-1).detach().cpu().tolist()]
    p = pos.detach().cpu().numpy()
    try:
        mol = build_rdkit_mol_from_z_pos(z_list, p)
        return True, mol_to_canonical_smiles(mol), ""
    except Exception as e:
        return False, "", str(e)


# ---------------- model I/O (reuse your project loader) ----------------
def load_model_and_dataset(ae_ckpt: str, data_root: str, start_index: int, end_index: int, device: torch.device):
    """
    复用 train_latent_diffusion.py 的 loader/create_dataset，避免 ckpt 格式与数据集构建不一致。
    """
    from train_latent_diffusion import load_ae_from_ckpt, create_dataset

    ae, cfg = load_ae_from_ckpt(ae_ckpt, device=device)
    ae.eval()

    ds = create_dataset(data_root, start_index=start_index, end_index=end_index)
    return ae, cfg, ds


@torch.no_grad()
def encode_latent_low_dense(model: torch.nn.Module, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回:
      latent_low_dense: (B, Nmax, D)
      node_mask:        (B, Nmax)  True=real atom
    与旧脚本一致：out[-2] / out[4]
    """
    out = model(batch, return_latent=True)
    latent_low_dense = out[-2]
    node_mask = out[4]
    return latent_low_dense, node_mask


@torch.no_grad()
def decode_from_latent_low_dense(model: torch.nn.Module, z_low_dense: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """
    z_low_dense: (B,N,D)
    node_mask:   (B,N) bool
    返回 coords_dense: (B,N,3)
    """
    node_mask_3d = node_mask.float().unsqueeze(-1)  # (B,N,1)
    coords_dense = model.forward_from_latent(
        z_low_dense,
        node_mask=node_mask_3d,
        latent_is_low=True,
        return_dense=True,
    )
    return coords_dense


# ---------------- main workflow ----------------
@torch.no_grad()
def pick_valid_sample_a(ds, pad_idx: int, rng: random.Random, max_trials: int) -> Tuple[Data, Chem.Mol, str, int]:
    """
    随机选 A，要求能从 (z,pos) 构建 RDKit mol 并 sanitize。
    返回 (data_a_trim, mol_a, smiles_a, raw_index).
    """
    n = len(ds)
    if n <= 0:
        raise RuntimeError("Dataset is empty.")

    for _ in range(max_trials):
        raw_idx = rng.randrange(n)
        data = ds[raw_idx]
        data_trim, _ = trim_padding_like_model(data, pad_idx)

        ok, _smiles, _err = rdkit_check_from_tensor(data_trim.x.view(-1), data_trim.pos)
        if not ok:
            continue

        mol_a = build_rdkit_mol_from_z_pos(
            [int(v + 1) for v in data_trim.x.view(-1).tolist()],
            data_trim.pos.detach().cpu().numpy(),
        )
        smiles_a = mol_to_canonical_smiles(mol_a)
        return data_trim, mol_a, smiles_a, int(raw_idx)

    raise RuntimeError(f"Failed to find a valid sample A in {max_trials} trials.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ae_ckpt", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--end_index", type=int, default=-1)

    p.add_argument("--steps", type=int, default=11, help="插值点数（含端点），>=2")
    p.add_argument("--overlap_thresh", type=float, default=0.65, help="原子对距离小于该值视为重叠(Å)")
    p.add_argument("--mutation_mode", type=str, default="mixed", choices=["halogen_swap", "os_swap", "mixed"])
    p.add_argument("--max_sample_trials", type=int, default=256, help="寻找可 RDKit sanitize 的样本A的最大重采样次数")
    p.add_argument("--max_mut_trials", type=int, default=256, help="对A做替换并得到可 sanitize 的B的最大尝试次数")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_dir", type=str, default="linear_interp_out")

    p.add_argument("--save_xyz", action="store_true", help="输出 A/B 与每一步解码坐标的 xyz")
    p.add_argument("--save_smiles", action="store_true", help="输出每一步 RDKit 重建的 SMILES（若成功）")
    args = p.parse_args()

    if args.steps < 2:
        raise ValueError("--steps must be >= 2")

    set_seed(args.seed)
    rng = random.Random(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = ensure_dir(args.out_dir)
    (out_dir / "xyz").mkdir(parents=True, exist_ok=True)

    # ---- load model/dataset ----
    LOGGER.info("Loading AE ckpt and dataset ...")
    ae, _cfg, ds = load_model_and_dataset(
        ae_ckpt=args.ae_ckpt,
        data_root=args.data_root,
        start_index=args.start_index,
        end_index=args.end_index,
        device=device,
    )
    pad_idx = int(getattr(ae, "pad_elem_index", 0))
    LOGGER.info("pad_elem_index=%d, dataset_len=%d", pad_idx, len(ds))

    # ---- pick A ----
    LOGGER.info("Sampling a valid A (RDKit-sanitizable) ...")
    data_a, mol_a, smiles_a, raw_idx = pick_valid_sample_a(ds, pad_idx, rng, args.max_sample_trials)

    # ---- mutate to get B ----
    LOGGER.info("Mutating A to get a sanitizable B (mode=%s) ...", args.mutation_mode)
    mol_b, (mut_idx, old_z, new_z) = mutate_mol_keep_atoms(mol_a, args.mutation_mode, rng, max_trials=args.max_mut_trials)
    smiles_b = mol_to_canonical_smiles(mol_b)

    # B 的 Data：保持坐标不变，仅修改原子序号
    x_b = data_a.x.clone().view(-1)
    if int(x_b[mut_idx].item()) != int(old_z):
        LOGGER.warning(
            "Atom idx mapping mismatch: data_a.z[%d]=%d but RDKit old_z=%d. Will still set to new_z.",
            mut_idx, int(x_b[mut_idx].item()), int(old_z)
        )
    x_b[mut_idx] = int(new_z - 1)

    data_b = Data(**data_a.to_dict())
    data_b.x = x_b.view(-1, 1).long()
    data_b.pos = data_a.pos.clone()

    mut_info = MutateInfo(
        mode=args.mutation_mode,
        atom_idx=int(mut_idx),
        old_z=int(old_z),
        new_z=int(new_z),
        smiles_a=str(smiles_a),
        smiles_b=str(smiles_b),
    )

    # ---- encode zA / zB ----
    LOGGER.info("Encoding zA / zB ...")
    batch_ab = Batch.from_data_list([data_a, data_b]).to(device)
    z_dense, node_mask = encode_latent_low_dense(ae.to(device), batch_ab)

    n_atoms = int(data_a.x.numel())
    z_a = z_dense[0, :n_atoms].contiguous()
    z_b = z_dense[1, :n_atoms].contiguous()
    node_mask_a = node_mask[0, :n_atoms].bool()
    node_mask_interp = node_mask_a.unsqueeze(0).repeat(args.steps, 1)

    # ---- interpolate in latent space ----
    LOGGER.info("Linear interpolation: steps=%d", args.steps)
    alphas = torch.linspace(0.0, 1.0, args.steps, device=device)
    z_stack = torch.stack([(1.0 - a) * z_a + a * z_b for a in alphas], dim=0)  # (T,N,D)

    # ---- decode each step ----
    data_list = []
    for _ in range(args.steps):
        dd = Data(**data_a.to_dict())
        dd.x = data_a.x.clone()
        dd.pos = data_a.pos.clone()
        data_list.append(dd)
    _batch_static = Batch.from_data_list(data_list).to(device)  # 仅用于生成 batch 信息（模型内部需要）

    coords_dense = decode_from_latent_low_dense(ae, z_stack, node_mask_interp)  # (T,N,3)
    coords_dense = coords_dense[:, :n_atoms]

    # ---- evaluate & export ----
    LOGGER.info("Evaluating overlap + RDKit validity per step ...")
    z_a_cpu = data_a.x.view(-1).cpu()
    metrics: List[StepMetric] = []
    smiles_lines: List[str] = []

    if args.save_xyz:
        write_xyz(out_dir / "xyz" / "A.xyz", data_a.x.view(-1), data_a.pos, comment=f"raw_index={raw_idx} smiles={smiles_a}")
        write_xyz(out_dir / "xyz" / "B.xyz", data_b.x.view(-1), data_b.pos, comment=f"raw_index={raw_idx} smiles={smiles_b} (mut {old_z}->{new_z} at {mut_idx})")

    for i in range(args.steps):
        pos_i = coords_dense[i].detach().cpu()
        min_d, n_bad = overlap_stats(pos_i, args.overlap_thresh)

        ok, smi, err = rdkit_check_from_tensor(z_a_cpu, pos_i)
        if not ok:
            smiles_lines.append(f"{i}\t{float(alphas[i].item()):.6f}\tFAIL\t{err}")
            smi_out = ""
        else:
            smiles_lines.append(f"{i}\t{float(alphas[i].item()):.6f}\tOK\t{smi}")
            smi_out = smi

        metrics.append(
            StepMetric(
                step=i,
                alpha=float(alphas[i].item()),
                rdkit_ok=bool(ok),
                rdkit_smiles=str(smi_out),
                min_pair_dist=float(min_d),
                num_overlap_pairs=int(n_bad),
            )
        )

        if args.save_xyz:
            write_xyz(
                out_dir / "xyz" / f"interp_{i:03d}.xyz",
                data_a.x.view(-1),
                pos_i,
                comment=f"step={i} alpha={float(alphas[i].item()):.6f} rdkit_ok={ok} overlaps<{args.overlap_thresh}A={n_bad}",
            )

    if args.save_smiles:
        (out_dir / "interp_smiles.tsv").write_text(
            "step\talpha\trdkit_ok\tsmiles_or_error\n" + "\n".join(smiles_lines) + "\n",
            encoding="utf-8",
        )

    ok_rate = float(np.mean([m.rdkit_ok for m in metrics])) if metrics else 0.0
    overlap_any_rate = float(np.mean([m.num_overlap_pairs > 0 for m in metrics])) if metrics else 0.0
    min_dist_min = float(np.min([m.min_pair_dist for m in metrics])) if metrics else float("inf")

    summary: Dict[str, Any] = dict(
        args=vars(args),
        raw_index=raw_idx,
        mutate=asdict(mut_info),
        n_atoms=n_atoms,
        rdkit_ok_rate=ok_rate,
        overlap_any_rate=overlap_any_rate,
        min_pair_dist_min=min_dist_min,
        metrics=[asdict(m) for m in metrics],
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    LOGGER.info("Done.")
    LOGGER.info("A SMILES: %s", smiles_a)
    LOGGER.info("B SMILES: %s", smiles_b)
    LOGGER.info("Mutation: atom_idx=%d, %d -> %d", mut_idx, old_z, new_z)
    LOGGER.info("RDKit OK rate over steps: %.3f", ok_rate)
    LOGGER.info("Any-overlap rate over steps: %.3f (thresh=%.2f A)", overlap_any_rate, args.overlap_thresh)
    LOGGER.info("Min pair distance (min over steps): %.3f A", min_dist_min)
    LOGGER.info("Outputs saved to: %s", str(out_dir.resolve()))


if __name__ == "__main__":
    main()
