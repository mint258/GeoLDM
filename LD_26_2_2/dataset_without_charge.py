import os
import re
from torch_geometric.data import Dataset, Data
import torch
import periodictable
import logging

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')
DEBUG = False

def parse_xyz_stable(file_path):
    if DEBUG:
        logging.debug("Parsing stable file: %s", file_path)
    with open(file_path, 'r') as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise ValueError("文件内容不足")
    atom_num = int(lines[0].strip())
    energy_line = lines[1].strip()
    if "SCF Energy:" not in energy_line:
        raise ValueError("稳态文件缺少 SCF Energy 信息")
    try:
        energy = float(energy_line.split()[2])
    except Exception as e:
        raise ValueError(f"能量解析失败: {e}")
    atom_lines = lines[2:2 + atom_num]
    if len(atom_lines) < atom_num:
        raise ValueError("原子行数量不足")
    atom_types = []
    positions = []
    charges = []
    for line in atom_lines:
        parts = line.strip().split()
        if len(parts) < 5:
            raise ValueError("稳态文件中缺少电荷数据")
        atom_types.append(parts[0])
        pos = [float(x.replace('*^', 'e')) for x in parts[1:4]]
        positions.append(pos)
        charge = float(parts[4].replace('*^', 'e'))
        charges.append(charge)
    element_dict = {element.symbol: idx for idx, element in enumerate(periodictable.elements)}
    z = torch.tensor([element_dict[atom] for atom in atom_types], dtype=torch.long)
    pos = torch.tensor(positions, dtype=torch.float)
    charge = torch.tensor(charges, dtype=torch.float).view(-1, 1)
    data = Data(x=z.view(-1, 1), pos=pos, y=charge, energy=energy)
    data.charge_mask = torch.ones(data.pos.shape[0], 1)
    if DEBUG:
        logging.debug("Parsed stable Data: x shape %s, pos shape %s", data.x.shape, data.pos.shape)
    return data

def parse_xyz_transition(file_path):
    if DEBUG:
        logging.debug("Parsing transition file: %s", file_path)
    with open(file_path, 'r') as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise ValueError("文件内容不足")
    atom_num = int(lines[0].strip())
    energy_line = lines[1].strip()
    if "SCF Energy:" not in energy_line:
        raise ValueError("过渡态文件缺少 SCF Energy 信息")
    try:
        energy = float(energy_line.split()[2])
    except Exception as e:
        raise ValueError(f"能量解析失败: {e}")
    atom_lines = lines[2:2 + atom_num]
    if len(atom_lines) < atom_num:
        raise ValueError("原子行数量不足")
    atom_types = []
    positions = []
    for line in atom_lines:
        parts = line.strip().split()
        if len(parts) < 4:
            raise ValueError("过渡态文件数据不完整")
        atom_types.append(parts[0])
        pos = [float(x.replace('*^', 'e')) for x in parts[1:4]]
        positions.append(pos)
    element_dict = {element.symbol: idx for idx, element in enumerate(periodictable.elements)}
    z = torch.tensor([element_dict[atom] for atom in atom_types], dtype=torch.long)
    pos = torch.tensor(positions, dtype=torch.float)
    data = Data(x=z.view(-1, 1), pos=pos, energy=energy)
    data.charge_mask = torch.zeros(data.pos.shape[0], 1)
    if DEBUG:
        logging.debug("Parsed transition Data: x shape %s, pos shape %s", data.x.shape, data.pos.shape)
    return data

def standardize_filename(filename):
    return filename.replace('_new', '')

class MoleculeDataset(Dataset):
    """
    额外参数
    ----------
    start_idx : int | None
        仅保留文件名里 8 位编号 >= start_idx 的分子
    end_idx   : int | None
        仅保留编号 <= end_idx 的分子
    """
    def __init__(self, root,
                 stable_folder="train_result_geometric_optimization",
                 transition_folder="geom_frames",
                 start_idx=None, end_idx=None,
                 transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

        self.start_idx = start_idx
        self.end_idx   = end_idx
        stable_dir     = os.path.join(root, stable_folder)
        transition_dir = os.path.join(root, transition_folder)

        stable_files = [os.path.join(stable_dir, f)
                        for f in os.listdir(stable_dir) if f.endswith('.xyz')]
        transition_files = [os.path.join(transition_dir, f)
                            for f in os.listdir(transition_dir) if f.endswith('.xyz')]

        # ----------- 收集稳态 & 过渡态 -----------
        self.mol_groups = {}
        stable_pat     = re.compile(r'(.*_\d{8}_\d{2})(?:\.xyz)$')
        transition_pat = re.compile(r'(.*_\d{8}_\d{2})_(\d+)\.xyz')

        for f in stable_files:
            key = stable_pat.fullmatch(standardize_filename(os.path.basename(f)))
            if not key:
                continue
            key = key.group(1)
            if key not in self.mol_groups:
                self.mol_groups[key] = {'stable': [], 'transition': []}
            self.mol_groups[key]['stable'].append(f)

        for f in transition_files:
            m = transition_pat.fullmatch(standardize_filename(os.path.basename(f)))
            if not m:
                continue
            key, frame = m.group(1), int(m.group(2))
            if key not in self.mol_groups:
                self.mol_groups[key] = {'stable': [], 'transition': []}
            self.mol_groups[key]['transition'].append((frame, f))

        # ----------- 编号过滤 (新功能) -----------
        keep_keys = []
        idx_pat = re.compile(r'_(\d{8})_\d{2}$')
        for k in self.mol_groups.keys():
            idx = int(idx_pat.search(k).group(1))
            if (self.start_idx is not None and idx < self.start_idx):
                continue
            if (self.end_idx   is not None and idx > self.end_idx):
                continue
            keep_keys.append(k)

        # ----------- 整理文件列表 -----------
        self.group_files = {}
        for key in keep_keys:
            grp = self.mol_groups[key]
            if not grp['stable']:
                logging.warning("Skip %s — no stable xyz.", key); continue
            stable_list = sorted(grp['stable'])
            trans_list  = [f for _, f in sorted(grp['transition'])]
            self.group_files[key] = {'stable': stable_list, 'transition': trans_list}

        self.keys = list(self.group_files.keys())
        logging.info("MoleculeDataset(%s) loaded %d molecules "
                     "[filter %s-%s]", root, len(self.keys),
                     str(self.start_idx), str(self.end_idx))

    # ----------- PyG 接口 -----------
    def len(self):  return len(self.keys)

    def get(self, idx):
        key   = self.keys[idx]
        group = self.group_files[key]

        stable = parse_xyz_stable(group['stable'][-1])
        traj   = []
        for f in group['transition']:
            try:
                traj.append(parse_xyz_transition(f))
            except Exception as e:
                logging.error("Transition parse fail (%s): %s", f, e)
        stable.trajectory = traj
        return stable
