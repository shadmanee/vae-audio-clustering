import torch, numpy as np
from torch.utils.data import Dataset

from config import config
from utils.audio_data import check_global_min_max

class AudioSpectrogramDataset(Dataset):
    def __init__(self, dataset_dir, expected_shape=(config.INPUT_HEIGHT, config.INPUT_WIDTH), add_channel_dim=False):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.file_paths = sorted(self.dataset_dir.rglob("*npy"))
        self.expected_shape = expected_shape
        self.add_channel_dim = add_channel_dim
        
        global_min_max = check_global_min_max(self.file_paths, n_sub=200)
        self.global_min = global_min_max["min_min"]
        self.global_max = global_min_max["max_max"]
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        file_path = self.file_paths[index]
        x = np.load(file_path)
        x = (x - self.global_min) / (self.global_max - self.global_min)
        x = torch.tensor(x, dtype=torch.float32)
        if self.add_channel_dim: x = x.unsqueeze(0)
        
        return x, str(file_path), -1
    
    
class AudioSpectogramDatasetwithLabels(Dataset):
    def __init__(self, dataset_dir, labeled_df, expected_shape=(config.INPUT_HEIGHT, config.INPUT_WIDTH), add_channel_dim=False):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.expected_shape = expected_shape
        self.add_channel_dim = add_channel_dim
        self.labeled_df = labeled_df
        # print("\n"*3, "="*20, f"\n{self.labeled_df["label"].unique()}\n", "="*20, "\n"*3)        

        if labeled_df is not None:
            # Only load files that exist in labeled_df
            valid_stems = set(labeled_df["audio_file_stem"].values)
            self.file_paths = sorted([
                p for p in self.dataset_dir.rglob("*.npy")
                if str(p.stem).split("_")[0] in valid_stems
            ])
            # Build stem -> label lookup for O(1) access in __getitem__
            self.label_map = labeled_df.set_index("audio_file_stem")["label"].to_dict()
            # print("\n"*3, "="*20, f"\n{set(val for val in self.label_map.values())}\n", "="*20, "\n"*3)
        else:
            raise ValueError("No labels found. Use the `AudioSpectogramDataset` class.")

        global_min_max = check_global_min_max(self.file_paths, n_sub=200)
        self.global_min = global_min_max["min_min"]
        self.global_max = global_min_max["max_max"]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        x = np.load(file_path)
        x = (x - self.global_min) / (self.global_max - self.global_min)
        x = torch.tensor(x, dtype=torch.float32)
        if self.add_channel_dim:
            x = x.unsqueeze(0)

        if self.label_map is not None:
            label = self.label_map[str(file_path.stem).split("_")[0]]
            return x, str(file_path), label
        else:
            raise ValueError("No labels found. Use the `AudioSpectogramDataset` class.")