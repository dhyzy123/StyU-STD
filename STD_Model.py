import torch.nn as nn
import torch

def norm(f_dim, norm_type):
    if norm_type == "batch_norm" or norm_type == "bn":
        return nn.BatchNorm1d(f_dim)
    elif norm_type == "instance_norm" or norm_type == "in":
        return nn.InstanceNorm1d(f_dim)
    else:
        return nn.Identity()
    
def ConvBlock(f_dim, channels, kernel_size, stride, norm_type="batch_norm", padding=0, T=False):
    if T:
        return (
            nn.ConvTranspose1d(f_dim, channels, kernel_size, stride, padding),
            nn.ReLU(),
            norm(channels, norm_type),
        )
    else:
        return (
            nn.Conv1d(f_dim, channels, kernel_size, stride, padding),
            nn.ReLU(),
            norm(channels, norm_type),
        )

class ResBlock(nn.Module):
    def __init__(self, T, channels, kernel_size, stride, norm_type="batch_norm") -> None:
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.net = nn.Sequential(
            *ConvBlock(channels, channels, kernel_size, stride, norm_type, padding, T),
            *ConvBlock(channels, channels, kernel_size, stride, norm_type, padding, T),
        )

    def forward(self, x):
        y = self.net(x)
        return x + y

class ConvResEncoder(nn.Module):
    def __init__(self, f_dim: int, out_dim: int, norm_type: str, down_sample: int = 1) -> None:
        super().__init__()
        self.ds = down_sample
        self.net = nn.Sequential(
            nn.Conv1d(f_dim, 512, 5, 1, padding=2),
            ResBlock(False, 512, 5, 1, norm_type),
            ResBlock(False, 512, 5, 1, norm_type),
            ResBlock(False, 512, 5, 1, norm_type),
            *ConvBlock(512, out_dim, down_sample, down_sample,norm_type=norm_type),
        )
        self.regressor = nn.LSTM(
            input_size=out_dim,
            hidden_size=out_dim,
            bidirectional=True,
            num_layers=2,
            batch_first=True,
        )
        
    def forward(self, x):
        x = self.net(x)
        x_, _ = self.regressor(x.transpose(-1, -2))
        return x_.transpose(-1, -2), x

    def encode(self, x):
        return self.forward(x)

    def get_seq_len(self, seq_len):
        if seq_len is not None:
            seq_len = (seq_len / self.ds).ceil().int()
        return seq_len

class MatrixCosineSimilarity(nn.Module):
    def __init__(self, f_dim, t_dim) -> None:
        super().__init__()
        self.f_dim = f_dim
        self.t_dim = t_dim

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, mask=None):
        n1 = x1.norm(dim=self.f_dim, keepdim=True)
        n2 = x2.norm(dim=self.f_dim, keepdim=True)
        if self.f_dim < self.t_dim:
            norm = torch.matmul(n1.transpose(self.t_dim, self.f_dim), n2)
            prod = torch.matmul(x1.transpose(self.t_dim, self.f_dim), x2)
        else:
            norm = torch.matmul(n1, n2.transpose(self.t_dim, self.f_dim))
            prod = torch.matmul(x1, x2.transpose(self.t_dim, self.f_dim))
        sim = prod / norm
        if mask is not None:
            mask_tensor = torch.ones_like(sim)
            for i, (w, h) in enumerate(zip(mask[0], mask[1])):
                mask_tensor[i, w:] = mask[2]
                mask_tensor[i, :, h:] = mask[2]
            sim = sim * mask_tensor
        return sim

class STD_Model(nn.Module):
    def __init__(self, f_dim, ls) -> None:
        super().__init__()
        self.feat = ConvResEncoder(f_dim, ls, "in", 8)
        self.sim = MatrixCosineSimilarity(-2, -1)
        self.detector = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 2),
        )

    def forward(self, s, q, s_len, q_len):
        feat_s, _ = self.feat(s)
        feat_q, _ = self.feat(q)
        mask = ((s_len*feat_s.size(-1)/s.size(-1)).int(),
               (q_len*feat_q.size(-1)/q.size(-1)).int(),
               0)
        sim_map = self.sim(feat_s, feat_q, mask)
        # mask = ((q_len*feat_q.size(-1)/q.size(-1)).int(),
        #        (s_len*feat_s.size(-1)/s.size(-1)).int(),
        #        0)
        # sim_map = self.sim(feat_q, feat_s, mask)
        ext = self.detector(sim_map.unsqueeze(1))
        return {"pred": None, "ext": ext, "map": sim_map}
    
    def pred(self, s, q, s_len, q_len):
        scores = self.forward(s, q, s_len, q_len)["ext"]
        return scores