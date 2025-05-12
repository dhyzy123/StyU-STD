from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from model import FreeVC
from utils import seq_util as util
from utils.audio import (
    load_audio,
    random_seg,
    remove_silent,
    list_audio,
    NoiseAdder,
    random_drop,
    random_speed_change,
)
from torchvision.transforms.functional import resize as mel_reseize
import random


def random_drop_frames(sr, seq):
    drop_length = int(sr * 0.05)
    drop_times = random.randrange(0, int(seq.size(-1) / drop_length * 0.2) + 1)
    for i in range(drop_times):
        seq = random_drop(seq, drop_length)
    return seq


class UnlabeledSTD(Dataset):
    def __init__(
        self,
        root: list | str,
        sr,
        feat_s,
        feat_q,
        segment_duration=(3, 3),
        word_duration=(0.4, 2),
        noise_path=None,
        add_noise_on=(True, False),
    ) -> None:
        super().__init__()
        if isinstance(root, str):
            self.audio_files = list_audio(root)
        elif isinstance(root, list):
            self.audio_files = list_audio(*root)
        else:
            raise Exception(f"not support root type:{root}")
        self.feat_s = feat_s if feat_s is not None else nn.Identity()
        self.feat_q = feat_q if feat_q is not None else nn.Identity()
        self.sr = sr
        self.s_dur = segment_duration
        self.q_dur = word_duration
        self.add_noise_on = add_noise_on
        if noise_path is not None:
            self.bg_adder = NoiseAdder(noise_path, (10, 80))
        else:
            self.bg_adder = nn.Identity()
        print(len(self))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, n: int):
        file_name = self.audio_files[n]
        waveform, wlen = load_audio(file_name, self.sr)
        dup_times = int(0.5 + self.s_dur[0] / wlen)
        if dup_times > 0:
            waveform = waveform.repeat((1, dup_times))
        waveform, _ = remove_silent(waveform)
        s_seg_len = (int(self.sr * self.s_dur[0]), int(self.sr * self.s_dur[1]))
        q_seg_len = (int(self.sr * self.q_dur[0]), int(self.sr * self.q_dur[1]))
        if (waveform.size(-1) / self.sr) < self.s_dur[0]:
            waveform = util.zero_pad(waveform, s_seg_len[0])
        s, _ = random_seg(waveform, s_seg_len)
        mean_q = 0
        mean_s = s.abs().mean() * 2 / 3
        try_times = 5
        while mean_q < mean_s and try_times > 0:
            q, (ts, te) = random_seg(s, q_seg_len)
            mean_q = q.abs().mean()
            try_times -= 1
        if self.add_noise_on[0]:
            s = self.bg_adder(s)
        if self.add_noise_on[1]:
            q = self.bg_adder(q)
        feat_s = self.feat_s(s)
        feat_q = self.feat_q(q)
        return feat_s, feat_q, s, q, ts, te

    @classmethod
    def collect_fn(cls, batch_data):
        list_s = []
        list_q = []
        len_s = []
        len_q = []
        for s, q, _, _,_,_ in batch_data:
            # C,*,T -> T,*,C
            list_s.append(s.transpose(0, -1))
            list_q.append(q.transpose(0, -1))
            len_s.append(s.size(-1))
            len_q.append(q.size(-1))
        return (
            pad_sequence(list_s, True).transpose(1, -1),
            pad_sequence(list_q, True).transpose(1, -1),
            torch.LongTensor(len_s),
            torch.LongTensor(len_q),
        )


class STDDataConstructor(nn.Module):
    def __init__(self, sr, feat, noise_path, snr=(10, 80), var_len=0.5, **config) -> None:
        super().__init__()
        self.var_len = var_len
        self.sr = sr
        self.feat = feat
        self.config = {"RR": True, "CP": True, "NS": True, "ST": True}
        for k, v in config.items():
            self.config[k] = v
        print(self.config)
        noise_path = "/home/lxr/data/SpeechCommands/speech_commands_v0.02/_background_noise_"
        self.freevc = FreeVC()
        if noise_path is not None:
            self.bg_adder = NoiseAdder(noise_path, snr)
        else:
            self.bg_adder = torch.nn.Identity()

    def syn(self, wav, wav_len):
        batch_size = wav.size(0)
        ext_list = []
        new_qs_idx = []
        for i in range(batch_size):
            ext = random.randint(0, 1)
            if ext == 1:
                qs_id = i
            else:
                qs_id = -i - 1
            ext_list.append(ext)
            new_qs_idx.append(qs_id)
        if self.config["ST"]:
            new_lens = []
            new_qc = []
            r_style = self.freevc.get_s(wav)
            r_content = self.freevc.get_c(wav)
            for id in new_qs_idx:
                new_len = int(wav_len[id] / wav.size(-1) * r_content.size(-1))
                qc = r_content[id][..., :new_len].unsqueeze(0)
                len_var = 1 - self.var_len / 2 + self.var_len * random.random()
                new_lens.append(int(wav_len[i] * len_var))
                new_shape = (qc.size(-2), int(new_len * len_var))
                qc = mel_reseize(qc, new_shape, antialias=True).squeeze()
                new_qc.append(qc)
            new_qcs, mask = util.pad_seq_mask(new_qc)
            syn_s = self.freevc.syn(new_qcs, r_style[new_qs_idx], mask)
        else:
            syn_s = wav[torch.IntTensor(new_qs_idx)]
            new_lens = wav_len[torch.IntTensor(new_qs_idx)]
        ext = torch.LongTensor(ext_list).to(syn_s.device)
        wav_len = torch.Tensor(new_lens)
        return syn_s.squeeze(), wav_len, ext

    def forward(self, s, q, len_s, len_q):
        target_len_s = int(s.size(-1) * (1 + self.var_len / 2))
        target_len_q = int(q.size(-1) * (1 + self.var_len / 2))
        s = s.squeeze()
        q = q.squeeze()
        q, len_q, ext = self.syn(q, len_q)
        # if random.randint(0, 1) == 0:
        #     s, len_s, ext = self.syn(s, len_s)
        # else:
        #     q, len_q, ext = self.syn(q, len_q)
        if self.config["CP"]:
            s = random_drop_frames(self.sr, s)
            q = random_drop_frames(self.sr, q)
        if self.config["RR"]:
            q = random_speed_change(q.cpu(), int(self.sr * 0.35), self.sr, 0.2).to(q.device)
            s = random_speed_change(s.cpu(), int(self.sr * 0.35), self.sr, 0.2).to(s.device)
        if self.config["NS"]:
            s = self.bg_adder(s.squeeze())
            q = self.bg_adder(q.squeeze())
        s = util.zero_pad(s, target_len_s)
        q = util.zero_pad(q, target_len_q)
        fs = self.feat(s)
        fq = self.feat(q)
        len_s = (len_s * fs.size(-1) / s.size(-1)).int().to(fs.device)
        len_q = (len_q * fq.size(-1) / q.size(-1)).int().to(fq.device)
        return fs, fq, len_s, len_q, ext


if __name__ == "__main__":
    data = UnlabeledSTD("0622", None, file_suffix=".wav")
    print(data[0])
