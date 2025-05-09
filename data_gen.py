import torch

from utils.seq_util import pad_seq_mask, zero_pad

from . import utils
from .models import SynthesizerTrn
from .mel_processing import mel_spectrogram_torch
from .speaker_encoder.voice_encoder import SpeakerEncoder
from torchvision.transforms.functional import resize as mel_reseize
import random
from utils.audio import NoiseAdder

config_path = "configs/freevc-s.json"
ptfile_path = "checkpoints/freevc-s.pth"
smodel_path = "speaker_encoder/ckpt/pretrained_bak_5805000.pt"


class FreeVC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sr = 16000

        config = utils.get_path(config_path)
        ptfile = utils.get_path(ptfile_path)
        smodel = utils.get_path(smodel_path)
        self.hps = utils.get_hparams_from_file(config)
        print("Loading model...")
        self.net_g = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model,
        )
        _ = self.net_g.eval()
        print("Loading checkpoint...")
        _ = utils.load_checkpoint(ptfile, self.net_g, None, True)

        print("Loading WavLM for content...")
        self.cmodel = utils.get_cmodel()

        if self.hps.model.use_spk:
            print("Loading speaker encoder...")
            self.smodel = SpeakerEncoder(smodel)

    def syn(self, content, spk, c_mask=None):
        if c_mask is None:
            c_mask = torch.ones((content.size(0), content.size(-1)), device=content.device)
        z_p, m_p, logs_p, c_mask = self.net_g.enc_p(content, c_mask)
        z = self.net_g.flow(z_p, c_mask, g=spk, reverse=True)
        o = self.net_g.dec(z * c_mask, g=spk)
        o = o * c_mask.repeat_interleave(self.hps.data.hop_length, dim=-1)
        return o

    def recon(self, q):
        with torch.no_grad():
            q_s = self.get_s(q)
            q_c = self.get_c(q)
            syn_q = self.syn(q_c, q_s, None)
            syn_q = self.bg_adder(syn_q)
        return syn_q

    def forward(self, wav):
        return self.recon(wav)

    def get_c(self, x):
        content = utils.get_content(self.cmodel, x)
        return content

    def get_s(self, x):
        mel_tgt = mel_spectrogram_torch(
            x,
            self.hps.data.filter_length,
            self.hps.data.n_mel_channels,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            self.hps.data.mel_fmin,
            self.hps.data.mel_fmax,
        )
        spk = self.net_g.enc_spk.embed_utterance(mel_tgt.transpose(1, 2)).unsqueeze(-1)
        return spk
