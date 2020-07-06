
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from easydict import EasyDict as ED

@dataclass
class Decoder:
    pretrained = True

@dataclass
class Encoder:
    pretrained = True

@dataclass
class Config:
    exp_name: str
    data: str
    lang: str

    base_log_dir: str = '../logs'
    base_output_dir: str = '../output'

    #embed_dim: int = 100
    hidden_size: int = 256
    intermediate_size: int = 512
    num_attention_heads: int = 1
    num_hidden_layers: int = 1
    hidden_act: str = 'gelu'
    dropout_prob: float = 0.1

    epochs: int = 1
    batch_size: int = 64
    eval_size: int = 16
    lr: float = 1e-3


    decoder: Decoder = Decoder()
    encoder: Encoder = Encoder()

    def __post_init__(self):
        self.log_dir = Path(self.base_log_dir) / self.exp_name
        self.log_dir.mkdir(parents=False, exist_ok=True)

        output_dir = Path(self.base_output_dir) / self.exp_name
        self.model_output_dirs = ED({})

        for m in ['encoder', 'decoder']:
            out = output_dir / m
            out.mkdir(parents=True, exist_ok=True)
            self.model_output_dirs[m] = out


hc1 = Config(data='../data/hin-eng/',
                    exp_name='default',
                    lang='hi',
                )

#increasing hidden size needs change in lr, others only change time
hc20 = replace(hc1,
                    num_hidden_layers=6,
                    num_attention_heads=8
                )
hc21 = replace(hc20, num_hidden_layers=12) #learns ok
hc22 = replace(hc21, num_attention_heads=12, hidden_size=288) #learns ok
hc23 = replace(hc21, num_attention_heads=12, hidden_size=384, lr=5e-4) #learns ok, 1e-4 too small

hc24 = replace(hc23, intermediate_size=3072, lr=1e-4)

preEnc = replace(hc24, epochs=5, exp_name='pretrained-enc')
preEncDec = replace(hc24, epochs=2, exp_name='pretrained-enc-dec')


