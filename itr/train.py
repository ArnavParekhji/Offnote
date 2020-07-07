from time import time
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


def preproc_data():
    from data import split_data
    split_data('../data/hin-eng/hin.txt', '../data/hin-eng')


from data import IndicDataset, PadSequence
import model as M
from config import replace, preEnc, preEncDec

def main():
    rconf = preEncDec
    model, tokenizers = M.build_model(rconf)
    trainer = pl.Trainer(train_percent_check=0.1,  max_epochs=rconf.epochs)
    # trainer = pl.Trainer(max_epochs=rconf.epochs)
    trainer.fit(model)
    print(model.get_values())

    # model.save(tokenizers, rconf.model_output_dirs)

if __name__ == '__main__':
    # preproc_data()
    main()