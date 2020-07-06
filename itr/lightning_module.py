import torch
from transformers import BertConfig, BertModel, BertForMaskedLM, BertTokenizer
from easydict import EasyDict as ED
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.base import Callback
from data import IndicDataset, PadSequence
import model as M
from time import time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import WEIGHTS_NAME, CONFIG_NAME
import time
from config import replace, preEnc, preEncDec
import numpy as np


def save_model(model, output_dir):

    output_dir = Path(output_dir)
    # Step 1: Save a model, configuration and vocabulary that you have fine-tuned

    # If we have a distributed model, save only the encapsulated model
    # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
    model_to_save = model.module if hasattr(model, 'module') else model

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = output_dir / WEIGHTS_NAME
    output_config_file = output_dir / CONFIG_NAME

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    #src_tokenizer.save_vocabulary(output_dir)



class MyLightningModule(LightningModule):
    def __init__(self, encoder, decoder):
        super(MyLightningModule, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.training_loss_values = []
        self.validation_loss_values = []
        self.validation_accuracy_values = []

    def forward(self, encoder_input_ids, decoder_input_ids):
        encoder_hidden_states = self.encoder(encoder_input_ids)[0]
        loss, logits = self.decoder(decoder_input_ids,
                                    encoder_hidden_states=encoder_hidden_states, 
                                    masked_lm_labels=decoder_input_ids)
        return loss, logits

    def save(self, tokenizers, output_dirs):
        from train_util import save_model

        save_model(self.encoder, output_dirs.encoder)
        save_model(self.decoder, output_dirs.decoder)

    def flat_accuracy(self, preds, labels):
        preds = preds.detach().numpy()
        labels = labels.detach().numpy()
        pred_flat = np.argmax(preds, axis=2).flatten()
        labels_flat = labels.flatten()
        return np.sum(np.equal(pred_flat, labels_flat)) / len(labels_flat)
        
    def training_step(self, batch, batch_idx):
        loss, logits = self(batch[0], batch[1])
        return {"loss":loss}

    def validation_step(self, batch, batch_idx):
        val_loss, logits = self(batch[0], batch[1])
        tmp_eval_accuracy = self.flat_accuracy(logits, batch[1])
        tmp_eval_accuracy = torch.from_numpy(np.array(tmp_eval_accuracy, dtype="float64"))
        return {"val_loss":val_loss, "tmp_eval_accuracy":tmp_eval_accuracy}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.train_dataloader()), eta_min=config.lr)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_loader = DataLoader(IndicDataset(tokenizers.src, tokenizers.tgt, config.data, True), 
                            batch_size=config.batch_size, 
                            shuffle=False, 
                            collate_fn=pad_sequence)
        return train_loader

    def val_dataloader(self):
        eval_loader = DataLoader(IndicDataset(tokenizers.src, tokenizers.tgt, config.data, False), 
                           batch_size=config.eval_size, 
                           shuffle=False, 
                           collate_fn=pad_sequence)
        return eval_loader

    def training_epoch_end(self, outputs):
        train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.training_loss_values.append(train_loss)
        log = {"train_loss": train_loss}
        return {"log": log}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        eval_accuracy = torch.stack([x["tmp_eval_accuracy"] for x in outputs]).mean()
        self.validation_loss_values.append(val_loss)
        self.validation_accuracy_values.append(eval_accuracy)
        log = {"avg_val_loss": val_loss, "eval_accuracy": eval_accuracy}
        return {"log": log}

    def get_values(self):
        return self.training_loss_values, self.validation_loss_values, self.validation_accuracy_values



# class MyPrintingCallback(Callback):
#     def __init__(self):
#         super(MyPrintingCallback, self).__init__()
#         self.training_loss_values = []
#         self.validation_loss_values = []
#         self.validation_accuracy_values = []
#         self.total_loss = 0

#     def on_batch_end(self, trainer, pl_module):
#         pass

#     def on_epoch_start(self, trainer, pl_module):
#         self.total_loss = 0

#     def on_epoch_end(self, trainer, pl_module):
#         pass



#********************************************************************************

config = preEncDec

src_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tgt_tokenizer.bos_token = '<s>'
tgt_tokenizer.eos_token = '</s>'

#hidden_size and intermediate_size are both wrt all the attention heads. 
#Should be divisible by num_attention_heads
encoder_config = BertConfig(vocab_size=src_tokenizer.vocab_size,
                            hidden_size=config.hidden_size,
                            num_hidden_layers=config.num_hidden_layers,
                            num_attention_heads=config.num_attention_heads,
                            intermediate_size=config.intermediate_size,
                            hidden_act=config.hidden_act,
                            hidden_dropout_prob=config.dropout_prob,
                            attention_probs_dropout_prob=config.dropout_prob,
                            max_position_embeddings=512,
                            type_vocab_size=2,
                            initializer_range=0.02,
                            layer_norm_eps=1e-12)

decoder_config = BertConfig(vocab_size=tgt_tokenizer.vocab_size,
                            hidden_size=config.hidden_size,
                            num_hidden_layers=config.num_hidden_layers,
                            num_attention_heads=config.num_attention_heads,
                            intermediate_size=config.intermediate_size,
                            hidden_act=config.hidden_act,
                            hidden_dropout_prob=config.dropout_prob,
                            attention_probs_dropout_prob=config.dropout_prob,
                            max_position_embeddings=512,
                            type_vocab_size=2,
                            initializer_range=0.02,
                            layer_norm_eps=1e-12,
                            is_decoder=False)

#Create encoder and decoder embedding layers.
encoder_embeddings = torch.nn.Embedding(src_tokenizer.vocab_size, config.hidden_size, padding_idx=src_tokenizer.pad_token_id)
decoder_embeddings = torch.nn.Embedding(tgt_tokenizer.vocab_size, config.hidden_size, padding_idx=tgt_tokenizer.pad_token_id)

encoder = BertModel(encoder_config)
encoder.set_input_embeddings(encoder_embeddings)

decoder = BertForMaskedLM(decoder_config)
decoder.set_input_embeddings(decoder_embeddings)

tokenizers = ED({'src': src_tokenizer, 'tgt': tgt_tokenizer})

pad_sequence = PadSequence(tokenizers.src.pad_token_id, tokenizers.tgt.pad_token_id)

#******************************************************************************************

if __name__ == "__main__":
    # trainer = pl.Trainer(train_percent_check=0.1,  max_epochs=config.epochs, callbacks=[MyPrintingCallback()])
    trainer = pl.Trainer(train_percent_check=0.1,  max_epochs=config.epochs)
    lm = MyLightningModule(encoder, decoder)
    trainer.fit(lm)
    print(lm.get_values())