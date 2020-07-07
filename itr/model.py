import torch
from transformers import BertConfig, BertModel, BertForMaskedLM, BertTokenizer
from transformers import WEIGHTS_NAME, CONFIG_NAME

from easydict import EasyDict as ED
from data import IndicDataset, PadSequence
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np
from pathlib import Path


class MyLightningModule(LightningModule):
    def __init__(self, encoder, decoder, config, tokenizers, pad_sequence):
        super(MyLightningModule, self).__init__()
        self.config = config
        self.tokenizers = tokenizers
        self.pad_sequence = pad_sequence
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

    def save_model(self, model, output_dir):

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


    def save(self, tokenizers, output_dirs):
        self.save_model(self.encoder, output_dirs.encoder)
        self.save_model(self.decoder, output_dirs.decoder)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.train_dataloader()), eta_min=config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.train_dataloader()), eta_min=self.config.lr)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_loader = DataLoader(IndicDataset(self.tokenizers.src, self.tokenizers.tgt, self.config.data, True), 
                            batch_size=self.config.batch_size, 
                            shuffle=False, 
                            collate_fn=self.pad_sequence)
        return train_loader

    def val_dataloader(self):
        eval_loader = DataLoader(IndicDataset(self.tokenizers.src, self.tokenizers.tgt, self.config.data, False), 
                           batch_size=self.config.eval_size, 
                           shuffle=False, 
                           collate_fn=self.pad_sequence)
        return eval_loader

    def training_epoch_end(self, outputs):
        train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.training_loss_values.append(train_loss.item())
        self.logger.experiment.add_scalar("Loss/Train", train_loss.item(), self.current_epoch)
        for name, weights in self.named_parameters():
            self.logger.experiment.add_histogram(name, weights, self.current_epoch)
            print("Added: " + str(name))
        log = {"train_loss": train_loss.item()}
        return {"log": log}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        eval_accuracy = torch.stack([x["tmp_eval_accuracy"] for x in outputs]).mean()
        self.validation_loss_values.append(val_loss.item())
        self.validation_accuracy_values.append(eval_accuracy.item())
        self.logger.experiment.add_scalar("Loss/Val", val_loss.item(), self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Val", eval_accuracy.item(), self.current_epoch)
        self.logger.experiment.flush()
        log = {"avg_val_loss": val_loss.item(), "eval_accuracy": eval_accuracy.item()}
        return {"log": log}

    def get_values(self):
        return self.training_loss_values, self.validation_loss_values, self.validation_accuracy_values


def build_model(config):
    
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

    # model = TranslationModel(encoder, decoder)
    model = MyLightningModule(encoder, decoder, config, tokenizers, pad_sequence)
    # model.cuda()

    return model, tokenizers