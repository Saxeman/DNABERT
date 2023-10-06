import torch
from transformers import BertModel, get_linear_schedule_with_warmup, BertForSequenceClassification
from torchmetrics.classification import MulticlassF1Score
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np


class BaselineModelA(pl.LightningModule):
    def __init__(self,
                 dnabert_path=None, 
                 total_number_of_samples=None, 
                 max_epochs=None, 
                 batch_size=None,
                 fine_tune=True,
                 lr=0.2e-3, 
                 num_classes=2) -> None:
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(dnabert_path,
                                                                   num_labels=num_classes, 
                                                                   output_attentions = False, 
                                                                   output_hidden_states = False)
    
        self.loss_fcn = nn.CrossEntropyLoss()
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.total_number_of_samples = total_number_of_samples
        self.lr = lr
        self.f1_score = MulticlassF1Score(num_classes=num_classes, average='macro')
        
        # BERT Freezing layers, True for training, False for freezing
        for param in self.model.bert.parameters():
            param.requires_grad = fine_tune
        for param in self.model.bert.encoder.layer[-1].parameters():
            param.requires_grad = True
        
    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
        'weight_decay': 0.0001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
        'weight_decay': 0.0}
        ]  
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0.01, 
        num_training_steps=(self.total_number_of_samples // self.batch_size) * self.max_epochs
        )
        return [optimizer], [scheduler]
    
    def forward(self, seq, label):
        input_ids = seq['input_ids']
        attention_mask = seq['attention_mask']
        input_ids = input_ids.view(input_ids.shape[0], -1)
        output = self.model(input_ids, attention_mask=attention_mask, token_type_ids=None, labels=label)
        return output
        
    def training_step(self, batch, batch_idx):
        seq, label, _ = batch
        output = self.forward(seq, label)
        loss = self.loss_fcn(output.logits, label)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=len(batch))
        return loss
    
    def validation_step(self, batch, batch_idx):
        seq, label, _ = batch
        output = self.forward(seq, label)
        loss = self.loss_fcn(output.logits, label)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=len(batch))
        f1_score = self.f1_score(output.logits, label)
        self.log("val_f1score", f1_score, on_epoch=True, prog_bar=True, batch_size=len(batch))
        return loss
    
    def test_step(self, batch, batch_idx):
        seq, labels, seq_id = batch
        output = self.forward(seq, labels)
        loss = self.loss_fcn(output.logits, labels)
        self.log("test_loss", loss)
        preds = output.logits.argmax(dim=-1)
        self.log('test_acc', (preds == labels).float().mean(), on_epoch=True, prog_bar=True, batch_size=len(batch))

        # Store predictions in a variable
        if batch_idx == 0:
            self.test_preds = preds.detach().cpu().numpy()
            self.test_labels = labels.detach().cpu().numpy()
            self.test_logits = output.logits.detach().cpu().numpy()
            self.test_ids = list(seq_id)
            # self.test_ids = seq_id.detach().cpu().numpy()
        else:
            self.test_preds = np.concatenate((self.test_preds, preds.detach().cpu().numpy()))
            self.test_labels = np.concatenate((self.test_labels, labels.detach().cpu().numpy()))
            self.test_logits = np.concatenate((self.test_logits, output.logits.detach().cpu().numpy()))
            self.test_ids = self.test_ids + list(seq_id)
            # self.test_ids = np.concatenate((self.test_ids, seq_id.detach().cpu().numpy()))
            
            
class BaselineModelB(pl.LightningModule):
    def __init__(self, 
                 dnabert_path=None,
                 total_number_of_samples=None, 
                 max_epochs=None, 
                 batch_size=None,
                 fine_tune=True,
                 lr=0.2e-3, 
                 num_classes=1) -> None:
        super().__init__()
        self.model = BertModel.from_pretrained(dnabert_path,
                                               output_attentions = False, 
                                               output_hidden_states = True)
        self.head_layer = nn.Sequential(nn.Linear(768, 8),
                                        nn.BatchNorm1d(num_features=8),
                                        nn.Dropout(0.5),
                                        nn.ReLU(),
                                        nn.Linear(8, num_classes))
    
        self.loss_fcn = nn.MSELoss()
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.total_number_of_samples = total_number_of_samples
        self.lr = lr
        
        # BERT Freezing layers, True for training, False for freezing
        for param in self.model.parameters():
            param.requires_grad = fine_tune
        for param in self.model.encoder.layer[-1].parameters():
            param.requires_grad = True
        
    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
        'weight_decay': 0.0001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
        'weight_decay': 0.0}
        ]  
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0.01, 
        num_training_steps=(self.total_number_of_samples // self.batch_size) * self.max_epochs
        )
        return [optimizer], [scheduler]
    
    def forward(self, seq):
        input_ids = seq['input_ids']
        attention_mask = seq['attention_mask']
        input_ids = input_ids.view(input_ids.shape[0], -1)
        output = self.model(input_ids, attention_mask=attention_mask, token_type_ids=None)
        output = self.head_layer(output.pooler_output)
        return output
        
    def training_step(self, batch, batch_idx):
        seq, label, _ = batch
        output = self.forward(seq)
        loss = self.loss_fcn(output, label)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=len(batch))
        return loss
    
    def validation_step(self, batch, batch_idx):
        seq, label, _ = batch
        output = self.forward(seq)
        loss = self.loss_fcn(output, label)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=len(batch))
        return loss
    
    def test_step(self, batch, batch_idx):
        seq, labels, seq_id = batch
        output = self.forward(seq)
        loss = self.loss_fcn(output, labels)
        self.log("test_loss", loss)
        preds = output.argmax(dim=-1)
        self.log('test_acc', (preds == labels).float().mean(), on_epoch=True, prog_bar=True, batch_size=len(batch))

        # Store predictions in a variable
        if batch_idx == 0:
            self.test_preds = preds.detach().cpu().numpy()
            self.test_labels = labels.detach().cpu().numpy()
            self.test_logits = output.detach().cpu().numpy()
            self.test_ids = list(seq_id)
            # self.test_ids = seq_id.detach().cpu().numpy()
        else:
            self.test_preds = np.concatenate((self.test_preds, preds.detach().cpu().numpy()))
            self.test_labels = np.concatenate((self.test_labels, labels.detach().cpu().numpy()))
            self.test_logits = np.concatenate((self.test_logits, output.detach().cpu().numpy()))
            self.test_ids = self.test_ids + list(seq_id)
            # self.test_ids = np.concatenate((self.test_ids, seq_id.detach().cpu().numpy()))
