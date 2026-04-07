# 模型参数文件：存放BERT模型和相关模型参数
'''
定义模型结构
定义 forward
定义 loss
定义 optimizer
定义 scheduler
定义训练步骤
定义验证步骤
'''
import lightning as L          
import torch                   
from torch import nn     
from transformers import AutoModel
from transformers import get_linear_schedule_with_warmup  
from config import Config


class BertTextClassifier(L.LightningModule):
    def __init__(self, pretrained_model_name="bert-base-chinese", num_labels=10):
        super().__init__()
      
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.encoder.config.hidden_size
      
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self(input_ids, attention_mask)

        loss = nn.CrossEntropyLoss()(logits, labels)

        self.log("train_loss", loss, prog_bar=True)
        return loss  

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self(input_ids, attention_mask)

        loss = nn.CrossEntropyLoss()(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss 

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return loss


    def configure_optimizers(self):
        # 优化器
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)

        # 训练步数
        num_training_steps = self.trainer.estimated_stepping_batches

        num_warmup_steps = int(0.1 * num_training_steps)  # 前 10% 步数学习率从0是缓慢上升到目标值

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  
                "frequency": 1
            }
        }
  
