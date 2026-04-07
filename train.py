# 训练推理文件
'''
epoch 循环
batch 循环
backward
optimizer.step
scheduler.step
日志
验证
保存模型
'''

from lightning import Trainer, seed_everything
from swanlab.integration.pytorch_lightning import SwanLabLogger
from config import Config
from data import DataModule
from model import BertTextClassifier


def main():

    seed_everything(42)

    data_module = DataModule(
        train_path=Config.train_path,
        dev_path=Config.dev_path,
        test_path=Config.test_path,
        batch_size=Config.batch_size,
        max_length=Config.max_length,
        pretrained_model_name=Config.pretrained_model_name
    )

    model = BertTextClassifier(
        pretrained_model_name=Config.pretrained_model_name,
        num_labels=Config.num_labels
    )

    logger = SwanLabLogger(
        project="bert-text-classification",
        experiment_name="baseline"
    )

    trainer = Trainer(
        max_epochs=Config.max_epochs,
        accelerator=Config.accelerator,
        devices=Config.devices,
        log_every_n_steps=Config.log_every_n_steps,
        logger=logger
    )

    trainer.fit(model, data_module)

    trainer.test(model, data_module)


if __name__ == "__main__":
    main()
