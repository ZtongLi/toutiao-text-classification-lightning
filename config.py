# config.py
# 用于集中管理所有超参数，方便调参和实验记录

class Config:
    train_path = "train_3k.txt"
    dev_path = "dev_1k.txt"
    test_path = "test_1k.txt"

    pretrained_model_name = "bert-base-chinese"
    num_labels = 15
    max_length = 128
    dropout = 0.1

    batch_size = 16
    max_epochs = 3
    learning_rate = 2e-5

    warmup_ratio = 0.1
    scheduler_type = "linear"  # 可选：linear / cosine

    accelerator = "gpu"
    devices = 1

    log_every_n_steps = 10
