# 数据处理文件: 将原始数据转化为张量
# 用https://lightning.ai/docs/pytorch/stable/starter/introduction.html 的模块化方法

# data.py
import os
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class ToutiaoDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, text = self.data[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": label
        }

class DataModule(LightningDataModule):

    def __init__(self, train_path, dev_path, test_path, batch_size, max_length, pretrained_model_name="bert-base-chinese"):
        super().__init__()
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None

        # 你的标签映射
        self.label_map = {
            100: 0, 101: 1, 102: 2, 103: 3, 104: 4, 106: 5,
            107: 6, 108: 7, 109: 8, 110: 9, 112: 10,
            113: 11, 114: 12, 115: 13, 116: 14
        }

    def prepare_data(self):
        assert os.path.exists(self.train_path), f"{self.train_path} 不存在"
        assert os.path.exists(self.dev_path), f"{self.dev_path} 不存在"
        assert os.path.exists(self.test_path), f"{self.test_path} 不存在"

    def setup(self, stage=None):

        def load_file(path):
            data_list = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("_!_")
                    label, text = parts[1], parts[3]
                    data_list.append((self.label_map[int(label)], text))
            return data_list

        if stage == "fit" or stage is None:
            train_list = load_file(self.train_path)
            dev_list = load_file(self.dev_path)
            self.train_dataset = ToutiaoDataset(train_list, self.tokenizer, self.max_length)
            self.dev_dataset = ToutiaoDataset(dev_list, self.tokenizer, self.max_length)

        if stage == "test" or stage is None:
            test_list = load_file(self.test_path)
            self.test_dataset = ToutiaoDataset(test_list, self.tokenizer, self.max_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
