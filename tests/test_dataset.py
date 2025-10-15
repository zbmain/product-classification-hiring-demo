from src.dataset import ProductClassificationDataset
from tabulate import tabulate


class TestDataset:

    def test_dataset_train(self):
        train_dataset = ProductClassificationDataset(split="train")
        print(f"train_dataset size: {len(train_dataset)}")
        print(tabulate(train_dataset.dataframe().head(), headers='keys', tablefmt='psql', showindex=False))
        print(train_dataset.dataframe()['category'].unique())

    def test_dataset_test(self):
        test_dataset = ProductClassificationDataset(split="test")
        print(f"test_dataset size: {len(test_dataset)}")
        print(tabulate(test_dataset.dataframe().head(), headers='keys', tablefmt='psql', showindex=False))
