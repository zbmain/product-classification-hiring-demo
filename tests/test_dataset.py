from src.dataset import ProductClassificationDataset
from tabulate import tabulate


class TestDataset:

    def test_dataset_train(self):
        train_dataset = ProductClassificationDataset(split="train")
        print(f"\nTrain Category: {train_dataset.dataframe()['category'].unique()}")
        print(f"Train Size: {len(train_dataset)}")
        print(tabulate(train_dataset.dataframe().head(), headers='keys', tablefmt='psql', showindex=False))

    def test_dataset_test(self):
        test_dataset = ProductClassificationDataset(split="test")
        print(f"\nTest Category: {test_dataset.dataframe()['category'].unique()}")
        print(f"Test Size: {len(test_dataset)}")
        print(tabulate(test_dataset.dataframe().head(), headers='keys', tablefmt='psql', showindex=False))
