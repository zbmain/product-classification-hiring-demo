from modelscope.msdatasets import MsDataset


class ProductClassificationDataset:

    def __init__(self, split="train"):
        assert split or split in ["train", "test"]
        self.split = split
        self._dataset = None

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = MsDataset.load("winwin_inc/product-classification-hiring-demo", split=self.split)
        return self._dataset

    def __iter__(self):
        for sample in self.dataset:
            yield sample

    def dataframe(self):
        return self.dataset.to_pandas()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
