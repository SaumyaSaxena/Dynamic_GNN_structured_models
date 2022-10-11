import pytorch_lightning as pl

class DataModule(pl.LightningDataModule):
    """ Datamodule d4rl environments """

    def __init__(self, dataset, batch_size = 512, train_val_split = 0.95, num_workers = 8):
        super().__init__()

        self.batch_size = batch_size
        self.train_val_split = train_val_split

        self.num_workers = num_workers

        self.dataset = dataset

    def setup(self, stage = None):

        train_size = int(len(self.dataset) * self.train_val_split)
        val_size = len(self.dataset) - train_size

        self.train_data, self.val_data = torch.utils.data.random_split(self.dataset, (train_size, val_size))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)