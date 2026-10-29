import numpy as np
import torch
from sklearn.model_selection import KFold,train_test_split
from datasets.data_augment import get_2023data_dicts,get_2020data_dicts,data_transform,test_transform
from monai.data import Dataset, DataLoader, CacheDataset
from config.config import parse_args

class BraTSData:
    def __init__(self, dataset_name, batch_size, num_workers, train_size, n_splits=5,random_state=42,cache_data=False
                 ,worker_init_fn=None, generator=None,data_dir=None,test_dir=None):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.cache_data = cache_data  # Whether to use CacheLoader cache acceleration
        ds_name = self.dataset_name.lower()
        if ds_name == "brats2023" :
            self.all_data_dict = get_2023data_dicts(data_dir)
        elif ds_name == "brats2020":
            self.all_data_dict = get_2020data_dicts(data_dir)
        self.train_val_data = self.all_data_dict
        self.train_transforms, self.val_transforms = data_transform(train_size,ds_name)
        self.test_data = None
        self.test_transforms = None
        if test_dir is not None:
            self.test_data = get_2023data_dicts(test_dir)
            self.test_transforms = test_transform()

        self.worker_init_fn = worker_init_fn
        self.generator = generator

        all_data = np.array(self.all_data_dict)
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.splits = list(kfold.split(all_data))

    def _get_dataset(self, data, transform):
        if self.cache_data:
            return CacheDataset(data=data, transform=transform,
                                cache_rate=1.0, num_workers=self.num_workers)
        return Dataset(data=data, transform=transform)

    def _get_loader(self, dataset, shuffle):
        is_train = shuffle
        workers = self.num_workers if is_train else 2
        return DataLoader(dataset,
                          batch_size=self.batch_size if shuffle else 1,
                          shuffle=shuffle,
                          num_workers=workers,
                          pin_memory=False,
                          prefetch_factor=2,
                          persistent_workers=is_train,
                          worker_init_fn=self.worker_init_fn,
                          generator=self.generator
                          )

    def get_fold_loader(self, fold):

        if not (0 <= fold < self.n_splits):
            raise ValueError(f"fold should be in [0, {self.n_splits - 1}], currently: {fold}")

        all_data = np.array(self.train_val_data)
        train_idx, val_idx = self.splits[fold]
        train_data = all_data[train_idx].tolist()
        val_data = all_data[val_idx].tolist()
        train_ds = self._get_dataset(train_data, self.train_transforms)
        val_ds = self._get_dataset(val_data, self.val_transforms)
        train_loader = self._get_loader(train_ds, shuffle=True)
        val_loader = self._get_loader(val_ds, shuffle=False)

        return train_loader, val_loader

    def get_test_loader(self):
        if self.test_data is None:
            raise ValueError("test_dir was not passed in, the test set is unavailable")
        test_ds = self._get_dataset(self.test_data, self.test_transforms)
        return self._get_loader(test_ds, shuffle=False)

    def __repr__(self):
        test_num = len(self.test_data) if self.test_data is not None else 0
        return (f"BraTSData("
                f"total={len(self.all_data_dict)}, "
                f"train_val={len(self.train_val_data)},"
                f"test={test_num},"
                f"folds={self.n_splits}, "
                f"batch={self.batch_size}, "
                f"cache={self.cache_data})")



if __name__ == '__main__':
    cfg = parse_args()
    cfg.Dataset_name = "brats2023"

    Data = BraTSData(cfg.Dataset_name, batch_size=1, num_workers=4, train_size=(128,128,128), n_splits=5, random_state=42,
                     test_dir=cfg.test_dir,data_dir=cfg.BraTS2023)
    print(Data)









