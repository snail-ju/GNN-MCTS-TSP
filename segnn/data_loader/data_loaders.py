import os
import torch
from base import BaseDataLoader
from env.generator import DataGenerator
from torch_geometric.data import InMemoryDataset, Dataset


class TSPDataLoader(BaseDataLoader):
    def __init__(self, config, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.dataset = DiskDataset(config)
        super(TSPDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class DiskDataset(Dataset):
    def __init__(self, config, transform=None, pre_transform=None):
        self.config = config
        save_dir = os.path.join(config['data_loader']['data']['save_dir'],
                                config['data_loader']['data']['graph_type'],
                                "tsp_{}".format(config['arch']['args']['graph_size']))
        super(DiskDataset, self).__init__(save_dir, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        file_names = []
        file_count = 0
        for dir_path, dir_names, filenames in os.walk(self.processed_dir):
            for file in filenames:
                if "data" in file:
                    file_names.append("data_{}.pt".format(file_count))
                    file_count += 1

        if len(file_names) == 0:
            file_names.append("test.pt")

        return file_names

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        pass

    def process(self):
        DataGenerator(self.config).run(self.processed_dir)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

# class MemoryDataset(InMemoryDataset):
#     def __init__(self, config, transform=None, pre_transform=None):
#         self.config = config
#         save_dir = os.path.join(config['data_loader']['data']['save_dir'],
#                                 config['data_loader']['data']['graph_type'],
#                                 "tsp_{}".format(config['arch']['args']['graph_size']))
#         super(MemoryDataset, self).__init__(save_dir, transform, pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])
#
#     @property
#     def raw_file_names(self):
#         return []
#
#     @property
#     def processed_file_names(self):
#         return ['tsp.dataset']
#
#     def download(self):
#         pass
#
#     def process(self):
#         data_list = DataGenerator(self.config).run()
#
#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])
