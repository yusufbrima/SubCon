import torch
from torch.utils.data import Dataset
import numpy as np
import h5py


class EfficientCustom3DShapesDataset(Dataset):
    """
    A custom PyTorch dataset for 3DShapes.

    Args:
        file_path (str): Path to the HDF5 file containing the dataset.
        transform (callable, optional): A function/transform to apply to each data sample.
    """

    def __init__(self, file_path, transform=None):
        """
        Initializes the dataset.

        Args:
            file_path (str): Path to the HDF5 file containing the dataset.
            transform (callable, optional): A function/transform to apply to each data sample.
        """
        self.file_path = file_path
        self.transform = transform
        self._open_file()
        self._FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']
        self._NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                          'scale': 8, 'shape': 4, 'orientation': 15}

    def _open_file(self):
        """
        Opens the HDF5 file and initializes dataset attributes.
        """
        self.dataset = h5py.File(self.file_path, 'r')
        self.images = self.dataset['images']
        self.labels = self.dataset['labels']
        self.n_samples = len(self.labels)

    def __getitem__(self, idx):
        """
        Gets a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the image and its corresponding label.
        """
        image = self.images[idx]
        label = self.labels[idx]
        
        image = np.asarray(image) / 255.0
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        
        label = label.astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image), torch.tensor(label)

    def close(self):
        """
        Closes the HDF5 file.
        """
        self.dataset.close()

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return self.n_samples
    def __del__(self):
        """
        Destructor to close the HDF5 file when the dataset is destroyed.
        """
        if hasattr(self, 'dataset') and self.dataset is not None:
            self.dataset.close()
    def get_index(self, factors):
        """
        Converts factors to indices in the range [0, num_data-1].

        Args:
            factors (numpy.ndarray): Factors array of shape [6, batch_size].

        Returns:
            numpy.ndarray: Indices array of shape [batch_size].
        """
        indices = 0
        base = 1
        for factor, name in reversed(list(enumerate(self._FACTORS_IN_ORDER))):
            indices += factors[factor] * base
            base *= self._NUM_VALUES_PER_FACTOR[name]
        return indices

    def sample_random_batch(self, batch_size):
        """
        Samples a random batch of images.

        Args:
            batch_size (int): Number of images to sample.

        Returns:
            numpy.ndarray: Batch of images with shape [batch_size, 3, 64, 64].
        """
        indices = np.random.choice(self.n_samples, batch_size)
        ims = []
        for ind in indices:
            im = self.images[ind]
            im = np.asarray(im)
            im = im.transpose((2, 0, 1))  # Transpose to [3, 64, 64]
            ims.append(im)
        ims = np.stack(ims, axis=0)
        ims = ims / 255.0  # Normalize values to [0, 1]
        ims = ims.astype(np.float32)
        return ims

    def sample_batch(self, batch_size, fixed_factor, fixed_factor_value):
        """
        Samples a batch of images with a fixed factor value, while other factors vary randomly.

        Args:
            batch_size (int): Number of images to sample.
            fixed_factor (int): Index of the factor to be fixed (0-5).
            fixed_factor_value (int): Fixed value for the specified factor.

        Returns:
            numpy.ndarray: Batch of images with shape [batch_size, 3, 64, 64].
        """
        factors = np.zeros([len(self._FACTORS_IN_ORDER), batch_size], dtype=np.int32)
        for factor, name in enumerate(self._FACTORS_IN_ORDER):
            num_choices = self._NUM_VALUES_PER_FACTOR[name]
            factors[factor] = np.random.choice(num_choices, batch_size)
        factors[fixed_factor] = fixed_factor_value
        indices = self.get_index(factors)
        ims = []
        for ind in indices:
            im = self.images[ind]
            im = np.asarray(im)
            im = im.transpose((2, 0, 1))  # Transpose to [3, 64, 64]
            ims.append(im)
        ims = np.stack(ims, axis=0)
        ims = ims / 255.0  # Normalize values to [0, 1]
        ims = ims.astype(np.float32)
        return ims


class Custom3DShapesDataset(Dataset):
    """
    A custom PyTorch dataset for 3DShapes.

    Args:
        file_path (str): Path to the HDF5 file containing the dataset.
        transform (callable, optional): A function/transform to apply to each data sample.
    """

    def __init__(self, file_path, transform=None):
        """
        Initializes the dataset.

        Args:
            file_path (str): Path to the HDF5 file containing the dataset.
            transform (callable, optional): A function/transform to apply to each data sample.
        """
        self.dataset = h5py.File(file_path, 'r')
        self.images = self.dataset['images']  # [480000, 64, 64, 3]
        self.labels = self.dataset['labels']  # [480000, 6]
        self.transform = transform
        self.n_samples = len(self.labels)
        self._FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']
        self._NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                          'scale': 8, 'shape': 4, 'orientation': 15}

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return self.n_samples

    def __getitem__(self, idx):
        """
        Gets a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the image and its corresponding label.
        """
        image = self.images[idx]
        image = np.asarray(image) / 255.0  # Normalize to [0, 1]
        image = image.transpose((2, 0, 1))  # Transpose to [3, 64, 64]
        image = image.astype(np.float32)

        label = self.labels[idx]
        label = label.astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return image, label
    def __del__(self):
        """
        Destructor to close the HDF5 file when the dataset is destroyed.
        """
        if hasattr(self, 'dataset') and self.dataset is not None:
            self.dataset.close()
    def get_index(self, factors):
        """
        Converts factors to indices in the range [0, num_data-1].

        Args:
            factors (numpy.ndarray): Factors array of shape [6, batch_size].

        Returns:
            numpy.ndarray: Indices array of shape [batch_size].
        """
        indices = 0
        base = 1
        for factor, name in reversed(list(enumerate(self._FACTORS_IN_ORDER))):
            indices += factors[factor] * base
            base *= self._NUM_VALUES_PER_FACTOR[name]
        return indices

    def sample_random_batch(self, batch_size):
        """
        Samples a random batch of images.

        Args:
            batch_size (int): Number of images to sample.

        Returns:
            numpy.ndarray: Batch of images with shape [batch_size, 3, 64, 64].
        """
        indices = np.random.choice(self.n_samples, batch_size)
        ims = []
        for ind in indices:
            im = self.images[ind]
            im = np.asarray(im)
            im = im.transpose((2, 0, 1))  # Transpose to [3, 64, 64]
            ims.append(im)
        ims = np.stack(ims, axis=0)
        ims = ims / 255.0  # Normalize values to [0, 1]
        ims = ims.astype(np.float32)
        return ims

    def sample_batch(self, batch_size, fixed_factor, fixed_factor_value):
        """
        Samples a batch of images with a fixed factor value, while other factors vary randomly.

        Args:
            batch_size (int): Number of images to sample.
            fixed_factor (int): Index of the factor to be fixed (0-5).
            fixed_factor_value (int): Fixed value for the specified factor.

        Returns:
            numpy.ndarray: Batch of images with shape [batch_size, 3, 64, 64].
        """
        factors = np.zeros([len(self._FACTORS_IN_ORDER), batch_size], dtype=np.int32)
        for factor, name in enumerate(self._FACTORS_IN_ORDER):
            num_choices = self._NUM_VALUES_PER_FACTOR[name]
            factors[factor] = np.random.choice(num_choices, batch_size)
        factors[fixed_factor] = fixed_factor_value
        indices = self.get_index(factors)
        ims = []
        for ind in indices:
            im = self.images[ind]
            im = np.asarray(im)
            im = im.transpose((2, 0, 1))  # Transpose to [3, 64, 64]
            ims.append(im)
        ims = np.stack(ims, axis=0)
        ims = ims / 255.0  # Normalize values to [0, 1]
        ims = ims.astype(np.float32)
        return ims


if __name__ == "__main__":
    pass
