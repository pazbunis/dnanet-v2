import numpy as np
from os.path import join


class EnhancersData:

    class SeqDataSet:
        def __init__(self, dataset_folder, suffix):
            X_path = join(dataset_folder, 'X_bin_', suffix, '.npy')
            y_path = join(dataset_folder, 'Y_bin_', suffix, '.npy')
            headers_path = join(dataset_folder, 'headers_', suffix, '.npy')
            self._headers = np.load(headers_path)
            self._num_examples = len(self._headers)
            # un-compress data
            X_bin = np.load(X_path)
            Xrec = np.unpackbits(X_bin)
            self._seqs = np.reshape(Xrec, [self._num_examples, 4, 1000])
            y_bin = np.load(y_path)
            yrec = np.unpackbits(y_bin)
            
            self._labels = np.zeros((self._num_examples, 2))
            self._labels[np.arange(self._num_examples), yrec] = 1
            
            # init
            self._epochs_completed = 0
            self._index_in_epoch = 0

        @property
        def seqs(self):
            return self._seqs

        @property
        def labels(self):
            return self._labels

        @property
        def headers(self):
            return self._headers

        @property
        def num_examples(self):
            return self._num_examples
    
        @property
        def epochs_completed(self):
            return self._epochs_completed
    
        def next_batch(self, batch_size):
          """Return the next `batch_size` examples from this data set."""
          start = self._index_in_epoch
          self._index_in_epoch += batch_size
          if self._index_in_epoch > self._num_examples:
              # Finished epoch
              self._epochs_completed += 1
              # Shuffle the data
              perm = np.arange(self._num_examples)
              np.random.shuffle(perm)
              self._seqs = self._seqs[perm]
              self._labels = self._labels[perm]
              # Start next epoch
              start = 0
              self._index_in_epoch = batch_size
              assert batch_size <= self._num_examples
          end = self._index_in_epoch
          return self._seqs[start:end], self._labels[start:end]

    def __init__(self, dataset_folder):
        self.train = self.SeqDataSet(dataset_folder, "train")
        self.validation = self.SeqDataSet(dataset_folder, "validation")
        self.test = self.SeqDataSet(dataset_folder, "test")
