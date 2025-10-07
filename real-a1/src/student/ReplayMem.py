import pickle
import numpy as np
from collections import deque, defaultdict
import random
import tensorflow as tf


class ReplayMemory:
    """
    ReplayMemory class for storing and sampling transitions
    """

    def __init__(self, size):
        self.size = int(size)
        self.memory = None
        self.index = 0
        self.full = False

    def initialize(self, experience):
        example_shapes = [() if isinstance(exp, bool) else exp.shape for exp in experience]
        example_dtypes = [bool if isinstance(exp, bool) else exp.dtype for exp in experience]
        self.memory = [np.zeros((self.size, *shape), dtype=dtype) for shape, dtype in
                       zip(example_shapes, example_dtypes)]

    def add(self, experience):
        if self.memory is None:
            self.initialize(experience)

        for i, e in enumerate(experience):
            self.memory[i][self.index] = e

        self.index = (self.index + 1) % self.size
        if self.index == 0:
            self.full = True

    def sample(self, batch_size):
        max_index = self.size if self.full else self.index
        if batch_size > max_index:
            raise ValueError(
                f"Batch size ({batch_size}) cannot be greater than the number of available samples ({max_index})")
        random_idx = np.random.choice(max_index, size=batch_size, replace=False)
        return [mem[random_idx] for mem in self.memory]

    def get_size(self):
        return self.size if self.full else self.index

    def reset(self):
        # clear memory and reset index
        self.memory = None
        self.index = 0
        self.full = False


class RLBuffer:
    """
    Standard replay buffer for off-policy RL (e.g., SAC).
    Stores single-step transitions, optionally with context c.
    """

    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.storage = deque(maxlen=max_size)

    def add(self, s, a, r, s_next, done, c=None):
        """
        Add a single transition to the buffer.

        Args:
            s (np.ndarray): current state
            a (np.ndarray or scalar): action taken
            r (float): reward
            s_next (np.ndarray): next state
            done (bool): whether the episode terminated
            c: context label (can be None if you do not need it)
        """
        transition = (s, a, r, s_next, done, c)
        self.storage.append(transition)

    def sample(self, batch_size):
        """
        Sample a batch of transitions uniformly at random.

        Returns:
            A tuple (S, A, R, S_next, Done, C),
            each of which is a np.array or list of length batch_size.
        """
        batch = random.sample(self.storage, batch_size)

        S, A, R, S_next, Done, C = [], [], [], [], [], []
        for (s, a, r, s_next, done, c) in batch:
            S.append(s)
            A.append(a)
            R.append(r)
            S_next.append(s_next)
            Done.append(done)
            C.append(c)

        # Convert to np.array as needed. If states or actions have varying shapes,
        # you might keep them as a list of arrays instead.
        return (
            np.array(S, dtype=np.float32),
            np.array(A, dtype=np.float32),
            np.array(R, dtype=np.float32),
            np.array(S_next, dtype=np.float32),
            np.array(Done, dtype=np.float32),
            C  # context can stay as a list (esp. if c is not numeric)
        )

    def __len__(self):
        return len(self.storage)

    def save2file(self, file_path):
        with open(file_path, 'wb') as fp:
            pickle.dump(self.storage, fp)


class ContextualBuffer:
    """
    Buffer for storing trajectory slices for each context.
    Each context's data is stored as a list of slices, where each slice
    is a sub-trajectory (list of transitions).
    """

    def __init__(self, max_size=1000,
                 slice_size=30,
                 task='classification',
                 extra_info=False,
                 log_window=10):
        """
        Args:
            max_size (int): Maximum number of slices to store per context.
            slice_size (int): Length (#transitions) of each trajectory slice.
            task (str): 'classification' or 'regression'.
            extra_info (bool): Whether to include rewards in the output.
            log_window (int): Number of episodes to log for recent episode rewards and lengths.

        NOTE:
            Always call create_onehot_labels() after adding ALL contexts.
        """

        self.slice_size = slice_size
        self.max_size = max_size
        self.extra_info = extra_info
        self.log_info = defaultdict(lambda: deque(maxlen=log_window))

        # storage[context] -> list of slices
        self.storage = defaultdict(list)

        # temporary holder for transitions in the current episode
        self.episode_transitions = []

        # one-hot encoding for context labels
        self.task = task
        if task == 'classification':
            self.contexts_sorted = []
            self.context_to_idx = {}
            self.idx_to_context = {}
            self.num_contexts = 0

    def add(self, s, a, r, c):
        # Args: s, a , r, c(hashable context key)
        self.episode_transitions.append((s, a, r, c))

    def finish_episode(self, R):
        """
        Finish the current episode:
          - Break the stored transitions into slices of size `self.slice_size`
          - Store each slice into the buffer keyed by its context
        """
        num_transitions = len(self.episode_transitions)

        if num_transitions <= self.slice_size:  # If the episode is too short, discard it
            self.episode_transitions = []
            return

        context = self.episode_transitions[0][3]  # Get the context of the episode

        # log the episode reward and length
        self.log_info[context].append((np.sum(R), len(R)))

        num_full_slices = num_transitions // self.slice_size
        idx = 0
        for _ in range(num_full_slices):
            chunk = self.episode_transitions[idx: idx + self.slice_size]
            idx += self.slice_size
            self.storage[context].append(chunk)

            while len(self.storage[context]) > self.max_size:  # If the buffer is full, remove the oldest slice
                self.storage[context].pop(0)

        self.episode_transitions = []  # Clear the temporary holder

    def _flatten(self, slice_data):
        """ A helper function to turn a slice (list of transitions) into a single vector or array.

        Args:
            slice_data (list): List of transitions in the slice
            extra_info (bool): Whether to include rewards in the output
        """
        flat_slice = []
        for s, a, r, _ in slice_data:
            if a.ndim == 0:
                a = np.expand_dims(a, axis=0)
            if self.extra_info:
                flat_slice.append(np.concatenate([s, a, [r]]))
            else:
                flat_slice.append(np.concatenate([s, a]))
        return np.concatenate(flat_slice, axis=0)

    def sample_recent(self, c, num_slices=10, return_type=tf.Tensor):
        """
        Sample recent slices for a given context. (Used for rollouts)
        """
        if c not in self.storage:
            raise ValueError(f"Context {c} not found in the buffer.")

        if len(self.storage[c]) == 0:
            raise ValueError(f"No data available for context {c}.")

        num_slices = min(num_slices, len(self.storage[c]))
        recent_slices = self.storage[c][-num_slices:]
        X = [self._flatten(slice_data) for slice_data in recent_slices]
        random.shuffle(X)

        return np.array(X, dtype=np.float32) if return_type == np.array else tf.convert_to_tensor(X, dtype=tf.float32)

    def sample_by_context(self, batch_size, c=None):
        """
        Sample a batch of *flattened slices* (X) and their context labels (Y).

        Args:
            batch_size (int): Number of slices to sample.
            c can be:
                - None: sample from all contexts
                - single context key (e.g., a tuple)
                - a list of context keys. NOTE: len(c) should be == batch_size

        Returns:
        (X_batch, Y_batch):
            X_batch: np.ndarray of shape [batch_size, dX]
            Y_batch: np.ndarray of shape [batch_size, dY]
                     (dY could be num_contexts if one-hot, or dimension of c if regression)
        """
        all_contexts = list(self.storage.keys())

        if c is None:
            contexts = random.choices(all_contexts, k=batch_size)
        elif isinstance(c, list):
            if len(c) != batch_size:
                raise ValueError("Length of context list should be equal to batch_size.")
            contexts = c
        else:
            if c not in self.storage or len(self.storage[c]) == 0:
                raise ValueError(f"Context {c} not found or empty in the buffer.")
            contexts = [c] * batch_size

        X_batch, Y_batch = [], []

        for ctx in contexts:
            if len(self.storage[ctx]) == 0:
                raise ValueError(f"No data available for context {ctx}.")

            slice_data = random.choice(self.storage[ctx])
            X_batch.append(self._flatten(slice_data))

            if self.task == 'classification':
                Y_batch.append(self._context_to_onehot(ctx))
            else:
                ctx = np.array(ctx, dtype=np.float32)
                Y_batch.append(ctx)

        X_batch = np.array(X_batch, dtype=np.float32)
        Y_batch = np.array(Y_batch, dtype=np.float32)

        return X_batch, Y_batch

    def sample_contrastive(self, batch_size, mode="in-batch", neg_k=6):
        """
        Sample a batch of slices for contrastive learning. ctx is used as semantic label.

        Args:
            batch_size (int): number of anchor-positive (and possibly negative) sets.

            mode (str): "in-batch" or "explicit_negatives"
                - "in-batch": returns a single batch of slices and labels; negatives are identified within the batch.
                - "explicit_negatives": returns (anchor, positive, negatives) arrays.

            neg_k (int): number of negatives per anchor if mode = "explicit_negatives".

        Returns:
            if mode == "in-batch":
                (X_anchor, X_positive)
            if mode == "explicit_negatives":
                (X_anchor, X_positive, X_negatives)

        """
        all_contexts = list(self.storage.keys())
        if len(all_contexts) < 2:
            raise ValueError("Insufficient contexts in the buffer to sample contrasts.")

        if mode == "in-batch":
            anchorX, posX, labels = [], [], []

            for _ in range(batch_size):
                ctx = random.choice(all_contexts)
                if len(self.storage[ctx]) < 2:  # Need at least 2 slices to sample a positive pair
                    continue

                anchor_slice, pos_slice = random.sample(self.storage[ctx], 2)
                # slice_data = random.choice(self.storage[ctx])
                anchorX.append(self._flatten(anchor_slice))
                posX.append(self._flatten(pos_slice))

                if self.task == 'classification':
                    labels.append(self._context_to_onehot(ctx))
                else:
                    labels.append(np.array(ctx, dtype=np.float32))

            anchorX = np.array(anchorX, dtype=np.float32)
            posX = np.array(posX, dtype=np.float32)
            labels = np.array(labels, dtype=np.float32)
            return (anchorX, posX), labels

        elif mode == "explicit_negatives":
            all_contexts = list(self.storage.keys())
            anchorX, posX, negX, labels_anchor = [], [], [], []

            for _ in range(batch_size):
                valid_ctxs = [c for c in all_contexts if len(self.storage[c]) >= 2]
                if not valid_ctxs:
                    break

                ctx = random.choice(valid_ctxs)

                # Pick a random slice from the context
                anchor_slice = random.choice(self.storage[ctx])
                positive_slice = random.choice(self.storage[ctx])
                while positive_slice is anchor_slice:  # Ensure anchor and positive are different
                    positive_slice = random.choice(self.storage[ctx])

                anchor_flat, pos_flat = self._flatten(anchor_slice), self._flatten(positive_slice)

                neg_contexts = [c for c in all_contexts if c != ctx]
                neg_slices = []
                for _ in range(neg_k):
                    nc = random.choice(neg_contexts)
                    neg_slice = random.choice(self.storage[nc])
                    neg_slices.append(self._flatten(neg_slice))

                anchorX.append(anchor_flat)
                posX.append(pos_flat)
                negX.append(neg_slices)

                if self.task == 'classification':
                    labels_anchor.append(self._context_to_onehot(ctx))
                else:
                    labels_anchor.append(np.array(ctx, dtype=np.float32))

            anchorX = np.array(anchorX, dtype=np.float32)
            posX = np.array(posX, dtype=np.float32)
            negX = np.array(negX, dtype=np.float32)  # shape: [batch_size, neg_k, dX]
            labels_anchor = np.array(labels_anchor, dtype=np.float32)
            return (anchorX, posX, negX), labels_anchor

        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'in-batch' or 'explicit_negatives'.")

    def create_onehot_labels(self, ):
        """
        Create one-hot encoding for existing context labels in the buffer.
        NOTE: This should be called after all contexts have been added.
        """
        unique_contexts = sorted(list(self.storage.keys()))
        self.contexts_sorted = unique_contexts
        self.context_to_idx = {c: i for i, c in enumerate(unique_contexts)}
        self.idx_to_context = {i: c for c, i in self.context_to_idx.items()}
        self.num_contexts = len(unique_contexts)

    def _context_to_onehot(self, c):
        """ Helper function to convert a context label to one-hot encoding """
        onehot = np.zeros(self.num_contexts, dtype=np.float32)
        idx = self.context_to_idx[c]
        if idx is not None:
            onehot[idx] = 1.0
        return onehot

    def get_info(self, tabular=False):
        """
        Print or return information about the buffer contents.
        """
        info = {}
        for ctx, data in self.log_info.items():
            rewards, lengths = zip(*data)
            avg_reward = np.round(np.mean(rewards), 2)
            std_reward = np.round(np.std(rewards), 2)
            avg_length = np.round(np.mean(lengths), 2)
            total_num_samples = len(self.storage[ctx])
            info[ctx] = (avg_reward, std_reward, avg_length, total_num_samples)

        if tabular:
            from tabulate import tabulate
            headers = ["Context", "Avg. Reward", "Std. Reward", "Avg. Length", "Num. Samples"]
            table = [[str(ctx), f"{r:.2f}", f"{s:.2f}", f"{l:.2f}", n] for ctx, (r, s, l, n) in info.items()]
            print(tabulate(table, headers=headers))
        return info

    def clear(self):
        self.storage.clear()
        self.episode_transitions = []
        self.contexts_sorted = []
        self.context_to_idx = {}
        self.idx_to_context = {}
        self.num_contexts = 0

    def save_to_file(self, file_path):
        data = {
            'storage': self.storage,
            'slice_size': self.slice_size,
            'max_size': self.max_size,
            'contexts_sorted': self.contexts_sorted,
            'context_to_idx': self.context_to_idx,
            'idx_to_context': self.idx_to_context,
            'num_contexts': self.num_contexts
        }
        with open(file_path, 'wb') as fp:
            pickle.dump(data, fp)

    def load_from_file(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.storage = data['storage']
        self.slice_size = data['slice_size']
        self.max_size = data['max_size']
        self.contexts_sorted = data['contexts_sorted']
        self.context_to_idx = data['context_to_idx']
        self.idx_to_context = data['idx_to_context']
        self.num_contexts = data['num_contexts']

        self.episode_transitions = []  # clear the temporary holder

    def _test_sampling(self, batch_size):
        import time
        import tracemalloc
        results = {
            'by_context': {'time': None, 'memory': None},
            'contrastive_in_batch': {'time': None, 'memory': None},
            'contrastive_explicit': {'time': None, 'memory': None}
        }

        tracemalloc.start()
        start_time = time.time()

        try:
            X, Y = self.sample_by_context(batch_size)
            results['by_context']['time'] = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            current = current / 10 ** 6
            peak = peak / 10 ** 6
            results['by_context']['memory'] = (current, peak)
        except Exception as e:
            results['by_context']['error'] = str(e)

        tracemalloc.reset_peak()

        start_time = time.time()

        try:
            X, Y = self.sample_contrastive(batch_size, mode='in-batch')
            results['contrastive_in_batch']['time'] = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            current = current / 10 ** 6
            peak = peak / 10 ** 6
            results['contrastive_in_batch']['memory'] = (current, peak)
        except Exception as e:
            results['contrastive_in_batch']['error'] = str(e)

        tracemalloc.reset_peak()
        start_time = time.time()

        try:
            X, Y, Z = self.sample_contrastive(batch_size, mode='explicit_negatives', neg_k=3)
            results['contrastive_explicit']['time'] = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            current = current / 10 ** 6
            peak = peak / 10 ** 6
            results['contrastive_explicit']['memory'] = (current, peak)
        except Exception as e:
            results['contrastive_explicit']['error'] = str(e)

        tracemalloc.stop()
        for method, data in results.items():
            print(f"Method: {method}")
            if 'error' in data:
                print(f"Error: {data['error']}")
            else:
                print(f"Time: {data['time']:.4f} sec")
                print(f"Memory: {data['memory']}")
            print("\n")
        return results