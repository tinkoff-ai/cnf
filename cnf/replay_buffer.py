import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(2e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.storage = dict()
        self.storage['state'] = np.zeros((max_size, state_dim))
        self.storage['action'] = np.zeros((max_size, action_dim))
        self.storage['next_state'] = np.zeros((max_size, state_dim))
        self.storage['reward'] = np.zeros((max_size, 1))
        self.storage['done'] = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.storage['state'][self.ptr] = state
        self.storage['action'][self.ptr] = action
        self.storage['next_state'][self.ptr] = next_state
        self.storage['reward'][self.ptr] = reward
        self.storage['done'][self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.tensor(self.storage['state'][ind], dtype=torch.float32, device=device),
            torch.tensor(self.storage['action'][ind], dtype=torch.float32, device=device),
            torch.tensor(self.storage['reward'][ind], dtype=torch.float32, device=device),
            torch.tensor(self.storage['done'][ind], dtype=torch.float32, device=device),
            torch.tensor(self.storage['next_state'][ind], dtype=torch.float32, device=device)
        )

    def save(self, filename):
        np.save("./buffers/" + filename + ".npy", self.storage)

    def load(self, data, subtract_one: bool = False):
        assert('next_observations' in data.keys())
        for i in range(data['observations'].shape[0] - 1):
            r = data['rewards'][i]
            if subtract_one:
                r -= 1
            self.add(data['observations'][i], data['actions'][i], data['next_observations'][i],
                     r, data['terminals'][i])
        print("Dataset size:" + str(self.size))
