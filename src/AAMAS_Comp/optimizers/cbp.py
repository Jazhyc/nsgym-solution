"""
Taken from: https://github.com/balovess/continual-bp/blob/main/cbp.py

Continual Backprop acts as a wrapper around an existing optimizer.
Usage:
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
cbp = ContinualBackpropagation(model, optimizer=optimizer, reset_rate=0.01, maturity_threshold=50)

for step, (x, y) in enumerate(dataloader):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()       # Normal optimizer step
    cbp.step(step)         # CBP step: updates utility, resets low-utility weights
"""


import torch
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

class MemoryPool:
    def __init__(self):
        """
        Initializes the MemoryPool.
        """
        self.pool = {}

    def allocate(self, shape, dtype=torch.float16, device=None):
        """
        Allocates memory from the pool or creates new memory if not available.

        Args:
            shape (torch.Size): Shape of the tensor to allocate.
            dtype (torch.dtype): Data type of the tensor (default: torch.float16).
            device (torch.device): Device to allocate memory on (default: None, uses default device).

        Returns:
            torch.Tensor: A tensor with the requested shape, dtype, and device, either from the pool or newly created.
        """
        key = (shape, dtype, device)
        if key not in self.pool:
            self.pool[key] = torch.zeros(shape, dtype=dtype, device=device)
        return self.pool[key]

class ContinualBackpropagation:
    def __init__(self, model, optimizer=None, reset_rate=0.01, maturity_threshold=50, utility_decay=0.05,
                 reset_init='uniform', momentum=0.9, device=None):
        """
        Initializes the Continual Backpropagation (CBP) algorithm.

        Args:
            model (torch.nn.Module): PyTorch model to apply CBP to.
            optimizer (torch.optim.Optimizer, optional): Optimizer to use (default: None).
            reset_rate (float): Proportion of weights to reset in each reset step (default: 0.01).
            maturity_threshold (int): Maturity threshold for weights to be considered for reset (default: 50).
            utility_decay (float): Decay factor for the utility values (default: 0.05).
            reset_init (str): Initialization method for reset weights ('uniform', 'kaiming', 'orthogonal', 'normal', 'constant') (default: 'uniform').
            momentum (float): Momentum for smoothing utility value changes (default: 0.9).
            device (torch.device, optional): Device to perform computations on (default: None, uses default device).
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.reset_rate = reset_rate
        self.maturity_threshold = maturity_threshold
        self.utility_decay = utility_decay
        self.reset_init = reset_init
        self.momentum = momentum

        # Create memory pool
        self.memory_pool = MemoryPool()

        # Allocate all weight states using memory pool
        self.state = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                self.state[name] = {
                    'utility': self.memory_pool.allocate(param.shape, dtype=torch.float16, device=param.device),
                    'maturity': self.memory_pool.allocate(param.shape, dtype=torch.int16, device=param.device),
                    'is_frozen': self.memory_pool.allocate(param.shape, dtype=torch.bool, device=param.device)
                }

    def _reset_weights(self, param, indices):
        """
        Resets selected weights using the specified initialization strategy.

        Args:
            param (torch.Tensor): Weight parameter to reset.
            indices (torch.Tensor): Indices of the weights to reset (flattened).
        """
        with torch.no_grad():  # Disable gradient tracking during reset
            flat_param = param.flatten()
            if self.reset_init == 'uniform':
                new_values = (torch.rand(indices.size(), device=param.device) - 0.5) * 0.1
            elif self.reset_init == 'orthogonal':
                if param.dim() < 2:
                    raise ValueError("Orthogonal initialization requires at least 2D tensors.")
                else:
                    new_values = torch.empty(indices.size(), device=param.device)
                    init.orthogonal_(new_values.view(-1, 1))
            elif self.reset_init == 'normal':
                new_values = torch.randn(indices.size(), device=param.device) * 0.1
            elif self.reset_init == 'constant':
                new_values = torch.zeros(indices.size(), device=param.device)
            else:
                raise ValueError(f"Unsupported reset_init type: {self.reset_init}")

            flat_param[indices] = new_values
            # Print reset weight information
            # print(f"Resetting weights for {param.shape} at indices {indices}")

    def _update_utility(self):
        """
        Updates the utility values of weights based on gradient contribution.
        """
        for name, param in self.model.named_parameters():
            if param.grad is None or 'weight' not in name:
                continue

            grad_contribution = torch.abs(param.grad).detach() * (torch.abs(param).detach() + 1e-8).to(param.device)

            # Smoothing with momentum
            if self.optimizer and param in self.optimizer.state: # Check if param has state in optimizer
                state = self.optimizer.state[param]
                exp_avg_sq = state.get('exp_avg_sq', torch.ones_like(param, device=param.device))
                momentum_buffer = state.get('momentum_buffer', torch.zeros_like(param, device=param.device))
                grad_contribution *= (torch.abs(momentum_buffer) + 1e-8)
                grad_contribution /= (torch.sqrt(exp_avg_sq) + 1e-8)

            # Update utility value
            utility = self.state[name]['utility']
            # old_utility = utility.clone()  # Get utility value before updating
            utility = self.utility_decay * utility + (1 - self.utility_decay) * grad_contribution
            self.state[name]['utility'] = torch.clamp(utility, min=0, max=1e6)
            # Print utility update information
            # if not torch.equal(old_utility, utility):
            #     print(f"Utility updated for {name}: {old_utility.flatten()} -> {utility.flatten()}")


    def _selective_reset(self):
        """
        Selectively resets low-utility weights and freezes high-utility weights.
        """
        for name, param in self.model.named_parameters():
            if 'weight' not in name or name not in self.state:  # Only process weights
                continue

            utility = self.state[name]['utility'].flatten().to(param.device)
            maturity = self.state[name]['maturity'].flatten().to(param.device)
            is_frozen = self.state[name]['is_frozen'].flatten().to(param.device)

            # Freeze high-utility weights
            freeze_threshold = max(torch.quantile(utility, 0.9), 1e-6)  # Avoid all-zero utility causing freeze failure
            is_frozen[utility > freeze_threshold] = True
            # Print freeze information
            # if torch.any(is_frozen):
            #     print(f"Freezing weights for {name} at indices {is_frozen.nonzero(as_tuple=True)}")

            # Filter resettable weights
            valid_mask = (~is_frozen) & (maturity >= self.maturity_threshold)
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]

            # Dynamically determine the number of weights to reset
            num_to_reset = min(int(self.reset_rate * utility.numel()), valid_indices.numel())
            if num_to_reset > 0:
                reset_indices = valid_indices[torch.randperm(valid_indices.numel())[:num_to_reset]]
                self._reset_weights(param, reset_indices)
                utility[reset_indices] = 0
                maturity[reset_indices] = 0
                # Print reset information
                # print(f"Resetting {num_to_reset} weights for {name} at indices {reset_indices}")

            # Update state back
            self.state[name]['utility'] = utility.view_as(param)
            self.state[name]['maturity'] = maturity.view_as(param)
            self.state[name]['is_frozen'] = is_frozen.view_as(param)

    def _increment_maturity(self):
        """
        Dynamically increments maturity based on gradient contribution.
        """
        for name, param in self.model.named_parameters():
            if param.grad is None or name not in self.state:
                continue

            grad_contribution = torch.abs(param.grad).detach() * (torch.abs(param).detach() + 1e-8)

            # Use median and IQR as dynamic threshold
            median_contribution = torch.median(grad_contribution)
            q75, q25 = torch.quantile(grad_contribution, 0.75), torch.quantile(grad_contribution, 0.25)
            iqr_contribution = q75 - q25
            dynamic_threshold = median_contribution + 1.5 * iqr_contribution

            # Increment maturity
            self.state[name]['maturity'] += (grad_contribution > dynamic_threshold).long()

    def step(self, current_step, step_interval=1):
        """
        Performs a single update step of CBP.

        Args:
            current_step (int): Current training step number.
            step_interval (int): Interval at which CBP operations (utility update, reset, maturity increment) are performed (default: 1).
        """
        if current_step % step_interval == 0:
            self._update_utility()
            self._selective_reset()
            self._increment_maturity()

