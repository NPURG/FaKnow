import torch
from torch import nn


class PGD(object):
    def __init__(self, model: nn.Module, emb_name, epsilon=1., alpha=0.3):
        """Projected Gradient Descent (PGD) attack on a PyTorch model.

        Args:
            model (nn.Module): The PyTorch model to be attacked.
            emb_name: The name of the embedding parameter to be perturbed.
            epsilon (float, optional): The maximum perturbation allowed (default: 1.0).
            alpha (float, optional): The step size for each iteration of the attack (default: 0.3).
        """
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        """Perform the PGD attack on the model.

        Args:
            is_first_attack (bool, optional): If True, it creates a backup of the model's embeddings
                                              before performing the first attack (default: False).
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        """Restore the original embeddings of the model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name: str, param_data, epsilon: float):
        """Project the perturbed embeddings to stay within the allowed epsilon neighborhood.

        Args:
            param_name (str): Name of the embedding parameter.
            param_data: The perturbed embedding data.
            epsilon (float): The maximum allowed perturbation.

        Returns:
            Tensor: The projected embedding data.
        """
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        """Backup the gradients of the model's parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        """Restore the original gradients of the model's parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]
