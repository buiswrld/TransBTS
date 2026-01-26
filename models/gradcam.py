import torch
import torch.nn.functional as F
import numpy as np


class GradCAM3D:
	"""Simple Grad-CAM for 3D Conv models.

	Hooks a target module (by name) and computes a class-specific CAM
	upsampled to the input spatial dimensions.

	Usage:
		gradcam = GradCAM3D(model, target_layer_name='endconv', use_cuda=False)
		cam = gradcam.generate_cam(input_tensor, class_idx)
	where `input_tensor` is a torch.Tensor shaped (N, C, D, H, W).
	Returns: numpy array shaped (D, H, W) for the first batch element.
	"""

	def __init__(self, model, target_layer_name='endconv', use_cuda=False):
		self.model = model
		self.use_cuda = use_cuda and torch.cuda.is_available()
		base = getattr(model, 'module', model)
		modules = dict(base.named_modules())
		if target_layer_name not in modules:
			raise ValueError(f"target layer '{target_layer_name}' not found in model modules")
		self.target = modules[target_layer_name]
		self.activations = None
		self.gradients = None
		self._register_hooks()

	def _register_hooks(self):
		def forward_hook(module, inp, outp):
			# store activations tensor (we keep reference so backward fills grad)
			self.activations = outp

			# register hook on the activation tensor to capture gradients during backward
			def _backward_hook(grad):
				self.gradients = grad

			# outp may be a tensor or tuple
			if isinstance(outp, torch.Tensor):
				outp.register_hook(_backward_hook)

		self.target.register_forward_hook(forward_hook)

	def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = 0):
		"""Compute CAM for the first item in the batch.

		Args:
			input_tensor: torch.Tensor of shape (N, C, D, H, W)
			class_idx: target class index in model output channels
		Returns:
			numpy array (D, H, W) with values normalized to [0,1]
		"""
		with torch.enable_grad():
			if self.use_cuda:
				self.model = self.model.cuda()
				input_tensor = input_tensor.cuda()

			# clear previous state
			self.activations = None
			self.gradients = None
			# ensure gradients enabled
			self.model.zero_grad()

			# forward
			output = self.model(input_tensor)

			# output expected shape: (N, num_classes, Df, Hf, Wf)
			if output.dim() != 5:
				raise RuntimeError(f"Expected model output to be 5D (N,C,D,H,W), got {output.shape}")

			# pick the scalar score for target class (sum spatially to get scalar per-batch)
			score = output[:, class_idx, ...].sum()

			# backward to populate gradients on activations
			score.backward(retain_graph=True)

			if self.activations is None or self.gradients is None:
				raise RuntimeError("GradCAM failed to capture activations or gradients. Was a forward/backward run executed?")

			grads = self.gradients  # shape (N, C, Df, Hf, Wf)
			acts = self.activations  # shape (N, C, Df, Hf, Wf)

			# global-average-pool gradients over spatial dims -> weights
			weights = grads.mean(dim=(2, 3, 4), keepdim=True)  # (N, C, 1,1,1)

			# weighted combination of activations
			cam = (weights * acts).sum(dim=1, keepdim=True)  # (N,1,Df,Hf,Wf)
			cam = F.relu(cam)

			# upsample to input spatial size
			target_size = input_tensor.shape[2:]
			cam_up = F.interpolate(cam, size=target_size, mode='trilinear', align_corners=False)
			cam_up = cam_up.squeeze(1)  # (N, D, H, W)

			cam_vol = cam_up[0]
			cam_vol = cam_vol.detach().cpu()

			# normalize to [0,1]
			cam_np = cam_vol.numpy()
			cam_np -= cam_np.min()
			denom = cam_np.max() if cam_np.max() != 0 else 1.0
			cam_np = cam_np / (denom + 1e-8)

		return cam_np


__all__ = ["GradCAM3D"]