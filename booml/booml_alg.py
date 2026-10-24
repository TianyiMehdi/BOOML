import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from booml.common import math
from booml.common.scale import RunningScale
from booml.common.world_model import WorldModel
from booml.common import layers

class BOOML:
	"""
	Modified TD-MPC2 agent. Implements training + inference.
	Current implementation supports both state and pixel observations.
	Only support Single-task setting is supported.
	Adds a learned score function to fit Langevin gradients.
	"""

	def __init__(self, cfg, device=None):
		self.cfg = cfg
		if device is not None:
			self.device = device
		else:
			if torch.cuda.is_available():
				self.device = torch.device("cuda")
			else:
				self.device = torch.device("cpu")
		self.model = WorldModel(cfg, self.device).to(self.device)
		
		self.optim = torch.optim.Adam(
			[
				{
					"params": self.model._encoder.parameters(),
					"lr": self.cfg.lr * self.cfg.enc_lr_scale,
				},
				{"params": self.model._dynamics.parameters()},
				{"params": self.model._reward.parameters()},
				{"params": self.model._Qs.parameters()},
				{
					"params": self.model._task_emb.parameters()
					if self.cfg.multitask
					else []
				},
			],
			lr=self.cfg.lr,
		)
		self.score_optim = torch.optim.Adam(
			self.model._score_function.parameters(), lr=self.cfg.lr, eps=1e-5
		)
		self.pi_optim = torch.optim.Adam(
			self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5
		)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.log_pi_scale = RunningScale(cfg) # policy log-probability scale
		self.cfg.mppi_iterations += 2 * int(
			cfg.action_dim >= 20
		)  # Heuristic for large action spaces
		self.discount = (
			torch.tensor(
				[self._get_discount(ep_len) for ep_len in cfg.episode_lengths],
				device="cuda",
			)
			if self.cfg.multitask
			else self._get_discount(cfg.episode_length)
		)
		self.alpha = getattr(cfg, 'alpha', 1.0)
		self.use_score_function = getattr(cfg, 'use_score_function', True)
		self.train_score_function = getattr(cfg, 'train_score_function', True)
		self._latest_langevin_grad_norm = float("nan")
		self._latest_langevin_grad_norm_clipped = float("nan")

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
				episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
				float: Discount factor for the task.
		"""
		frac = episode_length / self.cfg.discount_denom
		return min(
			max((frac - 1) / (frac), self.cfg.discount_min), self.cfg.discount_max
		)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
				fp (str): Filepath to save state dict to.
		"""
		torch.save({
			"model": self.model.state_dict(),
		}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
				fp (str or dict): Filepath or state dict to load.
		"""
		state_dict = fp if isinstance(fp, dict) else torch.load(fp)
		self.model.load_state_dict(state_dict["model"])

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None, use_pi=False, use_diffusion=False):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
				obs (torch.Tensor): Observation from the environment.
				t0 (bool): Whether this is the first observation in the episode.
				eval_mode (bool): Whether to use the mean of the action distribution.
				task (int): Task index (only used for multi-task experiments).

		Returns:
				torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		z = self.model.encode(obs, task)
		if self.cfg.mpc and not use_pi and not use_diffusion:
			a, mu, std = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
		elif use_pi:
			mu, pi, log_pi, log_std = self.model.pi(z, task)
			if eval_mode:
				a = mu[0]
			else:
				a = pi[0]
			mu, std = mu[0], log_std.exp()[0]
		return a.cpu(), mu.cpu(), std.cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task, horizon, eval_mode=False):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G += discount * reward
			discount *= (
				self.discount[torch.tensor(task)]
				if self.cfg.multitask
				else self.discount
			)
		return G + discount * self.model.Q(
			z, self.model.pi(z, task)[1], task, return_type="avg"
		)
	
	def _estimate_value_with_action_rollout(self, z, a_0, task, horizon, eval_mode=False):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		action = a_0
		for t in range(horizon):
			reward = math.two_hot_inv(self.model.reward(z, action, task), self.cfg)
			z = self.model.next(z, action, task)
			# with torch.no_grad():
			action = self.model.pi(z, task)[1]
			G += discount * reward
			discount *= (
				self.discount[torch.tensor(task)]
				if self.cfg.multitask
				else self.discount
			)
		return G + discount * self.model.Q(
			z, self.model.pi(z, task)[1], task, return_type="avg"
		)

	def plan(self, z, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.
		Combines Langevin optimization on a_0 and MPPI optimization on full horizon.
		
		Args:
				z (torch.Tensor): Latent state from which to plan.
				t0 (bool): Whether this is the first observation in the episode.
				eval_mode (bool): Whether to use the mean of the action distribution.
				task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
				torch.Tensor: Action to take in the environment.
		"""
		# ========== Part 1: Initialize a_0 ==========
		if self.cfg.num_pi_trajs > 0:
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			pi_a0 = self.model.pi(_z, task)[1]

		z_langevin = z.repeat(self.cfg.num_pi_trajs, 1)
		
		# ========== Part 2: LaRO on a_0 ==========
		max_grad_norm = getattr(self.cfg, 'max_grad_norm', 25.0)
		min_grad_norm = getattr(self.cfg, 'min_grad_norm', 1e-4)
		eta = getattr(self.cfg, 'eta', 5e-3)
		
		z_eval_langevin = z_langevin.detach()
		langevin_grad_norms = []
		langevin_grad_norms_clipped = []
		
		if self.use_score_function:
			a_0 = pi_a0.detach()
			for _ in range(self.cfg.langevin_iterations):
				with torch.no_grad():
					if self.cfg.multitask:
						z_input = self.model.task_emb(z_eval_langevin, task)
					else:
						z_input = z_eval_langevin
					score_input = torch.cat([z_input, a_0], dim=-1)
					grad = self.model._score_function(score_input)
				
				grad_norm = grad.norm(dim=1, keepdim=True)
				grad_norm_mean = grad_norm.mean().item()

				clip_factor = torch.where(
					grad_norm > max_grad_norm,
					max_grad_norm / (grad_norm + 1e-12),
					torch.ones_like(grad_norm)
				)
				grad = grad * clip_factor
				clipped_grad_norm = grad.norm(dim=1, keepdim=True)
				langevin_grad_norms.append(grad_norm_mean)
				langevin_grad_norms_clipped.append(clipped_grad_norm.mean().item())

				a_0 = (a_0 - (eta / self.alpha) * grad + (2 * eta) ** 0.5 * torch.randn_like(a_0)).clamp(-1, 1)
				
				if grad_norm_mean < min_grad_norm:
					break
		else:
			a_0 = pi_a0.detach().requires_grad_(True)
			with torch.enable_grad():
				for _ in range(self.cfg.langevin_iterations):
					# Compute value and loss
					value = self._estimate_value_with_action_rollout(z_eval_langevin, a_0, task, self.cfg.horizon)
					loss = -value.sum()
					grad = torch.autograd.grad(loss, a_0, create_graph=False, retain_graph=False)[0]
					
					grad_norm = grad.norm(dim=1, keepdim=True)
					grad_norm_mean = grad_norm.mean().item()

					clip_factor = torch.where(
						grad_norm > max_grad_norm,
						max_grad_norm / (grad_norm + 1e-12),
						torch.ones_like(grad_norm)
					)
					grad = grad * clip_factor
					clipped_grad_norm = grad.norm(dim=1, keepdim=True)
					langevin_grad_norms.append(grad_norm_mean)
					langevin_grad_norms_clipped.append(clipped_grad_norm.mean().item())

					a_0 = (a_0 - eta * grad + (2 * eta) ** 0.5 * torch.randn_like(a_0)).clamp(-1, 1)
					a_0 = a_0.detach().requires_grad_(True)
					
					if grad_norm_mean < min_grad_norm:
						break

		if langevin_grad_norms:
			self._latest_langevin_grad_norm = float(np.mean(langevin_grad_norms))
			self._latest_langevin_grad_norm_clipped = float(np.mean(langevin_grad_norms_clipped))
		else:
			self._latest_langevin_grad_norm = float("nan")
			self._latest_langevin_grad_norm_clipped = float("nan")
		
		langevin_a0 = a_0.detach()
		# ========== Part 3: MPPI optimization (parallel, independent from Langevin) ==========
		z_mppi = z.repeat(self.cfg.num_samples, 1)
		
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = self.cfg.max_std * torch.ones(
			self.cfg.horizon, self.cfg.action_dim, device=self.device
		)
		if not t0:
			mean[:-1] = self._prev_mean[1:] if hasattr(self, '_prev_mean') else mean[:-1]
		
		actions = torch.empty(
			self.cfg.horizon,
			self.cfg.num_samples,
			self.cfg.action_dim,
			device=self.device,
		)
		
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(
				self.cfg.horizon,
				self.cfg.num_pi_trajs,
				self.cfg.action_dim,
				device=self.device,
			)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon - 1):
				pi_actions[t] = self.model.pi(_z, task)[1]
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1] = self.model.pi(_z, task)[1]
			actions[:, : self.cfg.num_pi_trajs] = pi_actions

		for _ in range(self.cfg.mppi_iterations):
			actions[:, self.cfg.num_pi_trajs :] = (
					mean.unsqueeze(1)
					+ std.unsqueeze(1)
					* torch.randn(
						self.cfg.horizon,
						self.cfg.num_samples - self.cfg.num_pi_trajs,
						self.cfg.action_dim,
						device=std.device,
					)
				).clamp(-1, 1)
			
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			value = self._estimate_value(z_mppi, actions, task, self.cfg.horizon).nan_to_num_(0)
			elite_idxs = torch.topk(
				value.squeeze(1), self.cfg.num_elites, dim=0
			).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			max_value = elite_value.max(0)[0]
			score = torch.exp(self.cfg.temperature * (elite_value - max_value))
			score /= score.sum(0)
			score = score.squeeze()
			score_sum = score.sum() + 1e-9

			mean = torch.einsum('e,hea->ha', score, elite_actions) / score_sum
			diff = elite_actions - mean.unsqueeze(1)
			variance = torch.einsum('e,hea->ha', score, diff ** 2) / score_sum
			std = torch.sqrt(variance).clamp_(self.cfg.min_std, self.cfg.max_std)
		# ========== Part 4: (MLAP) Evaluate and select between LaRO and MPPI ==========
		with torch.no_grad():
			langevin_std_precomputed = torch.std(langevin_a0, dim=0).clamp_(self.cfg.min_std, self.cfg.max_std)
			score_tensor = score.squeeze()
			mppi_index = torch.multinomial(score_tensor, 1).item()
			mppi_actions = elite_actions[:, mppi_index] 
			mppi_mu = mppi_actions[0]
			mppi_std = std[0]
			self._prev_mean = mean
			
			
			z_eval_langevin = z.repeat(self.cfg.num_pi_trajs, 1)
			langevin_value = self._estimate_value_with_action_rollout(z_eval_langevin, langevin_a0, task, self.cfg.horizon)
			langevin_value = langevin_value.squeeze() if langevin_value.dim() > 1 else langevin_value

			mppi_value = elite_value[mppi_index].squeeze()
			mppi_best_value = mppi_value.item() if mppi_value.numel() == 1 else mppi_value[0].item()
			
			k = min(self.cfg.topk_samples, langevin_value.size(0)) if hasattr(self.cfg, 'topk_samples') else min(5, langevin_value.size(0))
			topk_values, topk_indices = torch.topk(langevin_value, k)
			langevin_best_idx = topk_indices[0] if eval_mode else topk_indices[torch.randint(0, k, (1,)).item()]
			langevin_best_value = langevin_value[langevin_best_idx].item()
			langevin_mu = langevin_a0[langevin_best_idx]
			
			use_langevin = langevin_best_value > mppi_best_value
			if use_langevin:
				mu = langevin_mu
				std0 = langevin_std_precomputed
			else:
				mu = mppi_mu
				std0 = mppi_std
			
			if eval_mode or use_langevin:
				a = mu
			else:
				a = mu + std0 * torch.randn(self.cfg.action_dim, device=std0.device)

		return a.clamp_(-1, 1), mu, std0
	
	def update_pi(self, zs, action, mu, std, task, step):
		"""
		Update policy using a sequence of latent states.

		Args:
				zs (torch.Tensor): Sequence of latent states.
				action (torch.Tensor): Sequence of actions.
				task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
				float: Loss of the policy update.
		"""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)

		_, pis, log_pis, log_std = self.model.pi(zs, task)
		qs = self.model.Q(zs, pis, task, return_type="min")
		qs_mu = self.model.Q(zs, pis, task, return_type="min")
		
		self.scale.update(qs[0])
		qs = self.scale(qs)
		qs_mu = self.scale(qs_mu)
			
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))

		action_dims = None if not self.cfg.multitask else self.model._action_masks.size(-1)
		std = log_std.exp().detach()
		std = torch.max(std, self.cfg.min_std * torch.ones_like(std))
		eps = (pis - mu) / std
		log_pis_prior = math.gaussian_logprob(eps, std.log(), size=action_dims).mean(dim=-1)

		log_pis_prior = self.scale(log_pis_prior) if self.scale.value > self.cfg.scale_threshold else torch.zeros_like(log_pis_prior)

		log_pis_prior = torch.softmax(qs_mu.detach().squeeze(),dim=-1) * log_pis_prior
		prior_loss = - (log_pis_prior.sum(dim=-1) * rho).mean()

		q_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
		pi_loss = q_loss + (self.cfg.action_dim / self.cfg.res_coef) * prior_loss

		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(
			self.model._pi.parameters(), self.cfg.grad_clip_norm
		)
		
		self.pi_optim.step()
		self.model.track_q_grad(True)

		return pi_loss.item(), q_loss.item(), prior_loss.item()

	@torch.no_grad()
	def _td_target(self, next_z, reward, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
				next_z (torch.Tensor): Latent state at the following time step.
				reward (torch.Tensor): Reward at the current time step.
				task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
				torch.Tensor: TD-target.
		"""
		pi = self.model.pi(next_z, task)[1]
		discount = (
			self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		)
		return reward + discount * self.model.Q(
			next_z, pi, task, return_type="min", target=True
		)

	def update(self, buffer, replay_sample, logger, step):	
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
				buffer (common.buffer.Buffer): Replay buffer.

		Returns:
				dict: Dictionary of training statistics.
		"""
		self._step = step
		obs, action, mu, std, reward, task = replay_sample # mu and std are from Gaussian policy used for data collection	
		
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, task)
			
		# Prepare for update
		self.optim.zero_grad(set_to_none=True)
		self.model.train()

		# Latent rollout
		zs = torch.empty(
			self.cfg.horizon + 1,
			self.cfg.batch_size,
			self.cfg.latent_dim,
			device=self.device,
		)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t in range(self.cfg.horizon):
			z = self.model.next(z, action[t], task)
			consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
			zs[t + 1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type="all")
		reward_preds = self.model.reward(_zs, action, task)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t in range(self.cfg.horizon):
			reward_loss += (
				math.soft_ce(reward_preds[t], reward[t], self.cfg).mean()
				* self.cfg.rho**t
			)
			for q in range(self.cfg.num_q):
				value_loss += (
					math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean()
					* self.cfg.rho**t
				)
		consistency_loss *= 1 / self.cfg.horizon
		reward_loss *= 1 / self.cfg.horizon
		value_loss *= 1 / (self.cfg.horizon * self.cfg.num_q)

		# Combine all losses
		total_loss = (
			self.cfg.consistency_coef * consistency_loss
			+ self.cfg.reward_coef * reward_loss
			+ self.cfg.value_coef * value_loss
		)
			
		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(
			self.model.parameters(), self.cfg.grad_clip_norm
		)
		self.optim.step()

		# Update policy
		pi_loss, pi_loss_q, pi_loss_prior  = self.update_pi(_zs.detach(), action.detach(), mu.detach(), std.detach(), task, step)
		
		# ========== Compute score function loss ==========
		score_loss = torch.tensor(0.0, device=self.device)
		if self.train_score_function:
			self.score_optim.zero_grad(set_to_none=True)
			z_train = _zs[0].detach()  # [batch_size, latent_dim]
			a_train = action[0].detach()  # [batch_size, action_dim]
			
			# Prepare input for score function
			if self.cfg.multitask:
				z_input = self.model.task_emb(z_train, task)
			else:
				z_input = z_train
			score_input = torch.cat([z_input, a_train], dim=-1)
			
			pred_grad = self.model._score_function(score_input)
			
			a_train_grad = a_train.clone().requires_grad_(True)
			z_eval_train = z_train.detach()
			
			with torch.enable_grad():
				value_train = self._estimate_value_with_action_rollout(
					z_eval_train, a_train_grad, task, self.cfg.horizon
				)
				loss_train = -value_train.sum()
				true_grad = torch.autograd.grad(
					loss_train, a_train_grad, create_graph=False, retain_graph=False
				)[0]
			
			score_loss = F.mse_loss(pred_grad, true_grad.detach())

			score_loss.backward()
			score_grad_norm = torch.nn.utils.clip_grad_norm_(
				self.model._score_function.parameters(), self.cfg.grad_clip_norm
			)
			self.score_optim.step()
		
		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		return {
			"consistency_loss": float(consistency_loss.mean().item()),
			"reward_loss": float(reward_loss.mean().item()),
			"value_loss": float(value_loss.mean().item()),
			"pi_loss": pi_loss,
			"pi_loss_q": pi_loss_q,
			"pi_loss_prior": pi_loss_prior,
			"total_loss": float(total_loss.mean().item()),
			"grad_norm": float(grad_norm),
			"langevin_grad_norm": self._latest_langevin_grad_norm,
			"langevin_grad_norm_clipped": self._latest_langevin_grad_norm_clipped,
			"pi_scale": float(self.scale.value),
			"score_loss": float(score_loss.item())
		}
