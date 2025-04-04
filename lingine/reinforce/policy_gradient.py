import gymnasium as gym
import torch
from torch import nn
from torch import optim

from typing import Optional


class ActorCriticPolicy(nn.Module):
    def __init__(
        self,
        policy_net: nn.Module,
        value_net: nn.Module,
    ):
        super().__init__()
        self.policy_net = policy_net
        self.value_net = value_net

    def forward(self, state):
        return self.policy_net(state), self.value_net(state)

    def get_action(self, state, greedy=False):
        if greedy:
            return self.policy_net(state).argmax(dim=-1)
        else:
            return self.policy_net(state).sample()


class VPGLearner:
    def __init__(
        self,
        ac_policy: ActorCriticPolicy,
        env: gym.Env = gym.make("CartPole-v1"),
        optimizer: type[optim.Optimizer] = optim.Adam,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        gemma: float = 0.99,
    ):
        self.env = env
        self.policy = ac_policy
        self.optimizer = optimizer(self.policy.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size
        self.gemma = gemma

    def rollout(self):
        states, actions, rewards = [], [], []
        state = self.env.reset()
        done = False
        while not done:
            states.append(state)
            action = self.policy.get_action(state)
            state, reward, done, _ = self.env.step(action)
            actions.append(action)
            rewards.append(reward)
        return states, actions, rewards

    def train(self):
        for epoch in range(self.epochs):
            states, actions, rewards = self.rollout()
            self.update_policy(states, actions, rewards)


class PPOLearner:
    def __init__(
        self,
        ac_policy: ActorCriticPolicy,
        env: gym.Env = gym.make("CartPole-v1"),
        optimizer: type[optim.Optimizer] = optim.Adam,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        gemma: float = 0.99,
        lambda_: float = 0.95,
    ):
        self.env = env
        self.policy = ac_policy
        self.optimizer = optimizer(self.policy.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size
        self.gemma = gemma
        self.lambda_ = lambda_

    def generalized_advantage_estimate(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ):
        # rewards: (B, T)
        # values: (B, T+1)

        deltas = rewards + self.gemma * values[:, 1:] - values[:, :-1]
        advantages = torch.zeros_like(deltas)
        for t in reversed(range(len(deltas))):
            advantages[t] = deltas[t] + self.gemma * self.lambda_ * advantages[t + 1]
        return advantages

    def rollout(self):
        states, actions, rewards, values = [], [], [], []
        state = self.env.reset()
        done = False
        while not done:
            states.append(state)
            action = self.policy.get_action(state)
            state, reward, done, _ = self.env.step(action)
            actions.append(action)
            rewards.append(reward)
            values.append(self.policy.value_net(state))
        return states, actions, rewards, values

    def train(self):

        for epoch in range(self.epochs):
            states, actions, rewards, values = self.collect_trajectories()
            advantages = self.generalized_advantage_estimate(rewards, values)
            self.update_policy(states, actions, advantages, values)
