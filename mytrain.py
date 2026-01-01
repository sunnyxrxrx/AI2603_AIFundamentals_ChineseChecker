import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.categorical import Categorical
from ChineseChecker import chinese_checker_v1
import time
import os
from agents import GreedyPolicy
import random
import argparse


# import debugpy
# debugpy.listen(('localhost', 12345))
# print("Waiting for debugger attach")
# debugpy.wait_for_client()


class MyPPOPolicy:
    def __init__(self, model_path, env, triangle_size=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = Agent(env,triangle_size).to(self.device)
        self.triangle_size=triangle_size
        # 加载权重
        state_dict = torch.load(model_path, map_location=self.device)
        for k, v in self.agent.state_dict().items():
            assert k in state_dict, f"Missing key in checkpoint: {k}"
            assert v.shape == state_dict[k].shape, \
                f"Shape mismatch at {k}: model {v.shape}, ckpt {state_dict[k].shape}"
        self.agent.load_state_dict(state_dict)
        self.agent.eval()
        print(f"Successfully loaded custom PPO model from {model_path}")

    def compute_single_action(self, obs):
        # 数据预处理
        obs_data = obs['observation']
        mask = obs['action_mask']
        
        # 假设obs是扁平的，需要reshape 
        # 如果环境已经是3D的，就不需要 reshape
        dim = 4 * self.triangle_size + 1 # 
        
        if len(obs_data.shape) == 1:
            now_size = (np.sqrt(len(obs_data)//4)-1)//4
            assert len(obs_data) == dim * dim * 4, f"Triangle size must be 4 to run our pretrained model! Now size: {now_size}"
            obs_data = obs_data.reshape(dim, dim, 4)
             
        obs_tensor = torch.tensor(obs_data, dtype=torch.float32).to(self.device)
        mask_tensor = torch.tensor(mask, dtype=torch.bool).to(self.device)

        # 推理
        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(
                obs_tensor.unsqueeze(0), 
                mask_tensor.unsqueeze(0)
            )
        
        #print(f"returning{action.item()}")
        return [action.item()]

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, env, triangle_size=4):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * (4*triangle_size+1)**2, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, env.action_space(env.possible_agents[0]).n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        

    def get_value(self, x):
        # x shape: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action_mask, action=None):
        # x shape: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        hidden = self.network(x)
        end_turn_idx = action_mask.shape[1] - 1
        has_other_moves = (action_mask.sum(dim=1) > 1)
        action_mask[has_other_moves, end_turn_idx] = 0
        # ---------------------
        HUGE_NEG = -1e8
        if True: 
            
            logits = self.actor(hidden)
            bool_mask = (action_mask > 0.5)
            masked_logits = logits.masked_fill(~bool_mask, HUGE_NEG)
        
        probs = Categorical(logits=masked_logits)
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

def main(args):
    TRIANGLE_SIZE = args.triangle_size
    TOTAL_TIMESTEPS = 2000000  # 总步数
    LEARNING_RATE = 3e-4
    NUM_STEPS = 2048           # 每次采集多少步 (Batch Size)
    MINIBATCH_SIZE = 512       # GPU 每次训练的小批次
    UPDATE_EPOCHS = 10         # 每次采集的数据训练几轮
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_COEF = 0.2
    ENT_COEF = 0.01
    VF_COEF = 0.5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {DEVICE}")
    # 初始化贪心陪练
    greedy_policy = GreedyPolicy(TRIANGLE_SIZE)
    # 初始化环境
    env = chinese_checker_v1.env(triangle_size=TRIANGLE_SIZE, render_mode=None)
    env.reset()
    # 初始化 Agent
    agent = Agent(env,triangle_size=TRIANGLE_SIZE).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # 存储 Buffer
    obs_shape = (4*TRIANGLE_SIZE+1, 4*TRIANGLE_SIZE+1, 4)
    action_dim = env.action_space(env.possible_agents[0]).n
    
    # 预分配内存
    obs_buffer = torch.zeros((NUM_STEPS,) + obs_shape).to(DEVICE)
    mask_buffer = torch.zeros((NUM_STEPS, action_dim)).to(DEVICE)
    actions_buffer = torch.zeros((NUM_STEPS,)).to(DEVICE)
    logprobs_buffer = torch.zeros((NUM_STEPS,)).to(DEVICE)
    rewards_buffer = torch.zeros((NUM_STEPS,)).to(DEVICE)
    dones_buffer = torch.zeros((NUM_STEPS,)).to(DEVICE)
    values_buffer = torch.zeros((NUM_STEPS,)).to(DEVICE)

    # 训练循环
    global_step = 0
    start_time = time.time()
    
    # 获取初始状态
    env.reset()
    
    iter_count = 0
    while global_step < TOTAL_TIMESTEPS:
        iter_count += 1
        agent.train()
        step=0
        # 阶段1 采样
        while step < NUM_STEPS:
            global_step += 1
            
            # 获取当前玩家的 observation
            agent_id = env.agent_selection
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None # AEC 要求传 None
                env.step(action)
                # 如果游戏结束，我们重置环境
                env.reset()
                continue
            if agent_id == 'player_3' and random.random() < 0.8:
                # 1. 用 Greedy 计算动作
                # Greedy 需要 dict 格式的 obs
                greedy_action = greedy_policy.compute_single_action(observation)
                act_int = int(greedy_action[0]) # Greedy 返回的是 list
                # 2. 执行动作
                env.step(act_int)
                continue 
            # 处理结束情况
            
            
            # 准备数据转 Tensor
            obs_tensor = torch.tensor(observation['observation'], dtype=torch.float32).to(DEVICE)
            # 恢复 3D 形状 (如果 env 是扁平的，这里要 reshape)
            obs_tensor = obs_tensor.reshape(obs_shape)
            
            mask_tensor = torch.tensor(observation['action_mask'], dtype=torch.bool).to(DEVICE)
            mask = observation['action_mask']
            # 存入 Buffer
            obs_buffer[step] = obs_tensor
            mask_buffer[step] = mask_tensor
            dones_buffer[step] = 0 # 简化处理

            # 选择动作
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    obs_tensor.unsqueeze(0), 
                    mask_tensor.unsqueeze(0)
                )
                
            values_buffer[step] = value.flatten()
            actions_buffer[step] = action
            logprobs_buffer[step] = logprob

            # 执行动作
            act_int = action.item()
            prev_cum_reward = env._cumulative_rewards[agent_id]
            env.step(act_int)
            curr_cum_reward = env._cumulative_rewards[agent_id]
            step_reward = curr_cum_reward - prev_cum_reward 
            rewards_buffer[step] = torch.tensor(step_reward, dtype=torch.float32).to(DEVICE)
            step += 1
            if step_reward < -500:
                print(f"[ALERT] Step {step}: Agent {agent_id} took Action {act_int} and got PENALTY {step_reward}!")
                print(f"Mask value for this action: {mask_tensor[act_int]}")
            
        # 阶段2 计算优势函数
        with torch.no_grad():
            next_value = 0 # 简化，假设最后一步后价值为0
            advantages = torch.zeros_like(rewards_buffer).to(DEVICE)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 # 简化
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buffer[t + 1]
                    nextvalues = values_buffer[t + 1]
                
                delta = rewards_buffer[t] + GAMMA * nextvalues * nextnonterminal - values_buffer[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            
            returns = advantages + values_buffer

        # 阶段3 PPO 更新 
        b_obs = obs_buffer.reshape((-1,) + obs_shape)
        b_logprobs = logprobs_buffer.reshape(-1)
        b_actions = actions_buffer.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buffer.reshape(-1)
        b_masks = mask_buffer.reshape((-1, action_dim))

        # 索引列表
        b_inds = np.arange(NUM_STEPS)
        clipfracs = []
        
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, NUM_STEPS, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], 
                    b_masks[mb_inds], 
                    b_actions.long()[mb_inds]
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # 计算KL散度用于监控
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > CLIP_COEF).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                # 标准化优势函数
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -CLIP_COEF,
                    CLIP_COEF,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                # Entropy Loss
                entropy_loss = entropy.mean()
                
                # 总 Loss
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        print(f"Iter {iter_count}: Mean Reward = {rewards_buffer.sum().item():.2f}, Value Loss = {v_loss.item():.4f}")
        
        # 保存模型
        if iter_count % 10 == 0:
            torch.save(agent.state_dict(), f"ppo_agent_{iter_count}_size_{args.triangle_size}.pth")
            
        # TODO. 保存优化器 断点续训 比率调优

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='RLLib train script'
    )
    parser.add_argument('--triangle_size', type=int, required=True)  # 三角区域大小
    args = parser.parse_args()
    main(args)