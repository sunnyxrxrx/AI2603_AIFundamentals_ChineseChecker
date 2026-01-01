import datetime
import os
import numpy as np
import glob
import argparse

from gymnasium.spaces import Box, Discrete

from pettingzoo.classic import rps_v2

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import (
    PPO,
    PPOConfig,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.tune.logger import pretty_print
from ray.rllib.policy.policy import Policy

from ChineseChecker import chinese_checker_v0
from ChineseChecker.models.action_masking_rlm import TorchActionMaskRLM
from ChineseChecker.models.action_masking import ActionMaskModel
from ChineseChecker.logger import custom_log_creator
from rllib_marl import train

import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")

def create_config(env_name: str, triangle_size: int = 4, entropy_coeff: float = 0.0):
    """
    创建PPO训练配置
    
    参数:
        env_name: 环境名称
        triangle_size: 跳棋三角区域大小，决定棋盘规模
        entropy_coeff: 熵系数，用于鼓励探索
    """
    rlm_class = TorchActionMaskRLM  # 使用支持动作掩码的RL模块

    # 神经网络配置：
    model_config = {
        #"fcnet_hiddens": [64, 64]
        # "fcnet_hiddens":[256, 256, 128]
        "conv_filters": [
            [64, [3, 3], 1],   # 第1层：提取基础特征
            [128, [3, 3], 1],  # 第2层：提取复杂特征
            [128, [3, 3], 1],  # 第3层：加深网络
        ],
        "conv_activation": "relu",
        
        # CNN 提取完特征后，再接一个全连接层输出策略
        "fcnet_hiddens": [256, 128],
    }

    # 定义RL模块规格
    rlm_spec = SingleAgentRLModuleSpec(
        module_class=rlm_class, 
        model_config_dict=model_config
    )

    # 计算动作空间维度：(4n+1)^2 × 6方向 × 2移动类型 + 1结束回合
    action_space_dim = (4 * triangle_size + 1) * (4 * triangle_size + 1) * 6 * 2 + 1
    # 观察空间形状：(4n+1)^2 × 4个通道，扁平化
    # observation_space_shape = ((4 * triangle_size + 1) * (4 * triangle_size + 1) * 4,)
    dim = 4 * triangle_size + 1
    observation_space_shape = (dim, dim, 4) 
    # 主要配置：配置ActionMaskEnv和ActionMaskModel
    config = (
        PPOConfig()
        .environment(
            # 环境设置
            env=env_name, 
            clip_actions=True,  # 裁剪动作到合法范围
            env_config={
                "triangle_size": triangle_size,  # 三角区域大小
                "action_space": Discrete(action_space_dim),  # 动作空间
                "max_iters": 100,  # 最大迭代次数
                "render_mode": None,  # 渲染模式（训练时不渲染）
                # 环境配置的观察空间，实际环境会创建包含"observation"和"action_mask"的字典
                "observation_space": Box(low=0, high=1, shape=(observation_space_shape), dtype=np.int8),
            },
        )
        .training(
            train_batch_size=2048,      # 加大采样量，更稳
            lr=1e-4,                    # 加快学习速度
            gamma=0.995,                # 看得更远
            lambda_=0.95,               # 标准 GAE 参数
            use_gae=True,
            clip_param=0.2,             # 回归标准值，防止崩盘
            grad_clip=0.5,              # 防止梯度爆炸
            entropy_coeff=0.01,         # 鼓励探索新战术
            vf_loss_coeff=0.5,          # 重视局势判断
            sgd_minibatch_size=512,     # 喂饱 GPU
            num_sgd_iter=20,            # 多复习几遍
            _enable_learner_api=True
        )
        # 实验性设置：禁用预处理器，因为环境返回的是字典观察
        .experimental(
            _disable_preprocessor_api=True,
        )
        .framework("torch")  # 使用PyTorch框架
        .resources(
            num_gpus=1,  # 确保这里是 1
        )
        .rl_module(rl_module_spec=rlm_spec)  # 设置RL模块
        .rollouts(
            num_rollout_workers=20,  
            num_envs_per_worker=1
        )
    )
    return config
    
def main(args):
    """
    主函数：初始化环境、配置和开始训练
    """
    # 环境创建函数
    def env_creator(config):
        return chinese_checker_v0.env(**config)

    # 注册环境到RLlib
    env_name = 'chinese_checker_v0'
    # 模型名称：包含熵系数信息
    model_name = f'full_sharing{args.entropy_coeff}'
    register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))

    # 测试环境创建（用于验证）
    test_env = PettingZooEnv(env_creator({"triangle_size": 2}))

    # 初始化Ray
    #ray.init(num_cpus=1 or None, local_mode=True)
    #ray.init()
    ray.init(ignore_reinit_error=True, num_cpus=20) # 根据你的服务器核数设置
    
    # 创建配置
    config = create_config(env_name, args.triangle_size, args.entropy_coeff)
    
    # 训练配置参数
    train_config = {
        "triangle_size": args.triangle_size,
        "train_iters": args.train_iters,
        "entropy_coeff": args.entropy_coeff,
        "eval_period": args.eval_period,
        "eval_num_trials": args.eval_num_trials,
        "eval_max_iters": args.eval_max_iters,
        "render_mode": args.render_mode,
    }
    
    # 开始训练
    train(config, model_name, train_config)
    print("Finished successfully without selecting invalid actions.")
    
    # 关闭Ray
    ray.shutdown()

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(
        prog='RLLib train script'
    )
    parser.add_argument('-t', '--train',
                        action='store_true')  # 训练标志
    parser.add_argument('--train_iters', type=int, default=500)  # 训练迭代次数
    parser.add_argument('--triangle_size', type=int, required=True)  # 三角区域大小
    parser.add_argument('--eval_period', type=int, default=50)  # 评估周期（每多少次训练迭代评估一次）
    parser.add_argument('--eval_num_trials', type=int, default=10)  # 评估试验次数
    parser.add_argument('--eval_max_iters', type=int, default=400)  # 评估最大迭代次数
    parser.add_argument('--entropy_coeff', type=float, default=0.001)
    parser.add_argument('--render_mode', type=str, default=None)  # 渲染模式
    args = parser.parse_args()
    main(args)