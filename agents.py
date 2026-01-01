import csv
import datetime
import os
import numpy as np
import glob
import argparse
from pathlib import Path
from tqdm import tqdm

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
from gymnasium.spaces import Box, Discrete
from ChineseChecker.env.game import Direction, Move, Position, ChineseCheckers
import copy
import math
import time

# Random Policy 
class ChineseCheckersRandomPolicy(Policy):
    def __init__(self, triangle_size=4, config={}):
        observation_space = Box(low=0, high=1, shape=((4 * triangle_size + 1) * (4 * triangle_size + 1) * 4,), dtype=np.int8)
        action_space = Discrete((4 * triangle_size + 1) * (4 * triangle_size + 1) * 6 * 2 + 1)
        super().__init__(observation_space, action_space, config)
        self.action_space = action_space

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        actions = []
        for obs in obs_batch:
            action = self.action_space.sample(obs["action_mask"])
            actions.append(action)
        return actions, [], {}

    def compute_single_action(self, obs, state=None, prev_action=None, prev_reward=None, info=None, episode=None, **kwargs):
        return self.compute_actions([obs], state_batches=[state], prev_action_batch=[prev_action], prev_reward_batch=[prev_reward], info_batch=[info], episodes=[episode], **kwargs)[0]

# TODO: Greedy Policy
class GreedyPolicy(Policy):
    def __init__(self, triangle_size=4, config={}):
        # 观察空间：扁平化的棋盘状态 + 动作掩码
        observation_space = Box(low=0, high=1, shape=((4 * triangle_size + 1) * (4 * triangle_size + 1) * 4,), dtype=np.int8)
        # 动作空间：所有可能的移动 + 结束回合
        action_space = Discrete((4 * triangle_size + 1) * (4 * triangle_size + 1) * 6 * 2 + 1)
        super().__init__(observation_space, action_space, config)
        self.triangle_size = triangle_size
        self.action_space_dim = action_space.n
        
    def _action_to_move(self, action: int):
        """将动作索引转换为Move对象（从utils.py复制过来的逻辑）"""
        n = self.triangle_size
        # print(f"action {action}")
        if action == (4 * n + 1) ** 2 * 6 * 2:
            return Move.END_TURN
        
        index = action
        index, is_jump = divmod(index, 2)     # 提取是否跳跃
        index, direction = divmod(index, 6)   # 提取方向
        _q, _r = divmod(index, 4 * n + 1)     # 提取坐标索引
        q, r = _q - 2 * n, _r - 2 * n         # 转换为相对坐标
        return Move(q, r, direction, bool(is_jump))
    
    def _move_to_action(self, move: Move):
        """将Move对象转换为动作索引（从utils.py复制过来的逻辑）"""
        n = self.triangle_size
        
        if move == Move.END_TURN:
            return (4 * n + 1) ** 2 * 6 * 2
        
        q, r, direction, is_jump = move.position.q, move.position.r, move.direction, move.is_jump
        index = int(is_jump) + 2 * (direction + 6 * ((r + 2 * n) + (4 * n + 1) * (q + 2 * n)))
        return index
    
    def _calculate_move_score(self, move, player, board_observation):
        """
        计算移动的得分（奖励估计）
        基于环境中的奖励规则
        """
        n = self.triangle_size
        
        if move == Move.END_TURN:
            # 结束回合的得分较低，除非没有其他合法移动
            return 0.0
        
        # 基础得分
        score = 0.0
        # 1. 鼓励向目标区域移动，惩罚远离目标区域
        if move.direction in [Direction.DownLeft, Direction.DownRight]:
            move_distance = 2 if move.is_jump else 1
            score += 1.0 * move_distance
        elif move.direction in [Direction.UpLeft, Direction.UpRight]:
            move_distance = 2 if move.is_jump else 1
            score -= 1.0 * move_distance

        return score
    
    def _get_player_from_observation(self, observation):
        """
        从观察中推断当前玩家
        观察包含4个通道：当前玩家棋子、其他玩家棋子、跳跃起始位置、上次跳跃目标位置
        通过查找哪个通道有棋子来推断
        """
        n = self.triangle_size
        board_size = 4 * n + 1
        
        # 重塑观察为通道形式
        channels = observation["observation"].reshape(board_size, board_size, 4)
        
        # 第一个通道应该是当前玩家的棋子
        # 但为了安全，我们检查哪个通道有最多的棋子
        player0_pieces = np.sum(channels[:, :, 0])  # 通道0：当前玩家棋子
        player3_pieces = np.sum(channels[:, :, 1])  # 通道1：其他玩家棋子
        
        # 如果有棋子，返回对应玩家
        if player0_pieces > 0:
            return 0
        elif player3_pieces > 0:
            return 3
        else:
            # 默认返回0
            return 0
    
    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, 
                       prev_reward_batch=None, info_batch=None, episodes=None, **kwargs)-> list[int]:
        """
        计算一批观察的动作
        使用贪心策略：选择得分最高的合法动作
        """
        actions = []
        
        for i, obs in enumerate(obs_batch):
            # 获取哪些动作合法
            action_mask = obs["action_mask"]
            
            # 推断当前玩家
            player = self._get_player_from_observation(obs)
            
            # 初始化最佳动作和最高得分
            best_action = None
            best_score = -float('inf')
            
            # 遍历所有可能的动作
            for action_idx in range(self.action_space_dim):
                # 检查动作是否合法
                if action_mask[action_idx] == 1:
                    # 转换为Move对象
                    move = self._action_to_move(action_idx)
                    
                    # 计算动作得分
                    score = self._calculate_move_score(move, player, obs)
                    
                    # 更新最佳动作
                    if score > best_score:
                        best_score = score
                        best_action = action_idx
            
            # 如果没有找到合法动作（理论上不会发生），选择第一个合法动作
            if best_action is None:
                # 查找第一个合法动作
                for action_idx in range(self.action_space_dim):
                    if action_mask[action_idx] == 1:
                        best_action = action_idx
                        break
            
            # 如果没有合法动作，选择结束回合
            if best_action is None:
                best_action = self.action_space_dim - 1  # END_TURN动作
            
            actions.append(best_action)
        
        return actions, [], {}
    
    def compute_single_action(self, obs, state=None, prev_action=None, 
                              prev_reward=None, info=None, episode=None, **kwargs):
        """
        计算单个观察的动作
        """
        return self.compute_actions(
            [obs], 
            state_batches=[state], 
            prev_action_batch=[prev_action], 
            prev_reward_batch=[prev_reward], 
            info_batch=[info], 
            episodes=[episode], 
            **kwargs
        )[0]
#######################################################################
class MinimaxPolicy(GreedyPolicy):
    def __init__(self, triangle_size=4, config={}, depth=2):
        super().__init__(triangle_size, config)
        self.max_depth = depth
        self.time_limit = 5000.0 
        
        self.sim_offsets = {
            0: (1, 0), 1: (1, -1), 2: (0, -1),
            3: (-1, 0), 4: (-1, 1), 5: (0, 1)
        }

    def unwind_index(self, idx, dim):
        if idx > dim // 2: return idx - dim
        return idx

    def _obs_to_game(self, obs):
        """
        【核心魔法】将 Observation 逆向重构为 ChineseCheckers 游戏实例
        """
        # 1. 创建新游戏 (此时棋盘上有初始位置的棋子)
        game = ChineseCheckers(self.triangle_size)
        
        # 2. 【关键步骤】拔掉所有棋子！
        # 我们只保留地形 (EMPTY 和 OUT_OF_BOUNDS)，清除所有玩家 (0 和 3)
        game.board[game.board == 0] = ChineseCheckers.EMPTY_SPACE
        game.board[game.board == 3] = ChineseCheckers.EMPTY_SPACE
        
        # 3. 根据 obs 重新摆放棋子
        n = self.triangle_size
        dim = 4 * n + 1
        obs_board = obs["observation"].reshape(dim, dim, 4)
        offset = 2 * n
        
        # 遍历 obs 只有 1 的位置 (稀疏更新，效率更高)
        # Channel 0: 我方
        for r_idx, c_idx in np.argwhere(obs_board[:, :, 0] == 1):
            q = self.unwind_index(r_idx, dim)
            r = self.unwind_index(c_idx, dim)
            s = -q - r
            game.board[q+offset, r+offset, s+offset] = 0
            
        # Channel 1: 敌方
        for r_idx, c_idx in np.argwhere(obs_board[:, :, 1] == 1):
            q = self.unwind_index(r_idx, dim)
            r = self.unwind_index(c_idx, dim)
            s = -q - r
            game.board[q+offset, r+offset, s+offset] = 3
            
        return game

    def _evaluate_game(self, game):
        """评估函数"""
        score = 0
        n = self.triangle_size
        offset = 2 * n
        
        p0_indices = np.argwhere(game.board == 0)
        p3_indices = np.argwhere(game.board == 3)
        score += len(p0_indices) * 1000.0
        # 我方得分
        for idx in p0_indices:
            q = idx[0] - offset
            r = idx[1] - offset
            score += ((5.0 * r) - (5.0 * q))
            if r > 2: score += 20.0
            
            score-= 10* abs(r+2*q)
        if len(p0_indices) > 1:
            # 1. 计算重心 (Center of Mass)
            qs = [idx[0] - offset for idx in p0_indices]
            rs = [idx[1] - offset for idx in p0_indices]
            avg_q = sum(qs) / len(qs)
            avg_r = sum(rs) / len(rs)
            
            # 2. 计算离散度 (Variance-like)
            # 每个棋子到重心的曼哈顿距离之和
            scatter = 0
            for q, r in zip(qs, rs):
                scatter += abs(q - avg_q) + abs(r - avg_r)
            
            # 3. 施加惩罚
            # 权重建议：1.0 到 2.0
            # 如果太散 (scatter 大)，扣分
            score -= 6.0 * scatter


        # 敌方得分
        for idx in p3_indices:
            q = idx[0] - offset
            r = idx[1] - offset
            score += (10.0 * r) - (5.0 * q)
           
        return score

    def minimax_recursive(self, game, depth, alpha, beta, maximizing_player, start_time):
        # 1. 终止条件
        if depth == 0 or (time.time() - start_time > self.time_limit):
            return self._evaluate_game(game)
        
        current_player = 0 if maximizing_player else 3
        
        # 2. 获取合法移动 (递归层只能依赖引擎生成)
        # 注意：这里的 find_legal_moves 无法生成复杂的连跳，因为缺少历史
        # 但它能生成所有平移和单跳，这对于预测对手已经足够了
        game.find_legal_moves(current_player)
        legal_moves = game.get_legal_moves(current_player)
        
        if not legal_moves or (len(legal_moves) == 1 and legal_moves[0] == Move.END_TURN):
            return self._evaluate_game(game)

        # 3. 排序优化
        def move_priority(m):
            if m == Move.END_TURN: return -99
            if maximizing_player:
                if m.direction in [4, 5]: return 10
                if m.is_jump: return 5
            else:
                if m.direction in [4, 5]: return 10
            return 0
            
        legal_moves.sort(key=move_priority, reverse=True)
        beam_width = math.inf
        if len(legal_moves) > beam_width:
            # 确保 END_TURN 不会被切掉（如果它是唯一选择的话）
            # 但我们的排序把 END_TURN 放最后了，所以通常会被切掉，这正好
            legal_moves = legal_moves[:beam_width]

        if maximizing_player:
            max_eval = -math.inf
            for move in legal_moves:
                if move == Move.END_TURN: continue
                
                next_game = copy.deepcopy(game)
                # 在递归层，我们信任引擎生成的移动是合法的
                result = next_game.move(current_player, move)
                if result is None and move != Move.END_TURN: continue
                if move != Move.END_TURN and move.is_jump:
                    next_depth=depth
                    next_max = True
                else:
                    next_depth=depth-1
                    next_max=False
                eval_val = self.minimax_recursive(next_game, next_depth, alpha, beta, next_max, start_time)
                max_eval = max(max_eval, eval_val)
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in legal_moves:
                if move == Move.END_TURN: continue
                
                next_game = copy.deepcopy(game)
                result = next_game.move(current_player, move)
                if result is None and move != Move.END_TURN: continue
                if move != Move.END_TURN and move.is_jump:
                    next_depth = depth # 对手连跳也不减深度
                    next_max = False   # 还是对手
                else:
                    next_depth = depth - 1
                    next_max = True    # 换我方
                eval_val = self.minimax_recursive(next_game, next_depth, alpha, beta, next_max, start_time)
                min_eval = min(min_eval, eval_val)
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
            return min_eval

    def compute_single_action(self, obs, state=None, **kwargs):
        start_time = time.time()
        
        root_game = self._obs_to_game(obs)
        action_mask = obs["action_mask"]
        legal_indices = np.where(action_mask == 1)[0]
        
        if len(legal_indices) == 0:
            return [self.action_space_dim - 1]

        best_val = -math.inf
        best_action = legal_indices[0]
        
        def get_sort_key(idx):
            m = self._action_to_move(idx)
            if m == Move.END_TURN: return -1
            if m.direction == 4: return 2
            if m.direction == 5: return 1
            return 0

        sorted_indices = sorted(legal_indices, key=get_sort_key, reverse=True)

        alpha = -math.inf
        beta = math.inf

        for idx in sorted_indices:
            if time.time() - start_time > self.time_limit: break
            
            move = self._action_to_move(idx)
            
            next_game = copy.deepcopy(root_game)
            
            if move == Move.END_TURN:
                val = self._evaluate_game(next_game)
            else:
                result=next_game.move(0,move)
                if result is None:
                    print("手动移动")
                    src = move.position
                    dst = move.moved_position()
                    offset = 2 * self.triangle_size
                    
                    # 1. 计算数组索引
                    src_q, src_r, src_s = src.q + offset, src.r + offset, src.s + offset
                    dst_q, dst_r, dst_s = dst.q + offset, dst.r + offset, dst.s + offset
                    dst_idx=(dst_q, dst_r, dst_s)
                    # --- DEBUG 2: 幽灵检测 ---
                    if next_game.board[dst_idx] != ChineseCheckers.EMPTY_SPACE:
                        print(f"[WARNING] Ghost detected at {dst_idx}! Clearing it for move {move}.")
                        next_game.board[dst_idx] = ChineseCheckers.EMPTY_SPACE
                    # 2. 执行移动 (Player 0)
                    # 既然是合法移动，起点一定有子，终点一定为空(或为虚空)
                    # 我们直接暴力覆盖
                    next_game.board[src_q, src_r, src_s] = ChineseCheckers.EMPTY_SPACE
                    next_game.board[dst_q, dst_r, dst_s] = 0 # 0代表我方
                    if move.is_jump:
                        # 手动把这个跳跃加进历史，欺骗引擎
                        next_game._jumps.append(move)
                        
                        # 既然修补了历史，我们就可以自信地继续我方回合
                        next_max = True 
                        next_depth = self.max_depth # 深度不减，鼓励连跳
                    else:
                        # 平移不能连走
                        next_max = False
                        next_depth = self.max_depth - 1
                else:
                    # 引擎移动成功！_jumps 已更新
                    if move.is_jump:
                        next_max = True # 继续是我方
                        next_depth = self.max_depth # 深度不减
                    else:
                        next_max = False
                        next_depth = self.max_depth - 1
                
                # 进入递归 (Depth - 1)
                val = self.minimax_recursive(next_game, next_depth, alpha, beta, next_max, start_time)
                
                # 启发式微调
                if move.direction == 4: val += 0.5
                if move.direction == 5: val += 0.3
                if move.is_jump: val += 0.1

            if val > best_val:
                best_val = val
                best_action = idx
                # print(f"New Best: {move} Score: {val}")
            
            alpha = max(alpha, best_val)
        # print("finish finding")    
        return [best_action]