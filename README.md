# AI2603_AIFundamentals_ChineseChecker

train.py是原有的采用全连接神经网络实现的基于PPO的强化学习算法，配套采用的中国跳棋环境是ChineseChecker\env文件夹中'chinese_checker_env_primal'环境。
文件'checkpoint99size2'是采用train.py训练出来的triangle_size为2的RL模型，文件'checkpoint99'是采用train.py训练出来的triangle_size为4的RL模型。
可以在play.py中运行 play.py --use_rl --checkpoint (checkpoint) 进行模型验证。
