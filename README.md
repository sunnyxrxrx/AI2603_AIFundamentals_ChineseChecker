# AI2603_AIFundamentals_ChineseChecker
<<<<<<< HEAD
## Environment sutup
You need to run in Linux-based system, gpu is recommended but not necessary.

After creating a new environment with python>=3.10, run the following command:
```
conda install anaconda::swig 
pip install -r requirements.txt
```
## Run Minimax
To test our Minimax agent, you can run the command:
```
python play.py --triangle_size xx [--render_mode human]
```
- triangle size is default to 2
- For RL opponents, a pretrain model trained on size 2 board is provided, you can import it by changing line 35 into
  ```python
  rl_baseline_policy = load_policy(
        os.path.join(
            os.path.dirname(__file__),
            'pretrain',
        )
    )
  ```
- Render mode is optional.

## Run RL
We provide two RL agents, agent1 is trained on size 2 board, agent2 is trained on size 4 board.
- To test agent1, firstly, change line 47 of ``play.py`` into 
  ```python
  your_policy = load_policy(args.checkpoint)
  ```
    Then run 
    ```
  python play.py --triangle_size 4 --use_rl --checkpoint ppo_agent_99_size_2
  ```
  Where ``ppo_agent_99_size_2`` is our agent.
- To test agent2, firstly, change line 35 of ``play.py`` into 
  ```python
    rl_baseline_policy = load_policy(
          os.path.join(
              os.path.dirname(__file__),
              'checkpoint99',
          )
      )
    ```
    Where ``checkpoint99`` is a MLP based model trained on size 4 board, serving as a baseline opponent.
  
  Then run the following command
  ```
  python play.py --triangle_size 4 --use_rl --checkpoint ppo_agent_460_size_4.pth
  ```
  Where ``ppo_agent_460_size_4.pth`` is our agent.

## Train RL
You can chosse to train MLP based RL by running
```
python train.py --triangle_size xx
```
checkpoints will be saved in ``logs/``

You can also train a CNN based RL by running
```
python mytrain.py --triangle_size xx
```
checkpoints will be save in the form of ``ppo_agent_[iteration]_size_[size].pth``
=======

train.py是原有的采用全连接神经网络实现的基于PPO的强化学习算法，配套采用的中国跳棋环境是ChineseChecker\env文件夹中'chinese_checker_env_primal'环境。

文件'checkpoint99size2'是采用train.py训练出来的triangle_size为2的RL模型，文件'checkpoint99'是采用train.py训练出来的triangle_size为4的RL模型。

可以在play.py中运行 play.py --use_rl --checkpoint (checkpoint) 进行模型验证。
>>>>>>> 73c0362b6e717e9c1651d4172e8235a77e7a5681
