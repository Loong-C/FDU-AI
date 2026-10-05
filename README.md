# FDU-AI

复旦大学人工智能课程实验。本目录为 Project 1：搜索，基于 UC Berkeley AI Projects 改编的勇者斗恶龙教学框架。

## 运行环境

- Python 3；当前代码使用 Python 3.12 验证。
- 图形界面需要 Tkinter；自动评分可以使用 `--no-graphics`。
- Q0–Q3 不需要配置模型 API。`jev/` 是可选的模型辅助搜索实验，配置方式见 [Jev 说明](jev/README.md)。

在仓库根目录运行：

```bash
python hero.py
python hero.py -h
```

## 项目结构

| 文件或目录 | 用途 |
| --- | --- |
| `search.py` | BFS、A* 等通用搜索算法 |
| `searchAgents.py` | 搜索问题建模、搜索智能体与启发函数 |
| `multiAgents.py` | Minimax、Alpha-Beta、Expectimax 与局面评估 |
| `hero.py`、`game.py` | 游戏入口、状态接口和运行框架 |
| `util.py` | 队列、优先队列等工具 |
| `layouts/`、`images/` | 地图和显示资源 |
| `autograder.py`、`test_cases/` | 自动评分入口与测试数据 |
| `jev/` | 可选模型实验 |

## 分题验证

完成相应题目后，使用以下命令运行该题现有测试：

```bash
python autograder.py -q q1 --no-graphics
python autograder.py -q q2 --no-graphics
python autograder.py -q q3 --no-graphics
```

Q1 为 BFS，Q2 为 A*，Q3 为 Minimax。Minimax 的一层深度表示勇者及所有恶龙各行动一次。

完整评分命令为 `python autograder.py --no-graphics`；尚未完成的题目会报错或失分。

## 来源与许可

本项目改编自 [UC Berkeley AI Projects](http://ai.berkeley.edu/)。各源文件保留了原始作者信息和教学使用许可；具体条件请阅读文件头部的 Licensing Information。
