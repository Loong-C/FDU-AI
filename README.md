# FDU-AI

复旦大学人工智能课程实验。本目录为 Project 1：搜索，基于 UC Berkeley AI Projects 改编的勇者斗恶龙教学框架。

## 运行环境

- Python 3；当前代码使用 Python 3.12 验证。
- 图形界面需要 Tkinter；自动评分可以使用 `--no-graphics`。
- Q0–Q9 的当前实现不需要配置模型 API。`jev/` 是可选的模型辅助搜索实验，配置方式见 [Jev 说明](jev/README.md)。

在仓库根目录运行：

```bash
python hero.py
python hero.py -h
```

热身示例会初始化一张包含两条恶龙的地图，输出金币坐标列表和第二条恶龙的位置：

```bash
python warmup.py
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

## 完成情况

Q1–Q9 均已实现。2026-10-09 按最小路径代价记录修订 Q2 后，在 Python 3.12.14 下运行仓库现有完整自动评分，结果为 **37/37**。

| 题目 | 内容 | 实现位置 | 得分 |
| --- | --- | --- | --- |
| Q1 | BFS 图搜索 | `search.py`：`breadthFirstSearch` | 3/3 |
| Q2 | A* 图搜索 | `search.py`：`aStarSearch` | 3/3 |
| Q3 | Minimax | `multiAgents.py`：`MinimaxAgent` | 5/5 |
| Q4 | 四角问题建模 | `searchAgents.py`：`CornersProblem` | 3/3 |
| Q5 | 四角启发函数 | `searchAgents.py`：`cornersHeuristic` | 3/3 |
| Q6 | 全金币启发函数 | `searchAgents.py`：`coinHeuristic` | 4/4 |
| Q7 | Alpha-Beta 剪枝 | `multiAgents.py`：`AlphaBetaAgent` | 5/5 |
| Q8 | Expectimax | `multiAgents.py`：`ExpectimaxAgent` | 5/5 |
| Q9 | 改进局面评估 | `multiAgents.py`：`betterEvaluationFunction` | 6/6 |

实现要点：

- Q2 用 `best_cost` 记录各状态已发现的最小路径代价 `g`；仅将严格更低代价的路径加入队列，弹出时丢弃已被更低代价替代的旧条目。已扩展的状态仍可通过更短路径重新扩展；可采纳且目标值为 0 的启发函数不必一致，也可保持最优性。
- Q4 的状态为 `(当前位置, 四个角落的访问标记)`，使用不可变元组区分不同访问进度。
- Q5 枚举剩余角落的访问顺序，取最小曼哈顿距离总和；最多枚举 24 种顺序。
- Q6 取当前位置到剩余金币的最远迷宫距离，通过 BFS 求距离，并按两个端点缓存结果。
- 多智能体搜索的一层深度表示勇者及所有恶龙各行动一次；终局会提前结束搜索。
- Q7 按原有动作顺序逐个生成后继，保留祖先传下的剪枝边界，按本项目要求在严格不等时剪枝。
- Q8 的恶龙节点对合法动作的后继价值求平均。
- Q9 综合游戏分数、金币与宝剑距离、恶龙距离和剩余虚弱时间评价局面。

## 运行与验证

完整评分：

```bash
python autograder.py --no-graphics
```

单题评分（将 `q6` 替换为 `q1` 至 `q9` 中的目标题号）：

```bash
python autograder.py -q q6 --no-graphics
```

评分器会自动运行配置中声明的依赖题目，例如 Q5 会同时检查 Q1 和 Q4。

图形演示：

```bash
# BFS 与 A*
python hero.py -l mediumMaze -p SearchAgent -a fn=bfs
python hero.py -l mediumMaze -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic

# 四角问题与全金币搜索
python hero.py -l mediumCorners -p AStarCornersAgent
python hero.py -l trickySearch -p AStarCoinSearchAgent

# 使用改进评估函数的 Expectimax
python hero.py -l smallClassic -p ExpectimaxAgent -a depth=2,evalFn=better -k 1 -f
```

最近一次完整评分中的关键结果：

| 指标 | 结果 |
| --- | --- |
| Q4 测试路线长度 | 28 步 |
| Q5 路线长度 / 扩展节点数 | 106 步 / 741 |
| Q6 扩展节点数 | 4137 |
| Q9 获胜局数 | 10/10 |
| Q9 平均游戏分数 | 1075.0 |
| Q9 平均每局勇者累计决策时间 | 0.763 秒 |

Q9 的上述结果来自测试配置指定的 `smallClassic`、一条随机恶龙和固定随机种子。不同地图、随机种子或运行环境可能产生不同结果；图形演示命令也不等同于十局评分。Q9 的时间评分统计平均每局勇者智能体累计思考时间，耗时以实际运行输出为准。

## 范围说明

默认评分范围为 Q1–Q9。模板中仍保留未实现的 DFS、UCS、最近金币搜索，以及使用默认评分的 `ReflexAgent`；它们不属于当前默认评分入口。使用 `SearchAgent` 时应显式指定 `fn=bfs` 或 `fn=astar`，因为其默认算法 DFS 尚未实现。

自动评分通过表示现有测试通过，课堂验收、截图和报告提交仍需按课程正式要求完成。

## 来源与许可

本项目改编自 [UC Berkeley AI Projects](http://ai.berkeley.edu/)。各源文件保留了原始作者信息和教学使用许可；具体条件请阅读文件头部的 Licensing Information。
