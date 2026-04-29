# 2026 数据科学与工程算法 实验一：数据流统计算法

## 运行说明

### 依赖
仅使用 Python 3.10+ 标准库，无需额外安装第三方包。

### 快速运行（使用仓库内样本数据）

```bash
# 从仓库根目录运行
python code/main.py --input data.txt
```

### 完整流水线（大文件 + 外部排序）

```bash
python code/main.py \
    --input Gowalla_totalCheckins.txt \
    --do-sort \
    --sorted-output data_sorted.txt \
    --grid-step 0.001 \
    --topk 10 \
    --checkpoint-every 100000 \
    --output-dir output
```

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | `data.txt` | 输入数据文件路径 |
| `--do-sort` | — | 对输入文件按时间戳排序（外部归并排序，支持大文件） |
| `--sorted-output` | `<input>.sorted` | 排序后的输出文件路径 |
| `--grid-step` | `0.001` | 网格步长（度），如 0.001 ≈ 111 m |
| `--topk` | `10` | Top-K 查询的 K 值 |
| `--checkpoint-every` | `100000` | 每处理 N 条记录触发一次 checkpoint 评测 |
| `--sample-size` | `200` | 每次 checkpoint 抽样的 key 数量 |
| `--output-dir` | `output` | 输出目录（存放 checkpoints.csv） |
| `--cms-width` | `20000` | Count-Min Sketch 列数（越大误差越小） |
| `--cms-depth` | `7` | Count-Min Sketch 行数（越大置信度越高） |
| `--mg-k` | `500` | Misra-Gries 计数器数量（越大 Top-K 召回越好） |
| `--bf-capacity` | `300000` | Bloom Filter 预期元素数量 |
| `--bf-fpr` | `0.01` | Bloom Filter 目标误判率 |
| `--print-all-users` | — | 打印所有用户的精确签到次数 |

### 输出文件

- `output/checkpoints.csv`：每个 checkpoint 的 sketch 精度评测指标

### 代码结构

```
code/
├── main.py               # 命令行入口
├── stream_processor.py   # 流式处理器（精确结构 + sketch 结构 + checkpoint 评测）
├── grid.py               # 经纬度网格编码工具
├── sort_external.py      # 外部归并排序（支持超内存大文件）
└── sketches/
    ├── bloom.py          # Bloom Filter（成员查询，无假阴性）
    ├── cms.py            # Count-Min Sketch（频率估计，MAE/MRE 评测）
    └── misra_gries.py    # Misra-Gries（Top-K 候选检测）
```

### Checkpoint 评测指标

| 指标 | 结构 | 说明 |
|------|------|------|
| `bf_user_fpr` / `bf_grid_fpr` | Bloom Filter | 对负例采样的假阳性率（理论无假阴性） |
| `cms_user_mae` / `cms_grid_mae` | Count-Min Sketch | 平均绝对误差 |
| `cms_user_mre` / `cms_grid_mre` | Count-Min Sketch | 平均相对误差（true>0 时计算） |
| `topk_user_recall` / `topk_grid_recall` | MG + CMS | Recall@K：sketch Top-K 与精确 Top-K 的重叠率 |
| `topk_user_precision` / `topk_grid_precision` | MG + CMS | Precision@K |
| `topk_user_jaccard` / `topk_grid_jaccard` | MG + CMS | Jaccard@K |

---

------

## 基本信息

- **实验名称**：数据流统计算法
- **截止时间**：2026 年 4 月 30 日 23:59:59
- **提交方式**：水杉码园（数据科学与工程算法基础仓库）
- **提交分支**：`homework1`

------

## 数据集描述

### 数据集来源

Gowalla 是一个基于位置的社交网络平台，用户通过 ** 签到（check-in）** 分享位置信息。

### 数据规模

- 用户数量：**196591 位**
- 签到记录：**6,442,890 条**
- 时间范围：**2009 年 2 月 – 2010 年 10 月**

### 数据格式（每条记录）

```
[用户ID] [签到时间] [纬度] [经度] [位置ID]
```

### 数据示例

```
196514 2010-07-24T13:45:06Z 53.3648119 -2.2723465833 145064
196514 2010-07-24T13:44:58Z 53.360511233 -2.276369017 1275991
196514 2010-07-24T13:44:46Z 53.3653895945 -2.2754087046 376497
```

### 重要约束

数据以**流式**到达：

- 按时间先后**逐行读取、逐行处理**
- 不允许一次性加载全部数据到内存
- 预处理阶段可对原始数据按时间排序

------

## 任务描述

### 0. 数据预处理与流式模拟

1. 对原始文件预处理，确保签到记录**按时间升序排列**。
2. 程序逐行读取文件，每读入一条记录**实时更新统计**。

------

### 1. 用户签到统计

1. 统计所有出现过的用户（至少 1 条签到）。
2. 统计每个用户的**签到总次数**。
3. 找出**签到次数最多**的用户（并列全部列出）。

------

### 2. 空间网格划分

1. 将经纬度空间划分为固定大小网格：
	- 精度至少**百分位**
	- 建议：千分位 / 万分位
2. 统计每个网格：
	- 是否有签到
	- 签到总次数
3. 找出前 K 个热门网格（K ≥ 10），输出：
	- 网格坐标范围
	- 签到次数

------

### 3. 实验报告要求

需详细说明**数据结构与性能**：

1. **数据结构**：用户计数、网格计数所用结构（哈希表 / 字典 / 数组等）及选择理由。
2. **构建与更新时间**：单条记录处理的平均时间复杂度。
3. **查询时间**：获取最高频用户、热门网格的耗时。
4. **空间占用**：运行时内存估算（与数据集规模关联）。
5. **精度分析**：网格大小对统计结果的影响、覆盖性、边界处理。

------

## 注意事项

1. 时间解析
	- 时间为 ISO 8601 格式
	- 时区为 UTC（以 `Z` 结尾）
2. 网格边界
	- 建议使用**左闭右开区间**，保证每个点唯一落入一个网格
3. 输出格式
	- 结果清晰可读，打印用户统计、热门网格列表等
4. 数据集文件
	- `Gowalla_totalCheckins.txt`

------

## 评分与提交规则

### 评分细则

- 实验报告：规范、页数 ≤10 页（单栏）
- 功能完整、代码正确、可正常运行

### 扣分点

- 迟交：每天扣 **10 分**
- 超页数：每超 1 页扣 **1 分**，最多扣 10 分

### 提交格式

1. 进入水杉码园私人仓库
2. 切换到 `homework1` 分支
3. 文件组织：
	- `code/`：存放所有代码
	- `report.pdf`：实验报告
	- `README.md`（可选）