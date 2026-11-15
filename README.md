# 机器学习课设：四个任务的深度学习实现

东北大学《机器学习》课程设计 · 个人独立完成

| 项目 | 内容 |
|---|---|
| 姓名 | 汪开进 |
| 学号 | 20245658 |
| 专业 / 班级 | 人工智能 / 人工智能 2401 |
| 课程 | 机器学习课设 |

本仓库包含课设四个任务的完整实现：数据探查脚本、模型训练代码、提交文件生成与校验工具，以及各任务的实验结果与结论。四个任务的源码、提交文件和实验记录都在本仓库内，训练所需的数据集不随仓库分发（见下文「关于数据」）。

---

## 一、成绩总览

| 任务 | 题目 | 评价指标 | 本机交叉验证 | Kaggle 排行榜 |
|---|---|---|---|---|
| 任务一 | 植物幼苗分类（5 类） | macro-F1 / 准确率 | OOF macro-F1 **0.9010** | **0.9048** |
| 任务二 | 人脸表情分类（6 类） | 准确率 | OOF 准确率 **0.7253** | **0.741** |
| 任务三 | 眼底黄斑中心凹定位 | RMSE（越小越好） | 留一法 RMSE **18.9 px** | **13.5** |
| 任务四 | 眼底血管分割 | Dice | OOF Dice **0.8312** | **0.85052** |

四个任务全部使用深度学习方案，未使用任何外部标注、答案反查或他人代码。

---

## 二、目录结构

```
ml-course/
├── README.md                       本文件
├── .gitignore
├── 机器学习课设报告-新.docx          课程设计报告
│
├── task1-plant/                    任务一：植物幼苗分类
│   ├── src/classdesign4-1.nb.txt   notebook 源文件（按 #@cell 切分的纯文本）
│   ├── src/build_nb.js             由 .nb.txt 生成 .ipynb 的构建脚本
│   ├── classdesign4-1.ipynb        可直接导入 Kaggle 的 notebook
│   ├── tools/check_submission.py   提交文件格式校验（9 项）
│   ├── tools/compare_submission.py 与官方样例逐字节比对
│   ├── reference/                  官方参考 notebook 与样例提交
│   └── submission.csv              最终提交（排行榜 0.9048）
│
├── task2-emotion/                  任务二：人脸表情分类
│   ├── src/classdesign4-2.nb.txt
│   ├── src/build_nb.js
│   ├── classdesign4-2.ipynb
│   ├── kaggle_paste.py             16 个 code cell 拼成的单文件版
│   ├── tools/
│   ├── reference/
│   └── README.md                   该任务的详细实验记录
│
├── task3-fovea/                    任务三：黄斑中心凹定位
│   ├── t3_data.py                  数据加载、坐标变换、数据增强
│   ├── t3_geo.py / t3_geo2.py      视盘检测 + 几何先验基线
│   ├── t3_kaggle.py                最终交付版：两阶段热图 + 多候选自动选优
│   ├── t3_train.py / t3_refine.py / t3_wide.py   各阶段训练脚本
│   ├── t3_chain.py / t3_tta*.py    级联评估与 TTA 消融
│   ├── t3_submit.py                提交生成与校验
│   ├── t3_rmse.py                  指标计算（与官方口径一致）
│   ├── README.md                   该任务的详细实验记录
│   └── submission_*.csv            两版提交文件存档
│
└── task4-vessel/                   任务四：眼底血管分割
    ├── t4_kaggle.py                最终交付版：U-Net + 512 评测网格 + TTA
    ├── t4_blend.py / t4_post.py / t4_areacal.py   被证伪方向的验证脚本
    ├── t4_infer_exp.py             推理方式消融（整图 / 分块 / D4 TTA）
    ├── t4_tau_heldout.py / t4_tau_check2.py       阈值的留出检验
    ├── README.md                   该任务的详细实验记录
    └── submission_*.csv            三版提交文件存档
```

---

## 三、如何运行

### 任务一 / 任务二（notebook）

1. 打开 [Kaggle](https://www.kaggle.com)，新建 Notebook；
2. **File → Import Notebook** 导入对应的 `.ipynb`（注意：不要直接拖进编辑器，那样只会把 JSON 当成一个 cell 插进去）；
3. 右侧 **Add Input** 挂载比赛数据集；
4. **Settings** 里把 Accelerator 设为 **GPU T4 x2**，任务二还需把 **Internet 设为 On**（要下载预训练权重）；
5. **Save Version → Save & Run All (Commit)** 运行。

`src/*.nb.txt` 是可读的 notebook 源文件，`src/build_nb.js` 用 Node 把它重新构建成 `.ipynb`：

```bash
node task1-plant/src/build_nb.js task1-plant/src/classdesign4-1.nb.txt task1-plant/classdesign4-1.ipynb
```

### 任务三 / 任务四（单文件脚本）

两个任务各有一个自包含的 `*_kaggle.py`，把整个文件粘贴进 Kaggle 的一个 code cell（或作为脚本上传）即可运行，不需要额外文件。

- `task3-fovea/t3_kaggle.py`：整图热图粗定位 → 1:1 裁块精定位，5 折 + 翻转 TTA，按时间预算自动压缩每折 epoch，任何时刻都会保证输出一份合法提交。
- `task4-vessel/t4_kaggle.py`：U-Net 在 512 评测网格上训练（与官方评测口径一致）+ 4 向翻转 TTA + 阈值与最小连通域搜索。

可在文件顶部修改 `TIME_BUDGET_MIN` 控制运行时长；脚本会在预算用尽前停止训练，并写出当前最优的提交文件。

### 依赖

```
torch >= 2.0        timm >= 0.9        numpy        pandas
Pillow              scipy             nbformat     nodejs（仅构建 notebook 时需要）
```

---

## 四、提交文件格式校验

两个任务的 `tools/` 与各任务的 `check_submission` 都逐字节校验以下 9 项，避免因为格式问题白白浪费提交次数：

1. 表头与列数
2. 行数
3. ID 集合
4. **ID 行序与官方样例逐行一致**
5. 行尾符（CRLF）
6. 无 BOM
7. 结尾有换行
8. 无多余空行
9. 数值列可解析

任务四另需注意官方 RLE 约定：**1 基、行优先（C 序）**、连续段写成（起点, 长度）对，且**任何非零像素都算前景**——因此 mask 中血管必须写 255。

---

## 五、关于数据

训练与测试数据来自课程提供的比赛数据集（任务一、二、三为课程发布数据，任务四为 DRIVE 眼底数据集），**不随本仓库分发**，原因有二：

1. 体积远超 GitHub 仓库的合理范围（单个权重文件就已超过 100 MB 上限）；
2. 比赛数据通常不允许再次公开分发。

复现时请自行把数据放到脚本读取的位置，各任务 README 里写明了预期的目录结构。

---

## 六、几点说明

- 四个任务的结论都写在各自 README 的「实验结果」一节，包括**被证伪的方向**（例如任务四的形态学膨胀、架构混合，任务三的视盘否决、多尺度推理）。这些负结果与正结果一并保留，因为它们是判断方案上限的依据。
- 所有评估都使用留出/交叉验证口径，并对「在验证集上挑参数」这一动作本身做了留出检验（在 4 折上挑、第 5 折上评），以避免过拟合验证集。
- 排行榜第 2 名的 1.00000（任务四）与 4.52862（任务三）对应公开数据集的标注泄漏，本仓库未采用该路线。
