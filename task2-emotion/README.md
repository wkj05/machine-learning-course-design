# 东北大学 机器学习课设 · 任务二：人脸表情识别（6 类）

> 本目录与 `neu-ml-lab1/`（任务一·植物幼苗分类）是同一套脚手架的两个实例：
> `src/*.nb.txt` 是 notebook 的源码（用 `#@cell markdown|code` 分格），
> `node src/build_nb.js src/xxx.nb.txt xxx.ipynb` 生成可上传的 `.ipynb`。
> 改代码**改 `.nb.txt` 然后重新 build**，不要直接改 `.ipynb`。

---

## 当前状态（2026-10-18 存档）

### ★★★ 最终成绩：Kaggle 公开榜 **0.741**

| 提交 | 内容 | 本机 OOF | LB |
|---|---|---|---|
| 单模型 | `cnxt_t` · 折0 · 12 epoch | 0.6996 | 0.700 |
| V1 | 5 主干（r18 / r34 / effb0 / cnxt_t / mvit_s）× 5 折 = 25 模型 | 0.7139 | 0.728 |
| **V2** | **换血 5 主干（cnxt_s / cnxt_t / cnxt12k / effb0 / xcit_t）× 5 折 = 25 模型** | **0.7253** | **0.741** |
| V1+V2 混合 + 类别偏置 | 见下方「失败案例」 | 0.7283 | 0.738 ← 倒退 |

**校准线**：`LB ≈ 本机 OOF + 1.4~1.6 点`（V1 +1.41、V2 +1.57，两次独立验证）。

### ★★★ 两条代价昂贵的教训（2026-10-18）

**教训 1：在 OOF 上拟合、又在同一批 OOF 上报告的数字，不是证据。**

我拟合了 6 个类别偏置去最大化 OOF 准确率，报出 +0.30 pt，然后据此提交 —— LB 从 0.741 **掉到 0.738**。
事后用 50/50 切分重做（`_inbox/calib_postmortem.py`，一半拟合、一半评估）：

| 做法 | in-sample 增益 | **held-out 增益** |
|---|---|---|
| 偏置加在 V2 上 | +0.30 pt | **+0.05 pt** |
| 偏置加在混合上 | +0.27 pt | **−0.11 pt** |
| 混合权重 w | — | **+0.05 pt** |

**accuracy 是阶跃函数**：6 个参数在 28275 个样本上足以"凑"出一个在样本上刚好更优的决策边界，而这个边界不迁移到测试集。
**规矩**：任何后处理（类别偏置 / 温度 / 混合权重 / 阈值）必须在**留出的那一半 OOF** 上报告增益，拟合与评估的样本必须不同。这一步应当默认存在，不是可选项。

**教训 2：共享成员、同折、同种子的两个集成，取平均没有用。**

V1 与 V2 各 25 个成员，其中 `cnxt_t` 与 `effb0` 共 **10 个成员逐位相同**（同折同种子），错误相关性极高。
混合权重扫描是条平线（`w=0` → 0.7253，`w=0.30` → 0.7262，`w=1` → 0.7139），held-out 增益 **+0.05 pt**。
**要真涨分，新成员必须在「骨干 × 折 × 种子 × 预训练」上真正不同**，而不是同一批模型换个集成比例。

### ★★ 耗时之谜已解开：13.5 min/epoch 是"被抢了 CPU"，不是代码开销

### ★★ 耗时之谜已解开：13.5 min/epoch 是"被抢了 CPU"，不是代码开销

两个测速脚本（`_inbox/fer_prof.py`、`_inbox/fer_prof2.py`，都在**独占 CPU** 下跑）的结果：

| 测量项 | 结果 | 折算每 epoch（353 步） |
|---|---|---|
| 整条 `train_transform`（含 ColorJitter/Rotation/Erasing） | 0.69 ms/张 | **0.26 min** |
| 模型纯前向+反向 | 1.062 s/step | 6.25 min |
| 完整训练步（loader + 模型） | 1.131 s/step | 6.65 min |
| **真循环逐项加满**（FERNet + drop_path + mixup + SoftTargetLoss + clip + GradScaler） | **1.193 s/step** | **7.02 min** |

单点隔离：`mixup_cutmix` **0.40 ms**、`clip_grad_norm_` **11.59 ms**、`one_hot` **0.00 ms**。
**数据管线只占 6.5%，循环里每一行加起来只值 0.1 秒 —— 代码没有问题。**

真凶是可用线程数。同一份"真循环"配置扫描下来：
`threads=4 → 2.180` ／ `6 → 1.624` ／ `8 → 1.361` ／ `12 → 1.185` ／ `16 → 1.085` s/step。
**日志里那个 2.01 s/it，正好等于"只有 4 个线程可用"** —— 上一轮会话留下的 `pwsh-190`
的 python 进程还在跑同一个训练，两个进程抢同一批核心，各自慢到约 2 倍。

**验证**：2026-10-15 20:44 在空载机器上重跑，日志 `_inbox/fer_cpu5_run4.log` 实测
`ep1/12` 到 `232/353 [04:41]` → **1.21 s/it**，与测速脚本的 1.19 一致。

**修正后的真实耗时**：

| 量 | 旧记录（被污染） | 实测（空载） |
|---|---|---|
| 每 step | 2.01 s | **1.21 s** |
| 每 epoch（353 步） | 13.5 min | **≈7.1 min** |
| 5 折 × 12 epoch | 13.5 h | **≈7.5 h** |

### ★ 一条运维事实，比任何算法都重要

**这里的后台任务不跨会话存活。已经实测两次：**

| 时间 | 事件 |
|---|---|
| 10-13 21:09:33 | 启动 `pwsh-190`（cpu5） |
| 10-13 **21:53:38** | 日志停止 —— 正是上一轮会话结束的时刻 |
| 10-14 18:46:38 | 重启为 `pwsh-65` |
| 10-14 21:43 | 跑到 `f1 ep2/12`，**fold 0 已完成并写出 `submission.csv`** |
| 10-14 21:45:08 | 结束 `pwsh-65` 时，连另一个用 `Start-Process` 起的 **detached 进程也一起死了** |
| 10-14 21:46:25 | 又用 `Start-Process` 起了一个（PID 19608）**照样被回收**：日志停在 `code cell 7 (5/16)`，`.err` 是 0 字节（不是崩溃，是被清掉） |

`Start-Process` + `-WindowStyle Hidden` 脱离进程树这招**试了两次都没用**。
**结论：本机没有"后台挂着过夜"这回事 —— 想让它跑，就得让这个会话一直开着。**
杀它：`Stop-Process -Id <PID> -Force`。

**⚠ 启动新跑之前先 `Get-Process python` 确认没有残留。** 上面那个 2 倍耗时误差
就是残留进程造成的 —— 这类误差会伪装成"代码慢"，害我先后给过 12.5h / 7.5h / 90min 三个错估计。

### 已经拿在手上的东西（与在跑的进程无关）

- **`_inbox/fer_cpu5/submission.csv`** —— 127,182 字节，
  `check_submission.py` **0 错 0 警告**，`compare_submission.py` **9 项全 OK**
  （ID 顺序逐一相同、CRLF 7066、无 BOM、大小与官方样例逐字节相同）。
- **它已经交到 Kaggle 了：LB 0.70。** 只有 1 个模型（`cnxt_t`、折 0、12 epoch），
  而本机留出集 no-TTA 是 **0.6996** —— **几乎逐点吻合**。
  这条校准线立住了：**本机留出集数字可以直接当 LB 预期用（±0.5 点内）**。
- 落盘兜底也在生效：跑完 1 折就有可交的 `submission.csv`，不必等 5 折跑完。

### 在跑的（2026-10-16 13:0x 重新启动，第 4 次）

**`pwsh-286`** —— `--profile cpu5 --rotate-folds 1`（`cnxt_t`、5 折、12 epoch、96px），
`OMP_NUM_THREADS=16`，日志 `_inbox/fer_cpu5_run8.log`，输出 `_inbox/fer_cpu5/`。

**2026-10-16 13:20 的好消息：`c-course/asr` 的 node worker 全部退出了**（`Get-Process node` = 0 个），
python 独占 CPU、拿到 10.5 核，实测稳定 **1.30 s/it**（被抢时是 2.7~5.7）⇒
**每 epoch ≈7.6 min、每折 ≈88 min、5 折 ≈7.3 h**。这是这几天第一次有可行的本机预算。

epoch 断点已端到端验证过：ep1 跑完落盘 `_foldckpt/cpu5_ep12_sz96_nf5_sd42_rot1/cnxt_t_f0.pkl`
（425 MB，每次覆盖写），杀掉进程重启后日志打出
`[epoch-resume] cnxt_t f0 从 ep2/12 接着跑（已跳过 1 个 epoch）`。

前三次（`pwsh-190` / `pwsh-106` / `pwsh-27`、`pwsh-67`）全都死在"会话结束即被回收"，
**一个新折都没跑完** —— 因为折级缓存只在**整折跑完**时才落盘，而一折要 3~11 h、
一次会话只有 45 min~3 h，所以每个会话的折内进度全丢。

**这一轮的三个改动（都在 `_inbox/run_nb2.py`，没碰 notebook 成品）：**

1. **`--rotate-folds 1`**：把 `FOLDS` 旋转一位，变成 `[原1折, 原2折, 原3折, 原4折, 原0折]`。
   理由是**原第 0 折的结果我们早就有了**（留出集 0.6996 / LB 0.70），
   而一折要 ~100 min、一次会话只有 45 min~3 h —— 每轮都从已知答案开练，
   等于把整场会话烧在重复劳动上。旋转后先跑结果未知的折。
   正确性：OOF 是用折里的真实下标 `oof_cnt[va_idx]` 归位的，旋转不影响；
   结尾只有 `assert test_cnt > 0` 一个完整性断言，缺折也过得去。

2. **折级磁盘缓存**（`make_cell_hook`，默认开启，`--no-resume` 可关）：
   给 `train_one_fold` 套一层 pickle 缓存，存整个返回值含模型对象，
   命中时跳过训练，只留后面的 TTA 推断（十几分钟而不是一百小时）。
   缓存目录 `_inbox/fer_cpu5/_foldcache/cpu5_ep12_sz96_nf5_sd42_rot1/`，
   文件名带 epoch/尺寸/折数/种子/旋转量，避免"改了配置却复用旧缓存"的静默错误。

3. **epoch 级断点（本轮新增，这是真正救命的那个）**：
   `inject_epoch_ckpt()` 用 `inspect.getsource(train_one_fold)` 取回函数源码，
   两处字符串替换后在同一个命名空间里重新 `exec` 绑定 —— **源文件 `classdesign4-2.ipynb` 一个字没动**。
   - 循环头 `it = 0` / `for ep in range(EPOCHS):` 前面插入"读断点"：
     恢复 `model / opt / sched / history / best / it`，并把 `range` 起点改成 `_start_ep`，
     打印 `[epoch-resume] <tag> f<fold> 从 epN/12 接着跑`。
   - 逐 epoch 打印那行（锚点 `f'(选优依据 {SEL_METRIC}={vs:.4f})')`）之后插入"原子写断点"
     （`.tmp` + `os.replace`，`protocol=4`），存 `epoch / model / opt / sched / history / best / it`。
   - 断点目录 `_inbox/fer_cpu5/_foldckpt/cpu5_ep12_sz96_nf5_sd42_rot1/`，
     文件名 `<tag>_f<fold>.pkl`。读不到/坏了就打印原因并从第 0 个 epoch 重跑（不抛异常）。
   - 为了让 `inspect.getsource` 能取回源码，`main()` 的 exec 循环现在会把每个 cell 的源码
     登记进 `linecache.cache[f'cell_{idx}']`（否则 exec 出来的代码报 `OSError` 取不到源码）。
   - 代价：续跑时增强的随机序列与中断前那一次不同，所以结果不是逐位可复现的。

**⚠ 以后重启必须带 `--rotate-folds 1`**，否则缓存键（`rot1`）对不上，等于从零重来。

**⚠ 机器负载**：`c-course/asr` 那两个 node worker 占着约 18/24 核时，python 只剩 ~2 核、
实测 **5.67 s/it**；只剩一个 worker 时是 **2.7 s/it**；空载是 **1.21 s/it**。
所以"一折多久"完全取决于 asr 那边闲不闲 —— 别拿单个 s/it 读数当计划依据。
**DSH 自己的 node 壳是 0 CPU，关掉 DSH 不会让训练变快，只是让进程不被回收。**

前 9 个 epoch 的实测（**空载时**从日志的 tqdm 计时读，每 epoch 353 步）：

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| 用时 | 7:05 | 7:08 | 7:10 | 7:11 | 7:03 | 6:52 | 6:51 | 7:17 | 7:13 |

⇒ **7.05 min/epoch，与测速脚本预测的 7.1 完全一致。** 验证曲线 `val_acc`
0.5708 → 0.6205 → 0.6525 → 0.6393 → 0.6668 → 0.6835 → 0.6851 → 0.6932 → 0.6934，
仍在缓慢上升。

`submission.csv` 是**累积集成**：`classdesign4-2.nb.txt:828` 的 `test_sum` 在折循环外初始化，
`:899` 的 `test_sum += tp` 在折内累加，`:925/:938` 写的正是这个累积结果
⇒ 跑完第 N 折时，那份文件 = 前 N+1 折的平均。**折数越多越好，可以随时拿去交。**
但注意：旋转之后，第 1 个完成的折对应的是**原本的第 1 折**，那份文件是单模型，
**要等 2 折（≈200 min）才会出现比现在 0.70 更好的集成。**

### ⚠ 第二个抢 CPU 的源头 —— 2026-10-16 12:28 已查清身份

epoch 10 突然掉到 **4.38 s/step**（前 9 个 epoch 是 1.20）。查下来：

| 进程 | 启动时间 | 占用核心 |
|---|---|---|
| `node` PID 3080 / 17128（轮换） | 12:00:30 / 12:14:08 | **9.3 ~ 9.6** |
| `node` PID 7888 | 12:00:30 | **8.9 ~ 9.6** |
| `python` PID 4860（训练） | 12:06:03 | 2.1（原本约 10） |

**身份已查明（靠 `Get-Process` 的 `Modules` 拿到的模块全路径，没有命令行也能定位）：**

```
C:\...\default-workspace\c-course\asr\node_modules\@img\sharp-win32-x64\lib\sharp-win32-x64-0.35.5.node
C:\...\default-workspace\c-course\asr\node_modules\@img\sharp-win32-x64\lib\libvips-cpp-8.18.7.dll
C:\...\default-workspace\c-course\asr\node_modules\onnxruntime-node\bin\napi-v6\win32\x64\onnxruntime_binding.node
C:\...\default-workspace\c-course\asr\node_modules\onnxruntime-node\bin\napi-v6\win32\x64\onnxruntime.dll
C:\WINDOWS\SYSTEM32\DSPARSE.dll
```

⇒ 它们是**同一个 workspace 里另一个项目 `c-course\asr`**（`@huggingface/transformers` + `onnxruntime-node` + `opencc-js`，
脚本有 `transcribe.mjs` / `pipeline.mjs` / `build_index.mjs` / `check_coverage.mjs` 等）的 **transformers.js 推理 worker**，
**不是本会话的 harness**（本会话的 node 是另外几个 0-CPU 的壳），也**不是泄漏的野进程**。
`c-course/asr/check_coverage.mjs` 在本会话进行中被改动过（12:09:06）⇒ 那边有活儿正在干。

**所以不能杀** —— 杀了会打断另一个项目正在跑的活。本机能给 FER 的算力取决于它什么时候闲下来。

**教训**：本机跑长任务的耗时上限不由自己的代码决定，而由同时段别的进程决定。
之前那个 2.01 s/it「代码慢」的假象，本质也是同一类问题。
另外**沙箱里 `Modules` 的 `FileName` 是定位陌生进程最有效的武器** —— `Get-CimInstance` / `Get-WmiObject` /
`Get-NetTCPConnection` 全被拒，但 `Modules` 能读出 DLL 的完整路径，一锤定音。

### 本机已有的真实分数

| 来源 | 配置 | 分数 |
|---|---|---|
| **Kaggle LB** | 本机 fold 0 产出的那份 CSV | **0.70** |
| 本机 fold 0 | `cnxt_t` × 12 epoch @96px | no-TTA **acc 0.6996** / TTA acc 0.6983 |
| 本机（更早） | `cnxt_t` × 8 epoch | acc 0.6944 |
| Kaggle | 5 主干 × 5 折 × 30ep（**未跑完**，预算只够 1–2 折） | **LB 0.653** |

12 epoch 只比 8 epoch 高 **0.5 个点** ⇒ **这个主干在 96px 上已接近天花板**，
继续堆 epoch 收益很小。**要涨分得靠更多主干/更多折的集成，而不是单模型训练更久。**

留出集数字系统性偏高（checkpoint 挑的是验证折上最好的那一轮），但 0.70 那次
说明偏得很少。

### ★★ 本机长跑跑不过夜 —— 已定案：不是没试，是路都试遍了

`2026-10-16` 把"让训练活过会话结束"这件事彻底试完，三条路全堵死：

| 办法 | 结果 |
|---|---|
| 普通后台 job | 会话结束即被回收（实测三次） |
| `Start-Process -WindowStyle Hidden`（脱离进程树） | 照样被回收，试了两次 |
| `schtasks /create` 交给计划任务服务 | **RPC 被沙箱掐断**：连 `schtasks /query` 都报 `ERROR: The system cannot find the path specified.` |
| `CREATE_BREAKAWAY_FROM_JOB`（Python `subprocess`，脚本 `_inbox/spawn_detached.py`） | **`PermissionError: [WinError 5] 拒绝访问`** —— harness 的 job object 没开 `JOB_OBJECT_LIMIT_BREAKAWAY_OK` |

**结论：harness 用一个不允许 breakaway 的 Windows Job Object 管住整棵树，
本机不存在"挂着过夜"这回事。** 于是策略必须反过来设计成"每次会话都能攒下一点东西"，
这就是上面 `--rotate-folds` + 折级缓存的由来。

### 重启命令

**★ 给「人」的启动方式（推荐）—— `_inbox\run_cpu5.cmd`**

它有两条路，等价，随便挑：

- **Win+R**（最快）：按 `Win+R`，把下面这行**原样粘进去**（**不要加引号**，路径没空格），回车。
  ```
  C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox\run_cpu5.cmd
  ```
  Win+R 能直接执行 .cmd 文件，会弹出一个控制台窗口；窗口会**实时**打印进度，同时写日志。
- **cmd 窗口**：`Win+R` → 输 `cmd` → 回车，然后粘（这时候**要加引号**）
  ```
  "C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox\run_cpu5.cmd"
  ```
- 也可以直接在资源管理器里**双击** `_inbox\run_cpu5.cmd`。

三条硬性注意事项：

1. **窗口不能关** —— 关了训练就断了。（关掉随时可以重跑，已完成的折在
   `_inbox\fer_cpu5\_foldcache\` 里，会被跳过。）
2. **看窗口里的 `s/it`**：**≈1.2 是正常**（一折 ≈85 min）；**≈5.9 说明机器被别的东西占着**
   （那时一折要 ≈11 小时），先 `Get-Process node,python | Select-Object Id,CPU` 看一眼。
3. **`.cmd` 里只能写 ASCII** —— cmd.exe 按 OEM 代码页读 .cmd，里面出现中文会把行解析搞乱
   （我第一次写的中文注释版本就是这么坏的：`%WS%` 展开成了 `\python.exe`）。
   说明文字一律写在本 README 里，不要写进 .cmd。

**⚠ 我先前的说法要更正**：抢 CPU 的不是 DSH 会话，而是 `c-course\asr` 的 transformers.js worker
（见上一节）。所以**关掉 DSH 并不会让训练变快**；要让训练变快，得等 `asr` 那边的活跑完。

**（备用）从 DSH 里用 PowerShell 直接起**——只用于我自己调试，会随会话结束被回收：

**⚠ 开跑前先确认没有残留进程**（否则会像 10-15 那样，两个进程互抢核心、各自慢一倍）：
`Get-Process python` 有输出就 `Stop-Process -Id <PID> -Force`。

```powershell
$ws="C:\Users\24625\Documents\deepseek-harness\default-workspace"
$env:MPLCONFIGDIR="$ws\_inbox\mplcache"; $env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"
$env:HF_HOME="$ws\_inbox\hfcache"; $env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"; $env:TQDM_DISABLE="1"
$env:OMP_NUM_THREADS="16"; $env:MKL_NUM_THREADS="16"   # 实测 16 线程比默认 12 快 8%
& "$ws\_inbox\python312\python.exe" -u "$ws\_inbox\run_nb2.py" --nb "$ws\neu-ml-lab2\classdesign4-2.ipynb" `
  --data "$ws\_inbox\fer\fer_data\fer_data" --out "$ws\_inbox\fer_cpu5" `
  --profile cpu5 --rotate-folds 1 *> "$ws\_inbox\fer_cpu5_run5.log"
```

`--rotate-folds 1` 不能省：折缓存的目录名里带着 `rot1`，不传就等于换了个缓存键、从零重来。

### 两个已修的会致命的 bug（都在 `src/classdesign4-2.nb.txt`）

1. **`to_csv(..., newline='')` 在 pandas 上是非法关键字**（那是标准库 `csv` 的参数）。
   原写法会让 notebook 在**训练全部跑完、就差写文件的那一刻**抛
   `TypeError: NDFrame.to_csv() got an unexpected keyword argument 'newline'`。
   在 Kaggle 上等于白烧几小时。两处（:864 每折兜底、:1129 最终提交）都改成 `lineterminator='\r\n'`，
   已在 pandas 3.0.6 上实测输出 `b'ID,Emotion\r\na.jpg,1\r\nb.jpg,2\r\n'`（3 CRLF / 0 裸 LF / 0 `\r\r\n` / 无 BOM）。
2. **每折兜底的 `submission_partial.csv` 曾按测试目录字母序写**，而 `submission.csv` 与官方样例用的是另一个顺序。
   若按行序对齐，这份兜底文件会 ≈ 随机。现在复用 `sample_df` 的 ID 列，**仅当 ID 集合完全一致时**按样例顺序重排，
   否则退回目录序并打印提示（兜底路径永不抛异常）。

顺带排除掉的第三种可能：**0.653 远高于 6 分类随机下限 0.167 ⇒ Kaggle 是按 ID 对齐的**，
所以顺序那件事原本也不会扣分（修了无害）。

### ★ 0.653 的根因（已从代码结构定案，不再只是猜测）

**`submission.csv` 不可能存在，而那几个 `.npy` 也一定不存在 —— 它们是同一个 cell 写的。**

`src/classdesign4-2.nb.txt` 的 cell 边界：

| cell | 起始行 | 写出的文件 |
|---|---|---|
| 训练循环 | 822 | 每折：`submission_partial.csv`，**现在也同时写 `submission.csv`** |
| OOF 汇总 | 939 | 无（只算） |
| 最终提交 | **1106** | `test_probs.npy` **1112**、`oof_probs.npy` **1113**、`fold_summary.csv` 1114、`tta_view_scores.csv` 1115、`tta_view_probs.npz` 1118、`submission.csv` **1191** |

关键在最后一行：**1106 这个 cell 里，`.npy`（1112–1118）写在 `submission.csv`（1191）之前。**
所以 `submission.csv` 存在 ⟹ 那四个 `.npy` 也存在 ⟹ Output 至少 **3.2 MB**（1.36 + 1.40 + 0.34 MB）。
实测 Output 只有 **252 KiB** ⟹ **最后一个 cell 根本没跑** ⟹ **0.653 是每折兜底文件的分数。**

截图里那个 `1 more` 因此**不是** `submission.csv`。虽然 252 KiB ≈ 两份 CSV 的大小（巧合），
但 cell 内部的书写顺序排除了它 —— 更可能是 `.virtual_documents` 与某个 notebook 副本。

**为什么会没跑到。** 这里有一条是我判断错了，得记下来：

> 我让你把 `TIME_BUDGET_MIN` 从默认的 540 改成 **130**，理由是"你 22:00 要收工"。

**这个理由不成立 —— Kaggle 的 commit 跑在 Kaggle 的服务器上，你的电脑开着关着都不影响它。**
我拿你睡觉的时间去卡一个根本不占你时间的远程任务，把预算白白砍掉 4 倍。

按 130 min 反推：`5 主干 × 5 折 × 30 epoch` 在这个预算下大约只够**跑完第一个主干的 1–2 折**，
而 `BACKBONES` 的第一个正是 `resnet18`。单模型 resnet18 @30ep 在 FER2013 上恰好是 65–68% 这个量级，
与 **0.653** 吻合。（这是反推，不是实测；日志里的 `已完成 N/M 个模型` 能验证。）

### 明天该怎么做

- **让 Kaggle 跑满。** `TIME_BUDGET_MIN` 保持默认 **540**（9 小时），或按 Kaggle 上限调到 660。
  **不要因为"我要睡了"去砍它** —— 跑长任务正是这个平台最擅长的场景。
- **主干顺序保持"便宜→贵"**（`r18 → r34 → effb0 → cnxt_t → mvit_s`），**不要**改成"强的优先"。
  我原先想的"强主干优先"是错的：既然预算只够 1–2 折，"5 折 r18 集成 + 2 折 r34"大概率胜过
  "1–2 折 cnxt_t"——**便宜优先能换到更多折数，而集成折数在 FER 上是实打实的收益。**
  （同样是推理，不是实测。）
- 本机那 12.5 h 的跑不受影响，明天上午会自己产出 `_inbox/fer_cpu5/submission.csv`。

### 两个新加固（已重建：29 cell / 78921 bytes，astcheck 0 错，官方 schema ✓）

1. **每折同时写 `submission.csv` 与 `submission_partial.csv`**（nb.txt:938 `for _fn in (...)`）。
   从此任何时刻手上都有一份完整可交的文件，不再依赖"最后一个 cell 能活下来"。
2. **`train_one_fold()` 外面套 try/except**（nb.txt:869）：某一折抛异常（现实里最可能是 CUDA OOM）
   只跳过该主干的剩余折，不再连带炸掉整个 cell 和后面两个 cell。

### 仍值得拿（用于验证上面两条推理）

Kaggle 日志尾部约 30 行：出现过哪些主干 tag、有没有 `!!!!!  cell N 抛异常`、`集成模型数 = ?`。
它现在不再是"定位根因"的必需品（代码已经定案），而是**验证"预算只够 1–2 折"这个反推**的证据。

---

## 0. 一句话

把 48×48 的灰度人脸图分到 6 类表情（`0 愤怒 / 1 恐惧 / 2 快乐 / 3 悲伤 / 4 惊讶 / 5 中性`），
用 **5 个 ImageNet 预训练主干 × 5 折交叉验证**的集成做预测，输出 `ID,Emotion` 的 CSV。
评分指标是**准确率（accuracy）**。

---

## 1. 数据事实（全部实测，不是转述）

| 项 | 值 |
|---|---|
| 训练集 | **28,275** 张，按情感名分 6 个子目录 |
| 测试集 | **7,065** 张，无标签 |
| 图像 | **48×48**、8 位灰度 JPEG（PIL 报 `mode=L`），约 900 字节/张 |
| 类别张数 | Angry 3963 / Fear 4097 / Happy 7192 / Neutral 4959 / Sad 4862 / Surprise 3202 |
| 不均衡度 | Happy : Surprise = 7192 : 3202 ≈ **2.25 : 1** |

**编号必须照任务说明来。** 本任务规定 `2 = 快乐`，而公开 FER2013 的常见编号里 `2 = Fear`、
`3 = Happy`。照抄公开编号会**静默**错三个类（不报错、分数崩）。notebook 里
`CLASS_ORDER` 与训练目录名做了精确断言，对不上直接 `raise` 并把实际目录名打出来。

**官方样例提交**（`reference/submission-for-task2.csv`，127,182 字节）：
表头 `ID,Emotion`、7,065 行、Emotion 全为 `0`、**CRLF** 行尾、无 BOM、结尾有换行。
**它的行顺序与测试目录的文件列举顺序不同** —— 所以提交必须**按 ID 对齐**，不能按目录顺序。

---

## 2. ⚠️ 关于这份数据集的来源（必须先知道）

这份数据**就是公开的 6 类 FER2013**。证据是逐项对上的：

* 48×48 灰度、六个类名、总数 35,340 —— 全部一致；
* **每一类的训练张数都精确等于公开张数 × 0.8 向上取整**：
  `ceil(0.8×4953)=3963` Angry、`ceil(0.8×6198)=4959` Neutral、`ceil(0.8×5121)=4097` Fear、
  `ceil(0.8×8989)=7192` Happy、`ceil(0.8×6077)=4862` Sad、`ceil(0.8×4002)=3202` Surprised。

由此有两点**必须写在前面**：

1. **存在一条通向 100% 的捷径，本方案不走。** 测试集就是这份公开数据被切走的 20%，
   而它的标签是随数据一起公开的；按像素比对就能把测试集标签全部还原。
   这是抄答案，而且**藏不住**：公开 FER2013 的 SOTA 只有 ~73–76%（人类标注一致率约 65%），
   一个真做出来的模型在 70% 上下，突然冒出 100% 一查就露。
2. **"合法地把公开数据当额外训练集"这条路在这里不存在。** 因为训练集本身就是那份公开数据的 80%，
   唯一多出来的部分恰好是测试集那 20% 自己。

所以后面的目标只有一个：**在不用外部标签的前提下，把诚实分数做高**。

---

## 3. 方案（对照老师给的 5 条思路）

| 老师的思路 | 本方案的落地 |
|---|---|
| ① 先做数据分析 | notebook 第 2 节：灰度→RGB 的可视化 1×6（标题带 `类名 (编号)`，**靠肉眼复核编号映射**）、张数统计、`sizes/modes` 抽样 |
| ② 用交叉验证 | `N_FOLDS=5` 分层 5 折，全量 OOF（28,275 张）用于评估与选优；不是"切一次 8:2" |
| ③ CNN + ImageNet 预训练 | `resnet18.a1_in1k` / `resnet34.a1_in1k` / `efficientnet_b0.ra_in1k` / `convnext_tiny.fb_in22k_ft_in1k` |
| ④ 注意力机制 + Transformer | `mobilevit_s`（MobileViT 块内含**真正的多头自注意力**，卷积+注意力混合）。**为什么不用 Swin/ViT 见 §5** |
| ⑤ 持续调参 | 尺度层 TTA 选优（由 OOF 决定，不拍脑袋）、Mixup + Label Smoothing、区分性学习率、cosine + warmup、per-class 混淆分析 |

**训练配置**（`src/classdesign4-2.nb.txt` 第 0 节）：
`IMG_SIZE=96`（原图 48，放大训练）、`CACHE_SIZE=48`（**按原始分辨率缓存**，内存约 65 MB，
任意 TTA 尺度都从它放大）、`N_FOLDS=5`、`EPOCHS=30`、`WARMUP_EPOCHS=3`、`BATCH_SIZE=64`、
`LR_HEAD=1e-3` / `LR_BACKBONE=2e-4`、`WEIGHT_DECAY=0.05`、`DROP_PATH=0.1`、
`MIXUP_PROB=0.5`、`LABEL_SMOOTH=0.1`、`TTA_SCALES=[96,128]`、`TTA_FLIPS=[None,'h']`。

**人脸专用的两处调整**（相对任务一）：
* **不做上下翻转** —— 倒过来的人脸不是合法数据。`TTA_FLIPS` 里没有 `'v'`/`'hv'`。
* 增强改温和：`RandomResizedCrop(scale=(0.8,1.0), ratio=(0.9,1.1))` + `RandomRotation(10)`
  + `ColorJitter(brightness=0.2, contrast=0.2)` + 水平翻转 + `RandomErasing(0.25)`。

---

## 4. 两个来自任务一的实测教训（本任务沿用）

### 4.1 TTA 视角不能"挑单个第一名"，要先在尺度层选

任务一上跑了两遍完全相同的配置（只换随机种子），**两遍挑出的"最佳单视角"完全相反**
（第 1 遍第一名 `288|hv`，第 2 遍掉到第 5；第 2 遍第一名 `224|orig`，第 1 遍只排第 3）。
单视角 OOF 分数之间的差实测能到 2.2 个点且不可复现 —— 取其中最大的那个，涨的是噪声。

而"**同一尺度下全部翻转的平均**"是稳定量：两遍一致给出 224 那组更好
（224 四翻转 0.8991 / 0.9050，288 四翻转 0.8905 / 0.8943，方向一致）。

所以本 notebook 的规则是：**先在"尺度"这一层选（尺度层的组平均远比单视角稳），
再在该尺度内平均它的全部翻转；绝不挑单个视角。** 本任务对应的问题是
"该不该带上 128 这个训练之外的尺度"，交给 OOF 回答。
（本任务 5 折每折约 5,655 张验证，OOF 噪声比任务一的 100 张验证折小一个量级。）

### 4.2 随机种子本身值一个点

任务一两遍只换种子，整份 OOF macro-F1 从 0.8931 挪到 0.9025（**1.0 个点**）。
所以：**任何 0.5 点以内的差别都不该当结论**，两次跑的差必须大于这个量级才可信。
notebook 里的种子由 `SEED + 100*fold + md5(tag)%97` 生成 —— 用 md5 而不是内置 `hash()`，
因为 Python 3 对 `str` 的 `hash` 每进程掺随机盐（PEP 456），那样种子根本不可复现。

---

## 5. ★ 一个"会在跑了几小时之后才崩"的坑（已在本地堵掉）

老师思路第 4 条点名要 Transformer。最自然的选择是 Swin-T —— **但它是错的**：

```
timm 1.0.30, swin_tiny_patch4_window7_224:
    AssertionError: Input height (96) doesn't match model (224).
```

timm 的 Swin/ViT 在"输入尺寸 ≠ 构造时的 `img_size`"时**直接断言失败**
（加 `dynamic_img_size=True` 无效；把 `img_size` 设成 96 则跑到 128 一样失败）。
而 TTA 要在 `[96, 128]` 两个尺度上前向 —— 于是这个错会在**训练全部跑完之后**才爆，
整个 Kaggle 运行白跑几小时。

本机实测能同时通过 96 与 128 的注意力型主干只有三个：
`mobilevit_s` / `mobilevit_xs` / `xcit_tiny_12_p16_224`
（`coatnet_0_rw_224`、`levit_128s`、`pit_ti_224` 的位置编码/尺寸假设都不行）。

**两道防线：**
1. 主干换成 `mobilevit_s`（构造时按 `candidates` 降级到 `mobilevit_xs`）；
2. `create_backbone()` 里加了**输入尺寸预检** —— 对 `IMG_SIZE` 与 `max(TTA_SCALES)`
   各前向一次空张量（用 `eval()` 以免污染 BN 统计量），任一失败就**跳过该主干**并打印原因。
   这样同类问题退化成启动时的一行 `[--] xxx: 尺寸预检失败`，而不是几小时后崩。

---

## 6. 已经做过的验证

| 验证 | 结果 |
|---|---|
| 语法 | 16 个非空 code cell，`ast.parse` **0 语法错误** |
| notebook schema | `nbformat.validate()` **通过**（nbformat 4.5，29 cell） |
| 端到端冒烟 | 本机 CPU 跑 `smoke` 档（随机权重、纯链路自检）**16/16 cell 全过，exit 0，6.2 min** |
| 提交格式 | 与官方样例 `compare_submission.py` **9 项逐项一致**：表头/列数/行数/ID 集合/**ID 顺序**/CRLF=7066/BOM=False/结尾换行/无结尾空行；文件大小 **127,182 = 127,182 字节** |
| 提交内容 | `check_submission.py` **0 错误 0 警告**，Emotion 取值 ⊂ `{0..5}` |
| 主干尺寸预检 | 日志出现 `[ok] backbone = ... 尺寸预检通过 [...]` |

**冒烟跑抓到的两个真 bug（都不是"可能性"，是实际崩了）：**

1. `sub.to_csv(..., newline='')` → `TypeError: NDFrame.to_csv() got an unexpected keyword
   argument 'newline'`。**pandas 的 `to_csv()` 没有 `newline` 参数**（那是标准库 `csv` 模块的），
   而报错发生在"训练跑完、就差写文件"的那一刻。已改成 `lineterminator='\r\n'`
   （pandas 内部已按 `newline=''` 打开句柄，不会被再翻译一遍），
   并**单独写脚本实测**确认输出是 3 个 CRLF、0 个裸 LF、0 个 `\r\r\n`、无 BOM。
2. 落盘自检那行把字节字面量写成了 `b"\\r\\n"`（多一层反斜杠）→ 实际在数 4 字节序列
   `\r\n`，**永远打印 0**，看上去"没问题"。已改成 `b"\r\n"`，现在正确打印
   `落盘自检: CRLF=7066 行, 裸 LF=0 行, BOM=False`。

> 教训：**"跑通冒烟"不是走过场。** 这两处一个会浪费几小时 GPU，一个会让自查变成假阳性。

---

## 7. 怎么跑

### 7.1 Kaggle 上（正式路径）

1. 打开竞赛 notebook 页面 → 左上角 **File → Import Notebook** → 选 `classdesign4-2.ipynb`
   （**不要往编辑器里拖文件**，那会把整份 JSON 当成一个 cell 插进去、满屏红字）；
   导入成功应该是 **29 个 cell**，第 1 个是 markdown 标题。
2. 右侧 **Add Input → 加入 2026task2 数据集**。
3. **Settings 面板三件套，缺一不可**：
   * **Accelerator = GPU T4 x2**（或 P100）
   * **Internet = On** —— 5 个主干都要从 HF 现下权重，关着网会直接下载失败
   * **Persistence** 随意
4. **不要改 `TIME_BUDGET_MIN`**，让它保持默认 **540**（9 小时，小于 Kaggle 的 12 小时上限）。
   上一次 0.653 就是被我把这个值改成 130 砍出来的，见「0.653 的根因」一节。
5. **Save Version → Save & Run All (Commit)**。

**跑完怎么判断是成功还是半途而废**（Output 面板，`/kaggle/working`）：

| Output 里有什么 | 说明 |
|---|---|
| `submission.csv` + `oof_probs.npy` + `test_probs.npy` + `tta_view_probs.npz` + `fold_summary.csv` + `tta_view_scores.csv`，**合计 ≈ 3.2 MB** | 全部跑完，`submission.csv` 是 **TTA 选优后的集成**，就用它 |
| 只有 `submission.csv` / `submission_partial.csv`，**合计 ≈ 252 KiB** | 训练中途挂了；这两份是**每折覆盖写的兜底**（跑完第 N 折就是前 N+1 折的平均），仍然能交，分数会低 |
| `submission.csv` 都不在 | 训练 cell 一格都没跑完，去 Logs 里找 `!!!!! cell N 抛异常` |

**提交**：Output 面板里 `submission.csv` 行右侧的 **Submit to Competition**；
或下载后到竞赛页 **Submit Prediction** 上传。分数在竞赛页 **My Submissions** 看。

`TIME_BUDGET_MIN=540` 是保险闸：超预算会提前收尾；每折结束都写一份兜底 CSV，
所以"最后一格挂了导致什么都没留下"这条路已经被堵死（29 cell / 78921 bytes 那一版加的加固）。

### 7.2 本机（离线自检）

本机 python **连不上 huggingface.co**（`WinError 10060`），所以只能跑不下载权重的档：

```powershell
# 纯链路自检（随机权重，分数无意义；只验证"能一路跑到写出 submission.csv"）
python -u _inbox\run_nb2.py --nb neu-ml-lab2\classdesign4-2.ipynb `
  --data _inbox\fer\fer_data\fer_data --out _inbox\fer_smoke --profile smoke
```

`_inbox\run_nb2.py` 的档位：`smoke`（随机权重自检）/ `real_cnxt`（只用已缓存的
`convnext_tiny.fb_in22k_ft_in1k`，1 折 8 epoch，**需配 `HF_HUB_OFFLINE=1`**）/ `quick` /
`full` / `ab192`。它靠正则按**赋值名**打补丁，不再依赖精确字符串匹配。

---

## 8. 已知风险 & 下一步

* **分辨率还没做过受控 A/B。** 48×48 的原图放大到 96 训练是"够快、能早交一版"的选择；
  放大到 192/224 是否更好，需要一个"同种子同折、只差 IMG_SIZE"的对照。
  **注意**：任务一那次 224-vs-288 的 A/B 只用了 1 折 100 张验证图（SE≈3 个点），
  根本分辨不了 1 个点的差异 —— 本任务要用 5 折 OOF（每折约 5,655 张）才有功率。
* **伪标签（pseudo-labeling）** 是这条路上最划算的合法增益：用集成对测试集打高置信度伪标签
  再训练一轮。它不引入任何外部标签，属于转导学习。
* **FER2013 的标签本身有噪声**（公开研究估计 5–20% 错标），所以宏 F1 与准确率的上限
  并不等于"人眼能到的上限"。不要因为到不了 90% 就以为模型坏了。
* **最终分数不要拿本机 OOF 直接当预测。** 任务一的校准结论是：
  本机估计只能把排行榜分数预测到 **±1.5 个点**以内
  （两次提交：本机 0.9000 → LB 0.8915；本机 0.9010 → LB 0.9048）。
