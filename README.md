# 工业蒸汽量预测分析平台

基于 Flask + scikit-learn 的工业蒸汽量预测分析平台，提供从数据探索、特征工程、多模型训练对比到 Web 部署的完整机器学习流程。

## 项目背景

工业锅炉的蒸汽产量是衡量生产效率的关键指标。本项目基于火力发电厂工业传感器采集的 38 维特征数据（V0 ~ V37），通过机器学习方法对蒸汽量（target）进行回归预测，旨在为企业生产调度和能效优化提供数据支撑。

## 技术架构

```
前端（HTML/CSS/JS + Chart.js）  ←→  Flask RESTful API  ←→  scikit-learn 模型层
         （SPA 页面）                 （app.py）              （9种回归模型）
```

- **前端**：单页应用（SPA），Chart.js 可视化图表
- **后端**：Python Flask，提供 RESTful API 接口，ThreadPoolExecutor 并行训练
- **数据缓存**：内存级 DATA_CACHE / MODEL_CACHE 避免重复加载

## 项目结构

```
.
├── web_app/
│   ├── app.py              # Flask 后端（14个API接口，9种模型）
│   └── templates/
│       └── index.html      # 前端可视化页面
├── zhengqi_train.txt       # 训练集 2888条 × 39列（含 target）
├── zhengqi_test.txt        # 测试集 1925条 × 38列
└── README.md
```

## 数据集

| 数据集 | 样本数 | 特征数 | 目标列 | 格式 |
|--------|--------|--------|--------|------|
| zhengqi_train.txt | 2888 | 38 (V0~V37) | target | Tab分隔 |
| zhengqi_test.txt | 1925 | 38 (V0~V37) | 无 | Tab分隔 |

## 功能模块

### 第一章：数据探索分析

- **数据概览**：维度、字段类型、前 N 行数据预览
- **描述性统计**：均值、标准差、分位数、偏度、峰度
- **数据分布分析**：训练集与测试集的直方图、箱线图对比，检测分布偏移
- **缺失值分析**：缺失值统计与热力图可视化
- **相关系数矩阵**：特征间及特征与 target 的相关性热力图

### 第二章：特征工程

#### 2.1 异常值处理
- 基于 Ridge 回归残差的 Z-score 检测（`|z| > 3` 标记异常）
- 结合箱线图手动阈值过滤（如 `V9 > -7.5`）

#### 2.2 数据归一化
- **MinMaxScaler**：将特征映射到 `[0, 1]` 区间
- **StandardScaler**：标准化至均值 0、标准差 1
- 归一化公式：`X_norm = (X - X_min) / (X_max - X_min)`

#### 2.3 PCA 降维
- 主成分分析（PCA）：将 38 维特征降至 16 维，保留 95% 方差信息
- 有效缓解维度灾难，降低模型复杂度的同时保留绝大部分有效信息

#### 2.4 特征交叉构造
定义四类交叉操作对特征进行两两组合：

| 操作 | 公式 | 说明 |
|------|------|------|
| 加法交叉 | `x + y` | 线性组合 |
| 减法交叉 | `x - y` | 差异特征 |
| 除法交叉 | `x / (y + ε)` | 比例特征（ε=1e-5防除零） |
| 乘法交叉 | `x × y` | 非线性交互 |

对 38 个特征两两组合 C(38,2) = 703 对，每对生成 4 个交叉特征，共 2812 个新特征。配合 PCA 降维和特征选择，特征优化后验证 MSE 从 **0.231 降至 0.104**，精度提升 **54.8%**。

#### 2.5 低相关性特征筛选
- 基于相关系数阈值自动过滤低相关特征

### 第三章：模型训练

#### 3.1 数据划分
- `train_test_split` 按 8:2 划分训练集/验证集，`random_state=42` 保证可复现

#### 3.2 模型列表（9种）

| 模型 | 类型 | 关键参数 |
|------|------|----------|
| LinearRegression | 线性模型 | — |
| Ridge | 线性模型（L2正则） | α=1.0 |
| SGDRegressor | 线性模型（随机优化） | max_iter=1000 |
| KNeighborsRegressor | 基于距离 | n_neighbors=5 |
| DecisionTreeRegressor | 树模型 | max_depth=10 |
| RandomForestRegressor | Bagging 集成 | n_estimators=100 |
| GradientBoostingRegressor | Boosting 集成 | n_estimators=100, lr=0.1 |
| ExtraTreesRegressor | Bagging 集成 | n_estimators=100 |
| SVR | 核方法 | kernel='rbf', C=1.0 |

#### 3.3 评价指标
- **MSE**（均方误差）
- **MAE**（平均绝对误差）
- **RMSE**（均方根误差）
- **R²**（决定系数）

#### 3.4 实验结论
- **Boosting 类模型（GradientBoosting/LightGBM）** 擅长拟合自变量与蒸汽量之间隐藏的非线性关联，预测误差最小
- **线性回归** 表现稳定，训练-测试差异小，泛化能力强
- **随机森林** 通过 Bagging 降低方差，但缺少残差迭代修正机制
- **KNN** 对数据分布敏感，在该数据集泛化能力最差

### 第四章：模型验证

#### 4.1 K-Fold 交叉验证
- 5折交叉验证，`shuffle=True` 确保每折数据分布均匀
- 交叉验证能有效评估泛化能力，减少单次划分的偶然性

#### 4.2 模型对比分析
- 多模型指标横向对比（RMSE / R²）
- 预测值 vs 实际值散点图
- 自动选择最佳模型

#### 4.3 学习曲线
- 观察训练集/验证集性能随样本数量的变化趋势
- 判断过拟合/欠拟合状态

#### 4.4 特征重要性
- 基于随机森林的 `feature_importances_` 排名 Top 20 特征

### 第五章：结果预测

- **测试集批量预测**：一键预测全部测试样本
- **单样本预测**：手动输入 38 维特征值进行预测
- **自动最优模型选择**：基于验证集 RMSE 自动选用最优模型

### 第六章：Web 应用平台

#### 6.1 核心 API

| 路由 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 返回前端页面 |
| `/api/load_data` | POST | 加载数据文件 |
| `/api/data_stats` | POST | 描述性统计 |
| `/api/data_distribution` | POST | 直方图/箱线图数据 |
| `/api/correlation_matrix` | POST | 相关系数矩阵 |
| `/api/missing_heatmap` | POST | 缺失值热力图 |
| `/api/feature_engineering` | POST | 特征工程流水线 |
| `/api/train_models` | POST | 多模型并行训练 |
| `/api/cross_validation` | POST | K折交叉验证 |
| `/api/model_comparison` | POST | 模型对比数据 |
| `/api/prediction_vs_actual` | POST | 预测vs实际对比 |
| `/api/feature_importance` | POST | 特征重要性 |
| `/api/predict` | POST | 在线预测 |
| `/api/available_models` | GET | 可用模型列表 |

#### 6.2 设计亮点
- **内存缓存**（DATA_CACHE / MODEL_CACHE）：避免重复加载数据与训练模型
- **线程池并行训练**（ThreadPoolExecutor）：多模型训练从 ~90s 优化至 ~25s
- **前后端解耦**：RESTful API 设计，前端可灵活替换

## 快速开始

### 1. 安装依赖

```bash
pip install flask numpy pandas scikit-learn
```

### 2. 启动应用

```bash
cd web_app
python app.py
```

启动后浏览器自动打开 `http://127.0.0.1:5000`

### 3. 使用流程

```
加载数据 → 特征工程 → 模型训练 → 模型验证 → 结果预测
```

---

## 关键问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| PCA 降维后出现 NaN 值 | 近零方差特征的数值不稳定性 | `fillna(0)` 填充所有 NaN |
| 模型训练时间过长 | Flask 单线程串行训练 | ThreadPoolExecutor 并行训练 |
| 特征交叉后维度爆炸 | C(38,2)×4 = 2812 新特征 | PCA 降维 + 特征重要性筛选 |
| 模型过拟合 | 训练-验证指标差距大 | 正则化、早停、降低复杂度 |
| 模型融合过拟合风险 | 直接在训练集做 Stacking | K 折交叉验证生成 out-of-fold 预测 |

## 改进方向

1. 引入深度学习模型（全连接神经网络、Transformer）
2. AutoML 自动超参数搜索
3. 增加时序特征，考虑传感器时序依赖关系
4. 使用 SHAP 等方法增强模型可解释性
5. 支持模型在线更新和 A/B 测试

## 参考文献

[1] 周志华. 机器学习 [M]. 北京: 清华大学出版社, 2020.

[2] 何晓群. 应用多元统计分析 [M]. 北京: 中国人民大学出版社, 2021.

[3] Ke G, Meng Q, Finley T, et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree [C]// NIPS, 2017: 3146-3154.

[4] Breiman L. Random Forests [J]. Machine Learning, 2001, 45(1): 5-32.

[5] Pedregosa F, et al. Scikit-learn: Machine Learning in Python [J]. JMLR, 2011, 12: 2825-2830.

[6] 陈允杰, 王健. 机器学习实战: 基于 Scikit-Learn 与 Python [M]. 北京: 人民邮电出版社, 2022.
