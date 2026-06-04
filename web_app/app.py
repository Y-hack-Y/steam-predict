# -*- coding: utf-8 -*-
"""
工业蒸汽量预测分析平台 - Flask后端
"""
import os
import json
import base64
import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, session

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score, cross_val_predict
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.svm import SVR

app = Flask(__name__)
app.secret_key = 'steam_prediction_secret_key_2024'

# 全局数据缓存
DATA_CACHE = {}
MODEL_CACHE = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# ==================== 数据加载与探索模块 ====================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/load_data', methods=['POST'])
def load_data():
    """加载数据文件并返回基本概览"""
    try:
        req = request.get_json()
        file_path = req.get('file_path', '')

        # 默认路径
        if not file_path:
            train_path = os.path.join(PROJECT_DIR, 'zhengqi_train.txt')
            test_path = os.path.join(PROJECT_DIR, 'zhengqi_test.txt')
        else:
            train_path = file_path
            test_path = file_path.replace('train', 'test')

        # 读取训练数据
        train_df = pd.read_csv(train_path, sep='\t')
        test_df = pd.read_csv(test_path, sep='\t')

        DATA_CACHE['train'] = train_df
        DATA_CACHE['test'] = test_df
        DATA_CACHE['feature_cols'] = [c for c in train_df.columns if c.startswith('V')]
        DATA_CACHE['target_col'] = 'target' if 'target' in train_df.columns else None

        # 基本概览
        overview = {
            'train_shape': list(train_df.shape),
            'test_shape': list(test_df.shape),
            'feature_count': len(DATA_CACHE['feature_cols']),
            'feature_names': DATA_CACHE['feature_cols'],
            'target_name': DATA_CACHE['target_col'],
            'train_head': train_df.head(10).to_dict('records'),
            'train_dtypes': {k: str(v) for k, v in train_df.dtypes.to_dict().items()},
            'has_target': DATA_CACHE['target_col'] is not None,
        }
        return jsonify({'success': True, 'data': overview})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/data_stats', methods=['POST'])
def data_stats():
    """基本统计分析"""
    try:
        if 'train' not in DATA_CACHE:
            return jsonify({'success': False, 'error': '请先加载数据'})

        train_df = DATA_CACHE['train']
        test_df = DATA_CACHE['test']
        feature_cols = DATA_CACHE['feature_cols']
        target_col = DATA_CACHE['target_col']

        # 描述性统计
        train_desc = train_df.describe().to_dict()
        test_desc = test_df.describe().to_dict()

        # 偏度和峰度
        skewness = train_df[feature_cols].skew().to_dict()
        kurtosis = train_df[feature_cols].kurtosis().to_dict()

        # 缺失值统计
        train_missing = train_df.isnull().sum().to_dict()
        test_missing = test_df.isnull().sum().to_dict()

        # 目标变量统计
        target_stats = {}
        if target_col:
            target_stats = {
                'mean': float(train_df[target_col].mean()),
                'std': float(train_df[target_col].std()),
                'min': float(train_df[target_col].min()),
                'max': float(train_df[target_col].max()),
                'median': float(train_df[target_col].median()),
                'skew': float(train_df[target_col].skew()),
            }

        return jsonify({
            'success': True,
            'data': {
                'train_desc': train_desc,
                'test_desc': test_desc,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'train_missing': train_missing,
                'test_missing': test_missing,
                'target_stats': target_stats,
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/missing_heatmap', methods=['POST'])
def missing_heatmap():
    """缺失值热力图数据"""
    try:
        if 'train' not in DATA_CACHE:
            return jsonify({'success': False, 'error': '请先加载数据'})

        df = DATA_CACHE['train']
        missing_matrix = df.isnull().astype(int).values.tolist()
        columns = df.columns.tolist()[:50]  # 限制列数
        return jsonify({
            'success': True,
            'data': {
                'missing_matrix': missing_matrix,
                'columns': columns,
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/data_distribution', methods=['POST'])
def data_distribution():
    """数据分布数据 - 箱线图和直方图"""
    try:
        if 'train' not in DATA_CACHE:
            return jsonify({'success': False, 'error': '请先加载数据'})

        req = request.get_json() or {}
        feature_idx = req.get('feature_idx', 0)

        train_df = DATA_CACHE['train']
        test_df = DATA_CACHE['test']
        feature_cols = DATA_CACHE['feature_cols']

        if feature_idx >= len(feature_cols):
            return jsonify({'success': False, 'error': '特征索引越界'})

        col = feature_cols[feature_idx]

        # 直方图数据
        train_hist, train_bins = np.histogram(
            train_df[col].dropna(), bins=30, density=True
        )
        test_hist, test_bins = np.histogram(
            test_df[col].dropna(), bins=30, density=True
        )

        # 箱线图数据
        def get_box_data(series):
            q1 = float(series.quantile(0.25))
            q2 = float(series.median())
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower = float(max(series.min(), q1 - 1.5 * iqr))
            upper = float(min(series.max(), q3 + 1.5 * iqr))
            outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)].tolist()
            return {'q1': q1, 'q2': q2, 'q3': q3, 'lower': lower, 'upper': upper, 'outliers': outliers}

        return jsonify({
            'success': True,
            'data': {
                'feature_name': col,
                'train_hist': {'counts': train_hist.tolist(), 'bins': train_bins.tolist()},
                'test_hist': {'counts': test_hist.tolist(), 'bins': test_bins.tolist()},
                'train_box': get_box_data(train_df[col]),
                'test_box': get_box_data(test_df[col]),
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/correlation_matrix', methods=['POST'])
def correlation_matrix():
    """相关系数矩阵"""
    try:
        if 'train' not in DATA_CACHE:
            return jsonify({'success': False, 'error': '请先加载数据'})

        train_df = DATA_CACHE['train']
        feature_cols = DATA_CACHE['feature_cols']
        target_col = DATA_CACHE['target_col']

        cols = feature_cols + ([target_col] if target_col else [])
        corr_matrix = train_df[cols].corr()

        return jsonify({
            'success': True,
            'data': {
                'columns': cols,
                'corr_values': corr_matrix.values.tolist(),
                'target_corr': corr_matrix[target_col].to_dict() if target_col else {},
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==================== 特征工程模块 ====================

@app.route('/api/feature_engineering', methods=['POST'])
def feature_engineering():
    """特征工程处理"""
    try:
        if 'train' not in DATA_CACHE:
            return jsonify({'success': False, 'error': '请先加载数据'})

        req = request.get_json() or {}
        operations = req.get('operations', ['scale'])
        pca_components = req.get('pca_components', None)
        remove_outliers = req.get('remove_outliers', False)
        outlier_threshold = req.get('outlier_threshold', 3.0)
        custom_features = req.get('custom_features', False)
        corr_threshold = req.get('corr_threshold', None)

        train_df = DATA_CACHE['train'].copy()
        test_df = DATA_CACHE['test'].copy()
        feature_cols = DATA_CACHE['feature_cols']
        target_col = DATA_CACHE['target_col']

        log = []
        removed_features = []

        # 异常值处理
        if remove_outliers and target_col:
            original_len = len(train_df)
            # 使用 Ridge 回归检测异常值
            X = train_df[feature_cols].values
            y = train_df[target_col].values
            ridge = Ridge(alpha=1.0)
            ridge.fit(X, y)
            y_pred = ridge.predict(X)
            residuals = y - y_pred
            z_scores = np.abs((residuals - residuals.mean()) / residuals.std())
            train_df = train_df[z_scores < outlier_threshold].reset_index(drop=True)
            removed_count = original_len - len(train_df)
            log.append(f'移除 {removed_count} 个异常值 (z-score > {outlier_threshold})')

        # 低相关性特征筛选
        if corr_threshold and target_col:
            correlations = train_df[feature_cols + [target_col]].corr()[target_col].abs()
            low_corr_features = correlations[correlations < corr_threshold].index.tolist()
            removed_features = [f for f in low_corr_features if f != target_col]
            log.append(f'移除低相关性特征: {", ".join(removed_features)} (阈值 < {corr_threshold})')

        # 自定义交叉特征生成
        if custom_features and len(feature_cols) >= 2:
            log.append('生成交叉特征...')
            # 对前几个特征进行交叉
            cross_cols = feature_cols[:min(10, len(feature_cols))]
            for i in range(len(cross_cols)):
                for j in range(i + 1, len(cross_cols)):
                    ci, cj = cross_cols[i], cross_cols[j]
                    train_df[f'{ci}_add_{cj}'] = train_df[ci] + train_df[cj]
                    train_df[f'{ci}_mul_{cj}'] = train_df[ci] * train_df[cj]
                    test_df[f'{ci}_add_{cj}'] = test_df[ci] + test_df[cj]
                    test_df[f'{ci}_mul_{cj}'] = test_df[ci] * test_df[cj]
            log.append(f'生成 {len(cross_cols)*(len(cross_cols)-1)} 个交叉特征')

        # 更新特征列
        current_features = [c for c in train_df.columns if c not in [target_col] and c != 'target']
        current_features = [c for c in current_features if c not in removed_features]

        # 数据缩放
        scale_type = 'minmax' if 'scale' in operations else 'none'
        scaler = None
        if 'scale' in operations:
            scaler = MinMaxScaler()
            scaled_train = scaler.fit_transform(train_df[current_features])
            scaled_test = scaler.transform(test_df[current_features])
            train_df[current_features] = scaled_train
            test_df[current_features] = scaled_test
            log.append('应用 MinMaxScaler 归一化')
        elif 'standard' in operations:
            scaler = StandardScaler()
            scaled_train = scaler.fit_transform(train_df[current_features])
            scaled_test = scaler.transform(test_df[current_features])
            train_df[current_features] = scaled_train
            test_df[current_features] = scaled_test
            log.append('应用 StandardScaler 标准化')

        # PCA 降维
        pca = None
        if pca_components:
            pca = PCA(n_components=pca_components)
            train_pca = pca.fit_transform(train_df[current_features])
            test_pca = pca.transform(test_df[current_features])
            pca_cols = [f'PC{i+1}' for i in range(train_pca.shape[1])]
            train_pca_df = pd.DataFrame(train_pca, columns=pca_cols)
            test_pca_df = pd.DataFrame(test_pca, columns=pca_cols)
            if target_col:
                train_pca_df[target_col] = train_df[target_col].values
            train_df = train_pca_df
            test_df = test_pca_df
            log.append(f'PCA 降维至 {train_pca.shape[1]} 维 (解释方差比: {pca.explained_variance_ratio_.sum():.2%})')

        # 保存处理后的数据
        DATA_CACHE['train_processed'] = train_df
        DATA_CACHE['test_processed'] = test_df
        DATA_CACHE['current_features'] = current_features
        DATA_CACHE['scaler'] = scaler
        DATA_CACHE['pca'] = pca
        DATA_CACHE['feature_log'] = log
        DATA_CACHE['scale_type'] = scale_type

        # 获取处理后的特征列
        processed_features = [c for c in train_df.columns if c != target_col and c != 'target']

        return jsonify({
            'success': True,
            'data': {
                'train_shape': list(train_df.shape),
                'test_shape': list(test_df.shape),
                'feature_count': len(processed_features),
                'features': processed_features[:50],
                'log': log,
                'train_head': train_df.head(5).to_dict('records'),
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==================== 模型训练模块 ====================

# 默认模型配置
DEFAULT_MODELS = {
    'LinearRegression': {
        'class': LinearRegression,
        'params': {},
        'name_cn': '线性回归',
        'color': '#5470c6',
    },
    'Ridge': {
        'class': Ridge,
        'params': {'alpha': 1.0},
        'name_cn': '岭回归',
        'color': '#91cc75',
    },
    'KNN': {
        'class': KNeighborsRegressor,
        'params': {'n_neighbors': 5},
        'name_cn': 'K近邻回归',
        'color': '#fac858',
    },
    'DecisionTree': {
        'class': DecisionTreeRegressor,
        'params': {'max_depth': 10, 'random_state': 42},
        'name_cn': '决策树',
        'color': '#ee6666',
    },
    'RandomForest': {
        'class': RandomForestRegressor,
        'params': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1},
        'name_cn': '随机森林',
        'color': '#73c0de',
    },
    'GradientBoosting': {
        'class': GradientBoostingRegressor,
        'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': 42},
        'name_cn': '梯度提升树',
        'color': '#3ba272',
    },
    'AdaBoost': {
        'class': AdaBoostRegressor,
        'params': {'n_estimators': 50, 'random_state': 42},
        'name_cn': 'AdaBoost',
        'color': '#fc8452',
    },
    'ExtraTrees': {
        'class': ExtraTreesRegressor,
        'params': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1},
        'name_cn': '极端随机树',
        'color': '#9a60b4',
    },
    'SGDRegressor': {
        'class': SGDRegressor,
        'params': {'max_iter': 1000, 'tol': 1e-3, 'random_state': 42},
        'name_cn': 'SGD回归',
        'color': '#ea7ccc',
    },
    'SVR': {
        'class': SVR,
        'params': {'kernel': 'rbf', 'C': 1.0},
        'name_cn': '支持向量回归',
        'color': '#48b8d0',
    },
}


def train_single_model(name, config, X_train, y_train, X_test, y_test):
    """训练单个模型并返回评估结果"""
    try:
        model_cls = config['class']
        params = config.get('params', {})
        model = model_cls(**params)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        result = {
            'name': name,
            'name_cn': config.get('name_cn', name),
            'color': config.get('color', '#666'),
            'train_mse': float(mean_squared_error(y_train, y_train_pred)),
            'test_mse': float(mean_squared_error(y_test, y_test_pred)),
            'train_mae': float(mean_absolute_error(y_train, y_train_pred)),
            'test_mae': float(mean_absolute_error(y_test, y_test_pred)),
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            'train_r2': float(r2_score(y_train, y_train_pred)),
            'test_r2': float(r2_score(y_test, y_test_pred)),
            'model': model,
            'test_predictions': y_test_pred.tolist(),
            'test_actual': y_test.tolist(),
        }
        MODEL_CACHE[name] = model
        return result
    except Exception as e:
        return {
            'name': name,
            'name_cn': config.get('name_cn', name),
            'color': config.get('color', '#666'),
            'error': str(e),
        }


@app.route('/api/train_models', methods=['POST'])
def train_models():
    """并行训练多个模型"""
    try:
        req = request.get_json() or {}
        selected_models = req.get('models', list(DEFAULT_MODELS.keys()))
        custom_params = req.get('custom_params', {})
        test_size = req.get('test_size', 0.2)
        random_state = req.get('random_state', 42)

        # 获取处理后的数据
        if 'train_processed' in DATA_CACHE:
            train_df = DATA_CACHE['train_processed']
        else:
            train_df = DATA_CACHE['train']

        target_col = DATA_CACHE['target_col']
        if not target_col:
            return jsonify({'success': False, 'error': '训练数据中缺少目标变量 target'})

        feature_cols = [c for c in train_df.columns if c != target_col and c != 'target']

        X = train_df[feature_cols].values
        y = train_df[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        results = []
        # 使用线程池并行训练
        with ThreadPoolExecutor(max_workers=min(len(selected_models), 4)) as executor:
            futures = {}
            for name in selected_models:
                if name in DEFAULT_MODELS:
                    config = DEFAULT_MODELS[name].copy()
                    if name in custom_params:
                        config['params'] = {**config['params'], **custom_params[name]}
                    futures[name] = executor.submit(
                        train_single_model, name, config, X_train, y_train, X_test, y_test
                    )

            for name, future in futures.items():
                result = future.result()
                # 移除不可序列化的 model 对象
                if 'model' in result:
                    MODEL_CACHE[name] = result.pop('model')
                results.append(result)

        # 按测试集 RMSE 排序
        results.sort(key=lambda x: x.get('test_rmse', float('inf')))

        DATA_CACHE['train_results'] = results
        DATA_CACHE['X_train'] = X_train
        DATA_CACHE['X_test'] = X_test
        DATA_CACHE['y_train'] = y_train
        DATA_CACHE['y_test'] = y_test

        return jsonify({'success': True, 'data': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==================== 模型验证模块 ====================

@app.route('/api/cross_validation', methods=['POST'])
def cross_validation():
    """交叉验证"""
    try:
        req = request.get_json() or {}
        model_name = req.get('model', 'RandomForest')
        n_splits = req.get('n_splits', 5)
        cv_method = req.get('cv_method', 'kfold')

        if 'train_processed' in DATA_CACHE:
            train_df = DATA_CACHE['train_processed']
        else:
            train_df = DATA_CACHE['train']

        target_col = DATA_CACHE['target_col']
        feature_cols = [c for c in train_df.columns if c != target_col and c != 'target']

        X = train_df[feature_cols].values
        y = train_df[target_col].values

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_results = []
        all_metrics = {'mae': [], 'mse': [], 'rmse': [], 'r2': []}

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            if model_name in DEFAULT_MODELS:
                config = DEFAULT_MODELS[model_name]
                model = config['class'](**config.get('params', {}))
            else:
                return jsonify({'success': False, 'error': f'未知模型: {model_name}'})

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)

            fold_metrics = {
                'mae': float(mean_absolute_error(y_val, y_pred)),
                'mse': float(mean_squared_error(y_val, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_val, y_pred))),
                'r2': float(r2_score(y_val, y_pred)),
            }
            fold_results.append(fold_metrics)
            for k, v in fold_metrics.items():
                all_metrics[k].append(v)

        avg_metrics = {
            f'avg_{k}': float(np.mean(v)) for k, v in all_metrics.items()
        }
        std_metrics = {
            f'std_{k}': float(np.std(v)) for k, v in all_metrics.items()
        }

        return jsonify({
            'success': True,
            'data': {
                'fold_results': fold_results,
                'avg_metrics': avg_metrics,
                'std_metrics': std_metrics,
                'n_splits': n_splits,
                'model_name': model_name,
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/model_comparison', methods=['POST'])
def model_comparison():
    """模型对比分析数据"""
    try:
        if 'train_results' not in DATA_CACHE:
            return jsonify({'success': False, 'error': '请先训练模型'})

        results = DATA_CACHE['train_results']
        comparison = {
            'models': [],
            'metrics': {
                'test_mse': [], 'test_mae': [], 'test_rmse': [], 'test_r2': [],
                'train_mse': [], 'train_mae': [], 'train_rmse': [], 'train_r2': [],
            }
        }

        for r in results:
            if 'error' not in r:
                comparison['models'].append(r['name_cn'])
                for metric in comparison['metrics']:
                    comparison['metrics'][metric].append(r.get(metric, 0))

        # 找出最佳模型
        best_idx = np.argmin(comparison['metrics']['test_rmse'])
        comparison['best_model'] = comparison['models'][best_idx]
        comparison['best_rmse'] = comparison['metrics']['test_rmse'][best_idx]
        comparison['best_r2'] = comparison['metrics']['test_r2'][best_idx]

        return jsonify({'success': True, 'data': comparison})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/prediction_vs_actual', methods=['POST'])
def prediction_vs_actual():
    """预测值与实际值对比数据"""
    try:
        req = request.get_json() or {}
        model_name = req.get('model', None)

        if 'train_results' not in DATA_CACHE:
            return jsonify({'success': False, 'error': '请先训练模型'})

        results = DATA_CACHE['train_results']
        data = {'models': []}

        for r in results:
            if 'error' not in r and (model_name is None or r['name'] == model_name):
                data['models'].append({
                    'name': r['name_cn'],
                    'color': r['color'],
                    'actual': r.get('test_actual', []),
                    'predicted': r.get('test_predictions', []),
                    'rmse': r['test_rmse'],
                    'r2': r['test_r2'],
                })

        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/feature_importance', methods=['POST'])
def feature_importance():
    """特征重要性"""
    try:
        if 'train_processed' in DATA_CACHE:
            train_df = DATA_CACHE['train_processed']
        else:
            train_df = DATA_CACHE['train']

        target_col = DATA_CACHE['target_col']
        feature_cols = [c for c in train_df.columns if c != target_col and c != 'target']

        X = train_df[feature_cols].values
        y = train_df[target_col].values

        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importances = rf.feature_importances_

        # 排序
        indices = np.argsort(importances)[::-1]
        top_n = min(20, len(feature_cols))
        top_features = [feature_cols[i] for i in indices[:top_n]]
        top_importances = importances[indices[:top_n]].tolist()

        return jsonify({
            'success': True,
            'data': {
                'features': top_features,
                'importances': top_importances,
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==================== 结果预测模块 ====================

@app.route('/api/predict', methods=['POST'])
def predict():
    """单样本或批量预测"""
    try:
        req = request.get_json() or {}
        model_name = req.get('model', None)
        predict_type = req.get('type', 'test')  # 'test', 'single', 'batch'
        single_values = req.get('values', None)
        batch_data = req.get('batch_data', None)

        # 选择模型
        if model_name and model_name in MODEL_CACHE:
            model = MODEL_CACHE[model_name]
        elif MODEL_CACHE:
            # 使用测试 RMSE 最低的模型
            best_name = None
            best_rmse = float('inf')
            if 'train_results' in DATA_CACHE:
                for r in DATA_CACHE['train_results']:
                    if 'error' not in r and r['test_rmse'] < best_rmse:
                        best_rmse = r['test_rmse']
                        best_name = r['name']
            if best_name and best_name in MODEL_CACHE:
                model = MODEL_CACHE[best_name]
                model_name = best_name
            else:
                model_name = list(MODEL_CACHE.keys())[0]
                model = MODEL_CACHE[model_name]
        else:
            return jsonify({'success': False, 'error': '请先训练模型'})

        if predict_type == 'single':
            if not single_values:
                return jsonify({'success': False, 'error': '请提供单样本特征值'})
            X_pred = np.array(single_values).reshape(1, -1)
            prediction = float(model.predict(X_pred)[0])
            return jsonify({
                'success': True,
                'data': {
                    'type': 'single',
                    'prediction': prediction,
                    'model': model_name,
                }
            })

        elif predict_type == 'batch':
            if not batch_data:
                return jsonify({'success': False, 'error': '请提供批量样本数据'})
            X_pred = np.array(batch_data)
            predictions = model.predict(X_pred).tolist()
            return jsonify({
                'success': True,
                'data': {
                    'type': 'batch',
                    'predictions': predictions,
                    'model': model_name,
                    'count': len(predictions),
                }
            })

        else:  # test set prediction
            if 'X_test' not in DATA_CACHE:
                return jsonify({'success': False, 'error': '请先训练模型'})

            X_test = DATA_CACHE['X_test']
            y_test = DATA_CACHE.get('y_test')

            predictions = model.predict(X_test).tolist()
            actual = y_test.tolist() if y_test is not None else []

            result = {
                'type': 'test',
                'predictions': predictions[:100],
                'actual': actual[:100] if actual else [],
                'model': model_name,
                'count': len(predictions),
            }
            if actual:
                result['rmse'] = float(np.sqrt(mean_squared_error(actual, predictions)))
                result['mae'] = float(mean_absolute_error(actual, predictions))
                result['r2'] = float(r2_score(actual, predictions))

            return jsonify({'success': True, 'data': result})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/available_models', methods=['GET'])
def available_models():
    """获取可用模型列表"""
    models = []
    for name, config in DEFAULT_MODELS.items():
        models.append({
            'id': name,
            'name': config['name_cn'],
            'color': config['color'],
            'default_params': config['params'],
        })
    return jsonify({'success': True, 'data': models})


if __name__ == '__main__':
    import threading
    import webbrowser

    def open_browser():
        webbrowser.open('http://127.0.0.1:5000')

    # 1秒后自动打开浏览器
    threading.Timer(1.0, open_browser).start()
    app.run(debug=False, host='127.0.0.1', port=5000)
