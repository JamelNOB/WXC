import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel, RFECV, SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')


class ChurnPredictor:
    def __init__(self, categorical_features, random_state=42):
        self.categorical_features = categorical_features.copy()
        self.random_state = random_state
        self.preprocessor = None
        self.feature_names = None
        self.feature_selector = None
        self.best_threshold = 0.5
        self.selected_features = []
        
        # 默认使用XGBoost作为基础分类器
        self.classifier = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=self.random_state
        )

    def load_data(self, filepath):
        """加载数据"""
        return pd.read_csv(filepath)
    
    def analyze_data(self, df):
        """分析数据并可视化关键模式"""
        print("\n===== 数据分析 =====")
        
        # 数据基本信息
        print(f"数据维度: {df.shape}")
        
        # 缺失值分析
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        missing_df = pd.DataFrame({'缺失值数量': missing, '缺失比例%': missing_percent})
        print("\n缺失值分析:")
        print(missing_df[missing_df['缺失值数量'] > 0])
        
        # 分析目标变量
        if 'Churned' in df.columns:
            churn_count = df['Churned'].value_counts()
            print("\n流失分布:")
            print(churn_count)
            print(f"流失率: {churn_count[1] / len(df):.2%}")
            
            # 查看数值特征与目标的相关性
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            numeric_cols = [col for col in numeric_cols if col not in ['Customer_ID', 'Churned']]
            
            if numeric_cols:
                correlations = df[numeric_cols + ['Churned']].corr()['Churned'].sort_values(ascending=False)
                print("\n数值特征与流失的相关性:")
                print(correlations)
            
            # 分析各分类特征与流失的关系
            print("\n分类特征与流失的关系:")
            for cat in self.categorical_features:
                if cat in df.columns:
                    churn_by_cat = df.groupby(cat)['Churned'].mean().sort_values(ascending=False)
                    print(f"\n{cat}:")
                    print(churn_by_cat)
        
        return missing_df[missing_df['缺失值数量'] > 0].index.tolist()

    def feature_engineering(self, df):
        """创建新特征以提高模型性能"""
        print("\n===== 执行特征工程 =====")
        df_fe = df.copy()
        original_features = self.categorical_features.copy()
        
        # 1. 满意度相关特征 - 更精细的分组
        if 'Satisfaction_Score' in df_fe.columns:
            # 使用分位数创建满意度分组
            valid_scores = df_fe['Satisfaction_Score'].dropna()
            if len(valid_scores) > 0:
                df_fe['Satisfaction_Group'] = pd.qcut(
                    df_fe['Satisfaction_Score'].fillna(valid_scores.median()), 
                    q=4, 
                    labels=['很低', '低', '中', '高']
                )
                self.categorical_features.append('Satisfaction_Group')
            
            # 提取满意度极值特征
            df_fe['Is_Very_Satisfied'] = (df_fe['Satisfaction_Score'] > 8).astype(int)
            df_fe['Is_Very_Unsatisfied'] = (df_fe['Satisfaction_Score'] < 3).astype(int)
            self.categorical_features.extend(['Is_Very_Satisfied', 'Is_Very_Unsatisfied'])
            
            # 标记缺失的满意度 (可能是重要信号)
            df_fe['Satisfaction_Missing'] = df_fe['Satisfaction_Score'].isnull().astype(int)
            self.categorical_features.append('Satisfaction_Missing')
        
        # 2. 活动时间特征增强
        if 'Last_Activity' in df_fe.columns:
            # 标记缺失的活动记录
            df_fe['Activity_Missing'] = df_fe['Last_Activity'].isnull().astype(int)
            self.categorical_features.append('Activity_Missing')
            
            # 计算活动衰减系数 (距今时间越长越可能流失)
            valid_activity = df_fe['Last_Activity'].dropna()
            if not valid_activity.empty:
                max_activity = valid_activity.max()
                df_fe['Activity_Recency'] = (max_activity - df_fe['Last_Activity']) / max_activity
                df_fe['Activity_Recency'].fillna(1.0, inplace=True)  # 没有活动记录的客户视为最不活跃
                
                # 将活动时间分组
                df_fe['Activity_Group'] = pd.qcut(
                    df_fe['Last_Activity'].fillna(valid_activity.median()),
                    q=4,
                    labels=['最近活跃', '近期活跃', '较早活跃', '早期活跃']
                )
                self.categorical_features.append('Activity_Group')
        
        # 3. 消费行为相关特征
        if 'Monthly_Spend' in df_fe.columns:
            # 标记缺失的消费记录
            df_fe['Spend_Missing'] = df_fe['Monthly_Spend'].isnull().astype(int)
            self.categorical_features.append('Spend_Missing')
            
            # 将消费额分组
            valid_spend = df_fe['Monthly_Spend'].dropna()
            if not valid_spend.empty and len(valid_spend) > 4:
                df_fe['Spend_Group'] = pd.qcut(
                    df_fe['Monthly_Spend'].fillna(valid_spend.median()),
                    q=4,
                    labels=['低消费', '中低消费', '中高消费', '高消费']
                )
                self.categorical_features.append('Spend_Group')
                
                # 标记高价值和低价值客户
                df_fe['Is_High_Value'] = (df_fe['Monthly_Spend'] > valid_spend.quantile(0.75)).astype(int)
                df_fe['Is_Low_Value'] = (df_fe['Monthly_Spend'] < valid_spend.quantile(0.25)).astype(int)
                self.categorical_features.extend(['Is_High_Value', 'Is_Low_Value'])
        
        # 4. 客户支持相关特征
        if 'Support_Tickets_Raised' in df_fe.columns:
            # 根据支持票据数量创建分组
            df_fe['Support_Level'] = pd.cut(
                df_fe['Support_Tickets_Raised'],
                bins=[-1, 0, 2, 4, float('inf')],
                labels=['无支持', '低支持', '中等支持', '高支持']
            )
            self.categorical_features.append('Support_Level')
            
            # 支持票据与订阅时长的比率 (支持密度)
            if 'Subscription_Length' in df_fe.columns:
                df_fe['Ticket_Rate'] = df_fe['Support_Tickets_Raised'] / df_fe['Subscription_Length'].replace(0, 1)
                df_fe['Ticket_Rate'].fillna(0, inplace=True)
                
                # 高支持需求客户标记
                df_fe['High_Support_Need'] = (df_fe['Ticket_Rate'] > 0.15).astype(int)
                self.categorical_features.append('High_Support_Need')
                
                # 短期内高支持需求
                df_fe['Early_Support_Issues'] = ((df_fe['Subscription_Length'] < 12) & 
                                               (df_fe['Support_Tickets_Raised'] > 2)).astype(int)
                self.categorical_features.append('Early_Support_Issues')
        
        # 5. 折扣效率特征
        if 'Discount_Offered' in df_fe.columns and 'Monthly_Spend' in df_fe.columns:
            valid_spend = df_fe['Monthly_Spend'].dropna()
            if not valid_spend.empty:
                # 折扣与支出比率
                df_fe['Discount_Ratio'] = df_fe['Discount_Offered'] / df_fe['Monthly_Spend'].fillna(valid_spend.median())
                df_fe['Discount_Ratio'].replace([float('inf'), -float('inf')], np.nan, inplace=True)
                df_fe['Discount_Ratio'].fillna(0, inplace=True)
                
                # 折扣效率 (高折扣但低支出可能表示客户不满意)
                df_fe['Discount_Efficiency'] = df_fe['Monthly_Spend'].fillna(valid_spend.median()) / (df_fe['Discount_Offered'] + 1)
                
                # 折扣分组
                df_fe['Discount_Group'] = pd.qcut(
                    df_fe['Discount_Offered'], 
                    q=4, 
                    labels=['低折扣', '中低折扣', '中高折扣', '高折扣']
                )
                self.categorical_features.append('Discount_Group')
                
                # 高折扣低支出标记 - 可能是流失风险信号
                df_fe['High_Discount_Low_Spend'] = ((df_fe['Discount_Offered'] > df_fe['Discount_Offered'].quantile(0.75)) & 
                                                  (df_fe['Monthly_Spend'] < valid_spend.quantile(0.5))).astype(int)
                self.categorical_features.append('High_Discount_Low_Spend')
        
        # 6. 年龄和人口统计特征
        if 'Age' in df_fe.columns:
            # 年龄分组
            df_fe['Age_Group'] = pd.cut(
                df_fe['Age'].fillna(df_fe['Age'].median()),
                bins=[0, 25, 35, 45, 55, 65, 100],
                labels=['年轻人', '青年', '中青年', '中年', '中老年', '老年']
            )
            self.categorical_features.append('Age_Group')
            
            # 特定年龄段客户标记
            df_fe['Is_Young'] = (df_fe['Age'] < 30).astype(int)
            df_fe['Is_Senior'] = (df_fe['Age'] >= 60).astype(int)
            self.categorical_features.extend(['Is_Young', 'Is_Senior'])
            
            # 年龄缺失标记
            df_fe['Age_Missing'] = df_fe['Age'].isnull().astype(int)
            self.categorical_features.append('Age_Missing')
        
        # 7. 订阅相关特征
        if 'Subscription_Length' in df_fe.columns:
            # 订阅年限分组
            df_fe['Subscription_Years'] = df_fe['Subscription_Length'] / 12
            df_fe['Subscription_Group'] = pd.cut(
                df_fe['Subscription_Length'],
                bins=[0, 6, 12, 24, 36, float('inf')],
                labels=['半年内', '一年内', '两年内', '三年内', '长期客户']
            )
            self.categorical_features.append('Subscription_Group')
            
            # 新客户和长期客户标记
            df_fe['Is_New_Customer'] = (df_fe['Subscription_Length'] <= 6).astype(int)
            df_fe['Is_Long_Term'] = (df_fe['Subscription_Length'] >= 36).astype(int)
            self.categorical_features.extend(['Is_New_Customer', 'Is_Long_Term'])
        
        # 8. 交互特征
        # 年龄与订阅时长交互
        if 'Age' in df_fe.columns and 'Subscription_Length' in df_fe.columns:
            df_fe['Age_per_Subscription'] = df_fe['Age'].fillna(df_fe['Age'].median()) / df_fe['Subscription_Length'].replace(0, 1)
            
        # 满意度与支持票据交互
        if 'Satisfaction_Score' in df_fe.columns and 'Support_Tickets_Raised' in df_fe.columns:
            valid_scores = df_fe['Satisfaction_Score'].dropna()
            if len(valid_scores) > 0:
                df_fe['Satisfaction_per_Ticket'] = df_fe['Satisfaction_Score'].fillna(valid_scores.median()) / (df_fe['Support_Tickets_Raised'] + 1)
        
        # 订阅与支出的性价比
        if 'Monthly_Spend' in df_fe.columns and 'Subscription_Length' in df_fe.columns:
            valid_spend = df_fe['Monthly_Spend'].dropna()
            if not valid_spend.empty:
                df_fe['Value_Ratio'] = df_fe['Subscription_Length'] / (df_fe['Monthly_Spend'].fillna(valid_spend.median()) + 1)
        
        # 9. 复合风险评分
        risk_columns = []
        if 'Satisfaction_Score' in df_fe.columns: 
            risk_columns.append('Satisfaction_Score')
        if 'Monthly_Spend' in df_fe.columns: 
            risk_columns.append('Monthly_Spend')
        if 'Support_Tickets_Raised' in df_fe.columns: 
            risk_columns.append('Support_Tickets_Raised')
        if 'Discount_Offered' in df_fe.columns: 
            risk_columns.append('Discount_Offered')
        if 'Last_Activity' in df_fe.columns:
            risk_columns.append('Last_Activity')
        
        if risk_columns:
            # 标准化风险因子
            risk_df = df_fe[risk_columns].copy()
            for col in risk_df.columns:
                if col == 'Satisfaction_Score':
                    # 满意度越低风险越高
                    if not risk_df[col].isna().all():
                        max_val = risk_df[col].fillna(risk_df[col].median()).max()
                        risk_df[col] = (max_val - risk_df[col].fillna(risk_df[col].median())) / max_val
                elif col == 'Monthly_Spend':
                    # 消费越低风险越高
                    if not risk_df[col].isna().all():
                        max_val = risk_df[col].fillna(risk_df[col].median()).max()
                        risk_df[col] = (max_val - risk_df[col].fillna(risk_df[col].median())) / max_val
                elif col == 'Support_Tickets_Raised':
                    # 支持票越多风险越高
                    if not risk_df[col].isna().all():
                        max_val = risk_df[col].max()
                        if max_val > 0:
                            risk_df[col] = risk_df[col] / max_val
                elif col == 'Discount_Offered':
                    # 折扣越高风险越高
                    if not risk_df[col].isna().all():
                        max_val = risk_df[col].max()
                        if max_val > 0:
                            risk_df[col] = risk_df[col] / max_val
                elif col == 'Last_Activity':
                    # 最后活动越早风险越高
                    if not risk_df[col].isna().all():
                        valid_activity = risk_df[col].dropna()
                        if len(valid_activity) > 0:
                            max_val = valid_activity.max()
                            min_val = valid_activity.min()
                            if max_val > min_val:
                                # 将活动时间转换为0-1之间的风险分数，活动时间越早风险越高
                                risk_df[col] = 1 - ((risk_df[col].fillna(min_val) - min_val) / (max_val - min_val))
            
            # 计算综合风险评分
            df_fe['Churn_Risk_Score'] = risk_df.mean(axis=1)
            
            # 创建双重风险特征
            high_risk_cols = ['Support_Tickets_Raised', 'Churn_Risk_Score']
            if 'Satisfaction_Score' in df_fe.columns:
                df_fe['High_Risk_Low_Satisfaction'] = ((df_fe['Churn_Risk_Score'] > df_fe['Churn_Risk_Score'].quantile(0.7)) & 
                                                     (df_fe['Satisfaction_Score'] < 5)).astype(int)
                self.categorical_features.append('High_Risk_Low_Satisfaction')
            
            # 风险评分分组
            df_fe['Risk_Group'] = pd.qcut(
                df_fe['Churn_Risk_Score'], 
                q=4, 
                labels=['低风险', '中低风险', '中高风险', '高风险']
            )
            self.categorical_features.append('Risk_Group')
        
        # 输出新创建的特征
        new_features = [f for f in self.categorical_features if f not in original_features]
        print(f"创建了 {len(new_features)} 个新特征")
        print(f"新特征: {new_features}")
        return df_fe

    def select_features(self, X, y):
        """使用多种方法选择最重要的特征"""
        print("\n===== 特征选择 =====")
        
        # 使用随机森林进行特征重要性评估
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf.fit(X, y)
        
        # 获取特征重要性
        if hasattr(self, 'feature_names') and self.feature_names is not None:
            feature_importances = [(feature, importance) for feature, importance in 
                                  zip(range(X.shape[1]), rf.feature_importances_)]
            feature_importances.sort(key=lambda x: x[1], reverse=True)
            
            print("顶级特征重要性:")
            for idx, importance in feature_importances[:10]:
                print(f"特征 {idx}: {importance:.4f}")
            
            # 选择重要特征 - 只保留超过平均重要性的特征
            avg_importance = np.mean(rf.feature_importances_)
            selected_indices = [idx for idx, importance in feature_importances if importance > avg_importance]
            
            print(f"\n基于随机森林重要性选择了 {len(selected_indices)} 个特征")
            
            # 创建特征选择器
            self.feature_selector = SelectFromModel(rf, prefit=True, threshold=avg_importance)
        
        return X

    def preprocess(self, df, is_train=True):
        """预处理数据：处理缺失值，编码分类特征，缩放数值特征"""
        print("\n===== 数据预处理 =====")
        df_processed = df.copy()
        
        # 确定所有数值型列
        numeric_features = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_features = [col for col in numeric_features if col not in self.categorical_features 
                           and col not in ['Churned', 'Customer_ID']]
        
        print(f"数值特征: {len(numeric_features)}")
        print(f"分类特征: {len(self.categorical_features)}")
        
        # 保存特征名称 (仅在训练阶段)
        if is_train:
            self.feature_names = numeric_features + self.categorical_features
        
        # 数值特征处理流水线 - 使用KNN填充缺失值并进行鲁棒缩放
        numeric_pipeline = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5, weights='distance')),  # 使用加权KNN提高填充质量
            ('scaler', RobustScaler()),  # 鲁棒缩放以处理异常值
            ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))  # 添加交互特征
        ])
        
        # 分类特征处理流水线 - 使用最频繁值填充缺失值并进行OneHot编码
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # 合并处理步骤
        transformers = [
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, self.categorical_features)
        ]
        
        if is_train:
            self.preprocessor = ColumnTransformer(transformers=transformers)
            df_processed = self.preprocessor.fit_transform(df_processed)
            print(f"预处理后特征维度: {df_processed.shape[1]}")
        else:
            df_processed = self.preprocessor.transform(df_processed)
        
        return df_processed

    def handle_imbalance(self, X, y):
        """处理类别不平衡问题"""
        print("\n===== 处理类别不平衡 =====")
        
        # 确保y是numpy数组
        y_np = y.values if hasattr(y, 'values') else np.array(y)
        
        # 检查类别分布
        class_counts = np.bincount(y_np)
        
        if len(class_counts) > 1 and class_counts[1] < class_counts[0] * 0.75:
            print(f"检测到类别不平衡: 多数类={class_counts[0]}, 少数类={class_counts[1]}")
            print("应用SMOTE过采样来平衡类别...")
            
            # 使用SMOTE过采样少数类
            smote = SMOTE(random_state=self.random_state, sampling_strategy=0.8)  # 不完全平衡，保留一些不平衡
            X_resampled, y_resampled = smote.fit_resample(X, y_np)
            
            # 报告重采样后的类别分布
            new_class_counts = np.bincount(y_resampled)
            print(f"重采样后类别分布: 多数类={new_class_counts[0]}, 少数类={new_class_counts[1]}")
            
            return X_resampled, y_resampled
        else:
            print("类别分布相对平衡，无需重采样")
            return X, y_np

    def select_best_model(self, X, y):
        """选择最佳模型类型"""
        print("\n===== 模型选择 =====")
        
        # 确保y是numpy数组
        y_np = y.values if hasattr(y, 'values') else np.array(y)
        
        # 候选模型
        models = {
            "XGBoost": XGBClassifier(
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=self.random_state
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state
            ),
            "随机森林": RandomForestClassifier(
                n_estimators=100, 
                class_weight='balanced',
                random_state=self.random_state
            ),
            "梯度提升": GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                random_state=self.random_state
            ),
            "逻辑回归": LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=2000,  # 增加最大迭代次数解决收敛问题
                solver='saga',  # 更换求解器提高收敛性
                penalty='l2',
                random_state=self.random_state
            )
        }
        
        # 使用5折交叉验证评估模型
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        best_model_name = None
        best_score = 0
        best_model = None
        results = {}
        
        for name, model in models.items():
            accuracies = []
            precisions = []
            recalls = []
            f1s = []
            roc_aucs = []
            
            # 使用交叉验证评估
            for train_idx, val_idx in cv.split(X, y_np):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y_np[train_idx], y_np[val_idx]
                
                # 训练模型
                model.fit(X_train, y_train)
                
                # 预测
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                # 计算指标
                accuracies.append(accuracy_score(y_val, y_pred))
                precisions.append(precision_score(y_val, y_pred))
                recalls.append(recall_score(y_val, y_pred))
                f1s.append(f1_score(y_val, y_pred))
                roc_aucs.append(roc_auc_score(y_val, y_proba))
            
            # 计算平均值和标准差
            avg_accuracy = np.mean(accuracies)
            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)
            avg_f1 = np.mean(f1s)
            avg_roc_auc = np.mean(roc_aucs)
            
            results[name] = {
                'accuracy': avg_accuracy,
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'roc_auc': avg_roc_auc
            }
            
            # 输出结果
            print(f"{name}:")
            print(f"  准确率: {avg_accuracy:.4f} ± {np.std(accuracies):.4f}")
            print(f"  精确率: {avg_precision:.4f} ± {np.std(precisions):.4f}")
            print(f"  召回率: {avg_recall:.4f} ± {np.std(recalls):.4f}")
            print(f"  F1分数: {avg_f1:.4f} ± {np.std(f1s):.4f}")
            print(f"  ROC AUC: {avg_roc_auc:.4f} ± {np.std(roc_aucs):.4f}")
            
            # 选择F1最高的模型
            if avg_f1 > best_score:
                best_score = avg_f1
                best_model_name = name
                best_model = models[name]  # 使用新的模型实例，而不是已经训练过的模型
        
        # 尝试堆叠集成 (使用StackingClassifier)
        print("\n尝试堆叠集成模型...")
        # 选择前3个性能最好的模型
        top_models = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)[:3]
        top_model_names = [name for name, _ in top_models]
        
        print(f"选择顶级模型进行堆叠: {', '.join(top_model_names)}")
        
        # 创建基础分类器
        estimators = []
        for name in top_model_names:
            # 重新初始化模型以避免问题
            if name == "XGBoost":
                model = XGBClassifier(
                    n_estimators=100, 
                    learning_rate=0.1,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=self.random_state
                )
            elif name == "LightGBM":
                model = LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state
                )
            elif name == "随机森林":
                model = RandomForestClassifier(
                    n_estimators=100, 
                    class_weight='balanced',
                    random_state=self.random_state
                )
            elif name == "梯度提升":
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=3,
                    random_state=self.random_state
                )
            elif name == "逻辑回归":
                model = LogisticRegression(
                    C=1.0,
                    class_weight='balanced',
                    max_iter=2000,  # 增加迭代次数解决收敛问题
                    solver='saga',   # 使用saga求解器提高收敛性和性能
                    random_state=self.random_state
                )
            estimators.append((name, model))
        
        # 使用有效的L2正则化逻辑回归作为最终元分类器
        final_estimator = LogisticRegression(
            C=0.1,  # 增强正则化
            solver='saga',  # saga求解器对大规模数据效果好
            max_iter=2000,  # 增加迭代次数
            random_state=self.random_state
        )
        
        # 创建堆叠分类器
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            n_jobs=-1,
            passthrough=False  # 不传递原始特征，减少过拟合
        )
        
        # 评估堆叠分类器
        stacking_accuracies = []
        stacking_precisions = []
        stacking_recalls = []
        stacking_f1s = []
        stacking_roc_aucs = []
        
        try:
            for train_idx, val_idx in cv.split(X, y_np):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y_np[train_idx], y_np[val_idx]
                
                # 训练模型
                stacking_clf.fit(X_train, y_train)
                
                # 预测
                y_pred = stacking_clf.predict(X_val)
                y_proba = stacking_clf.predict_proba(X_val)[:, 1]
                
                # 计算指标
                stacking_accuracies.append(accuracy_score(y_val, y_pred))
                stacking_precisions.append(precision_score(y_val, y_pred))
                stacking_recalls.append(recall_score(y_val, y_pred))
                stacking_f1s.append(f1_score(y_val, y_pred))
                stacking_roc_aucs.append(roc_auc_score(y_val, y_proba))
            
            # 计算堆叠模型的平均性能
            avg_stacking_accuracy = np.mean(stacking_accuracies)
            avg_stacking_precision = np.mean(stacking_precisions)
            avg_stacking_recall = np.mean(stacking_recalls)
            avg_stacking_f1 = np.mean(stacking_f1s)
            avg_stacking_roc_auc = np.mean(stacking_roc_aucs)
            
            print("\n堆叠集成模型性能:")
            print(f"  准确率: {avg_stacking_accuracy:.4f} ± {np.std(stacking_accuracies):.4f}")
            print(f"  精确率: {avg_stacking_precision:.4f} ± {np.std(stacking_precisions):.4f}")
            print(f"  召回率: {avg_stacking_recall:.4f} ± {np.std(stacking_recalls):.4f}")
            print(f"  F1分数: {avg_stacking_f1:.4f} ± {np.std(stacking_f1s):.4f}")
            print(f"  ROC AUC: {avg_stacking_roc_auc:.4f} ± {np.std(stacking_roc_aucs):.4f}")
            
            # 如果堆叠模型性能更好，则使用堆叠模型
            if avg_stacking_f1 > best_score:
                print("堆叠集成模型表现最佳，已选择堆叠模型")
                # 重新创建一个完整的堆叠模型
                final_stack = StackingClassifier(
                    estimators=estimators,
                    final_estimator=final_estimator,
                    cv=5,
                    n_jobs=-1,
                    passthrough=False
                )
                self.classifier = final_stack
                return "堆叠集成模型"
        except Exception as e:
            print(f"堆叠模型评估出错: {e}")
            print("回退到最佳单一模型")
        
        # 如果堆叠模型失败或性能不足，选择单一最佳模型
        print(f"单一模型 {best_model_name} 表现最佳，已选择该模型")
        self.classifier = best_model
        return best_model_name

    def tune_hyperparameters(self, X, y):
        """对最佳模型进行超参数调优"""
        print("\n===== 超参数调优 =====")
        
        # 确保y是numpy数组
        y_np = y.values if hasattr(y, 'values') else np.array(y)
        
        # 如果是堆叠分类器，直接返回
        if isinstance(self.classifier, StackingClassifier):
            print("堆叠分类器不进行超参数调优")
            return self.classifier
        
        # 根据不同模型类型定义参数网格
        if isinstance(self.classifier, XGBClassifier):
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
            
        elif isinstance(self.classifier, LGBMClassifier):
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
            
        elif isinstance(self.classifier, RandomForestClassifier):
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 8, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
        elif isinstance(self.classifier, GradientBoostingClassifier):
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'subsample': [0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10]
            }
            
        elif isinstance(self.classifier, LogisticRegression):
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0],
                'solver': ['saga', 'liblinear'],
                'penalty': ['l1', 'l2'],
                'class_weight': ['balanced', None]
            }
            
        else:
            print("不支持的模型类型，跳过超参数调优")
            return self.classifier
        
        # 执行网格搜索
        grid_search = GridSearchCV(
            estimator=self.classifier,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        
        print("执行网格搜索以找到最佳参数，这可能需要一些时间...")
        grid_search.fit(X, y_np)
        
        # 输出结果
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳F1分数: {grid_search.best_score_:.4f}")
        
        # 更新分类器
        self.classifier = grid_search.best_estimator_
        
        return self.classifier

    def find_optimal_threshold(self, X, y):
        """找到最佳决策阈值以平衡精确率和召回率"""
        print("\n===== 寻找最佳决策阈值 =====")
        
        # 确保y是numpy数组
        y_np = y.values if hasattr(y, 'values') else np.array(y)
        
        # 获取预测概率
        if hasattr(self.classifier, 'predict_proba'):
            y_proba = self.classifier.predict_proba(X)[:, 1]
        else:
            print("当前模型不支持概率预测，无法优化阈值")
            return 0.5
        
        # 计算不同阈值下的精确率和召回率
        precisions, recalls, thresholds = precision_recall_curve(y_np, y_proba)
        
        # 计算F1分数
        f1_scores = []
        for i in range(len(precisions) - 1):
            p = precisions[i]
            r = recalls[i]
            if p + r == 0:
                f1 = 0
            else:
                f1 = 2 * (p * r) / (p + r)
            f1_scores.append(f1)
        
        # 找到F1最高的阈值
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        best_precision = precisions[best_idx]
        best_recall = recalls[best_idx]
        
        print(f"最佳阈值: {best_threshold:.4f}")
        print(f"最佳阈值下指标: 精确率={best_precision:.4f}, 召回率={best_recall:.4f}, F1分数={best_f1:.4f}")
        
        # 保存最佳阈值
        self.best_threshold = best_threshold
        
        return best_threshold

    def train(self, X, y, perform_tuning=True):
        """训练模型的完整流程"""
        # 确保y是numpy数组 - 解决索引错误
        y_np = y.values if hasattr(y, 'values') else np.array(y)
        
        # 特征选择
        if self.feature_selector is None and X.shape[1] > 20:  # 当特征较多时执行特征选择
            X = self.select_features(X, y_np)
        elif self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        
        # 处理类别不平衡
        X_balanced, y_balanced = self.handle_imbalance(X, y_np)
        
        # 选择最佳模型
        best_model = self.select_best_model(X_balanced, y_balanced)
        
        # 超参数调优
        if perform_tuning:
            self.tune_hyperparameters(X_balanced, y_balanced)
        
        # 用全部平衡后的数据训练最终模型
        print("\n===== 训练最终模型 =====")
        self.classifier.fit(X_balanced, y_balanced)
        
        # 寻找最佳决策阈值
        self.find_optimal_threshold(X, y_np)  # 使用原始数据集找最佳阈值
        
        return self.classifier

    def predict(self, X):
        """使用最佳阈值预测类别标签"""
        if hasattr(self.classifier, 'predict_proba'):
            proba = self.classifier.predict_proba(X)[:, 1]
            return (proba >= self.best_threshold).astype(int)
        else:
            return self.classifier.predict(X)

    def predict_proba(self, X):
        """预测概率"""
        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(X)[:, 1]
        else:
            return self.predict(X).astype(float)

    def evaluate(self, X, y):
        """评估模型性能"""
        # 确保y是numpy数组
        y_np = y.values if hasattr(y, 'values') else np.array(y)
        
        predictions = self.predict(X)
        predictions_proba = self.predict_proba(X)
        
        # 计算评估指标
        metrics = {
            'accuracy': accuracy_score(y_np, predictions),
            'precision': precision_score(y_np, predictions),
            'recall': recall_score(y_np, predictions),
            'f1': f1_score(y_np, predictions),
            'roc_auc': roc_auc_score(y_np, predictions_proba)
        }
        
        # 输出结果
        print("\n===== 模型评估结果 =====")
        for metric, value in metrics.items():
            print(f'{metric.capitalize()}: {value:.4f}')
        
        # 混淆矩阵
        cm = confusion_matrix(y_np, predictions)
        print("\n混淆矩阵:")
        print(cm)
        
        # 分类报告
        print("\n分类报告:")
        print(classification_report(y_np, predictions))
        
        return metrics


def main():
    # 原始分类特征
    categorical_features = ['Gender', 'Region', 'Payment_Method']
    
    # 创建预测器
    predictor = ChurnPredictor(categorical_features=categorical_features)

    # 加载数据
    train_data = predictor.load_data('dataset/train.csv')
    test_data = predictor.load_data('dataset/test.csv')
    
    # 数据分析
    predictor.analyze_data(train_data)
    
    # 特征工程
    train_data_fe = predictor.feature_engineering(train_data)
    test_data_fe = predictor.feature_engineering(test_data)
    
    # 分离特征与目标变量
    train_X = train_data_fe.drop(['Churned', 'Customer_ID'], axis=1)
    test_X = test_data_fe.drop(['Churned', 'Customer_ID'], axis=1)
    train_y = train_data_fe['Churned']
    test_y = test_data_fe['Churned']
    
    # 数据预处理
    train_X_processed = predictor.preprocess(train_X, is_train=True)
    test_X_processed = predictor.preprocess(test_X, is_train=False)
    
    # 训练模型
    predictor.train(train_X_processed, train_y, perform_tuning=True)
    
    # 测试集评估
    print("\n===== 测试集评估 =====")
    test_metrics = predictor.evaluate(test_X_processed, test_y)
    
    print("\n客户流失预测模型训练完成。")


if __name__ == '__main__':
    main()