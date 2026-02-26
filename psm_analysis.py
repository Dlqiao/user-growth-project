"""
元宝APP引导文案AB测试 - PSM倾向性评分匹配
功能：消除用户特征差异导致的混杂偏差，准确估计处理效应
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

class PSM_Analyzer:
    """倾向性评分匹配分析类"""
    
    def __init__(self, df, treatment_col, outcome_col, feature_cols):
        """
        初始化
        
        Parameters:
        -----------
        df : DataFrame
            包含用户特征、分组和结果的数据
        treatment_col : str
            实验组标识列名 (1=实验组, 0=对照组)
        outcome_col : str
            结果变量列名 (1=留存, 0=未留存)
        feature_cols : list
            用于匹配的用户特征列名
        """
        self.df = df.copy()
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.feature_cols = feature_cols
        self.ps_model = None
        self.matched_df = None
        
    def fit_propensity_score(self):
        """计算倾向性得分 (用户被分到实验组的概率)"""
        X = self.df[self.feature_cols]
        y = self.df[self.treatment_col]
        
        # 处理缺失值
        X = X.fillna(X.mean())
        
        # 训练逻辑回归模型
        self.ps_model = LogisticRegression(random_state=42, max_iter=1000)
        self.ps_model.fit(X, y)
        
        # 预测倾向性得分
        self.df['propensity_score'] = self.ps_model.predict_proba(X)[:, 1]
        
        print("倾向性得分计算完成")
        print(f"实验组平均倾向得分: {self.df[self.df[self.treatment_col]==1]['propensity_score'].mean():.4f}")
        print(f"对照组平均倾向得分: {self.df[self.df[self.treatment_col]==0]['propensity_score'].mean():.4f}")
        
        return self
    
    def match(self, caliper=0.05, ratio=1):
        """
        最近邻匹配
        
        Parameters:
        -----------
        caliper : float
            允许的最大倾向得分差异 (标准差倍数)
        ratio : int
            每个实验组样本匹配的对照组样本数
        """
        treated = self.df[self.df[self.treatment_col] == 1].copy()
        control = self.df[self.df[self.treatment_col] == 0].copy()
        
        # 标准化倾向得分
        ps_mean = self.df['propensity_score'].mean()
        ps_std = self.df['propensity_score'].std()
        
        treated['ps_scaled'] = (treated['propensity_score'] - ps_mean) / ps_std
        control['ps_scaled'] = (control['propensity_score'] - ps_mean) / ps_std
        
        # 使用KNN进行匹配
        nn = NearestNeighbors(n_neighbors=ratio, metric='euclidean')
        nn.fit(control[['ps_scaled']])
        
        distances, indices = nn.kneighbors(treated[['ps_scaled']])
        
        # 应用卡尺
        mask = distances <= caliper
        matched_indices = indices[mask]
        
        # 构建匹配后的数据集
        matched_control = control.iloc[matched_indices.flatten()].drop_duplicates()
        self.matched_df = pd.concat([treated, matched_control])
        
        print(f"匹配完成:")
        print(f"  实验组样本量: {len(treated)}")
        print(f"  匹配后对照组样本量: {len(matched_control)}")
        
        return self
    
    def calculate_ate(self):
        """计算平均处理效应 (ATE)"""
        if self.matched_df is None:
            raise ValueError("请先运行match()方法进行匹配")
        
        # 匹配后
        treated_outcome = self.matched_df[self.matched_df[self.treatment_col]==1][self.outcome_col].mean()
        control_outcome = self.matched_df[self.matched_df[self.treatment_col]==0][self.outcome_col].mean()
        
        # 匹配前
        treated_outcome_before = self.df[self.df[self.treatment_col]==1][self.outcome_col].mean()
        control_outcome_before = self.df[self.df[self.treatment_col]==0][self.outcome_col].mean()
        
        results = {
            '匹配前处理效应': treated_outcome_before - control_outcome_before,
            '匹配后处理效应': treated_outcome - control_outcome,
            '匹配前实验组留存率': treated_outcome_before,
            '匹配前对照组留存率': control_outcome_before,
            '匹配后实验组留存率': treated_outcome,
            '匹配后对照组留存率': control_outcome
        }
        
        return results
    
    def plot_ps_distribution(self):
        """绘制匹配前后的倾向得分分布"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 匹配前
        sns.kdeplot(
            data=self.df[self.df[self.treatment_col]==1], 
            x='propensity_score', 
            label='实验组', 
            ax=axes[0]
        )
        sns.kdeplot(
            data=self.df[self.df[self.treatment_col]==0], 
            x='propensity_score', 
            label='对照组', 
            ax=axes[0]
        )
        axes[0].set_title('匹配前倾向得分分布')
        axes[0].legend()
        
        # 匹配后
        if self.matched_df is not None:
            sns.kdeplot(
                data=self.matched_df[self.matched_df[self.treatment_col]==1], 
                x='propensity_score', 
                label='实验组', 
                ax=axes[1]
            )
            sns.kdeplot(
                data=self.matched_df[self.matched_df[self.treatment_col]==0], 
                x='propensity_score', 
                label='对照组', 
                ax=axes[1]
            )
            axes[1].set_title('匹配后倾向得分分布')
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('psm_distribution.png', dpi=100)
        plt.show()


# 示例数据生成和运行
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 10000
    
    # 用户特征
    data = {
        'user_id': range(n_samples),
        'age': np.random.normal(35, 10, n_samples).astype(int),
        'device_price': np.random.choice([1000, 2000, 3000, 5000], n_samples),
        'channel_code': np.random.choice([1, 2, 3, 4], n_samples),
        'previous_visits': np.random.poisson(5, n_samples),
        'is_treatment': np.random.binomial(1, 0.5, n_samples),
        'retained': np.random.binomial(1, 0.15, n_samples)  # 基准留存率15%
    }
    
    df = pd.DataFrame(data)
    
    # 添加处理效应 (实验组额外提升)
    df.loc[df['is_treatment'] == 1, 'retained'] = np.random.binomial(1, 0.20, sum(df['is_treatment']==1))
    
    # 运行PSM分析
    psm = PSM_Analyzer(
        df=df,
        treatment_col='is_treatment',
        outcome_col='retained',
        feature_cols=['age', 'device_price', 'channel_code', 'previous_visits']
    )
    
    psm.fit_propensity_score()
    psm.match(caliper=0.2)
    results = psm.calculate_ate()
    
    print("\n" + "="*50)
    print("PSM分析结果")
    print("="*50)
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
    
    psm.plot_ps_distribution()