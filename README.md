# 元宝APP用户增长分析项目 - 智能引导文案AB测试

## 项目背景
本项目模拟了元宝APP新用户引导文案的AB测试全流程，通过科学的实验设计和因果推断方法，探索提升新用户次日留存率的最优策略。

## 项目目标
- 核心指标：新用户次日留存率提升 ≥ 5%
- 实验方案：3版差异化引导文案（A:功能罗列、B:利益诱导、C:情感共鸣）
- 技术要点：AB测试、因果推断(PSM)、MAB动态调流、统计显著性检验

## 技术栈
- **SQL**：数据提取、用户行为日志清洗
- **Python**：数据预处理、PSM倾向性评分匹配
- **R语言**：统计检验、可视化、置信区间计算
- **统计学方法**：T检验、倾向性评分匹配、MAB算法原理

## 文件说明
├── sql/
│ └── experiment_analysis.sql # 实验数据提取SQL
├── python/
│ ├── psm_analysis.py # PSM因果推断代码
│ └── requirements.txt # Python依赖
├── r/
│ └── abtest_significance.R # R统计分析代码
└── docs/
└── project_details.md # 详细项目文档

## 核心结论
- **情感共鸣版(C版)** 显著提升次日留存率4.5% (p=0.01<0.05)
- **分群洞察**：在高龄用户群体中留存提升达8%
- **业务建议**：全量上线C版文案，并针对高龄用户设计深度运营策略

## 运行方法
### SQL执行
在Hive/MySQL中执行`sql/experiment_analysis.sql`

### Python运行
```bash
cd python
pip install -r requirements.txt
python psm_analysis.py
