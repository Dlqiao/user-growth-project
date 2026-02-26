
---

## 文件2: `sql/experiment_analysis.sql`

```sql
/*
 * 元宝APP引导文案AB测试 - 数据提取SQL
 * 功能：关联实验分组与次日留存行为，计算各组留存率
 * 数据表说明：
 *   - experiment_exposure: 实验曝光表（user_id, variant, exp_date）
 *   - user_daily_activity: 用户活跃表（user_id, active_date）
 */

WITH 
-- 1. 提取实验期内曝光用户及其分组
exp_data AS (
  SELECT 
    user_id,
    variant,
    DATE(log_date) AS exp_date
  FROM experiment_exposure
  WHERE exp_id = 'homepage_guide_v1' 
    AND log_date BETWEEN '2024-01-01' AND '2024-01-07'
),

-- 2. 提取次日留存数据
retention_data AS (
  SELECT 
    user_id,
    DATE(log_date) AS active_date
  FROM user_daily_activity
  WHERE log_date BETWEEN '2024-01-02' AND '2024-01-08'
)

-- 3. 关联计算留存率
SELECT 
  e.variant,
  COUNT(DISTINCT e.user_id) AS exposed_users,
  COUNT(DISTINCT r.user_id) AS retained_users,
  ROUND(COUNT(DISTINCT r.user_id) / COUNT(DISTINCT e.user_id), 4) AS retention_rate,
  -- 计算置信区间 (Wilson Score)
  ROUND( 
    (COUNT(DISTINCT r.user_id) + 1.96^2/2) / (COUNT(DISTINCT e.user_id) + 1.96^2) 
    - 1.96 * SQRT( 
      (COUNT(DISTINCT r.user_id) * (COUNT(DISTINCT e.user_id) - COUNT(DISTINCT r.user_id))) / COUNT(DISTINCT e.user_id) 
      + 1.96^2/4 
    ) / (COUNT(DISTINCT e.user_id) + 1.96^2), 4
  ) AS ci_lower,
  ROUND(
    (COUNT(DISTINCT r.user_id) + 1.96^2/2) / (COUNT(DISTINCT e.user_id) + 1.96^2)
    + 1.96 * SQRT(
      (COUNT(DISTINCT r.user_id) * (COUNT(DISTINCT e.user_id) - COUNT(DISTINCT r.user_id))) / COUNT(DISTINCT e.user_id)
      + 1.96^2/4
    ) / (COUNT(DISTINCT e.user_id) + 1.96^2), 4
  ) AS ci_upper
FROM exp_data e
LEFT JOIN retention_data r 
  ON e.user_id = r.user_id 
  AND r.active_date = DATE_ADD(e.exp_date, 1)
GROUP BY e.variant
ORDER BY retention_rate DESC;
