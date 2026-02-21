---
name: fund-screener
description: 中国公募基金全市场筛选与深度分析工具，支持夏普/索提诺/卡玛等量化指标筛选
user-invocable: yes
metadata:
  {
    "openclaw":
      {
        "emoji": "📊",
        "os": ["darwin", "linux", "win32"],
        "requires": {
          "anyBins": ["python3", "python"]
        },
        "install": [
          {
            "id": "pip",
            "kind": "uv",
            "package": "akshare pandas numpy",
            "bins": ["python3"],
            "label": "pip install akshare pandas numpy"
          }
        ]
      }
  }
---

# 基金分析技能 (Fund Screener)

通过 AkShare 获取中国公募基金实时数据，基于量化指标筛选优质基金产品。

## 功能

1. **单基金深度分析**: 对单只基金进行全方位诊断，包括风险指标、阶段收益、年度收益、资产配置、基金经理、持仓信息
2. **全市场批量筛选**: 从全市场开放式基金中智能筛选，支持多种模式
3. **结果导出**: 自动保存 CSV 文件

## 使用方法

运行此技能需要 Python 3.8+ 和依赖: `pip install akshare pandas numpy`

```bash
# 单基金分析
python fund_screener.py <基金代码> [基金名称]

# 纯债基金筛选（低风险）
python fund_screener.py --bond

# 固收+基金筛选
python fund_screener.py --gushou-plus

# 股票类基金筛选
python fund_screener.py --stock

# Alpha策略筛选（卡玛比率）
python fund_screener.py --stock-alpha

# 自定义参数
python fund_screener.py --bond --min-sharpe 1.0 --max-dd 2 --min-return 2.5
python fund_screener.py --gushou-plus --min-sortino 1.5
python fund_screener.py --stock-alpha --min-calmar 0.8 --min-return 10

# 数量控制
python fund_screener.py --bond --max 50
```

## 筛选模式与指标

| 模式 | 参数 | 核心指标 | 默认筛选标准 |
|------|------|----------|------------|
| 纯债基金 | `--bond` | 夏普比率 | 夏普>=1.0, 回撤<2%, 收益>2.5% |
| 固收+ | `--gushou-plus` | 索提诺比率 | 夏普>=0.8, 索提诺>=1.2, 回撤<5%, 收益>3.5% |
| 股票Alpha | `--stock-alpha` | 卡玛比率 | 卡玛>=1.2, 回撤<30%, 收益>10% |
| 默认/股票 | 无/`--stock` | 夏普比率 | 夏普>=0.5, 回撤<15%, 收益>3% |

## 自定义参数

| 参数 | 说明 |
|------|------|
| `--min-sharpe <value>` | 最小夏普比率 |
| `--min-sortino <value>` | 最小索提诺比率 |
| `--min-calmar <value>` | 最小卡玛比率 |
| `--max-dd <value>` | 最大回撤百分比（正数，如5表示<5%） |
| `--min-return <value>` | 最小年化收益率(%) |
| `--max <value>` | 最大分析数量（50-500） |

## 作为 Python 模块使用

```python
from fund_screener import get_fund_metrics, analyze_single_fund, calculate_sharpe_ratio

# 分析单只基金
metrics = get_fund_metrics('004010', '华泰柏瑞鼎利混合A')

# 深度分析
analysis = analyze_single_fund('004010')
```

## 数据来源

- 基金净值: 东方财富网 (via AkShare)
- 基金经理/规模: 基金经理持仓数据
- 资产配置: 基金季度持仓报告

## 风险提示

历史业绩不代表未来表现。本工具仅供学习研究，不构成投资建议。
