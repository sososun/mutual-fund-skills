"""
基金分析技能 (Mutual Fund Skills)

基于 Python + AkShare 的全市场基金分析工具
"""

from .fund_screener import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_annualized_return,
    get_fund_metrics,
    get_fund_scale,
    get_fund_asset_allocation,
    analyze_funds,
    filter_quality_funds,
    get_default_fund_pool,
    main
)

__version__ = '1.0.0'
__all__ = [
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'calculate_annualized_return',
    'get_fund_metrics',
    'get_fund_scale',
    'get_fund_asset_allocation',
    'analyze_funds',
    'filter_quality_funds',
    'get_default_fund_pool',
    'main'
]
