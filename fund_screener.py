#!/usr/bin/env python3
"""
基金分析技能 - 核心代码
基于 AkShare 的全市场基金分析工具
"""

import akshare as ak
import pandas as pd
import numpy as np
import warnings
import time
from typing import Dict, Optional, List, Tuple
warnings.filterwarnings('ignore')

# 默认无风险利率（年化）
RISK_FREE_RATE = 0.025


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """
    计算夏普比率
    
    Args:
        returns: 日收益率序列
        risk_free_rate: 无风险利率（年化）
    
    Returns:
        夏普比率，值越大表示风险调整后收益越好
    """
    if len(returns) < 2 or returns.std() == 0:
        return np.nan
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / returns.std()


def calculate_max_drawdown(nav_values: pd.Series) -> float:
    """
    计算最大回撤
    
    Args:
        nav_values: 净值序列
    
    Returns:
        最大回撤比例（负数），如 -0.15 表示最大回撤15%
    """
    if len(nav_values) < 2:
        return np.nan
    peak = nav_values.expanding(min_periods=1).max()
    drawdown = (nav_values - peak) / peak
    return drawdown.min()


def calculate_annualized_return(nav_values: pd.Series, days: int = 252) -> float:
    """
    计算年化收益率
    
    Args:
        nav_values: 净值序列
        days: 每年交易日数量
    
    Returns:
        年化收益率
    """
    if len(nav_values) < 2:
        return np.nan
    total_return = (nav_values.iloc[-1] / nav_values.iloc[0]) - 1
    n_years = len(nav_values) / days
    if n_years <= 0:
        return np.nan
    return (1 + total_return) ** (1 / n_years) - 1


def get_fund_metrics(fund_code: str, fund_name: str, min_days: int = 500) -> Optional[Dict]:
    """
    获取基金风险指标
    
    Args:
        fund_code: 基金代码
        fund_name: 基金名称
        min_days: 最小数据天数要求
    
    Returns:
        包含各项指标的字典，失败返回 None
    """
    try:
        df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
        
        if df is None or len(df) < min_days:
            return None
            
        df['净值日期'] = pd.to_datetime(df['净值日期'])
        df = df.sort_values('净值日期')
        df['日收益率'] = df['单位净值'].pct_change()
        
        returns = df['日收益率'].dropna()
        sharpe = calculate_sharpe_ratio(returns)
        max_dd = calculate_max_drawdown(df['单位净值'])
        annual_return = calculate_annualized_return(df['单位净值'])
        volatility = returns.std() * np.sqrt(252)
        
        # 计算各期收益
        nav_now = df['单位净值'].iloc[-1]
        return_1y = (nav_now / df['单位净值'].iloc[-252] - 1) * 100 if len(df) >= 252 else None
        return_2y = (nav_now / df['单位净值'].iloc[-504] - 1) * 100 / 2 if len(df) >= 504 else None
        
        return {
            '基金代码': fund_code,
            '基金名称': fund_name,
            '夏普比率': round(sharpe, 2) if not np.isnan(sharpe) else None,
            '最大回撤(%)': round(max_dd * 100, 2) if not np.isnan(max_dd) else None,
            '年化收益率(%)': round(annual_return * 100, 2) if not np.isnan(annual_return) else None,
            '年化波动率(%)': round(volatility * 100, 2) if not np.isnan(volatility) else None,
            '近1年收益(%)': round(return_1y, 2) if return_1y else None,
            '近2年年化(%)': round(return_2y, 2) if return_2y else None,
            '数据天数': len(df)
        }
    except Exception as e:
        return None


def get_fund_scale(fund_code: str) -> Dict:
    """
    获取基金规模
    
    Args:
        fund_code: 基金代码
    
    Returns:
        包含基金规模的字典
    """
    try:
        # 缓存机制避免重复请求
        if not hasattr(get_fund_scale, 'manager_cache'):
            get_fund_scale.manager_cache = None
            get_fund_scale.cache_time = 0
        
        import time as time_module
        if get_fund_scale.manager_cache is None or time_module.time() - get_fund_scale.cache_time > 600:
            get_fund_scale.manager_cache = ak.fund_manager_em()
            get_fund_scale.cache_time = time_module.time()
        
        mgr_df = get_fund_scale.manager_cache
        fund_info = mgr_df[mgr_df['现任基金代码'] == fund_code]
        
        if len(fund_info) > 0:
            scale_str = fund_info.iloc[0]['现任基金资产总规模']
            return {'基金规模(亿元)': scale_str}
    except:
        pass
    return {'基金规模(亿元)': None}


def get_fund_asset_allocation(fund_code: str) -> Dict:
    """
    获取基金资产配置（从股票持仓推算）
    
    Args:
        fund_code: 基金代码
    
    Returns:
        包含股票仓位、债券仓位的字典
    """
    try:
        hold_df = ak.fund_portfolio_hold_em(symbol=fund_code, date="")
        if hold_df is not None and len(hold_df) > 0:
            latest_quarter = hold_df['季度'].iloc[0]
            latest_hold = hold_df[hold_df['季度'] == latest_quarter]
            
            # 计算股票总仓位
            stock_ratio = latest_hold['占净值比例'].astype(float).sum()
            bond_ratio = max(0, 100 - stock_ratio - 5)
            
            return {
                '股票仓位(%)': round(stock_ratio, 1),
                '估算债券仓位(%)': round(bond_ratio, 1),
                '报告期': latest_quarter
            }
    except:
        pass
    return {'股票仓位(%)': None, '估算债券仓位(%)': None, '报告期': None}


def analyze_funds(fund_list: List[Tuple[str, str, str]], 
                  min_sharpe: float = 0.3,
                  max_drawdown: float = -25,
                  min_return: float = 2,
                  delay: float = 0.3) -> pd.DataFrame:
    """
    批量分析基金
    
    Args:
        fund_list: 基金列表，格式 [(代码, 名称, 类型), ...]
        min_sharpe: 最小夏普比率
        max_drawdown: 最大回撤限制（负数）
        min_return: 最小年化收益率
        delay: 请求间隔（秒）
    
    Returns:
        分析结果 DataFrame
    """
    print(f"开始分析 {len(fund_list)} 只基金...")
    print("-" * 100)
    
    results = []
    for idx, (code, name, ftype) in enumerate(fund_list, 1):
        print(f"[{idx:3d}/{len(fund_list)}] {code} {name[:30]}", end=" ")
        
        metrics = get_fund_metrics(code, name)
        if not metrics:
            print("✗ 无数据")
            continue
        
        metrics['类型'] = ftype
        
        # 获取规模（每5个基金请求一次）
        if idx % 5 == 1:
            scale_info = get_fund_scale(code)
            metrics.update(scale_info)
        else:
            metrics['基金规模(亿元)'] = None
        
        # 获取资产配置
        asset_info = get_fund_asset_allocation(code)
        metrics.update(asset_info)
        
        results.append(metrics)
        print(f"✓ 夏普:{metrics['夏普比率']:.2f} 回撤:{metrics['最大回撤(%)']:.1f}%")
        
        time.sleep(delay)
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    df = df.sort_values('夏普比率', ascending=False)
    
    return df


def filter_quality_funds(df: pd.DataFrame, 
                         min_sharpe: float = 0.3,
                         max_drawdown: float = -25,
                         min_return: float = 2) -> pd.DataFrame:
    """
    筛选优质基金
    
    Args:
        df: 分析结果 DataFrame
        min_sharpe: 最小夏普比率
        max_drawdown: 最大回撤限制
        min_return: 最小年化收益率
    
    Returns:
        筛选后的 DataFrame
    """
    return df[
        (df['夏普比率'] >= min_sharpe) & 
        (df['最大回撤(%)'] >= max_drawdown) &
        (df['年化收益率(%)'] > min_return)
    ].copy()


def get_default_fund_pool() -> List[Tuple[str, str, str]]:
    """
    获取默认基金池
    
    Returns:
        基金列表 [(代码, 名称, 类型), ...]
    """
    return [
        # 华泰柏瑞
        ('004010', '华泰柏瑞鼎利混合A', '偏债混合'),
        ('004011', '华泰柏瑞鼎利混合C', '偏债混合'),
        ('001822', '华泰柏瑞惠利混合A', '偏债混合'),
        ('002340', '华泰柏瑞享利混合A', '偏债混合'),
        
        # 南方
        ('002015', '南方荣光A', '偏债混合'),
        ('014681', '南方誉稳一年持有混合A', '偏债混合'),
        ('016927', '南方誉泰稳健6个月持有混合A', '偏债混合'),
        
        # 华夏
        ('000047', '华夏鼎泓债券A', '二级债基'),
        ('002459', '华夏鼎利债券A', '二级债基'),
        ('000121', '华夏永福混合A', '偏债混合'),
        
        # 招商
        ('002657', '招商安本增利债券C', '二级债基'),
        ('217008', '招商安本增利债券A', '二级债基'),
        ('003859', '招商招旭纯债A', '纯债'),
        
        # 易方达
        ('000171', '易方达裕丰回报债券', '二级债基'),
        ('002351', '易方达裕祥回报债券', '二级债基'),
        ('001316', '安信稳健增值混合A', '偏债混合'),
        ('001182', '易方达安心回馈混合', '偏债混合'),
        ('110007', '易方达稳健收益债券A', '二级债基'),
        ('110027', '易方达安心回报债券A', '二级债基'),
        
        # 景顺长城
        ('000385', '景顺长城景颐双利债券A', '二级债基'),
        ('000386', '景顺长城景颐双利债券C', '二级债基'),
        
        # 广发
        ('270002', '广发稳健增长混合A', '偏债混合'),
        ('270006', '广发策略优选混合', '偏债混合'),
        ('000215', '广发趋势优选灵活配置混合A', '偏债混合'),
        
        # 富国
        ('100035', '富国优化增强债券A', '二级债基'),
        ('100036', '富国优化增强债券B', '二级债基'),
        
        # 交银施罗德
        ('004225', '交银增利增强债券A', '二级债基'),
        ('519682', '交银增利债券A', '二级债基'),
        ('519732', '交银定期支付双息平衡混合', '股债平衡'),
        
        # 工银瑞信
        ('485111', '工银瑞信双利债券A', '二级债基'),
        ('485011', '工银瑞信双利债券B', '二级债基'),
        
        # 博时
        ('050011', '博时信用债券A', '二级债基'),
        ('050111', '博时信用债券C', '二级债基'),
        
        # 鹏华
        ('000297', '鹏华可转债债券A', '可转债'),
        ('206013', '鹏华宏泰灵活配置混合', '偏债混合'),
        ('002018', '鹏华弘盛混合A', '偏债混合'),
        
        # 嘉实
        ('070009', '嘉实超短债债券C', '纯债'),
        ('000009', '嘉实超短债债券', '纯债'),
        
        # 其他
        ('002961', '蜂巢恒利债券A', '二级债基'),
        ('002440', '金鹰鑫瑞混合A', '偏债混合'),
        ('005833', '华泰保兴尊合债券A', '二级债基'),
        ('001711', '安信新趋势混合A', '偏债混合'),
        ('000190', '中银新回报混合A', '偏债混合'),
        ('002364', '华安安康灵活配置混合A', '偏债混合'),
        ('110017', '易方达增强回报债券A', '二级债基'),
        ('001717', '天弘精选混合', '偏债混合'),
        ('519069', '汇添富价值精选混合A', '偏债混合'),
        ('166006', '中欧行业成长混合(LOF)A', '偏债混合'),
        ('166005', '中欧价值发现混合A', '偏债混合'),
        ('163406', '兴全合润混合(LOF)', '偏债混合'),
        ('163407', '兴全合宜混合(LOF)A', '偏债混合'),
    ]


def get_fund_manager_info(fund_code: str) -> Dict:
    """获取基金经理信息"""
    try:
        if not hasattr(get_fund_manager_info, 'manager_cache'):
            get_fund_manager_info.manager_cache = None
            get_fund_manager_info.cache_time = 0
        
        import time as time_module
        if get_fund_manager_info.manager_cache is None or time_module.time() - get_fund_manager_info.cache_time > 600:
            get_fund_manager_info.manager_cache = ak.fund_manager_em()
            get_fund_manager_info.cache_time = time_module.time()
        
        mgr_df = get_fund_manager_info.manager_cache
        fund_info = mgr_df[mgr_df['现任基金代码'] == fund_code]
        
        if len(fund_info) > 0:
            info = fund_info.iloc[0]
            years = int(info['累计从业时间']) / 365
            return {
                '基金经理': info['姓名'],
                '所属公司': info['所属公司'],
                '从业年限': f"{years:.1f}年",
                '管理规模': f"{info['现任基金资产总规模']}亿元",
                '最佳回报': f"{info['现任基金最佳回报']}%"
            }
    except:
        pass
    return {}


def get_fund_holding(fund_code: str) -> Dict:
    """获取基金持仓信息"""
    try:
        hold_df = ak.fund_portfolio_hold_em(symbol=fund_code, date="")
        if hold_df is not None and len(hold_df) > 0:
            latest_quarter = hold_df['季度'].iloc[0]
            latest_hold = hold_df[hold_df['季度'] == latest_quarter]
            
            # 计算股票总仓位
            stock_ratio = latest_hold['占净值比例'].astype(float).sum()
            
            return {
                '报告期': latest_quarter,
                '持仓数量': len(latest_hold),
                '股票仓位': f"{stock_ratio:.1f}%",
                '前10大重仓': latest_hold.head(10)[['股票代码', '股票名称', '占净值比例']].to_dict('records')
            }
    except:
        pass
    return {}


def analyze_single_fund(fund_code: str, fund_name: Optional[str] = None) -> Dict:
    """
    深度分析单个基金
    
    Args:
        fund_code: 基金代码
        fund_name: 基金名称（可选，如果不提供会自动查找）
    
    Returns:
        完整的基金分析报告字典
    """
    # 如果未提供基金名称，尝试查找
    if fund_name is None:
        try:
            fund_list = ak.fund_open_fund_daily_em()
            info = fund_list[fund_list['基金代码'] == fund_code]
            if len(info) > 0:
                fund_name = info.iloc[0]['基金简称']
            else:
                fund_name = fund_code
        except:
            fund_name = fund_code
    
    # 获取最新净值信息
    try:
        fund_list = ak.fund_open_fund_daily_em()
        info = fund_list[fund_list['基金代码'] == fund_code]
        if len(info) > 0:
            latest_nav = info.iloc[0].get('2026-02-13-单位净值', 'N/A')
            accum_nav = info.iloc[0].get('2026-02-13-累计净值', 'N/A')
            daily_change = info.iloc[0].get('日增长率', 'N/A')
            fee = info.iloc[0].get('手续费', 'N/A')
            purchase_status = info.iloc[0].get('申购状态', 'N/A')
            redeem_status = info.iloc[0].get('赎回状态', 'N/A')
        else:
            latest_nav = accum_nav = daily_change = fee = purchase_status = redeem_status = 'N/A'
    except:
        latest_nav = accum_nav = daily_change = fee = purchase_status = redeem_status = 'N/A'
    
    # 获取历史数据用于计算各项指标
    try:
        df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
        if df is not None and len(df) >= 250:
            df['净值日期'] = pd.to_datetime(df['净值日期'])
            df = df.sort_values('净值日期')
            df['日收益率'] = df['单位净值'].pct_change()
            
            # 计算风险指标
            returns = df['日收益率'].dropna()
            sharpe = calculate_sharpe_ratio(returns)
            max_dd = calculate_max_drawdown(df['单位净值'])
            annual_return = calculate_annualized_return(df['单位净值'])
            volatility = returns.std() * np.sqrt(252)
            
            # 计算阶段收益
            nav_now = df['单位净值'].iloc[-1]
            periods = {}
            if len(df) >= 252:
                periods['近1年'] = round((nav_now / df['单位净值'].iloc[-252] - 1) * 100, 2)
            if len(df) >= 504:
                periods['近2年'] = round((nav_now / df['单位净值'].iloc[-504] - 1) * 100 / 2, 2)
            if len(df) >= 756:
                periods['近3年'] = round((nav_now / df['单位净值'].iloc[-756] - 1) * 100 / 3, 2)
            
            # 年度收益
            df['年份'] = df['净值日期'].dt.year
            annual_returns = {}
            for year in sorted(df['年份'].unique())[-5:]:
                year_df = df[df['年份'] == year]
                if len(year_df) > 50:
                    year_return = (year_df['单位净值'].iloc[-1] / year_df['单位净值'].iloc[0] - 1) * 100
                    annual_returns[str(year)] = round(year_return, 2)
        else:
            sharpe = max_dd = annual_return = volatility = None
            periods = {}
            annual_returns = {}
    except:
        sharpe = max_dd = annual_return = volatility = None
        periods = {}
        annual_returns = {}
    
    # 获取资产配置
    allocation = get_fund_asset_allocation(fund_code)
    
    # 获取基金经理信息
    manager = get_fund_manager_info(fund_code)
    
    # 获取持仓信息
    holding = get_fund_holding(fund_code)
    
    return {
        '基本信息': {
            '基金代码': fund_code,
            '基金名称': fund_name,
            '最新净值': latest_nav,
            '累计净值': accum_nav,
            '日增长率': daily_change,
            '手续费': fee,
            '申购状态': purchase_status,
            '赎回状态': redeem_status
        },
        '风险指标': {
            '夏普比率': round(sharpe, 2) if sharpe is not None else None,
            '最大回撤(%)': round(max_dd * 100, 2) if max_dd is not None else None,
            '年化收益率(%)': round(annual_return * 100, 2) if annual_return is not None else None,
            '年化波动率(%)': round(volatility * 100, 2) if volatility is not None else None
        },
        '阶段收益': periods,
        '年度收益': annual_returns,
        '资产配置': allocation,
        '基金经理': manager,
        '持仓信息': holding
    }


def print_fund_analysis(analysis: Dict):
    """打印基金分析报告"""
    print("=" * 80)
    print(" " * 25 + "基金深度分析报告")
    print("=" * 80)
    print()
    
    # 1. 基本信息
    print("【1. 基本信息】")
    print("-" * 80)
    basic = analysis['基本信息']
    for key, value in basic.items():
        print(f"{key}: {value}")
    print()
    
    # 2. 风险指标
    if analysis['风险指标'].get('夏普比率'):
        print("【2. 风险指标】")
        print("-" * 80)
        risk = analysis['风险指标']
        for key, value in risk.items():
            if value is not None:
                print(f"{key}: {value}")
        
        # 风险评级
        sharpe = risk.get('夏普比率', 0)
        max_dd = risk.get('最大回撤(%)', 0)
        if sharpe > 1.0 and max_dd > -10:
            risk_level = '低风险（优秀）'
        elif sharpe > 0.5 and max_dd > -20:
            risk_level = '中等风险（良好）'
        else:
            risk_level = '较高风险（需谨慎）'
        print(f"风险评级: {risk_level}")
        print()
    
    # 3. 阶段收益
    if analysis['阶段收益']:
        print("【3. 阶段收益】")
        print("-" * 80)
        for period, return_pct in analysis['阶段收益'].items():
            print(f"{period}: {return_pct}%")
        print()
    
    # 4. 年度收益
    if analysis['年度收益']:
        print("【4. 年度收益表现】")
        print("-" * 80)
        for year, return_pct in analysis['年度收益'].items():
            print(f"{year}年: {return_pct:+.2f}%")
        print()
    
    # 5. 资产配置
    if analysis['资产配置']:
        print("【5. 资产配置】")
        print("-" * 80)
        alloc = analysis['资产配置']
        for key, value in alloc.items():
            if value is not None:
                print(f"{key}: {value}")
        print()
    
    # 6. 基金经理
    if analysis['基金经理']:
        print("【6. 基金经理】")
        print("-" * 80)
        mgr = analysis['基金经理']
        for key, value in mgr.items():
            print(f"{key}: {value}")
        print()
    
    # 7. 持仓信息
    if analysis['持仓信息']:
        print("【7. 持仓信息】")
        print("-" * 80)
        hold = analysis['持仓信息']
        print(f"报告期: {hold.get('报告期', 'N/A')}")
        print(f"持仓数量: {hold.get('持仓数量', 'N/A')}只")
        print(f"股票仓位: {hold.get('股票仓位', 'N/A')}")
        
        if hold.get('前10大重仓'):
            print()
            print("前10大重仓股:")
            for i, stock in enumerate(hold['前10大重仓'][:10], 1):
                print(f"  {i}. {stock['股票名称']}({stock['股票代码']}) - {stock['占净值比例']}%")
        print()
    
    # 8. 综合评价
    print("【8. 综合评价】")
    print("-" * 80)
    
    # 优点
    print("✅ 优点:")
    stock_pos = analysis['资产配置'].get('股票仓位(%)')
    if stock_pos is not None and stock_pos < 20:
        print(f"  • 股票仓位适中({stock_pos}%)，风险可控")
    bond_pos = analysis['资产配置'].get('估算债券仓位(%)')
    if bond_pos is not None and bond_pos > 70:
        print(f"  • 债券仓位较高({bond_pos}%)，收益稳定")
    sharpe = analysis['风险指标'].get('夏普比率')
    if sharpe is not None and sharpe > 0.5:
        print(f"  • 夏普比率良好({sharpe})，风险调整后收益较好")
    
    # 注意
    print()
    print("⚠️  注意:")
    max_dd = analysis['风险指标'].get('最大回撤(%)')
    if max_dd is not None and max_dd < -15:
        print(f"  • 最大回撤{max_dd}%较大，需关注风险控制")
    if sharpe is not None and sharpe < 0.5:
        print(f"  • 夏普比率{sharpe}处于中等水平，风险调整后收益一般")
    
    print()
    print("=" * 80)


def get_all_gushou_funds(max_funds: int = 200) -> List[Tuple[str, str, str]]:
    """
    从所有基金中智能筛选优质基金
    
    筛选逻辑：
    1. 从所有开放式基金中筛选
    2. 排除明显股票型、指数型、行业主题型
    3. 保留债券型、稳健型、混合型中风险较低的
    4. 按近1年收益率排序，优先分析表现稳定的
    
    Args:
        max_funds: 最大筛选数量（默认200，避免耗时过长）
    
    Returns:
        基金列表 [(代码, 名称, 类型), ...]
    """
    print("\n正在从全市场基金中智能筛选优质产品...")
    print("-" * 100)
    
    try:
        # 获取所有开放式基金排名
        fund_rank = ak.fund_open_fund_rank_em()
        total = len(fund_rank)
        print(f"✓ 获取到 {total} 只开放式基金")
        
        # 第一步：排除明显股票型/行业主题型基金
        exclude_keywords = [
            '股票', '指数', 'ETF', 'ETF联接', 'LOF', '分级', '医药', '医疗', '科技', 
            '新能源', '半导体', '芯片', '军工', '传媒', '光伏', '白酒', '消费', 
            '制造', '高端制造', '先进制造', '材料', '化工', '有色', '稀土', '煤炭',
            '钢铁', '地产', '银行', '证券', '保险', '金融', '互联网', '传媒', '游戏',
            '生物医药', '医疗器械', '电子', '计算机', '通信', '5G', '人工智能',
            '大数据', '云计算', '新能源', '电动车', '新能源汽车', '碳中和',
            '价值', '成长', '红利', '量化', '对冲', '绝对收益'
        ]
        
        # 先排除明显股票型和行业主题型
        filtered = fund_rank[
            ~fund_rank['基金简称'].str.contains('|'.join(exclude_keywords), na=False, case=False)
        ].copy()
        
        print(f"✓ 排除股票型/行业主题型后剩余: {len(filtered)} 只")
        
        # 第二步：筛选稳健型基金特征
        include_keywords = ['债', '债券', '稳健', '增利', '鼎', '丰', '裕', '兴', 
                           '益', '瑞', '祥', '安', '合', '享', '顺', '优', '增强', 
                           '回报', '精选', '添', '稳', '利', '盈', '富', '悦', '泰',
                           '恒', '盛', '荣', '华', '嘉', '悦', '怡', '和', '康',
                           '宁', '静', '怡', '乐', '悦', '欣', '嘉', '祥', '福',
                           '双债', '强债', '信用债', '可转债', '纯债']
        
        # 包含稳健型基金特征的基金
        gushou_funds = filtered[
            filtered['基金简称'].str.contains('|'.join(include_keywords), na=False, case=False)
        ].copy()
        
        print(f"✓ 筛选出稳健型特征基金: {len(gushou_funds)} 只")
        
        # 第三步：通过收益率筛选，保留风险适中的
        gushou_funds['近1年'] = pd.to_numeric(gushou_funds['近1年'], errors='coerce')
        
        # 稳健型基金特征：近1年收益通常在 -5% 到 30% 之间
        # 太低可能是纯债，太高可能是股票型
        gushou_funds = gushou_funds[
            (gushou_funds['近1年'] >= -5) & 
            (gushou_funds['近1年'] <= 30)
        ].sort_values('近1年', ascending=False)  # 按收益排序，优先分析表现好的
        
        print(f"✓ 收益率筛选后剩余: {len(gushou_funds)} 只")
        
        # 限制数量
        selected = gushou_funds.head(max_funds)
        print(f"✓ 将分析前 {len(selected)} 只基金\n")
        
        # 转换为需要的格式
        result = []
        for idx, row in selected.iterrows():
            name = row['基金简称']
            # 更精确地判断基金类型
            if '纯债' in name or '短债' in name or '中短债' in name:
                ftype = '纯债型'
            elif '可转债' in name:
                ftype = '可转债'
            elif '债' in name or '债券' in name:
                ftype = '债券型'
            elif '混合' in name:
                ftype = '混合型'
            else:
                ftype = '其他'
            
            result.append((str(row['基金代码']).zfill(6), name, ftype))
        
        return result
        
    except Exception as e:
        print(f"✗ 获取基金列表失败: {e}")
        print("将使用备选基金池...")
        # 如果失败，返回一个小的备选池
        return [
            ('004010', '华泰柏瑞鼎利混合A', '混合型'),
            ('002015', '南方荣光A', '混合型'),
            ('000047', '华夏鼎泓债券A', '债券型'),
            ('003859', '招商招旭纯债A', '纯债型'),
            ('001316', '安信稳健增值混合A', '混合型'),
        ]


def main():
    """主函数 - 默认从全基金池筛选"""
    import sys
    
    print("=" * 120)
    print(" " * 40 + "基金分析技能")
    print(" " * 35 + "全市场智能筛选 + 深度分析")
    print("=" * 120)
    
    # 检查是否有命令行参数（单个基金分析模式）
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        # 单个基金分析模式
        fund_code = sys.argv[1]
        fund_name = sys.argv[2] if len(sys.argv) > 2 else None
        
        print(f"\n正在分析基金: {fund_code}\n")
        analysis = analyze_single_fund(fund_code, fund_name)
        print_fund_analysis(analysis)
        return
    
    # 批量筛选模式 - 默认从全市场筛选
    print("\n🚀 全市场基金智能筛选")
    print("-" * 120)
    print("将从所有开放式基金中智能筛选优质产品进行分析\n")
    
    # 解析参数
    max_funds = 200  # 默认200只
    if '--max' in sys.argv:
        idx = sys.argv.index('--max')
        if idx + 1 < len(sys.argv):
            try:
                max_funds = int(sys.argv[idx + 1])
                max_funds = max(50, min(max_funds, 500))  # 限制50-500
            except:
                pass
    
    print(f"📊 分析数量: {max_funds} 只基金")
    print(f"⏱️  预计耗时: {max_funds * 6 // 60} 分钟左右")
    print()
    
    # 从全市场获取基金池
    fund_pool = get_all_gushou_funds(max_funds=max_funds)
    
    if len(fund_pool) == 0:
        print("\n❌ 未获取到基金数据")
        return
    
    # 去重
    seen = set()
    unique_funds = []
    for fund in fund_pool:
        if fund[0] not in seen:
            seen.add(fund[0])
            unique_funds.append(fund)
    
    print(f"🎯 实际分析 {len(unique_funds)} 只基金\n")
    
    # 分析基金
    df = analyze_funds(unique_funds)
    
    if len(df) == 0:
        print("\n❌ 未获取到有效数据")
        return
    
    # 筛选优质基金（更严格的标准）
    high_quality = filter_quality_funds(df, min_sharpe=0.5, max_drawdown=-15)
    
    # 保存结果
    output_file = "基金筛选结果.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # 显示结果
    print("\n" + "=" * 120)
    print(" " * 50 + "🎉 筛选结果")
    print("=" * 120)
    print(f"\n【分析 {len(df)} 只基金，{len(high_quality)} 只优质基金（夏普>=0.5 & 回撤<15%）】\n")
    
    if len(high_quality) > 0:
        display_cols = ['基金代码', '基金名称', '类型', '夏普比率', '最大回撤(%)', 
                       '年化收益率(%)', '近1年收益(%)', '基金规模(亿元)', '股票仓位(%)']
        print("🏆 优质基金 TOP 20（推荐关注）：")
        print("-" * 120)
        print(high_quality[display_cols].head(20).to_string(index=False))
    else:
        print("⚠️  未找到符合严格标准（夏普>=0.5 & 回撤<15%）的基金")
        print("   建议放宽条件查看全部结果\n")
    
    print("\n\n📊 全部基金按夏普比率排序（TOP 30）：")
    print("-" * 120)
    display_cols2 = ['基金代码', '基金名称', '类型', '夏普比率', '最大回撤(%)', 
                     '年化收益率(%)', '股票仓位(%)', '基金规模(亿元)']
    print(df[display_cols2].head(30).to_string(index=False))
    
    # 统计
    print("\n\n" + "=" * 120)
    print(" " * 50 + "📈 统计摘要")
    print("=" * 120)
    print(f"分析基金总数: {len(df)}")
    print(f"夏普比率>=1.0: {len(df[df['夏普比率'] >= 1.0])} 只 (优秀)")
    print(f"夏普比率>=0.5: {len(df[df['夏普比率'] >= 0.5])} 只 (良好)")
    print(f"夏普比率>=0.3: {len(df[df['夏普比率'] >= 0.3])} 只 (可接受)")
    print(f"平均夏普比率: {df['夏普比率'].mean():.2f}")
    print(f"平均最大回撤: {df['最大回撤(%)'].mean():.2f}%")
    print(f"平均年化收益: {df['年化收益率(%)'].mean():.2f}%")
    
    # 按类型统计
    print("\n按类型统计：")
    type_stats = df.groupby('类型').agg({
        '夏普比率': 'mean',
        '最大回撤(%)': 'mean',
        '年化收益率(%)': 'mean',
        '基金代码': 'count'
    }).round(2)
    type_stats.columns = ['平均夏普', '平均回撤', '平均收益', '数量']
    print(type_stats.to_string())
    
    print(f"\n✅ 详细结果已保存到: {output_file}")
    print("\n💡 使用建议:")
    print("   • 优先关注夏普比率>0.5且回撤<15%的基金")
    print("   • 可通过 'npx fund-screener <基金代码>' 进行单基金深度分析")
    print("=" * 120)


if __name__ == "__main__":
    import sys
    
    # 检查是否有命令行参数（单个基金分析模式）
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        # 单个基金分析模式
        fund_code = sys.argv[1]
        fund_name = sys.argv[2] if len(sys.argv) > 2 else None
        
        print(f"正在分析基金: {fund_code}")
        analysis = analyze_single_fund(fund_code, fund_name)
        print_fund_analysis(analysis)
    else:
        # 批量筛选模式
        main()
