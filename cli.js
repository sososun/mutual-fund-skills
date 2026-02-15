#!/usr/bin/env node
/**
 * 基金分析技能 CLI
 * 通过 npx mutual-fund-skills 直接运行
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// 颜色输出
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m'
};

function printBanner() {
  console.log(`${colors.cyan}
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║           基金分析技能 (Mutual Fund Skills) v1.3.0                 ║
║                                                                  ║
║     基于 AkShare 的高夏普比率、低回撤基金筛选工具                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝${colors.reset}\n`);
}

function checkPython() {
  return new Promise((resolve, reject) => {
    const python = spawn('python3', ['--version']);
    let version = '';
    
    python.stdout.on('data', (data) => {
      version += data.toString();
    });
    
    python.on('close', (code) => {
      if (code === 0) {
        resolve(version.trim());
      } else {
        // 尝试 python
        const python2 = spawn('python', ['--version']);
        python2.on('close', (code2) => {
          if (code2 === 0) {
            resolve('python');
          } else {
            reject(new Error('未找到 Python，请先安装 Python 3.8+'));
          }
        });
      }
    });
  });
}

async function main() {
  printBanner();
  
  try {
    // 检查 Python
    const pythonVersion = await checkPython();
    console.log(`${colors.green}✓ 检测到 ${pythonVersion}${colors.reset}\n`);
    
    // 获取脚本路径
    const scriptPath = path.join(__dirname, 'fund_screener.py');
    
    if (!fs.existsSync(scriptPath)) {
      console.error(`${colors.red}✗ 错误: 找不到 fund_screener.py${colors.reset}`);
      process.exit(1);
    }
    
    // 运行 Python 脚本
    console.log(`${colors.yellow}正在启动基金筛选器...${colors.reset}\n`);
    
    const pythonCmd = pythonVersion === 'python' ? 'python' : 'python3';
    const child = spawn(pythonCmd, [scriptPath], {
      stdio: 'inherit',
      cwd: __dirname
    });
    
    child.on('close', (code) => {
      if (code !== 0) {
        console.error(`${colors.red}\n✗ 程序异常退出 (code: ${code})${colors.reset}`);
        process.exit(code);
      }
    });
    
  } catch (error) {
    console.error(`${colors.red}✗ 错误: ${error.message}${colors.reset}`);
    console.log(`\n${colors.yellow}请先安装 Python 3.8+:${colors.reset}`);
    console.log('  macOS: brew install python');
    console.log('  Linux: sudo apt-get install python3');
    console.log('  Windows: https://python.org/downloads\n');
    process.exit(1);
  }
}

// 处理命令行参数
const args = process.argv.slice(2);

if (args.includes('--help') || args.includes('-h')) {
  console.log(`${colors.cyan}
基金分析技能 - 使用方法

命令:
  npx mutual-fund-skills                    全市场智能筛选（默认200只，约20分钟）
  npx mutual-fund-skills --max 300          指定分析数量（50-500只）
  npx mutual-fund-skills <基金代码>         分析单个基金
  npx mutual-fund-skills <代码> <名称>      分析单个基金（指定名称）
  npx mutual-fund-skills --help             显示帮助信息
  npx mutual-fund-skills --version          显示版本

功能:
  • 全市场筛选: 从所有基金中智能筛选优质产品（默认200只）
  • 自定义数量: 支持分析50-500只基金
  • 单个分析: 深度分析指定基金的所有指标
  • 智能识别: 自动排除股票型、行业主题型基金
  • 计算夏普比率、最大回撤、年化收益
  • 获取基金规模、股票/债券仓位、基金经理、持仓等
  • 输出 CSV 格式结果

示例:
  # 全市场筛选（默认200只，约20分钟）
  npx mutual-fund-skills
  
  # 分析300只基金
  npx mutual-fund-skills --max 300
  
  # 快速筛选100只
  npx mutual-fund-skills --max 100

  # 单个基金分析
  npx mutual-fund-skills 000215
  npx mutual-fund-skills 000215 "广发趋势优选灵活配置混合A"

依赖:
  • Python 3.8+
  • akshare, pandas, numpy

安装依赖:
  pip install akshare pandas numpy
${colors.reset}`);
  process.exit(0);
}

if (args.includes('--version') || args.includes('-v')) {
  console.log('v1.2.0');
  process.exit(0);
}

// 检查是否是单个基金分析模式
if (args.length > 0 && !args[0].startsWith('--')) {
  // 单个基金分析模式
  const fundCode = args[0];
  const fundName = args[1] || null;
  
  printBanner();
  
  checkPython().then((pythonVersion) => {
    console.log(`${colors.green}✓ 检测到 ${pythonVersion}${colors.reset}\n`);
    
    const scriptPath = path.join(__dirname, 'fund_screener.py');
    
    if (!fs.existsSync(scriptPath)) {
      console.error(`${colors.red}✗ 错误: 找不到 fund_screener.py${colors.reset}`);
      process.exit(1);
    }
    
    console.log(`${colors.yellow}正在分析基金: ${fundCode}${colors.reset}\n`);
    
    const pythonCmd = pythonVersion === 'python' ? 'python' : 'python3';
    const pythonArgs = [scriptPath, fundCode];
    if (fundName) {
      pythonArgs.push(fundName);
    }
    
    const child = spawn(pythonCmd, pythonArgs, {
      stdio: 'inherit',
      cwd: __dirname
    });
    
    child.on('close', (code) => {
      if (code !== 0) {
        console.error(`${colors.red}\n✗ 程序异常退出 (code: ${code})${colors.reset}`);
        process.exit(code);
      }
    });
  }).catch((error) => {
    console.error(`${colors.red}✗ 错误: ${error.message}${colors.reset}`);
    console.log(`\n${colors.yellow}请先安装 Python 3.8+:${colors.reset}`);
    console.log('  macOS: brew install python');
    console.log('  Linux: sudo apt-get install python3');
    console.log('  Windows: https://python.org/downloads\n');
    process.exit(1);
  });
} else {
  // 批量筛选模式 - 默认全市场筛选
  printBanner();
  
  checkPython().then((pythonVersion) => {
    console.log(`${colors.green}✓ 检测到 ${pythonVersion}${colors.reset}\n`);
    
    const scriptPath = path.join(__dirname, 'fund_screener.py');
    
    // 构建参数
    const pythonArgs = [scriptPath];
    
    // 如果有 --max 参数，传递给 Python
    if (args.includes('--max')) {
      const idx = args.indexOf('--max');
      if (idx + 1 < args.length) {
        pythonArgs.push('--max', args[idx + 1]);
      }
    }
    
    console.log(`${colors.yellow}启动全市场基金智能筛选...${colors.reset}\n`);
    
    const pythonCmd = pythonVersion === 'python' ? 'python' : 'python3';
    const child = spawn(pythonCmd, pythonArgs, {
      stdio: 'inherit',
      cwd: __dirname
    });
    
    child.on('close', (code) => {
      if (code !== 0) {
        console.error(`${colors.red}\n✗ 程序异常退出 (code: ${code})${colors.reset}`);
        process.exit(code);
      }
    });
  }).catch((error) => {
    console.error(`${colors.red}✗ 错误: ${error.message}${colors.reset}`);
    console.log(`\n${colors.yellow}请先安装 Python 3.8+:${colors.reset}`);
    console.log('  macOS: brew install python');
    console.log('  Linux: sudo apt-get install python3');
    console.log('  Windows: https://python.org/downloads\n');
    process.exit(1);
  });
}
