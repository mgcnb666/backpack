import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime
from functools import wraps
from typing import Dict, List, Set, Tuple

from bpx.bpx import *
from bpx.bpx_pub import *
from dotenv import load_dotenv
from requests.exceptions import ConnectionError, RequestException, Timeout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("grid_trading.log"), logging.StreamHandler()],
)
logger = logging.getLogger("grid_trading")


running = True
trade_history = []
total_profit_loss = 0
total_fee = 0


def retry_on_network_error(max_retries=5, backoff_factor=1.5, max_backoff=60):
    """网络错误重试装饰器"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            backoff = 1

            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
                except (RequestException, ConnectionError, Timeout) as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(f"达到最大重试次数 {max_retries}，操作失败: {e}")
                        raise

                    backoff = min(backoff * backoff_factor, max_backoff)
                    logger.warning(
                        f"网络错误: {e}. 将在 {backoff:.1f} 秒后重试 (尝试 {retries}/{max_retries})"
                    )
                    await asyncio.sleep(backoff)

            raise Exception("重试机制逻辑错误")

        return wrapper

    return decorator


class MartingaleBot:
    """马丁格尔交易机器人类"""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str,
        initial_investment: float,
        max_level: int = 5,
        profit_percentage: float = 2.0,
        multiplier: float = 2.0,
        pair_accuracy: int = 2,
        fee_rate: float = 0.0008,
        use_all_balance: bool = False,
        total_investment: float = 0,
        min_order_quantity: float = 0.01,
        price_drop_threshold: float = 2.0,
    ):
        """
        初始化马丁格尔交易机器人

        Args:
            api_key: API密钥
            api_secret: API密钥
            symbol: 交易对，如 "SOL_USDC"
            initial_investment: 初始投资额(USDC)
            max_level: 最大马丁格尔级别
            profit_percentage: 目标利润百分比
            multiplier: 每级下注倍数
            pair_accuracy: 交易对价格精度
            fee_rate: 手续费率
            use_all_balance: 是否使用全部账户余额，False表示仅使用设置的投资金额
            total_investment: 总投资额(USDC)，所有级别加起来最多使用的资金总量
            min_order_quantity: 最小订单数量，取决于交易所和交易对
            price_drop_threshold: 价格下跌多少百分比时触发加仓
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.initial_investment = initial_investment
        self.max_level = max_level
        self.profit_percentage = profit_percentage / 100.0
        self.multiplier = multiplier
        self.pair_accuracy = pair_accuracy
        self.fee_rate = fee_rate
        self.use_all_balance = use_all_balance
        self.total_investment = total_investment
        self.min_order_quantity = min_order_quantity
        self.price_drop_threshold = price_drop_threshold / 100.0  # 转换为小数形式
        self.price_drop_factor = 1 - self.price_drop_threshold  # 计算价格下跌因子

        self.base_currency = symbol.split("_")[0]
        self.quote_currency = symbol.split("_")[1]

        self.bpx = BpxClient()
        self.bpx.init(api_key=api_key, api_secret=api_secret)

        self.current_level = 0
        self.avg_buy_price = 0
        self.total_base_bought = 0
        self.total_quote_spent = 0
        self.active_buy_order = None
        self.active_sell_order = None
        self.target_sell_price = 0
        self.last_buy_price = 0  # 新增：记录最后一次买入成交价格
        self.next_buy_price = 0  # 新增：记录下一次加仓价格
        self.last_planned_buy_price = 0  # 新增：记录最后一次买入的计划价格

        self.history_file = f"trade_history_{symbol}.json"
        self.load_trade_history()

        logger.info(
            f"马丁格尔交易机器人初始化完成: {symbol}, 初始投资: {initial_investment}, 最大级别: {max_level}, 倍数: {multiplier}, 总投资: {total_investment}, 最小订单量: {min_order_quantity}, 价格下跌阈值: {price_drop_threshold}%"
        )

    def load_trade_history(self):
        """加载交易历史"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, "r") as f:
                    data = json.load(f)
                    global trade_history, total_profit_loss, total_fee
                    trade_history = data.get("history", [])
                    total_profit_loss = data.get("total_profit_loss", 0)
                    total_fee = data.get("total_fee", 0)
                    logger.info(
                        f"已加载交易历史: {len(trade_history)}条记录, 总盈亏: {total_profit_loss}, 总手续费: {total_fee}"
                    )
        except Exception as e:
            logger.error(f"加载交易历史失败: {e}")

    def save_trade_history(self):
        """保存交易历史"""
        try:
            data = {
                "history": trade_history,
                "total_profit_loss": total_profit_loss,
                "total_fee": total_fee,
            }
            with open(self.history_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"保存交易历史失败: {e}")

    @retry_on_network_error(max_retries=5, backoff_factor=1.5)
    async def get_market_price(self) -> float:
        """获取当前市场价格"""
        try:
            market_depth = Depth(self.symbol)
            current_price = round(
                (float(market_depth["asks"][0][0]) + float(market_depth["bids"][-1][0]))
                / 2,
                self.pair_accuracy,
            )
            return current_price
        except Exception as e:
            logger.error(f"获取市场价格失败: {e}")
            if hasattr(self, "last_price") and self.last_price:
                return self.last_price
            raise

    @retry_on_network_error(max_retries=5, backoff_factor=1.5)
    async def get_account_balance(self) -> Tuple[float, float]:
        """获取账户余额"""
        account_balance = self.bpx.balances()
        base_available = float(account_balance[self.base_currency]["available"])
        quote_available = float(account_balance[self.quote_currency]["available"])
        return base_available, quote_available

    async def cancel_order(self, order_id):
        """取消指定ID的订单"""
        try:
            logger.info(f"正在取消订单: {order_id}")
            max_retries = 3
            retry_delay = 1  # 初始延迟1秒
            
            for attempt in range(max_retries):
                try:
                    response = await self.bpx.orderCancel(self.symbol, order_id)
                    if response:
                        logger.info(f"订单 {order_id} 取消成功: {response}")
                        return True
                    else:
                        logger.warning(f"订单 {order_id} 取消请求没有返回响应")
                    break  # 如果成功，直接退出循环
                except Exception as e:
                    if "Order not found" in str(e):
                        logger.info(f"订单 {order_id} 不存在或已完成，无需取消")
                        return True
                    elif attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"取消订单 {order_id} 失败 (尝试 {attempt+1}/{max_retries}): {e}. 将在 {wait_time} 秒后重试...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"取消订单 {order_id} 失败，已达到最大重试次数: {e}")
                        raise
            
            return True
        except Exception as e:
            logger.error(f"取消订单 {order_id} 时发生异常: {e}", exc_info=True)
            return False

    async def cancel_all_orders(self):
        """取消所有活跃订单"""
        try:
            logger.info(f"正在取消所有订单...")
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    # 获取当前活跃订单
                    active_orders = await self.get_active_orders()
                    
                    if not active_orders:
                        logger.info("没有活跃订单需要取消")
                        return True
                    
                    # 尝试通过API批量取消所有订单
                    try:
                        response = self.bpx.orderCancelAll(self.symbol)
                        logger.info(f"批量取消订单成功: {response}")
                        return True
                    except Exception as batch_e:
                        logger.warning(f"批量取消订单失败，将逐个取消: {batch_e}")
                        
                        # 如果批量取消失败，逐个取消订单
                        success = True
                        for order in active_orders:
                            order_id = order["id"]
                            if not await self.cancel_order(order_id):
                                success = False
                        
                        return success
                    
                except Exception as e:
                    if attempt < max_retries - 1:  # 如果不是最后一次尝试
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"取消所有订单失败 (尝试 {attempt+1}/{max_retries}): {e}. 将在 {wait_time} 秒后重试...")
                        await asyncio.sleep(wait_time)
                    else:
                        # 最后一次尝试也失败
                        logger.error(f"取消所有订单失败，已达到最大重试次数: {e}")
                        raise
            
            return True
        except Exception as e:
            logger.error(f"取消所有订单时发生异常: {e}", exc_info=True)
            return False

    async def place_buy_order(self, current_price) -> None:
        """下买单"""
        try:
            # 先检查是否有足够的资金
            base_available, quote_available = await self.get_account_balance()
            
            # 计算当前应该买入的金额
            buy_amount_quote = self.initial_investment * (self.multiplier ** self.current_level)
            if buy_amount_quote > self.total_investment:
                buy_amount_quote = self.total_investment
                logger.warning(f"买入金额 {buy_amount_quote} 已达到最大投资额 {self.total_investment}")
            
            if quote_available < buy_amount_quote:
                logger.error(f"账户余额不足，需要 {buy_amount_quote} {self.quote_currency}，但只有 {quote_available} {self.quote_currency}")
                return False
            
            logger.info(f"当前马丁格尔级别: {self.current_level}, 买入金额: {buy_amount_quote} {self.quote_currency}")
            logger.info(f"当前市场价格: {current_price}")
            
            # 使用市场价格作为计划买入价格
            planned_buy_price = current_price
            
            # 计算能买到的数量
            buy_amount_base = buy_amount_quote / planned_buy_price
            
            # 使用配置的最小订单量
            min_qty = self.min_order_quantity
            
            if buy_amount_base < min_qty:
                logger.warning(f"计算出的购买数量 {buy_amount_base} {self.base_currency} 低于最小订单量 {min_qty}，将使用最小订单量")
                buy_amount_base = min_qty
                # 重新计算买入总额
                buy_amount_quote = planned_buy_price * buy_amount_base
            
            # 买入金额不超过可用资金
            if buy_amount_quote > quote_available:
                buy_amount_quote = quote_available
                buy_amount_base = buy_amount_quote / planned_buy_price
                logger.warning(f"买入金额超过可用资金，已调整为: {buy_amount_quote} {self.quote_currency}")
            
            # 调整精度
            buy_amount_base = round(buy_amount_base, self.pair_accuracy)
            
            # 使用通用方法处理数量精度，避免"Quantity decimal too long"错误
            buy_amount_base = self.get_valid_quantity(buy_amount_base, 2, 0)
            
            # 计算实际消耗的quote资产（按照计划价格）
            planned_quote_amount = round(planned_buy_price * buy_amount_base, 2)
            
            # 尝试执行买单
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"正在下买单: 计划价格={planned_buy_price}, 数量={buy_amount_base} {self.base_currency}, 预计总金额≈{planned_quote_amount} {self.quote_currency}")
                    
                    # 使用限价单，价格设置在当前价格略高一点，确保成交
                    limit_buy_price = round(current_price * 1.005, 2)  # 比市场价高0.5%
                    
                    order = self.bpx.ExeOrder(
                        symbol=self.symbol,
                        side="Bid",
                        orderType="Limit",
                        quantity=str(buy_amount_base),
                        price=str(limit_buy_price),
                        timeInForce="GTC"
                    )
                    
                    logger.info(f"买单已提交: {order}")
                    
                    # 使用订单返回的实际价格或执行价格
                    executed_price = 0
                    actual_quantity = 0
                    actual_quote_amount = 0
                    
                    # 获取实际执行价格
                    if "executedPrice" in order and order["executedPrice"] and float(order["executedPrice"]) > 0:
                        executed_price = float(order["executedPrice"])
                    elif "avgPrice" in order and order["avgPrice"] and float(order["avgPrice"]) > 0:
                        executed_price = float(order["avgPrice"])
                    elif "price" in order and order["price"]:
                        executed_price = float(order["price"])
                    else:
                        executed_price = limit_buy_price
                    
                    # 获取实际执行数量
                    if "executedQuantity" in order and order["executedQuantity"] and float(order["executedQuantity"]) > 0:
                        actual_quantity = float(order["executedQuantity"])
                    else:
                        actual_quantity = float(order["quantity"])
                    
                    # 获取实际执行金额
                    if "executedQuoteQuantity" in order and order["executedQuoteQuantity"] and float(order["executedQuoteQuantity"]) > 0:
                        actual_quote_amount = float(order["executedQuoteQuantity"])
                    else:
                        # 估算实际成交金额
                        actual_quote_amount = round(executed_price * actual_quantity, 2)
                    
                    logger.info(f"买单详情 - 订单ID: {order['id']}")
                    logger.info(f"计划购买: 价格={planned_buy_price}, 数量={buy_amount_base}, 金额≈{planned_quote_amount}")
                    logger.info(f"实际成交: 价格={executed_price}, 数量={actual_quantity}, 金额={actual_quote_amount}")
                    
                    self.active_buy_order = {
                        "id": order["id"],
                        "price": executed_price,  # 使用实际执行价格
                        "planned_price": planned_buy_price,  # 保存计划价格
                        "quantity": actual_quantity,
                        "quote_amount": actual_quote_amount,  # 记录实际消耗的金额
                        "level": self.current_level,  # 添加级别信息
                        "created_at": datetime.now().timestamp()
                    }
                    
                    # 关键修改: 使用计划价格而非实际成交价格计算下次加仓价格
                    self.next_buy_price = round(planned_buy_price * self.price_drop_factor, self.pair_accuracy)
                    logger.info(f"下次加仓价格: {self.next_buy_price} (基于计划买入价格: {planned_buy_price})")
                    
                    # 保存最后一次买入的计划价格
                    self.last_planned_buy_price = planned_buy_price
                    
                    return True
                except Exception as e:
                    if "Insufficient" in str(e):
                        logger.error(f"买单失败-资金不足: {e}")
                        return False
                    elif attempt < max_retries - 1:
                        wait_time = 1 * (2 ** attempt)
                        logger.warning(f"买单失败 (尝试 {attempt+1}/{max_retries}): {e}. 将在 {wait_time} 秒后重试...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"买单失败，已达到最大重试次数: {e}")
                        return False
            
            return False
        except Exception as e:
            logger.error(f"下买单时发生异常: {e}", exc_info=True)
            return False

    async def place_sell_order(self) -> None:
        """卖出交易"""
        try:
            if self.total_base_bought <= 0:
                logger.warning(f"没有持仓，无法下卖单")
                return False
            
            # 确认有足够的资产可卖
            base_available, quote_available = await self.get_account_balance()
            if base_available < self.total_base_bought:
                logger.warning(f"账户可用 {self.base_currency} 不足: 需要 {self.total_base_bought} 但只有 {base_available}")
                # 如果实际持有量比记录的少，调整卖出量
                if base_available > 0:
                    self.total_base_bought = base_available
                    logger.info(f"已调整卖出量为当前可用余额: {base_available} {self.base_currency}")
                else:
                    return False
                
            # 获取当前市场价格 - 使用get_market_price替代ticker
            current_market_price = await self.get_market_price()
            logger.info(f"当前市场价格: {current_market_price}")
            
            # 计算目标卖出价格 (成本价 + 利润百分比)
            target_sell_price = round(self.avg_buy_price * (1 + self.profit_percentage), self.pair_accuracy)
            
            logger.info(f"卖单计算: 总成本均价={self.avg_buy_price}, 目标卖出价格={target_sell_price}, 当前市场价格={current_market_price}")
            
            # 处理订单数量精度问题 - 修复"Quantity decimal too long"错误
            # 使用通用方法处理数量精度
            sell_quantity = self.get_valid_quantity(self.total_base_bought, 2, 0)
            
            # 检查卖单价格是否低于当前市场价
            if target_sell_price < current_market_price:
                logger.info(f"目标卖出价格 {target_sell_price} 低于当前市场价格 {current_market_price}，将使用市价单卖出")
                
                try:
                    # 使用市价单卖出
                    order = self.bpx.ExeOrder(
                        symbol=self.symbol,
                        side="Ask",
                        orderType="Market",
                        quantity=str(sell_quantity),  # 使用舍入后的数量
                        timeInForce="IOC"  # 立即成交或取消
                    )
                    
                    logger.info(f"市价卖单已下单: {order}")
                    
                    # 记录卖单信息
                    self.active_sell_order = {
                        "id": order["id"],
                        "price": current_market_price,  # 使用当前市场价格作为预估价格
                        "quantity": sell_quantity,  # 使用舍入后的数量
                        "is_market": True,
                        "created_at": datetime.now().timestamp()
                    }
                    
                    return True
                except Exception as e:
                    error_message = str(e)
                    logger.error(f"市价卖单失败: {error_message}")
                    
                    # 如果是精度错误，尝试降低精度
                    if "decimal too long" in error_message.lower() or "INVALID_CLIENT_REQUEST" in error_message:
                        logger.info(f"尝试降低数量精度并重试")
                        for precision in range(2, -1, -1):  # 尝试使用2,1,0位小数精度
                            try:
                                sell_quantity = round(self.total_base_bought, precision)
                                logger.info(f"调整订单数量精度至{precision}位小数: {sell_quantity}")
                                
                                order = self.bpx.ExeOrder(
                                    symbol=self.symbol,
                                    side="Ask",
                                    orderType="Market",
                                    quantity=str(sell_quantity),
                                    timeInForce="IOC"
                                )
                                
                                logger.info(f"市价卖单已下单(调整精度后): {order}")
                                
                                self.active_sell_order = {
                                    "id": order["id"],
                                    "price": current_market_price,
                                    "quantity": sell_quantity,
                                    "is_market": True,
                                    "created_at": datetime.now().timestamp()
                                }
                                
                                return True
                            except Exception as e2:
                                error_message2 = str(e2)
                                logger.warning(f"精度{precision}尝试失败: {error_message2}")
                                continue
                    
                    # 如果市价单失败，尝试限价单 (价格设为市场价)
                    logger.info(f"尝试使用限价单 (市场价) 卖出")
                    target_sell_price = current_market_price
            
            # 使用限价单
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"正在下限价卖单: 价格={target_sell_price}, 数量={sell_quantity} {self.base_currency}")
                    
                    order = self.bpx.ExeOrder(
                        symbol=self.symbol,
                        side="Ask",
                        orderType="Limit",
                        quantity=str(sell_quantity),  # 使用舍入后的数量
                        price=str(target_sell_price),
                        timeInForce="GTC"
                    )
                    
                    logger.info(f"限价卖单已下单: {order}")
                    
                    # 记录卖单信息
                    self.active_sell_order = {
                        "id": order["id"],
                        "price": float(order["price"]),
                        "quantity": float(order["quantity"]),
                        "is_market": False,
                        "created_at": datetime.now().timestamp()
                    }
                    
                    return True
                
                except Exception as e:
                    error_message = str(e)
                    
                    # 处理精度错误
                    if "decimal too long" in error_message.lower() or "INVALID_CLIENT_REQUEST" in error_message:
                        if "Quantity decimal too long" in error_message:
                            logger.warning(f"订单数量精度错误: {error_message}")
                            
                            # 使用更低精度重试
                            sell_quantity = self.get_valid_quantity(self.total_base_bought, 1, 0)  # 降低起始精度至1
                            logger.info(f"降低数量精度: {sell_quantity}")
                            continue
                        else:
                            logger.error(f"请求无效: {error_message}")
                            return False
                    
                    # 处理"即时成交"错误
                    elif "would immediately match" in error_message.lower():
                        logger.warning(f"卖单会即时成交: {error_message}")
                        
                        # 调整卖单价格到市场价格上方1%
                        target_sell_price = round(current_market_price * 1.01, self.pair_accuracy)
                        logger.info(f"调整卖单价格到市场价格上方1%: {target_sell_price}")
                        
                        # 不立即重试，继续下一次循环尝试
                        continue
                    
                    # 处理其他错误
                    elif attempt < max_retries - 1:
                        wait_time = 1 * (2 ** attempt)
                        logger.warning(f"卖单失败 (尝试 {attempt+1}/{max_retries}): {e}. 将在 {wait_time} 秒后重试...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"卖单失败，已达到最大重试次数: {e}")
                        return False
            
            return False
        except Exception as e:
            logger.error(f"下卖单时发生异常: {e}", exc_info=True)
            return False

    async def get_order_details(self, order_id, original_order=None):
        """获取订单详细信息，包括实际成交价格、成交数量和成交金额"""
        try:
            # 初始化结果，使用原始订单信息作为默认值
            result = {
                "order_id": order_id,
                "executed_price": original_order.get("price", 0) if original_order else 0,
                "executed_quantity": original_order.get("quantity", 0) if original_order else 0,
                "executed_quote_quantity": original_order.get("quote_amount", 0) if original_order else 0,
                "status": "Unknown"
            }
            
            # 查询订单
            order_detail = None
            try:
                order_detail = self.bpx.orderQuery(self.symbol, order_id)
                logger.info(f"获取到订单详情: {order_detail}")
                
                # 如果成功获取订单详情
                if order_detail and not isinstance(order_detail, dict) or not order_detail.get("code"):
                    # 提取订单信息
                    # 获取执行价格
                    if "executedPrice" in order_detail and order_detail["executedPrice"] and float(order_detail["executedPrice"]) > 0:
                        result["executed_price"] = float(order_detail["executedPrice"])
                    elif "avgPrice" in order_detail and order_detail["avgPrice"] and float(order_detail["avgPrice"]) > 0:
                        result["executed_price"] = float(order_detail["avgPrice"])
                    elif "price" in order_detail and order_detail["price"]:
                        result["executed_price"] = float(order_detail["price"])
                    
                    # 获取执行数量
                    if "executedQuantity" in order_detail and order_detail["executedQuantity"] and float(order_detail["executedQuantity"]) > 0:
                        result["executed_quantity"] = float(order_detail["executedQuantity"])
                    elif "quantity" in order_detail and order_detail["quantity"]:
                        result["executed_quantity"] = float(order_detail["quantity"])
                    
                    # 获取执行金额
                    if "executedQuoteQuantity" in order_detail and order_detail["executedQuoteQuantity"] and float(order_detail["executedQuoteQuantity"]) > 0:
                        result["executed_quote_quantity"] = float(order_detail["executedQuoteQuantity"])
                    elif result["executed_price"] > 0 and result["executed_quantity"] > 0:
                        # 如果没有明确的执行金额，使用价格和数量计算
                        result["executed_quote_quantity"] = round(result["executed_price"] * result["executed_quantity"], 2)
                    
                    result["status"] = order_detail.get("status", "Unknown")
            except Exception as e:
                logger.warning(f"通过orderQuery获取订单详情失败: {e}")
                
                # 尝试使用交易历史查询，如果存在该方法
                if hasattr(self.bpx, 'fillsQuery'):
                    try:
                        fills = self.bpx.fillsQuery(self.symbol, limit=10)
                        if fills:
                            logger.info(f"通过fillsQuery获取到订单成交历史: {fills}")
                            fill_result = self._process_order_fills(fills)
                            
                            # 使用成交历史的数据
                            if fill_result:
                                if fill_result["executed_price"] > 0:
                                    result["executed_price"] = fill_result["executed_price"]
                                if fill_result["executed_quantity"] > 0:
                                    result["executed_quantity"] = fill_result["executed_quantity"]
                                if fill_result["executed_quote_quantity"] > 0:
                                    result["executed_quote_quantity"] = fill_result["executed_quote_quantity"]
                    except Exception as fill_e:
                        logger.warning(f"通过fillsQuery获取订单成交历史失败: {fill_e}")
                else:
                    logger.warning("BpxClient没有fillsQuery方法，无法通过成交历史获取订单详情")
            
            # 如果没有获取到有效的成交信息，但有原始订单信息，使用原始订单的值
            if result["executed_quantity"] == 0 and original_order and original_order.get("quantity"):
                result["executed_quantity"] = float(original_order["quantity"])
                logger.info(f"使用原始订单的数量: {result['executed_quantity']}")
                
            if result["executed_price"] == 0 and original_order and original_order.get("price"):
                result["executed_price"] = float(original_order["price"])
                logger.info(f"使用原始订单的价格: {result['executed_price']}")
                
            if result["executed_quote_quantity"] == 0 and result["executed_price"] > 0 and result["executed_quantity"] > 0:
                result["executed_quote_quantity"] = round(result["executed_price"] * result["executed_quantity"], 2)
                logger.info(f"计算订单的quote金额: {result['executed_quote_quantity']}")
            elif result["executed_quote_quantity"] == 0 and original_order and original_order.get("quote_amount"):
                result["executed_quote_quantity"] = float(original_order["quote_amount"])
                logger.info(f"使用原始订单的quote金额: {result['executed_quote_quantity']}")
            
            logger.info(f"订单 {order_id} 详情: 执行价格={result['executed_price']}, 执行数量={result['executed_quantity']}, 执行金额={result['executed_quote_quantity']}")
            return result
        except Exception as e:
            logger.error(f"获取订单详情失败: {e}")
            
            # 如果完全失败，但有原始订单信息，使用原始订单的值
            if original_order:
                result = {
                    "order_id": order_id,
                    "executed_price": float(original_order["price"]) if original_order.get("price") else 0,
                    "executed_quantity": float(original_order["quantity"]) if original_order.get("quantity") else 0,
                    "executed_quote_quantity": float(original_order["quote_amount"]) if original_order.get("quote_amount") else 0,
                    "status": "Filled"  # 假设已成交
                }
                logger.info(f"使用原始订单数据作为订单详情: {result}")
                return result
            return None

    def _process_order_fills(self, fills):
        """处理订单成交历史记录，计算平均成交价格和总成交量"""
        if not fills:
            return None
        
        total_quantity = 0
        total_quote_quantity = 0
        
        for fill in fills:
            if "quantity" in fill and fill["quantity"]:
                quantity = float(fill["quantity"])
                total_quantity += quantity
            
            if "quoteQuantity" in fill and fill["quoteQuantity"]:
                quote_quantity = float(fill["quoteQuantity"])
                total_quote_quantity += quote_quantity
            
        # 计算平均成交价格
        avg_price = 0
        if total_quantity > 0:
            avg_price = round(total_quote_quantity / total_quantity, self.pair_accuracy)
        
        return {
            "executed_price": avg_price,
            "executed_quantity": total_quantity,
            "executed_quote_quantity": total_quote_quantity
        }

    async def check_order_status(self) -> None:
        """检查订单状态"""
        try:
            current_orders = self.bpx.ordersQuery(self.symbol)
            current_order_ids = [order.get("id") for order in current_orders if order.get("id")]
            
            # 获取当前市场价格
            current_price = await self.get_market_price()
            
            # 使用参考价格计算下一次加仓价格 - 使用计划价格而不是实际成交价格
            reference_price = self.last_planned_buy_price if self.last_planned_buy_price > 0 else current_price
            next_level_price = round(reference_price * self.price_drop_factor, self.pair_accuracy)
            
            # 将计算出的下次加仓价格保存到属性中
            self.next_buy_price = next_level_price
            
            # 计算当前买单和卖单状态
            buy_status = "活跃" if self.active_buy_order else "无"
            sell_status = "活跃" if self.active_sell_order else "无"
            
            # 显示更丰富的状态信息，包括下一次加仓价格
            if self.active_buy_order:
                order_price = self.active_buy_order["price"]
                planned_price = self.active_buy_order.get("planned_price", order_price)
                price_gap_percent = round((current_price - next_level_price) / next_level_price * 100, 2)
                logger.info(f"当前活跃订单数: {len(current_orders)}, 买单状态: {buy_status}, 卖单状态: {sell_status}")
                logger.info(f"当前价格: {current_price}, 买单价格: {order_price}, 计划价格: {planned_price}, 下一次加仓价格: {next_level_price} (再下跌 {price_gap_percent}%)")
            elif self.active_sell_order:
                price_gap_percent = round((current_price - next_level_price) / next_level_price * 100, 2)
                sell_price = self.active_sell_order["price"]
                logger.info(f"当前活跃订单数: {len(current_orders)}, 买单状态: {buy_status}, 卖单状态: {sell_status}")
                reference_price_str = f"上次计划买入价格: {self.last_planned_buy_price}, " if self.last_planned_buy_price > 0 else ""
                logger.info(f"当前价格: {current_price}, {reference_price_str}下一次加仓价格: {next_level_price} (还需下跌 {price_gap_percent}%), 卖单价格: {sell_price}")
            else:
                price_gap_percent = round((current_price - next_level_price) / next_level_price * 100, 2)
                logger.info(f"当前活跃订单数: {len(current_orders)}, 买单状态: {buy_status}, 卖单状态: {sell_status}")
                reference_price_str = f"上次计划买入价格: {self.last_planned_buy_price}, " if self.last_planned_buy_price > 0 else ""
                logger.info(f"当前价格: {current_price}, {reference_price_str}下一次加仓价格: {next_level_price} (再下跌 {price_gap_percent}%)")
            
            # ===== 加仓条件判断，使用记录的计划买入价格 =====
            # 如果有买单，基于买单的计划价格计算；否则基于上次计划买入价格
            if self.active_buy_order and "planned_price" in self.active_buy_order:
                reference_price = self.active_buy_order["planned_price"]
            else:
                reference_price = self.last_planned_buy_price if self.last_planned_buy_price > 0 else current_price
            
            price_drop_trigger = reference_price * self.price_drop_factor
            
            # 检查当前价格是否已经达到加仓条件
            if current_price <= price_drop_trigger and self.current_level < self.max_level - 1:
                logger.info(f"价格已下跌超过{self.price_drop_threshold * 100}%，达到加仓条件，当前价格：{current_price}，触发价格：{price_drop_trigger}，参考价格：{reference_price}")
                
                # 保存原有卖单和持仓信息（用于后续合并计算）
                has_active_sell_order = self.active_sell_order is not None
                original_sell_order = self.active_sell_order
                original_base_bought = self.total_base_bought
                original_quote_spent = self.total_quote_spent
                original_avg_buy_price = self.avg_buy_price
                
                # 先增加马丁格尔级别
                self.current_level += 1
                logger.info(f"升级到马丁格尔级别 {self.current_level}")
                
                # 执行加仓买入前，临时重置持仓数据，避免影响新买单的计算
                if has_active_sell_order:
                    temp_base_bought = self.total_base_bought
                    temp_quote_spent = self.total_quote_spent
                    temp_avg_buy_price = self.avg_buy_price
                    self.total_base_bought = 0
                    self.total_quote_spent = 0
                    self.avg_buy_price = 0
                
                # 执行加仓买入
                buy_success = await self.place_buy_order(current_price)
                
                # 检查买入是否成功
                if buy_success and self.active_buy_order:
                    logger.info(f"加仓买单放置成功: {self.active_buy_order['id']}")
                    
                    # 如果有活跃卖单，取消它
                    if has_active_sell_order:
                        try:
                            # 恢复原始持仓数据
                            self.total_base_bought = temp_base_bought
                            self.total_quote_spent = temp_quote_spent
                            self.avg_buy_price = temp_avg_buy_price
                            
                            # 取消卖单
                            if original_sell_order:
                                await self.cancel_order(original_sell_order["id"])
                                logger.info(f"已取消原有卖单: {original_sell_order['id']}")
                                self.active_sell_order = None
                            
                            # 等待买单成交
                            logger.info("等待加仓买单成交...")
                            # 注意：这里不需要等待，因为系统会在下一次检查中处理买单成交逻辑
                        except Exception as e:
                            logger.error(f"取消卖单失败: {e}")
                    
                    return  # 执行加仓后直接返回，下一个周期检查买单成交状态
                else:
                    logger.error("加仓买单放置失败")
                    # 加仓失败，恢复原始数据
                    if has_active_sell_order:
                        self.total_base_bought = temp_base_bought
                        self.total_quote_spent = temp_quote_spent
                        self.avg_buy_price = temp_avg_buy_price
                        self.active_sell_order = original_sell_order
                    # 恢复级别
                    self.current_level -= 1
                    logger.info(f"恢复到马丁格尔级别 {self.current_level}")
            
            # 检查买单状态
            if self.active_buy_order:
                # 检查是否为市价单或订单ID不在当前活跃订单中
                is_market_order = self.active_buy_order.get("is_market", False)
                order_not_active = self.active_buy_order["id"] not in current_order_ids
                
                if is_market_order or order_not_active:
                    # 获取订单详细信息，确保拿到最终成交价格和数量
                    order_id = self.active_buy_order["id"]
                    order_details = await self.get_order_details(order_id, self.active_buy_order)
                    
                    # 如果无法获取详细信息，使用已保存的信息
                    if not order_details:
                        logger.warning(f"无法获取订单 {order_id} 的详细信息，将使用已保存的信息")
                        executed_price = self.active_buy_order["price"]
                        actual_quantity = self.active_buy_order["quantity"]
                        actual_quote_amount = self.active_buy_order["quote_amount"]
                        planned_price = self.active_buy_order.get("planned_price", executed_price)
                    else:
                        # 使用获取到的详细成交信息
                        executed_price = order_details["executed_price"]
                        actual_quantity = order_details["executed_quantity"]
                        actual_quote_amount = order_details["executed_quote_quantity"]
                        planned_price = self.active_buy_order.get("planned_price", executed_price)
                        
                        # 如果仍然无法获得实际成交金额，根据价格和数量计算
                        if actual_quote_amount == 0 and executed_price > 0 and actual_quantity > 0:
                            actual_quote_amount = round(executed_price * actual_quantity, 2)
                    
                    # 确保数量和金额必须大于0
                    if actual_quantity <= 0:
                        logger.warning(f"订单 {order_id} 的成交数量为0，使用原始订单的数量")
                        actual_quantity = self.active_buy_order["quantity"]
                        
                    if actual_quote_amount <= 0:
                        logger.warning(f"订单 {order_id} 的成交金额为0，使用价格和数量计算")
                        actual_quote_amount = round(executed_price * actual_quantity, 2)
                    
                    if is_market_order:
                        logger.info(f"市价买单已成交: 级别 {self.active_buy_order['level']}, 实际成交价格 {executed_price}, 计划价格 {planned_price}, 数量 {actual_quantity}, 金额 {actual_quote_amount}")
                    else:
                        logger.info(f"限价买单已成交: 级别 {self.active_buy_order['level']}, 实际成交价格 {executed_price}, 计划价格 {planned_price}, 数量 {actual_quantity}, 金额 {actual_quote_amount}")
                    
                    # 更新最后买入价格 - 保存实际成交价格和计划价格
                    self.last_buy_price = executed_price
                    self.last_planned_buy_price = planned_price
                
                    # 更新平均买入价格和总持仓
                    new_base = actual_quantity
                    new_quote = actual_quote_amount
                
                    # 更新总花费和总购买数量
                    self.total_quote_spent += new_quote
                    self.total_base_bought += new_base
                    
                    # 确保不会出现除以零的情况
                    if self.total_base_bought <= 0:
                        logger.warning("检测到总持仓数量为0或负数，将其设置为实际购买数量")
                        self.total_base_bought = new_base
                    
                    # 重新计算平均买入价格 - 使用实际花费金额除以总持仓数量
                    if self.total_base_bought > 0:
                        self.avg_buy_price = round(self.total_quote_spent / self.total_base_bought, self.pair_accuracy)
                    else:
                        # 如果总持仓量为0，使用本次交易价格作为平均价格
                        self.avg_buy_price = executed_price
                        logger.warning(f"总持仓为0，使用本次交易价格({executed_price})作为平均价格")
                    
                    # 计算下一次加仓价格 - 使用计划价格而不是实际成交价格
                    self.next_buy_price = round(planned_price * self.price_drop_factor, self.pair_accuracy)
                
                    logger.info(f"更新后的平均买入价格: {self.avg_buy_price}, 总持仓: {self.total_base_bought} {self.base_currency}, 已用资金: {self.total_quote_spent:.2f} {self.quote_currency}")
                    logger.info(f"下一次加仓价格: {self.next_buy_price} (基于计划买入价格 {planned_price})")
                
                    # 买单成交后，尝试放置卖单
                    self.active_buy_order = None
                    logger.info("准备放置卖单...")
                    await self.place_sell_order()
                    logger.info(f"卖单放置结果: {'成功' if self.active_sell_order else '失败，稍后重试'}")
                    # 原有的检查价格下跌逻辑可以移除，因为已经在上面处理了
            
            # 买单已成交但未放置卖单，重新尝试放置卖单
            elif self.active_buy_order is None and self.active_sell_order is None and self.total_base_bought > 0:
                logger.info(f"检测到买单已成交但卖单未成功放置，重新尝试放置卖单")
                logger.info(f"当前持仓: {self.total_base_bought} {self.base_currency}, 平均买入价格: {self.avg_buy_price}")
                await self.place_sell_order()
                logger.info(f"卖单重新放置结果: {'成功' if self.active_sell_order else '失败，稍后重试'}")
                
            # 检查卖单状态
            if self.active_sell_order:
                # 如果是市价单或者订单ID不在当前活跃订单中，进一步验证是否真的成交
                is_market_order = self.active_sell_order.get("is_market", False)
                order_not_active = self.active_sell_order["id"] not in current_order_ids
                
                # 额外验证：检查卖单是否真的成交了
                order_executed = False
                if order_not_active:
                    try:
                        # 1. 检查交易历史
                        try:
                            # 获取最近的交易历史
                            order_trades = self.bpx.fillsQuery(self.symbol, limit=10)
                            for trade in order_trades:
                                if trade.get("orderId") == self.active_sell_order["id"]:
                                    order_executed = True
                                    logger.info(f"通过交易历史确认卖单 {self.active_sell_order['id']} 确实成交")
                                    break
                        except Exception as e:
                            logger.error(f"查询交易历史失败: {e}")
                        
                        # 2. 如果交易历史查询不到，检查余额变化
                        if not order_executed:
                            base_available, quote_available = await self.get_account_balance()
                            sell_value = self.active_sell_order["price"] * self.active_sell_order["quantity"]
                            
                            # 获取之前的余额记录（如果有）
                            prev_quote_balance = getattr(self, 'prev_quote_balance', None)
                            
                            # 如果余额增加，可能卖单成交了
                            if prev_quote_balance is not None and quote_available > prev_quote_balance:
                                increase = quote_available - prev_quote_balance
                                expected_increase = sell_value * (1 - self.fee_rate)  # 考虑手续费
                                
                                # 如果余额增加接近预期的卖单值
                                if abs(increase - expected_increase) / expected_increase < 0.1:  # 允许10%的误差
                                    order_executed = True
                                    logger.info(f"通过余额变化确认卖单可能已成交, 预期增加: {expected_increase}, 实际增加: {increase}")
                        
                            # 更新余额记录
                            self.prev_quote_balance = quote_available
                        
                        # 3. 如果还是无法确认，检查订单历史
                        if not order_executed:
                            # 使用orderQuery直接查询订单状态
                            try:
                                order_status = self.bpx.orderQuery(self.symbol, self.active_sell_order["id"])
                                if order_status and order_status.get("status") == "Filled":
                                    order_executed = True
                                    logger.info(f"通过订单查询确认卖单 {self.active_sell_order['id']} 已成交")
                            except Exception as e:
                                logger.error(f"订单查询失败: {e}")
                        
                        # 如果通过以上三种方法都无法确认订单状态
                        if not order_executed:
                            logger.warning(f"卖单 {self.active_sell_order['id']} 不在活跃订单中，但未能确认是否成交，请检查订单状态")
                            
                            # 我们做一个延迟处理的机制，如果订单不在活跃列表中已经超过一定时间（例如30秒），
                            # 我们可以认为它很可能是已经成交了
                            current_time = datetime.now().timestamp()
                            order_time = self.active_sell_order.get('created_at', current_time)
                            time_passed = current_time - order_time
                            
                            if time_passed > 30:  # 30秒后如果还不在活跃列表中，可能已成交
                                logger.info(f"卖单已从活跃列表中消失超过30秒，假定已成交")
                                order_executed = True
                            else:
                                # 继续等待确认
                                return
                    except Exception as e:
                        logger.error(f"验证卖单状态失败: {e}")
                        # 保守处理：如果无法确认，不认为订单已成交
                        return
                
                if is_market_order or (order_not_active and order_executed):
                    if is_market_order:
                        logger.info(f"市价卖单已成交: 预估价格 {self.active_sell_order['price']}, 数量 {self.active_sell_order['quantity']}")
                    else:
                        logger.info(f"限价卖单已成交: 价格 {self.active_sell_order['price']}, 数量 {self.active_sell_order['quantity']}")
                    
                    # 再次检查账户余额，确保卖单真的成交
                    base_available, quote_available = await self.get_account_balance()
                    if base_available >= self.active_sell_order["quantity"]:
                        logger.warning(f"账户中仍有足够的 {self.base_currency}，卖单可能未真正成交。将继续保持卖单状态。")
                        return
                
                    # 计算利润
                    sell_value = self.active_sell_order["price"] * self.active_sell_order["quantity"]
                    buy_value = self.total_quote_spent
                    profit = sell_value - buy_value
                    
                    # 计算手续费
                    fee = sell_value * self.fee_rate
                    
                    # 记录交易
                    global total_profit_loss, total_fee, trade_history
                    total_profit_loss += profit
                    total_fee += fee
                    
                    trade_record = {
                        "timestamp": datetime.now().isoformat(),
                        "type": "martingale_cycle",
                        "levels_used": self.current_level + 1,
                        "avg_buy_price": self.avg_buy_price,
                        "sell_price": self.active_sell_order["price"],
                        "quantity": self.active_sell_order["quantity"],
                        "buy_value": buy_value,
                        "sell_value": sell_value,
                        "profit": profit,
                        "fee": fee,
                        "is_market_order": is_market_order
                    }
                    trade_history.append(trade_record)
                    
                    logger.info(f"完成一个马丁格尔周期, 利润: {profit:.2f} {self.quote_currency}, 手续费: {fee:.2f} {self.quote_currency}")
                    logger.info(f"总计: 利润 {total_profit_loss:.2f}, 手续费 {total_fee:.2f} {self.quote_currency}")
                    
                    # 保存交易历史
                    self.save_trade_history()
                    
                    # 重置状态，准备新的交易周期
                    self.current_level = 0
                    self.total_base_bought = 0
                    self.total_quote_spent = 0
                    self.active_sell_order = None
                    self.avg_buy_price = 0
                    
                    # 开始新的买入周期
                    current_price = await self.get_market_price()
                    logger.info(f"准备开始新的买入周期，当前市场价格: {current_price}")
                    await self.place_buy_order(current_price)
                    logger.info(f"新买单放置结果: {'成功' if self.active_buy_order else '失败'}")
                else:
                    # 卖单存在但尚未成交，显示等待状态
                    price_gap_to_target = round((self.active_sell_order['price'] - current_price) / current_price * 100, 2)
                    time_waiting = datetime.now() - datetime.fromtimestamp(self.active_sell_order.get('created_at', datetime.now().timestamp()))
                    minutes_waiting = round(time_waiting.total_seconds() / 60, 1)
                    
                    # 如果没有created_at字段，显示简化信息
                    if 'created_at' in self.active_sell_order:
                        logger.info(f"卖单仍在等待成交: 价格 {self.active_sell_order['price']}, 当前市价 {current_price}, 差距 {price_gap_to_target}%, 已等待 {minutes_waiting} 分钟")
                    else:
                        logger.info(f"卖单仍在等待成交: 价格 {self.active_sell_order['price']}, 当前市价 {current_price}, 差距 {price_gap_to_target}%")
        
        except Exception as e:
            logger.error(f"检查订单状态失败: {e}")
            import traceback
            traceback.print_exc()

    async def run(self) -> None:
        """运行马丁格尔交易机器人"""
        try:
            base_available, quote_available = await self.get_account_balance()
            logger.info(
                f"初始余额: {base_available} {self.base_currency}, {quote_available} {self.quote_currency}"
            )

            # 检查总投资额和初始投资额
            if self.total_investment > 0:
                # 计算所有级别的总投资需求
                total_needed = 0
                for level in range(self.max_level):
                    level_investment = self.initial_investment * (self.multiplier ** level)
                    total_needed += level_investment
                
                if abs(total_needed - self.total_investment) > 0.01:
                    logger.warning(
                        f"警告: 设置的初始投资额({self.initial_investment})与总投资额({self.total_investment})不匹配"
                    )
                    logger.info(f"所有级别投资总和将是: {total_needed:.2f}")
            
            # 检查初始资金是否足够
            if not self.use_all_balance:
                check_amount = self.total_investment if self.total_investment > 0 else self.initial_investment
                if check_amount > quote_available:
                    logger.warning(
                        f"设置的投资金额({check_amount:.2f} {self.quote_currency})超过可用余额({quote_available:.2f} {self.quote_currency})"
                    )
                    if self.total_investment > 0:
                        self.total_investment = quote_available * 0.95
                        # 重新计算初始投资额
                        self.initial_investment = calculate_initial_investment(
                            self.total_investment, self.max_level, self.multiplier
                        )
                        logger.info(
                            f"已调整总投资额为: {self.total_investment:.2f} {self.quote_currency}"
                        )
                        logger.info(
                            f"已调整初始投资额为: {self.initial_investment:.2f} {self.quote_currency}"
                        )
                    else:
                        self.initial_investment = quote_available * 0.95 / ((self.multiplier ** self.max_level - 1) / (self.multiplier - 1))
                        logger.info(
                            f"已调整初始投资额为: {self.initial_investment:.2f} {self.quote_currency}"
                        )

            # 取消所有现有订单，确保干净开始
            await self.cancel_all_orders()
            
            # 获取当前市场价格
            current_price = await self.get_market_price()
            logger.info(f"当前市场价格: {current_price}")
            
            # 第一轮使用当前价格初始化计划价格和下次加仓价格
            self.last_planned_buy_price = current_price  # 使用当前价格作为初始计划价格
            self.next_buy_price = round(current_price * self.price_drop_factor, self.pair_accuracy)
            logger.info(f"初始化下一次加仓价格: {self.next_buy_price}（基于当前价格: {current_price}）")
            
            # 开始第一级买入
            await self.place_buy_order(current_price)

            # 主循环，监控订单状态并根据需要调整
            while running:
                await self.check_order_status()
                
                # 如果买单未成交，检查是否需要取消并以更低价格重新买入（马丁格尔策略逻辑）
                if self.active_buy_order and not self.active_sell_order:
                    current_price = await self.get_market_price()
                    order_price = self.active_buy_order["price"]
                    planned_price = self.active_buy_order.get("planned_price", order_price)
                    
                    # 计算下一次加仓的触发价格 - 使用计划价格
                    next_level_price = round(planned_price * self.price_drop_factor, self.pair_accuracy)
                    
                    # 计算当前价格距离下一次加仓价格的百分比
                    price_gap_percent = round((current_price - next_level_price) / next_level_price * 100, 2)
                    
                    logger.info(f"当前价格: {current_price}, 买单价格: {order_price}, 计划价格: {planned_price}, 下一次加仓价格: {next_level_price} (再下跌 {price_gap_percent}%)")
                    
                    # 如果价格下跌超过设定阈值，取消当前订单并以更高级别重新买入
                    if current_price < planned_price * self.price_drop_factor and self.current_level < self.max_level - 1:
                        logger.info(f"价格下跌超过{self.price_drop_threshold * 100}%，取消当前买单并升级马丁格尔级别")
                        
                        # 取消现有买单
                        try:
                            await self.cancel_order(self.active_buy_order["id"])
                            logger.info(f"取消买单: {self.active_buy_order['id']}")
                        except Exception as e:
                            logger.error(f"取消买单失败: {e}")
                        
                        # 增加级别
                        self.current_level += 1
                        logger.info(f"升级到马丁格尔级别 {self.current_level}")
                        
                        # 重新放置买单
                        await self.place_buy_order(current_price)
                
                # 检查卖单状态
                if self.active_sell_order:
                    current_price = await self.get_market_price()
                    order_price = self.active_sell_order["price"]
                    
                    # 如果市场价格已经超过目标卖出价格但订单未成交，可能需要调整卖单价格
                    if current_price > order_price * 1.05:
                        logger.info(f"市场价格上涨较多，调整卖单价格")
                        
                        try:
                            # 取消现有卖单
                            self.bpx.orderCancel(self.symbol, self.active_sell_order["id"])
                            logger.info(f"取消卖单: {self.active_sell_order['id']}")
                            self.active_sell_order = None
                            
                            # 重新放置卖单，价格更高
                            await self.place_sell_order()
                        except Exception as e:
                            logger.error(f"调整卖单失败: {e}")
                
                # 睡眠一段时间再检查
                await asyncio.sleep(3)  # 从30秒减少到3秒，更快速地响应市场变化

        except Exception as e:
            logger.error(f"运行马丁格尔交易机器人失败: {e}")
            import traceback
            traceback.print_exc()

    async def orderQuery(self, order_id):
        """查询订单状态"""
        try:
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    response = await self.bpx.orderQuery(self.symbol, order_id)
                    return response
                except Exception as e:
                    if "Order not found" in str(e):
                        logger.info(f"订单 {order_id} 不存在")
                        return None
                    elif attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"查询订单 {order_id} 失败 (尝试 {attempt+1}/{max_retries}): {e}. 将在 {wait_time} 秒后重试...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"查询订单 {order_id} 失败，已达到最大重试次数: {e}")
                        raise
        except Exception as e:
            logger.error(f"查询订单 {order_id} 时发生异常: {e}", exc_info=True)
            return None

    @retry_on_network_error(max_retries=5, backoff_factor=1.5)
    async def get_active_orders(self):
        """获取当前活跃的订单列表"""
        try:
            active_orders = self.bpx.ordersQuery(self.symbol)
            return active_orders if active_orders else []
        except Exception as e:
            logger.error(f"获取活跃订单失败: {e}")
            return []

    def get_valid_quantity(self, quantity, start_precision=2, min_precision=0):
        """获取有效的订单数量精度
        
        Args:
            quantity: 原始数量
            start_precision: 起始精度
            min_precision: 最小可接受精度
            
        Returns:
            有效精度的数量
        """
        for precision in range(start_precision, min_precision - 1, -1):
            valid_quantity = round(quantity, precision)
            if valid_quantity > 0:  # 确保数量大于0
                return valid_quantity
                
        # 如果所有精度都尝试过仍然失败
        logger.error(f"无法找到合适的数量精度，原始数量: {quantity}")
        return round(quantity, min_precision)  # 返回最小精度结果

    async def initialize(self):
        """初始化配置"""
        try:
            base_accuracy = 2  # SOL的精度为2位小数
            quote_accuracy = 2  # USDC的精度为2位小数
            
            # 设置交易对精度 
            self.pair_accuracy = base_accuracy
            
            logger.info(f"初始化交易对精度: {self.base_currency}精度为{base_accuracy}位, {self.quote_currency}精度为{quote_accuracy}位")
            
            # 获取账户余额
            await self.get_account_balance()
            
            # 获取程序启动时的当前价格
            current_price = await self.get_market_price()
            
            # 其他初始化操作...
            return True
        except Exception as e:
            logger.error(f"初始化失败: {e}", exc_info=True)
            return False


def signal_handler(sig, frame):
    """处理退出信号"""
    global running
    logger.info("接收到退出信号，正在退出...")
    running = False

    sys.exit(0)


def calculate_initial_investment(total_investment: float, max_level: int, multiplier: float) -> float:
    """
    根据总投资额、最大级别和倍数计算初始投资额
    
    公式推导:
    total = initial + initial*multiplier + initial*multiplier^2 + ... + initial*multiplier^(max_level-1)
    total = initial * (1 + multiplier + multiplier^2 + ... + multiplier^(max_level-1))
    total = initial * (multiplier^max_level - 1) / (multiplier - 1)
    
    所以:
    initial = total * (multiplier - 1) / (multiplier^max_level - 1)
    """
    if max_level <= 0:
        return total_investment
    
    if multiplier == 1:
        return total_investment / max_level
    
    denominator = (multiplier ** max_level - 1) / (multiplier - 1)
    initial = total_investment / denominator
    
    return round(initial, 2)


async def main():
    """主函数"""

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    load_dotenv()

    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")

    if not api_key or not api_secret:
        logger.error("错误: API密钥未正确加载，请检查.env文件")
        sys.exit(1)

    symbol = os.getenv("TRADING_PAIR", "SOL_USDC")
    pair_accuracy = int(os.getenv("PAIR_ACCURACY", "2"))

    try:
        bpx_temp = BpxClient()
        bpx_temp.init(api_key=api_key, api_secret=api_secret)

        test_balance = bpx_temp.balances()
        if not test_balance:
            logger.error("错误: 无法获取账户余额，API连接失败")
            sys.exit(1)

        market_depth = Depth(symbol)
        current_price = round(
            (float(market_depth["asks"][0][0]) + float(market_depth["bids"][-1][0]))
            / 2,
            pair_accuracy,
        )
        logger.info(f"当前市场价格: {current_price}")

    except Exception as e:
        logger.error(f"获取市场价格失败: {e}")
        current_price = 0

    # 马丁格尔策略参数
    total_investment = float(os.getenv("TOTAL_INVESTMENT", "100"))
    max_level = int(os.getenv("MAX_LEVEL", "5"))
    multiplier = float(os.getenv("LEVEL_MULTIPLIER", "2.0"))
    
    # 根据总投资额计算初始投资额
    env_initial_investment = os.getenv("INITIAL_INVESTMENT")
    if env_initial_investment:
        initial_investment = float(env_initial_investment)
        logger.info(f"使用配置的初始投资额: {initial_investment}")
    else:
        initial_investment = calculate_initial_investment(total_investment, max_level, multiplier)
        logger.info(f"根据总投资额计算的初始投资额: {initial_investment}")
    
    profit_percentage = float(os.getenv("PROFIT_PERCENTAGE", "2.0"))
    fee_rate = float(os.getenv("FEE_RATE", "0.0008"))
    use_all_balance = os.getenv("USE_ALL_BALANCE", "false").lower() == "true"
    min_order_quantity = float(os.getenv("MIN_ORDER_QUANTITY", "0.01"))
    price_drop_threshold = float(os.getenv("PRICE_DROP_THRESHOLD", "2.0"))

    # 计算所有级别可能的总投资
    sum_investment = 0
    for level in range(max_level):
        level_investment = initial_investment * (multiplier ** level)
        sum_investment += level_investment
    
    logger.info(f"交易对: {symbol}")
    logger.info(f"总投资额: {total_investment}")
    logger.info(f"初始投资: {initial_investment}")
    logger.info(f"最大马丁格尔级别: {max_level}")
    logger.info(f"级别倍数: {multiplier}")
    logger.info(f"所有级别投资总和: {sum_investment:.2f}")
    logger.info(f"目标利润百分比: {profit_percentage}%")
    logger.info(f"使用全部余额: {use_all_balance}")
    logger.info(f"手续费率: {fee_rate * 100}%")
    logger.info(f"最小订单量: {min_order_quantity}")
    logger.info(f"价格下跌触发阈值: {price_drop_threshold}%")

    bot = MartingaleBot(
        api_key=api_key,
        api_secret=api_secret,
        symbol=symbol,
        initial_investment=initial_investment,
        max_level=max_level,
        profit_percentage=profit_percentage,
        multiplier=multiplier,
        pair_accuracy=pair_accuracy,
        fee_rate=fee_rate,
        use_all_balance=use_all_balance,
        total_investment=total_investment,
        min_order_quantity=min_order_quantity,
        price_drop_threshold=price_drop_threshold,
    )

    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
