import numpy as np
import pandas as pd
import pandas_ta as ta
from base_strategy import Strategy

class SMACrossoverStrategy(Strategy):
    """
    Stratégie de Croisement de Moyennes Mobiles Simples (ex: SMA 50 / SMA 200).
    L'IA XGBoost filtrera les faux signaux de croisement via le Meta-Labeling.
    """

    def __init__(self, fast_period=50, slow_period=200, hold_period=24, sl_mult=2.0, tp_mult=3.0, atr_period=14):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.hold_period = hold_period
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult
        self.atr_period = atr_period

    def generate_dataset(self, ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        # Logged close price to match the logic (or just use raw)
        close_raw = ohlcv['close'].to_numpy()
        close = np.log(close_raw) # Use log prices to calculate return consistency
        
        # Calculate SMAs
        sma_fast = ta.sma(ohlcv['close'], length=self.fast_period).to_numpy()
        sma_slow = ta.sma(ohlcv['close'], length=self.slow_period).to_numpy()
        
        # Calculate Base contextual Features
        atr_raw = ta.atr(ohlcv['high'], ohlcv['low'], ohlcv['close'], length=self.atr_period)
        atr_arr = (atr_raw / ohlcv['close']).to_numpy() # Log scale equivalent ATR
        
        rsi_14 = ta.rsi(ohlcv['close'], length=14).to_numpy()
        adx_14 = ta.adx(ohlcv['high'], ohlcv['low'], ohlcv['close'], length=14)['ADX_14'].to_numpy()
        vol_arr = (ohlcv['volume'] / ohlcv['volume'].rolling(50).median()).to_numpy()
        
        # Distance de l'Asset vs sa SMA lente (exprime si le prix est tiré)
        price_to_sma200 = (close_raw - sma_slow) / sma_slow
        sma_diff        = (sma_fast - sma_slow) / sma_slow

        trades = pd.DataFrame()
        trade_i = 0
        in_trade = False
        tp_price = sl_price = hp_i = None

        # Start loop from where all indicators are filled
        start_idx = max(self.slow_period, self.atr_period)
        
        for i in range(start_idx, len(ohlcv)):
            # Croisement Haussier (Fast cross over Slow)
            bull_cross = (sma_fast[i-1] <= sma_slow[i-1]) and (sma_fast[i] > sma_slow[i])
            
            # Entry logic
            if not in_trade and bull_cross:
                # Log-space TP/SL execution
                tp_price = close[i] + atr_arr[i] * self.tp_mult
                sl_price = close[i] - atr_arr[i] * self.sl_mult
                hp_i     = i + self.hold_period
                in_trade = True

                trades.loc[trade_i, 'entry_i']   = i
                trades.loc[trade_i, 'entry_p']   = close[i]
                trades.loc[trade_i, 'atr']       = atr_arr[i]
                trades.loc[trade_i, 'sl']        = sl_price
                trades.loc[trade_i, 'tp']        = tp_price
                trades.loc[trade_i, 'hp_i']      = hp_i
                
                # Assign contextual features to trade metadata
                trades.loc[trade_i, 'rsi']             = rsi_14[i]
                trades.loc[trade_i, 'adx']             = adx_14[i]
                trades.loc[trade_i, 'vol']             = vol_arr[i]
                trades.loc[trade_i, 'price_to_sma200'] = price_to_sma200[i]
                trades.loc[trade_i, 'sma_diff']        = sma_diff[i]

            # Exit logic
            if in_trade:
                if close[i] >= tp_price or close[i] <= sl_price or i >= hp_i:
                    trades.loc[trade_i, 'exit_i'] = i
                    trades.loc[trade_i, 'exit_p'] = close[i]
                    in_trade = False
                    trade_i += 1

        if len(trades) == 0:
            return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=int)

        trades['return'] = trades['exit_p'] - trades['entry_p']

        # Construct ML Training Features matrix X and Target vector y
        feature_cols = ['rsi', 'adx', 'vol', 'price_to_sma200', 'sma_diff']
        data_x = trades[feature_cols]
        
        data_y = pd.Series(0, index=trades.index)
        data_y.loc[trades['return'] > 0] = 1

        return trades, data_x, data_y
