# ============================================================
# mt5_connector.py — Interface MetaTrader5
# ============================================================

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
from typing import Optional, Tuple, List, Dict
import config

logger = logging.getLogger(__name__)

# Mapping des timeframes
TIMEFRAME_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}


class MT5Connector:
    """Gère toutes les interactions avec MetaTrader 5."""

    def __init__(self):
        self.connected = False
        self.account_info = None

    # ── Connexion ──────────────────────────────────────────────

    def connect(self) -> bool:
        """Initialise et connecte à MT5."""
        if not mt5.initialize():
            logger.error(f"Échec initialisation MT5 : {mt5.last_error()}")
            return False

        if config.MT5_LOGIN:
            authorized = mt5.login(
                login=config.MT5_LOGIN,
                password=config.MT5_PASSWORD,
                server=config.MT5_SERVER
            )
            if not authorized:
                logger.error(f"Échec connexion compte : {mt5.last_error()}")
                mt5.shutdown()
                return False

        self.account_info = mt5.account_info()
        if self.account_info is None:
            logger.error("Impossible de récupérer les infos du compte.")
            return False

        self.connected = True
        logger.info(
            f"[OK] Connecte MT5 | Compte: {self.account_info.login} | "
            f"Broker: {self.account_info.company} | "
            f"Balance: {self.account_info.balance:.2f} {self.account_info.currency}"
        )
        return True

    def disconnect(self):
        """Ferme la connexion MT5."""
        mt5.shutdown()
        self.connected = False
        logger.info("Déconnexion MT5.")

    # ── Données de Marché ──────────────────────────────────────

    def get_historical_data(
        self,
        symbol: str = config.SYMBOL,
        timeframe: str = config.TIMEFRAME,
        years: int = config.TRAINING_YEARS
    ) -> Optional[pd.DataFrame]:
        """Télécharge X années d'historique OHLCV."""
        tf = TIMEFRAME_MAP.get(timeframe, mt5.TIMEFRAME_M15)
        utc_to   = datetime.now(pytz.utc)
        utc_from = utc_to - timedelta(days=365 * years)

        logger.info(f"[DL] Telechargement de {years} ans de données {symbol} [{timeframe}]...")

        # S'assurer que le symbole est actif
        mt5.symbol_select(symbol, True)
        import time
        time.sleep(0.5)

        # Calculer le nombre de barres nécessaires selon le timeframe
        bars_per_year = {
            "M1": 525600, "M5": 105120, "M15": 35040,
            "M30": 17520, "H1": 8760, "H4": 2190, "D1": 365
        }
        n_bars = bars_per_year.get(timeframe, 35040) * years

        # Méthode fiable : copy_rates_from_pos (pas de problème de timezone)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)

        if rates is None or len(rates) == 0:
            logger.error(f"copy_rates_from_pos échoué : {mt5.last_error()}")
            # Fallback : essayer avec moins de barres
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, 10000)
            if rates is None or len(rates) == 0:
                logger.error(f"Fallback échoué aussi : {mt5.last_error()}")
                return None
            logger.warning(f"Fallback utilisé : {len(rates)} barres seulement")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={
            "open": "Open", "high": "High",
            "low": "Low",  "close": "Close", "tick_volume": "Volume"
        }, inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.dropna(inplace=True)

        logger.info(f"[OK] {len(df)} barres chargées ({df.index[0]} → {df.index[-1]})")
        return df

    def get_latest_bars(
        self,
        symbol: str = config.SYMBOL,
        timeframe: str = config.TIMEFRAME,
        n_bars: int = config.LOOKBACK_BARS + 50
    ) -> Optional[pd.DataFrame]:
        """Récupère les N dernières barres OHLCV."""
        tf = TIMEFRAME_MAP.get(timeframe, mt5.TIMEFRAME_M15)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)

        if rates is None:
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={
            "open": "Open", "high": "High",
            "low": "Low",  "close": "Close", "tick_volume": "Volume"
        }, inplace=True)
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def get_tick(self, symbol: str = config.SYMBOL) -> Optional[Dict]:
        """Retourne le tick courant (bid/ask)."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return {
            "bid":   tick.bid,
            "ask":   tick.ask,
            "last":  tick.last,
            "spread": round((tick.ask - tick.bid) / mt5.symbol_info(symbol).point, 1),
            "time":  datetime.fromtimestamp(tick.time)
        }

    def get_symbol_info(self, symbol: str = config.SYMBOL) -> Optional[Dict]:
        """Retourne les informations du symbole."""
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return {
            "point":        info.point,
            "digits":       info.digits,
            "trade_contract_size": info.trade_contract_size,
            "volume_min":   info.volume_min,
            "volume_max":   info.volume_max,
            "volume_step":  info.volume_step,
        }

    # ── Compte ─────────────────────────────────────────────────

    def get_account_stats(self) -> Dict:
        """Retourne les statistiques du compte en temps réel."""
        info = mt5.account_info()
        if info is None:
            return {}
        return {
            "balance":  info.balance,
            "equity":   info.equity,
            "margin":   info.margin,
            "free_margin": info.margin_free,
            "profit":   info.profit,
            "leverage": info.leverage,
            "currency": info.currency,
        }

    # ── Ordres & Positions ─────────────────────────────────────

    def get_open_positions(self, symbol: str = config.SYMBOL) -> List[Dict]:
        """Retourne la liste des positions ouvertes."""
        positions = mt5.positions_get(symbol=symbol, magic=config.MAGIC_NUMBER)
        if positions is None:
            return []
        result = []
        for p in positions:
            result.append({
                "ticket":    p.ticket,
                "type":      "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume":    p.volume,
                "open_price": p.price_open,
                "sl":        p.sl,
                "tp":        p.tp,
                "profit":    p.profit,
                "open_time": datetime.fromtimestamp(p.time),
            })
        return result

    def place_order(
        self,
        action: str,         # "BUY" ou "SELL"
        lot_size: float,
        sl: float,
        tp: float,
        comment: str = "AI_BOT"
    ) -> Optional[Dict]:
        """Place un ordre au marché avec SL/TP."""
        symbol_info = mt5.symbol_info(config.SYMBOL)
        if symbol_info is None:
            logger.error(f"Symbole {config.SYMBOL} introuvable.")
            return None

        if not symbol_info.visible:
            mt5.symbol_select(config.SYMBOL, True)

        tick = mt5.symbol_info_tick(config.SYMBOL)
        if tick is None:
            logger.error("Impossible de récupérer le tick courant.")
            return None

        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
        price      = tick.ask if action == "BUY" else tick.bid

        # Normaliser le lot_size
        lot_size = round(
            max(config.LOT_MIN, min(config.LOT_MAX,
                round(lot_size / config.LOT_STEP) * config.LOT_STEP
            )), 2
        )

        request = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    config.SYMBOL,
            "volume":    lot_size,
            "type":      order_type,
            "price":     price,
            "sl":        round(sl, symbol_info.digits),
            "tp":        round(tp, symbol_info.digits),
            "deviation": config.DEVIATION,
            "magic":     config.MAGIC_NUMBER,
            "comment":   comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"[ERR] Ordre échoué ({action}): retcode={result.retcode}, comment={result.comment}")
            return None

        logger.info(
            f"[OK] Ordre exécuté | {action} {lot_size} lots @ {price:.2f} | "
            f"SL={sl:.2f} TP={tp:.2f} | Ticket={result.order}"
        )
        return {
            "ticket":    result.order,
            "action":    action,
            "lot_size":  lot_size,
            "price":     price,
            "sl":        sl,
            "tp":        tp,
        }

    def close_position(self, ticket: int) -> bool:
        """Ferme une position spécifique par son ticket."""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.warning(f"Position {ticket} introuvable.")
            return False

        pos = position[0]
        tick = mt5.symbol_info_tick(config.SYMBOL)

        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price      = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

        request = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    config.SYMBOL,
            "volume":    pos.volume,
            "type":      order_type,
            "position":  ticket,
            "price":     price,
            "deviation": config.DEVIATION,
            "magic":     config.MAGIC_NUMBER,
            "comment":   "AI_BOT_CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"[OK] Position {ticket} fermée @ {price:.2f}")
            return True
        else:
            logger.error(f"[ERR] Fermeture {ticket} échouée: {result.retcode}")
            return False

    def close_all_positions(self) -> int:
        """Ferme toutes les positions ouvertes. Retourne le nombre de fermetures."""
        positions = self.get_open_positions()
        closed = 0
        for pos in positions:
            if self.close_position(pos["ticket"]):
                closed += 1
        logger.info(f"close_all_positions : {closed} position(s) fermee(s).")
        return closed

    def get_daily_pnl(self, start_balance: float) -> Tuple[float, float]:
        """Retourne (PnL absolu, PnL en %) depuis le début de journée."""
        stats    = self.get_account_stats()
        equity   = stats.get("equity", start_balance)
        pnl_abs  = equity - start_balance
        pnl_pct  = pnl_abs / start_balance if start_balance > 0 else 0.0
        return pnl_abs, pnl_pct