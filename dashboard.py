# ============================================================
# dashboard.py — Dashboard Console (Rich)
# ============================================================

import logging
import threading
import time
from datetime import datetime
from typing import List, Optional, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.live import Live
from rich import box
import config

logger = logging.getLogger(__name__)
console = Console()


class TradingDashboard:
    """
    Dashboard console temps-réel avec Rich.
    Affiche : prix, sentiment, compte, positions, logs récents.
    """

    def __init__(self):
        self._state: Dict = {
            "price":       0.0,
            "bid":         0.0,
            "ask":         0.0,
            "spread":      0.0,
            "sentiment":   0.0,
            "sent_label":  "⚪ NEUTRE",
            "balance":     0.0,
            "equity":      0.0,
            "daily_pnl":   0.0,
            "daily_pnl_pct": 0.0,
            "drawdown":    0.0,
            "open_trades": 0,
            "positions":   [],
            "ai_action":   "HOLD",
            "ai_probs":    [0.25, 0.25, 0.25, 0.25],
            "last_news":   [],
            "status":      "🟢 EN COURS",
            "total_steps": 0,
            "last_update": datetime.now(),
        }
        self._log_buffer: List[str] = []
        self._lock = threading.Lock()
        self._running = False

    def update(self, **kwargs):
        """Met à jour l'état du dashboard (thread-safe)."""
        with self._lock:
            self._state.update(kwargs)
            self._state["last_update"] = datetime.now()

    def add_log(self, message: str):
        """Ajoute une ligne au buffer de logs."""
        with self._lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self._log_buffer.append(f"[dim]{timestamp}[/dim] {message}")
            if len(self._log_buffer) > 20:
                self._log_buffer.pop(0)

    def _build_layout(self) -> Layout:
        """Construit le layout Rich du dashboard."""
        with self._lock:
            s = dict(self._state)
            logs = list(self._log_buffer)

        layout = Layout()
        layout.split_column(
            Layout(name="header",   size=3),
            Layout(name="main",     ratio=1),
            Layout(name="footer",   size=3),
        )
        layout["main"].split_row(
            Layout(name="left",   ratio=2),
            Layout(name="right",  ratio=1),
        )
        layout["left"].split_column(
            Layout(name="market", size=9),
            Layout(name="positions"),
        )

        # ── Header ─────────────────────────────────────────────
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_text = Text(
            f"  🤖 XAUUSD AI TRADING BOT  |  {now}  |  {s['status']}",
            style="bold white on dark_blue",
            justify="center"
        )
        layout["header"].update(Panel(header_text, box=box.HEAVY))

        # ── Marché & Compte ────────────────────────────────────
        market_table = Table(box=box.SIMPLE_HEAVY, expand=True, show_header=False)
        market_table.add_column("Clé",    style="bold cyan",  width=20)
        market_table.add_column("Valeur", style="bold white")
        market_table.add_column("Clé2",   style="bold cyan",  width=20)
        market_table.add_column("Valeur2",style="bold white")

        pnl_color = "green" if s["daily_pnl"] >= 0 else "red"
        pnl_str   = f"[{pnl_color}]{s['daily_pnl']:+.2f} ({s['daily_pnl_pct']:+.2f}%)[/{pnl_color}]"
        dd_color  = "green" if s["drawdown"] < 1.0 else ("yellow" if s["drawdown"] < 2.0 else "red")

        ai_action_colors = {"HOLD": "white", "BUY": "green", "SELL": "red", "CLOSE": "yellow"}
        ai_color = ai_action_colors.get(s["ai_action"], "white")

        market_table.add_row("💰 XAUUSD",   f"{s['price']:.2f}",
                              "📊 Balance",  f"{s['balance']:.2f} USD")
        market_table.add_row("📈 Bid",       f"{s['bid']:.2f}",
                              "💳 Equity",   f"{s['equity']:.2f} USD")
        market_table.add_row("📉 Ask",       f"{s['ask']:.2f}",
                              "📅 PnL jour", pnl_str)
        market_table.add_row("↔️  Spread",    f"{s['spread']:.1f} pts",
                              "📉 Drawdown", f"[{dd_color}]{s['drawdown']:.2f}%[/{dd_color}]")
        market_table.add_row("📰 Sentiment", s["sent_label"],
                              "🤖 Action IA", f"[bold {ai_color}]{s['ai_action']}[/bold {ai_color}]")

        layout["market"].update(Panel(market_table, title="[bold]MARCHÉ & COMPTE[/bold]", box=box.ROUNDED))

        # ── Positions Ouvertes ─────────────────────────────────
        pos_table = Table(box=box.SIMPLE, expand=True)
        pos_table.add_column("Ticket",    style="dim",        width=10)
        pos_table.add_column("Type",      style="bold",       width=6)
        pos_table.add_column("Lots",      justify="right",    width=8)
        pos_table.add_column("Ouvert @",  justify="right",    width=10)
        pos_table.add_column("SL",        justify="right",    width=10)
        pos_table.add_column("TP",        justify="right",    width=10)
        pos_table.add_column("P&L",       justify="right",    width=10)

        positions = s.get("positions", [])
        if positions:
            for p in positions:
                pnl_clr = "green" if p["profit"] >= 0 else "red"
                type_clr = "green" if p["type"] == "BUY" else "red"
                pos_table.add_row(
                    str(p["ticket"]),
                    f"[{type_clr}]{p['type']}[/{type_clr}]",
                    str(p["volume"]),
                    f"{p['open_price']:.2f}",
                    f"{p['sl']:.2f}",
                    f"{p['tp']:.2f}",
                    f"[{pnl_clr}]{p['profit']:+.2f}[/{pnl_clr}]",
                )
        else:
            pos_table.add_row("—", "—", "—", "—", "—", "—", "—")

        layout["positions"].update(
            Panel(pos_table, title=f"[bold]POSITIONS OUVERTES ({len(positions)})[/bold]", box=box.ROUNDED)
        )

        # ── Probabilités IA & News ─────────────────────────────
        right_text = Text()
        probs = s.get("ai_probs", [0.25] * 4)
        action_names = ["HOLD", "BUY", "SELL", "CLOSE"]
        action_emojis = ["⏸", "🟢", "🔴", "🔵"]
        right_text.append("── PROBABILITÉS IA ──\n", style="bold cyan")
        for i, (name, prob, emoji) in enumerate(zip(action_names, probs, action_emojis)):
            bar_len = int(prob * 20)
            bar     = "█" * bar_len + "░" * (20 - bar_len)
            clr     = "green" if name == "BUY" else ("red" if name == "SELL" else "white")
            right_text.append(f" {emoji} {name:<6} ", style=f"bold {clr}")
            right_text.append(f"{bar} {prob*100:5.1f}%\n", style=clr)

        right_text.append("\n── DERNIÈRES NEWS ──\n", style="bold cyan")
        for news in s.get("last_news", [])[:3]:
            sent = news.get("sentiment_score", 0)
            clr  = "green" if sent > 0.1 else ("red" if sent < -0.1 else "white")
            title = news.get("title", "")[:45]
            right_text.append(f" [{clr}]{sent:+.2f}[/{clr}] {title}…\n")

        layout["right"].update(Panel(right_text, title="[bold]IA & NEWS[/bold]", box=box.ROUNDED))

        # ── Footer — Logs ──────────────────────────────────────
        log_text = Text()
        for line in logs[-3:]:
            log_text.append(line + "\n")

        layout["footer"].update(Panel(log_text, title="[bold]DERNIÈRES DÉCISIONS[/bold]", box=box.ROUNDED))

        return layout

    def run(self, refresh_rate: float = config.DASHBOARD_REFRESH):
        """Lance le dashboard en mode Live (bloque le thread)."""
        self._running = True
        try:
            with Live(self._build_layout(), refresh_per_second=1/refresh_rate, screen=True) as live:
                while self._running:
                    live.update(self._build_layout())
                    time.sleep(refresh_rate)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False

    def start_background(self):
        """Lance le dashboard dans un thread daemon."""
        t = threading.Thread(target=self.run, daemon=True)
        t.start()

    def stop(self):
        self._running = False