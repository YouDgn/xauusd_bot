# ============================================================
# web_dashboard.py — Dashboard Web Temps Réel
# ============================================================
# Serveur FastAPI + WebSocket → s'ouvre automatiquement dans
# le navigateur. Contrôle total du bot (start/stop/positions).
# ============================================================

import asyncio
import json
import logging
import threading
import time
import webbrowser
from datetime import datetime
from typing import Optional, Set

logger = logging.getLogger(__name__)

# ── HTML Frontend ────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>XAUUSD AI BOT</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:        #080c10;
    --bg2:       #0d1117;
    --bg3:       #141b24;
    --border:    #1e2d3d;
    --gold:      #f0b429;
    --gold2:     #ffd166;
    --green:     #00e676;
    --red:       #ff3d57;
    --blue:      #40a9ff;
    --cyan:      #00e5ff;
    --gray:      #4a5568;
    --text:      #c9d1d9;
    --text2:     #8b949e;
    --font-mono: 'IBM Plex Mono', monospace;
    --font-sans: 'IBM Plex Sans', sans-serif;
  }
  * { margin:0; padding:0; box-sizing:border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 13px;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── Header ── */
  header {
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    padding: 0 24px;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .logo {
    font-family: var(--font-sans);
    font-weight: 600;
    font-size: 16px;
    color: var(--gold);
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .logo-icon {
    width: 28px; height: 28px;
    background: linear-gradient(135deg, var(--gold), #e67e00);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
  }

  .header-time {
    font-size: 12px;
    color: var(--text2);
    border-left: 1px solid var(--border);
    padding-left: 16px;
  }

  .status-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    border: 1px solid transparent;
    transition: all 0.3s;
  }

  .status-badge.running {
    background: rgba(0, 230, 118, 0.1);
    border-color: var(--green);
    color: var(--green);
  }

  .status-badge.stopped {
    background: rgba(255, 61, 87, 0.1);
    border-color: var(--red);
    color: var(--red);
  }

  .status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: currentColor;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
  }

  /* ── Controls ── */
  .controls {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .btn {
    padding: 7px 18px;
    border-radius: 4px;
    border: 1px solid;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn-start {
    background: rgba(0, 230, 118, 0.12);
    border-color: var(--green);
    color: var(--green);
  }
  .btn-start:hover { background: rgba(0, 230, 118, 0.25); }

  .btn-stop {
    background: rgba(255, 61, 87, 0.12);
    border-color: var(--red);
    color: var(--red);
  }
  .btn-stop:hover { background: rgba(255, 61, 87, 0.25); }

  .btn-close-all {
    background: rgba(240, 180, 41, 0.12);
    border-color: var(--gold);
    color: var(--gold);
  }
  .btn-close-all:hover { background: rgba(240, 180, 41, 0.25); }

  /* ── Layout Grid ── */
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    grid-template-rows: auto auto auto;
    gap: 1px;
    background: var(--border);
    padding: 0;
  }

  .card {
    background: var(--bg2);
    padding: 16px 20px;
  }

  .card-full   { grid-column: 1 / -1; }
  .card-half   { grid-column: span 2; }
  .card-third  { grid-column: span 1; }

  .card-title {
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text2);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .card-title::before {
    content: '';
    display: inline-block;
    width: 3px; height: 12px;
    background: var(--gold);
    border-radius: 2px;
  }

  /* ── Price Display ── */
  .price-main {
    font-size: 42px;
    font-weight: 600;
    color: var(--gold2);
    letter-spacing: -1px;
    line-height: 1;
    margin-bottom: 6px;
    font-variant-numeric: tabular-nums;
  }

  .price-change {
    font-size: 13px;
    color: var(--text2);
  }

  .price-change.up   { color: var(--green); }
  .price-change.down { color: var(--red); }

  .bid-ask {
    display: flex;
    gap: 24px;
    margin-top: 12px;
  }

  .bid-ask-item label {
    font-size: 10px;
    letter-spacing: 1px;
    color: var(--text2);
    display: block;
    margin-bottom: 2px;
  }

  .bid-val  { color: var(--red);   font-size: 16px; font-weight: 500; }
  .ask-val  { color: var(--green); font-size: 16px; font-weight: 500; }
  .spread-val { color: var(--text2); font-size: 16px; }

  /* ── Metrics Row ── */
  .metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--border);
  }

  .metric {
    background: var(--bg2);
    padding: 14px 18px;
  }

  .metric-label {
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text2);
    margin-bottom: 6px;
  }

  .metric-value {
    font-size: 22px;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
    color: var(--text);
  }

  .metric-value.positive { color: var(--green); }
  .metric-value.negative { color: var(--red); }
  .metric-value.gold     { color: var(--gold); }

  /* ── AI Probabilities ── */
  .ai-probs {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .prob-row {
    display: grid;
    grid-template-columns: 60px 1fr 48px;
    align-items: center;
    gap: 10px;
  }

  .prob-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
  }

  .prob-label.hold  { color: var(--text2); }
  .prob-label.buy   { color: var(--green); }
  .prob-label.sell  { color: var(--red); }
  .prob-label.close { color: var(--blue); }

  .prob-bar-bg {
    height: 6px;
    background: var(--bg3);
    border-radius: 3px;
    overflow: hidden;
  }

  .prob-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
  }

  .prob-bar-fill.hold  { background: var(--gray); }
  .prob-bar-fill.buy   { background: var(--green); }
  .prob-bar-fill.sell  { background: var(--red); }
  .prob-bar-fill.close { background: var(--blue); }

  .prob-pct {
    font-size: 12px;
    font-weight: 600;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }

  /* Current Action */
  .action-display {
    margin-top: 20px;
    padding: 12px 16px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border: 1px solid var(--border);
    background: var(--bg3);
  }

  .action-label { font-size: 10px; color: var(--text2); letter-spacing: 1px; }

  .action-value {
    font-size: 18px;
    font-weight: 600;
    letter-spacing: 2px;
  }

  .action-value.BUY   { color: var(--green); }
  .action-value.SELL  { color: var(--red); }
  .action-value.HOLD  { color: var(--text2); }
  .action-value.CLOSE { color: var(--blue); }

  /* ── Sentiment ── */
  .sentiment-bar-wrap {
    position: relative;
    height: 8px;
    background: linear-gradient(to right, var(--red), var(--gray) 50%, var(--green));
    border-radius: 4px;
    margin: 14px 0 6px;
  }

  .sentiment-needle {
    position: absolute;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 14px; height: 14px;
    background: white;
    border-radius: 50%;
    border: 2px solid var(--bg2);
    box-shadow: 0 0 6px rgba(255,255,255,0.5);
    transition: left 0.5s ease;
  }

  .sentiment-labels {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--text2);
  }

  .sentiment-value {
    text-align: center;
    font-size: 20px;
    font-weight: 600;
    margin-top: 10px;
  }

  /* ── Positions Table ── */
  .positions-table {
    width: 100%;
    border-collapse: collapse;
  }

  .positions-table th {
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text2);
    text-align: left;
    padding: 6px 10px;
    border-bottom: 1px solid var(--border);
  }

  .positions-table td {
    padding: 10px 10px;
    border-bottom: 1px solid rgba(30, 45, 61, 0.5);
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }

  .positions-table tr:last-child td { border-bottom: none; }

  .type-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
  }

  .type-badge.buy  { background: rgba(0,230,118,0.15); color: var(--green); }
  .type-badge.sell { background: rgba(255,61,87,0.15);  color: var(--red); }

  .no-positions {
    text-align: center;
    padding: 28px;
    color: var(--text2);
    font-size: 12px;
  }

  /* ── Logs ── */
  .log-container {
    max-height: 200px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
  }

  .log-entry {
    padding: 5px 0;
    border-bottom: 1px solid rgba(30,45,61,0.3);
    display: flex;
    gap: 12px;
    font-size: 11px;
    animation: fadeIn 0.3s ease;
  }

  @keyframes fadeIn { from { opacity:0; transform: translateY(-4px); } to { opacity:1; } }

  .log-time  { color: var(--text2); flex-shrink: 0; }
  .log-msg   { color: var(--text); }

  /* ── Macro Info ── */
  .macro-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
  }

  .macro-item {
    background: var(--bg3);
    padding: 10px 14px;
    border-radius: 4px;
    border: 1px solid var(--border);
  }

  .macro-item label {
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text2);
    display: block;
    margin-bottom: 4px;
  }

  .macro-item .val {
    font-size: 15px;
    font-weight: 500;
    color: var(--gold);
  }

  /* ── Session Indicators ── */
  .sessions {
    display: flex;
    gap: 8px;
    margin-top: 12px;
  }

  .session-pill {
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
    border: 1px solid var(--border);
    color: var(--text2);
    transition: all 0.3s;
  }

  .session-pill.active {
    background: rgba(0, 229, 255, 0.12);
    border-color: var(--cyan);
    color: var(--cyan);
  }

  /* ── Connection Status ── */
  .ws-status {
    font-size: 10px;
    color: var(--text2);
    display: flex;
    align-items: center;
    gap: 5px;
  }

  .ws-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--red);
    transition: background 0.3s;
  }

  .ws-dot.connected { background: var(--green); }

  /* ── PnL Chart ── */
  #pnlChart {
    width: 100%;
    height: 80px;
  }

  /* ── Responsive ── */
  @media (max-width: 1100px) {
    .grid { grid-template-columns: 1fr 1fr; }
    .metrics { grid-template-columns: repeat(2, 1fr); }
    .macro-grid { grid-template-columns: repeat(2, 1fr); }
  }

  /* ── Settings Panel ── */

  /* ── Active Params Bar ── */


  .param-chip {
    display: flex;
    align-items: center;
    gap: 5px;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 3px 10px;
    font-size: 11px;
    white-space: nowrap;
  }

  .param-chip .chip-label {
    color: var(--text2);
    font-size: 9px;
    letter-spacing: 1px;
    text-transform: uppercase;
  }

  .param-chip .chip-val {
    color: var(--gold);
    font-weight: 600;
  }

  .param-chip.active-manual {
    border-color: var(--cyan);
  }

  .param-chip.active-manual .chip-val {
    color: var(--cyan);
  }

  .param-divider {
    width: 1px;
    height: 16px;
    background: var(--border);
    margin: 0 4px;
  }
  /* ── LOGIN SCREEN ────────────────────────────────────── */
  .login-overlay {
    position: fixed; inset: 0; z-index: 9999;
    background: var(--bg);
    display: flex; align-items: center; justify-content: center;
  }
  .login-overlay.hidden { display: none; }
  .login-box {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 40px;
    width: 400px;
    display: flex; flex-direction: column; gap: 20px;
  }
  .login-logo {
    font-family: var(--font-mono);
    font-size: 22px;
    color: var(--gold);
    text-align: center;
    letter-spacing: 2px;
    margin-bottom: 8px;
  }
  .login-subtitle {
    font-size: 11px;
    color: var(--text2);
    text-align: center;
    margin-top: -14px;
  }
  .login-field { display: flex; flex-direction: column; gap: 6px; }
  .login-field label {
    font-size: 11px; color: var(--text2);
    font-family: var(--font-mono); letter-spacing: 1px;
  }
  .login-field input, .login-field select {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 13px;
    outline: none;
    transition: border 0.2s;
  }
  .login-field input:focus, .login-field select:focus {
    border-color: var(--gold);
  }
  .login-btn {
    background: var(--gold);
    color: #000;
    border: none;
    border-radius: 6px;
    padding: 12px;
    font-family: var(--font-mono);
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    letter-spacing: 1px;
    transition: opacity 0.2s;
    margin-top: 4px;
  }
  .login-btn:hover { opacity: 0.85; }
  .login-error {
    color: var(--red);
    font-size: 11px;
    text-align: center;
    font-family: var(--font-mono);
    display: none;
  }
  .login-error.visible { display: block; }

  /* ── TRADING PARAMS PANEL ────────────────────────────── */
  .tpanel-overlay {
    position: fixed; inset: 0; z-index: 1000;
    background: rgba(0,0,0,0.7);
    display: none; align-items: center; justify-content: center;
  }
  .tpanel-overlay.open { display: flex; }
  .tpanel-box {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px;
    width: 380px;
    display: flex; flex-direction: column; gap: 18px;
  }
  .tpanel-title {
    font-family: var(--font-mono);
    font-size: 13px; color: var(--gold);
    display: flex; justify-content: space-between; align-items: center;
    letter-spacing: 1px;
  }
  .tpanel-close {
    background: none; border: none;
    color: var(--text2); font-size: 16px; cursor: pointer;
  }
  .tpanel-field { display: flex; flex-direction: column; gap: 6px; }
  .tpanel-field label {
    font-size: 11px; color: var(--text2);
    font-family: var(--font-mono); letter-spacing: 1px;
  }
  .tpanel-field input, .tpanel-field select {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 9px 12px;
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 13px;
    outline: none;
  }
  .tpanel-field input:focus, .tpanel-field select:focus {
    border-color: var(--gold);
  }
  .tpanel-hint { font-size: 10px; color: var(--gray); margin-top: 2px; }
  .tpanel-footer { display: flex; gap: 10px; margin-top: 4px; }
  .tpanel-save {
    flex: 1; background: var(--gold); color: #000;
    border: none; border-radius: 6px; padding: 10px;
    font-family: var(--font-mono); font-size: 12px;
    font-weight: 600; cursor: pointer; letter-spacing: 1px;
  }
  .tpanel-cancel {
    flex: 1; background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 6px; padding: 10px; color: var(--text2);
    font-family: var(--font-mono); font-size: 12px; cursor: pointer;
  }
  .tpanel-applied {
    font-size: 11px; color: var(--green);
    text-align: center; font-family: var(--font-mono);
  }

</style>
</head>
<body>


<!-- ── LOGIN SCREEN ─────────────────────────────────── -->
<div class="login-overlay" id="loginOverlay">
  <div class="login-box">
    <div class="login-logo">⬡ XAUUSD AI BOT</div>
    <div class="login-subtitle">Connexion MetaTrader 5</div>

    <div class="login-field">
      <label>LOGIN (numéro de compte)</label>
      <input type="number" id="lLogin" placeholder="104235460">
    </div>
    <div class="login-field">
      <label>MOT DE PASSE</label>
      <input type="password" id="lPassword" placeholder="••••••••">
    </div>
    <div class="login-field">
      <label>SERVEUR</label>
      <input type="text" id="lServer" placeholder="MetaQuotes-Demo">
    </div>

    <div class="login-error" id="loginError">Identifiants invalides — réessayez</div>
    <button class="login-btn" onclick="doLogin()">SE CONNECTER →</button>
  </div>
</div>

<!-- ── TRADING PARAMS PANEL ──────────────────────────── -->
<div class="tpanel-overlay" id="tpanelOverlay" onclick="closeTpanelOutside(event)">
  <div class="tpanel-box">
    <div class="tpanel-title">
      ⚙ PARAMÈTRES TRADING
      <button class="tpanel-close" onclick="closeTpanel()">✕</button>
    </div>

    <div class="tpanel-field">
      <label>RISK : REWARD</label>
      <select id="tpRR">
        <option value="1.5">1 : 1.5</option>
        <option value="2.0">1 : 2.0</option>
        <option value="3.0" selected>1 : 3.0 (défaut IA)</option>
        <option value="4.0">1 : 4.0</option>
        <option value="5.0">1 : 5.0</option>
      </select>
      <div class="tpanel-hint">Ratio SL/TP — plus élevé = gains plus grands mais moins fréquents</div>
    </div>

    <div class="tpanel-field">
      <label>TAILLE DE LOT</label>
      <input type="number" id="tpLot" value="0.05" min="0.01" max="100" step="0.01">
      <div class="tpanel-hint">0.01 = micro lot | 0.10 = mini lot | 1.00 = lot standard</div>
    </div>

    <div class="tpanel-field">
      <label>PROTECTION COMPTE (%)</label>
      <input type="number" id="tpProtection" value="5" min="1" max="20" step="0.5">
      <div class="tpanel-hint">Bot s'arrête si perte session > ce % du compte</div>
    </div>

    <div class="tpanel-footer">
      <button class="tpanel-cancel" onclick="closeTpanel()">Annuler</button>
      <button class="tpanel-save" onclick="saveTpanel()">Appliquer</button>
    </div>
    <div class="tpanel-applied" id="tpanelApplied"></div>
  </div>
</div>

<header>
  <div class="header-left">
    <div class="logo">
      <div class="logo-icon">⚡</div>
      XAUUSD AI BOT
    </div>
    <div class="header-time" id="headerTime">--:--:--</div>
    <div class="ws-status">
      <div class="ws-dot" id="wsDot"></div>
      <span id="wsLabel">Connexion...</span>
    </div>
  </div>
  <div class="controls">
    <div class="status-badge stopped" id="statusBadge">
      <div class="status-dot"></div>
      <span id="statusText">ARRÊTÉ</span>
    </div>
    <button class="btn btn-start"     onclick="sendCmd('start')">▶ Démarrer</button>
    <button class="btn btn-stop"      onclick="sendCmd('stop')">■ Arrêter</button>
    <button class="btn btn-close-all" onclick="sendCmd('close_all')">✕ Fermer Positions</button>
    <button class="btn btn-settings" onclick="openTpanel()">⚙ Paramètres</button>
  </div>
</header>

<!-- Metrics Row -->
<div class="metrics">
  <div class="metric">
    <div class="metric-label">Balance MT5</div>
    <div class="metric-value gold" id="balance">0.00 $</div>
  </div>
  <div class="metric">
    <div class="metric-label">PnL Journalier</div>
    <div class="metric-value" id="dailyPnl">+0.00 $</div>
  </div>
  <div class="metric">
    <div class="metric-label">Drawdown</div>
    <div class="metric-value" id="drawdown">0.00%</div>
  </div>
  <div class="metric">
    <div class="metric-label">Positions Ouvertes</div>
    <div class="metric-value gold" id="openTrades">0</div>
  </div>
</div>

<!-- Main Grid -->
<div class="grid">

  <!-- Prix -->
  <div class="card card-third">
    <div class="card-title">XAUUSD</div>
    <div class="price-main" id="priceMain">0.00</div>
    <div class="bid-ask">
      <div class="bid-ask-item">
        <label>BID</label>
        <div class="bid-val" id="bid">0.00</div>
      </div>
      <div class="bid-ask-item">
        <label>ASK</label>
        <div class="ask-val" id="ask">0.00</div>
      </div>
      <div class="bid-ask-item">
        <label>SPREAD</label>
        <div class="spread-val" id="spread">0.0 pts</div>
      </div>
    </div>
  </div>

  <!-- IA Probabilities -->
  <div class="card card-third">
    <div class="card-title">Décision IA</div>
    <div class="ai-probs">
      <div class="prob-row">
        <span class="prob-label hold">HOLD</span>
        <div class="prob-bar-bg"><div class="prob-bar-fill hold" id="barHold" style="width:25%"></div></div>
        <span class="prob-pct" id="pctHold">25%</span>
      </div>
      <div class="prob-row">
        <span class="prob-label buy">BUY</span>
        <div class="prob-bar-bg"><div class="prob-bar-fill buy" id="barBuy" style="width:25%"></div></div>
        <span class="prob-pct" id="pctBuy">25%</span>
      </div>
      <div class="prob-row">
        <span class="prob-label sell">SELL</span>
        <div class="prob-bar-bg"><div class="prob-bar-fill sell" id="barSell" style="width:25%"></div></div>
        <span class="prob-pct" id="pctSell">25%</span>
      </div>
      <div class="prob-row">
        <span class="prob-label close">CLOSE</span>
        <div class="prob-bar-bg"><div class="prob-bar-fill close" id="barClose" style="width:25%"></div></div>
        <span class="prob-pct" id="pctClose">25%</span>
      </div>
    </div>
    <div class="action-display">
      <span class="action-label">ACTION IA</span>
      <span class="action-value HOLD" id="aiAction">HOLD</span>
    </div>
  </div>

  <!-- Sentiment + Macro -->
  <div class="card card-third">
    <div class="card-title">Sentiment & Macro</div>
    <div class="sentiment-bar-wrap">
      <div class="sentiment-needle" id="sentimentNeedle" style="left:50%"></div>
    </div>
    <div class="sentiment-labels">
      <span>BEARISH</span><span>NEUTRE</span><span>BULLISH</span>
    </div>
    <div class="sentiment-value" id="sentimentVal">0.00</div>

    <div class="sessions" id="sessions">
      <div class="session-pill" id="sessAsie">ASIE</div>
      <div class="session-pill" id="sessLondres">LONDRES</div>
      <div class="session-pill" id="sessNY">NEW YORK</div>
    </div>

    <div style="margin-top:14px; display:grid; grid-template-columns:1fr 1fr; gap:10px;">
      <div class="macro-item">
        <label>DXY</label>
        <div class="val" id="dxyVal">--</div>
      </div>
      <div class="macro-item">
        <label>US 10 ANS</label>
        <div class="val" id="us10yVal">--</div>
      </div>
    </div>
  </div>

  <!-- Positions -->
  <div class="card card-half">
    <div class="card-title">Positions Ouvertes</div>
    <table class="positions-table">
      <thead>
        <tr>
          <th>Ticket</th><th>Type</th><th>Lots</th>
          <th>Entrée</th><th>SL</th><th>TP</th><th>P&amp;L</th>
        </tr>
      </thead>
      <tbody id="positionsBody">
        <tr><td colspan="7" class="no-positions">Aucune position ouverte</td></tr>
      </tbody>
    </table>
  </div>

  <!-- PnL Mini Chart -->
  <div class="card card-third">
    <div class="card-title">Courbe PnL</div>
    <canvas id="pnlChart"></canvas>
    <div style="display:flex; justify-content:space-between; margin-top:6px; font-size:10px; color:var(--text2);">
      <span id="pnlMin">Min: 0</span>
      <span id="pnlMax">Max: 0</span>
    </div>
  </div>

  <!-- Logs -->
  <div class="card card-full">
    <div class="card-title">Journal des Décisions</div>
    <div class="log-container" id="logContainer"></div>
  </div>

</div>


<script>
// ── WebSocket ──────────────────────────────────────────────
let ws = null;
let pnlHistory = [];
let reconnectTimer = null;

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);

  ws.onopen = () => {
    document.getElementById('wsDot').classList.add('connected');
    document.getElementById('wsLabel').textContent = 'Connecté';
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  };

  ws.onclose = () => {
    document.getElementById('wsDot').classList.remove('connected');
    document.getElementById('wsLabel').textContent = 'Déconnecté';
    reconnectTimer = setTimeout(connect, 3000);
  };

  ws.onerror = () => ws.close();

  ws.onmessage = (e) => {
    try { updateDashboard(JSON.parse(e.data)); }
    catch(err) { console.error('Parse error:', err); }
  };
}

function sendCmd(cmd) {
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({ command: cmd }));
  } else {
    addLog('⚠️ WebSocket non connecté');
  }
}

// ── Dashboard Update ───────────────────────────────────────
function updateDashboard(d) {
  // ── Gestion réponse Login ──
  if (d.login_ok) {
    hideLogin();
    addLog('✅ Connexion MT5 réussie');
  }
  if (d.login_error) {
    showLoginError(d.login_error);
  }

  // Prix
  if (d.price) {
    document.getElementById('priceMain').textContent = d.price.toFixed(2);
    document.getElementById('bid').textContent       = d.bid?.toFixed(2) ?? '--';
    document.getElementById('ask').textContent       = d.ask?.toFixed(2) ?? '--';
    document.getElementById('spread').textContent    = (d.spread?.toFixed(1) ?? '0') + ' pts';
  }

  // Compte
  if (d.balance !== undefined) {
    const bal = document.getElementById('balance');
    bal.textContent = formatMoney(d.balance);
  }

  if (d.daily_pnl !== undefined) {
    const el = document.getElementById('dailyPnl');
    el.textContent = (d.daily_pnl >= 0 ? '+' : '') + formatMoney(d.daily_pnl) +
                     ' (' + (d.daily_pnl_pct >= 0 ? '+' : '') + (d.daily_pnl_pct ?? 0).toFixed(2) + '%)';
    el.className = 'metric-value ' + (d.daily_pnl >= 0 ? 'positive' : 'negative');

    // PnL Chart
    pnlHistory.push(d.daily_pnl);
    if (pnlHistory.length > 80) pnlHistory.shift();
    drawPnlChart();
  }

  if (d.drawdown !== undefined) {
    const el = document.getElementById('drawdown');
    el.textContent  = (d.drawdown * 100).toFixed(2) + '%';
    el.className    = 'metric-value ' + (d.drawdown > 0.02 ? 'negative' : '');
  }

  if (d.open_trades !== undefined)
    document.getElementById('openTrades').textContent = d.open_trades;

  // IA Probs
  if (d.ai_probs) {
    const labels = ['Hold','Buy','Sell','Close'];
    const ids    = ['Hold','Buy','Sell','Close'];
    d.ai_probs.forEach((p, i) => {
      const pct = (p * 100).toFixed(1) + '%';
      document.getElementById('bar' + ids[i]).style.width = pct;
      document.getElementById('pct' + ids[i]).textContent = pct;
    });
  }

  if (d.ai_action) {
    const el = document.getElementById('aiAction');
    el.textContent  = d.ai_action;
    el.className    = 'action-value ' + d.ai_action;
  }

  // Sentiment
  if (d.sentiment !== undefined) {
    const pct = ((d.sentiment + 1) / 2 * 100).toFixed(1);
    document.getElementById('sentimentNeedle').style.left = pct + '%';
    const sval = document.getElementById('sentimentVal');
    sval.textContent = (d.sentiment >= 0 ? '+' : '') + d.sentiment.toFixed(3);
    sval.style.color = d.sentiment > 0.1 ? 'var(--green)' :
                       d.sentiment < -0.1 ? 'var(--red)' : 'var(--text2)';
  }

  // Sessions
  if (d.macro) {
    document.getElementById('sessAsie').classList.toggle('active',    d.macro.session_asia === 1);
    document.getElementById('sessLondres').classList.toggle('active', d.macro.session_london === 1);
    document.getElementById('sessNY').classList.toggle('active',      d.macro.session_newyork === 1);
    if (d.macro.dxy_price)  document.getElementById('dxyVal').textContent   = d.macro.dxy_price.toFixed(2);
    if (d.macro.us10y_rate) document.getElementById('us10yVal').textContent = d.macro.us10y_rate.toFixed(2) + '%';
  }

  // Positions
  if (d.positions !== undefined) renderPositions(d.positions);

  // Status
  if (d.status) {
    const badge = document.getElementById('statusBadge');
    const txt   = document.getElementById('statusText');
    const s = d.status.toUpperCase();
    // EN COURS / RUNNING → vert
    // STOPPING / ARRET   → orange clignotant
    // STOPPED / ARRETE   → rouge
    const isRunning  = s === 'EN COURS' || s === 'RUNNING';
    const isStopping = s === 'STOPPING' || s === 'ARRET EN COURS';
    const isStopped  = s === 'STOPPED'  || s === 'ARRETE' || s === 'ARRÊTÉ';
    if (isRunning) {
      badge.className = 'status-badge running';
      txt.textContent = 'EN COURS';
    } else if (isStopping) {
      badge.className = 'status-badge stopping';
      txt.textContent = 'ARRÊT...';
    } else {
      badge.className = 'status-badge stopped';
      txt.textContent = 'ARRÊTÉ';
    }
  }

  // Logs
  if (d.log_message) addLog(d.log_message);
}

function renderPositions(positions) {
  const tbody = document.getElementById('positionsBody');
  if (!positions || positions.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="no-positions">Aucune position ouverte</td></tr>';
    return;
  }
  tbody.innerHTML = positions.map(p => {
    const typeClass = p.type === 'buy' ? 'buy' : 'sell';
    const pnlClass  = p.profit >= 0 ? 'color:var(--green)' : 'color:var(--red)';
    return `<tr>
      <td>${p.ticket}</td>
      <td><span class="type-badge ${typeClass}">${p.type.toUpperCase()}</span></td>
      <td>${p.volume?.toFixed(2) ?? '--'}</td>
      <td>${p.price_open?.toFixed(2) ?? '--'}</td>
      <td>${p.sl?.toFixed(2) ?? '--'}</td>
      <td>${p.tp?.toFixed(2) ?? '--'}</td>
      <td style="${pnlClass}">${p.profit >= 0 ? '+' : ''}${p.profit?.toFixed(2) ?? '--'} $</td>
    </tr>`;
  }).join('');
}

function addLog(msg) {
  const container = document.getElementById('logContainer');
  const now = new Date().toLocaleTimeString('fr-FR');
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.innerHTML = `<span class="log-time">${now}</span><span class="log-msg">${msg}</span>`;
  container.insertBefore(entry, container.firstChild);
  // Garder max 50 entrées
  while (container.children.length > 50) container.removeChild(container.lastChild);
}

// ── PnL Mini Chart ─────────────────────────────────────────
function drawPnlChart() {
  const canvas = document.getElementById('pnlChart');
  const ctx    = canvas.getContext('2d');
  const w = canvas.offsetWidth;
  const h = canvas.offsetHeight;
  canvas.width  = w;
  canvas.height = h;

  if (pnlHistory.length < 2) return;

  const min = Math.min(...pnlHistory);
  const max = Math.max(...pnlHistory);
  const range = max - min || 1;

  document.getElementById('pnlMin').textContent = 'Min: ' + min.toFixed(2) + '$';
  document.getElementById('pnlMax').textContent = 'Max: ' + max.toFixed(2) + '$';

  ctx.clearRect(0, 0, w, h);

  // Grid line at 0
  const zeroY = h - ((0 - min) / range) * h;
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, zeroY); ctx.lineTo(w, zeroY); ctx.stroke();

  // Line
  ctx.beginPath();
  pnlHistory.forEach((v, i) => {
    const x = (i / (pnlHistory.length - 1)) * w;
    const y = h - ((v - min) / range) * h;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  const lastVal = pnlHistory[pnlHistory.length - 1];
  ctx.strokeStyle = lastVal >= 0 ? '#00e676' : '#ff3d57';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Fill under line
  ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  const col  = lastVal >= 0 ? '0,230,118' : '255,61,87';
  grad.addColorStop(0, `rgba(${col},0.2)`);
  grad.addColorStop(1, `rgba(${col},0)`);
  ctx.fillStyle = grad;
  ctx.fill();
}

// ── Utils ──────────────────────────────────────────────────
function formatMoney(v) {
  return (v >= 0 ? '' : '') + v.toFixed(2) + ' $';
}

// ── Clock ──────────────────────────────────────────────────
setInterval(() => {
  document.getElementById('headerTime').textContent =
    new Date().toLocaleTimeString('fr-FR');
}, 1000);

// ── Settings ───────────────────────────────────────────────

function updateCapitalDisplay(capital) {
  document.getElementById('capitalAlloc').textContent = capital.toFixed(0) + ' $';
}

function refreshParamsBar(s) {
  if (!s) return;
  const rr = parseFloat(s.rr_ratio || 2.0);
  const sl = 1.5;
  const tp = parseFloat((sl * rr).toFixed(2));
  const isManual = s.use_manual_lot;

  document.getElementById('pCapital').textContent   = (s.capital || 10000).toLocaleString() + ' $';
  document.getElementById('pRisk').textContent       = (s.risk_pct || 2.0).toFixed(1) + '%';
  document.getElementById('pRR').textContent         = '1 : ' + rr.toFixed(1);
  document.getElementById('pSL').textContent         = sl.toFixed(1) + ' × ATR';
  document.getElementById('pTP').textContent         = tp.toFixed(1) + ' × ATR';
  document.getElementById('pMaxTrades').textContent  = s.max_trades || 3;
  document.getElementById('pKill').textContent       = '-' + (s.kill_switch || 3.0).toFixed(1) + '%';
  document.getElementById('pLeverage').textContent   = '1:' + (s.leverage || 100);

  const chipLotMode = document.getElementById('chipLotMode');
  const chipLotMin  = document.getElementById('chipLotMin');
  const chipLotMax  = document.getElementById('chipLotMax');

  if (isManual) {
    document.getElementById('pLotMode').textContent = 'Manuel : ' + (s.manual_lot || 0.1).toFixed(2);
    chipLotMode.classList.add('active-manual');
    chipLotMin.style.display  = 'none';
    chipLotMax.style.display  = 'none';
  } else {
    document.getElementById('pLotMode').textContent = 'Auto';
    chipLotMode.classList.remove('active-manual');
    document.getElementById('pLotMin').textContent  = s.lot_min || 0.01;
    document.getElementById('pLotMax').textContent  = s.lot_max || 1.0;
    chipLotMin.style.display  = 'flex';
    chipLotMax.style.display  = 'flex';
  }
}


// ── Init ───────────────────────────────────────────────────
// Charger les params actifs au démarrage
fetch('/settings').then(r => r.json()).then(s => {
  refreshParamsBar(s);
  updateCapitalDisplay(s.capital || 10000);
}).catch(() => {});

connect();

// ── LOGIN ──────────────────────────────────────────────
function doLogin() {
  const login    = document.getElementById('lLogin').value.trim();
  const password = document.getElementById('lPassword').value.trim();
  const server   = document.getElementById('lServer').value.trim();

  if (!login || !password || !server) {
    showLoginError('Remplissez tous les champs');
    return;
  }

  // Envoyer les credentials au bot via WS
  sendRaw({ command: 'mt5_login', login: parseInt(login), password, server });

  // Sauvegarder localement (session only)
  sessionStorage.setItem('mt5_creds', JSON.stringify({ login, password, server }));
}

function showLoginError(msg) {
  const el = document.getElementById('loginError');
  el.textContent = msg;
  el.classList.add('visible');
  setTimeout(() => el.classList.remove('visible'), 3000);
}

function hideLogin() {
  document.getElementById('loginOverlay').classList.add('hidden');
}

// ── TRADING PARAMS PANEL ──────────────────────────────
let tradingParams = {
  rr: 3.0,
  lot: 0.05,
  protection: 5.0
};

function openTpanel() {
  document.getElementById('tpRR').value        = tradingParams.rr;
  document.getElementById('tpLot').value       = tradingParams.lot;
  document.getElementById('tpProtection').value= tradingParams.protection;
  document.getElementById('tpanelApplied').textContent = '';
  document.getElementById('tpanelOverlay').classList.add('open');
}

function closeTpanel() {
  document.getElementById('tpanelOverlay').classList.remove('open');
}

function closeTpanelOutside(e) {
  if (e.target === document.getElementById('tpanelOverlay')) closeTpanel();
}

function saveTpanel() {
  tradingParams.rr         = parseFloat(document.getElementById('tpRR').value);
  tradingParams.lot        = parseFloat(document.getElementById('tpLot').value);
  tradingParams.protection = parseFloat(document.getElementById('tpProtection').value);

  sendRaw({ command: 'set_params', ...tradingParams });

  document.getElementById('tpanelApplied').textContent =
    `✓ Appliqué — RR 1:${tradingParams.rr} | Lot ${tradingParams.lot} | Protection ${tradingParams.protection}%`;

  setTimeout(closeTpanel, 1200);
}

function sendRaw(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

</script>
</body>
</html>
"""


class WebDashboardServer:
    """
    Serveur FastAPI + WebSocket pour le dashboard web.
    S'ouvre automatiquement dans le navigateur.
    """

    def __init__(self, port: int = 8765, bot_ref=None):
        self.port     = port
        self.bot_ref  = bot_ref   # Référence vers XAUUSDBot pour start/stop
        self._clients: Set = set()
        self._app     = None
        self._thread: Optional[threading.Thread] = None
        self._loop    = None

    def set_bot(self, bot):
        self.bot_ref = bot

    def _build_app(self):
        """Construit l'application FastAPI."""
        try:
            from fastapi import FastAPI, WebSocket, WebSocketDisconnect
            from fastapi.responses import HTMLResponse
        except ImportError:
            logger.error("FastAPI non installé. pip install fastapi uvicorn")
            return None

        app = FastAPI(title="XAUUSD AI Bot Dashboard")

        @app.get("/", response_class=HTMLResponse)
        async def root():
            return DASHBOARD_HTML


        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._clients.add(websocket)
            logger.info(f"Dashboard: nouveau client connecté ({len(self._clients)} total)")

            # Envoyer le statut actuel au nouveau client
            bot = self.bot_ref
            is_running = bot and getattr(bot, "_run_event", None) and bot._run_event.is_set()
            await websocket.send_text(json.dumps({
                "status": "EN COURS" if is_running else "STOPPED",
                "log_message": "Connecte au dashboard" if is_running else "Bot en attente — clique sur Demarrer"
            }))

            try:
                while True:
                    data = await websocket.receive_text()
                    await self._handle_command(json.loads(data), websocket)
            except Exception:
                pass
            finally:
                self._clients.discard(websocket)
                logger.info(f"Dashboard: client déconnecté ({len(self._clients)} restants)")

        return app

    async def _handle_command(self, data: dict, websocket):
        """Traite les commandes envoyées depuis le dashboard."""
        cmd = data.get("command")
        bot = self.bot_ref
        logger.info(f"Commande reçue : {cmd} | bot={bot is not None}")

        if cmd == "start":
            is_running = bot and getattr(bot, "_run_event", None) and bot._run_event.is_set()
            logger.info(f"start cmd | is_running={is_running}")
            if bot and not is_running:
                await self.broadcast({"log_message": "Demarrage du bot...", "status": "EN COURS"})
                t = threading.Thread(target=bot.start, daemon=True, name="BotStartThread")
                t.start()
                logger.info(f"Thread demarrage lance : {t.name}")
            else:
                await self.broadcast({"log_message": "Bot deja en cours"})

        elif cmd == "stop":
            if bot and getattr(bot, "_run_event", None) and bot._run_event.is_set():
                await self.broadcast({
                    "log_message": "Arret en cours...",
                    "status": "STOPPING"
                })
                def do_stop():
                    bot.stop()
                    self.broadcast_sync({
                        "status": "STOPPED",
                        "log_message": "Bot arrete. Cliquez Demarrer pour relancer."
                    })
                threading.Thread(target=do_stop, daemon=True).start()
            else:
                await self.broadcast({"status": "STOPPED", "log_message": "Bot non demarre"})

        elif cmd == "close_all":
            if bot and hasattr(bot, 'mt5'):
                await self.broadcast({"log_message": "Fermeture des positions en cours..."})
                def do_close():
                    n = bot.mt5.close_all_positions()
                    self.broadcast_sync({"log_message": f"Positions fermees : {n}"})
                threading.Thread(target=do_close, daemon=True).start()
            else:
                await self.broadcast({"log_message": "MT5 non connecte"})

        elif cmd == "mt5_login":
            # Credentials envoyés depuis l'écran de login
            login    = data.get("login", 0)
            password = data.get("password", "")
            server   = data.get("server", "")
            try:
                import config as _cfg
                import MetaTrader5 as _mt5
                
                # Nettoyer les données reçues
                login = int(login) if login else 0
                password = str(password).strip() if password else ""
                server = str(server).strip() if server else ""
                
                if not login or not password or not server:
                    await websocket.send_text(json.dumps({
                        "login_error": "Tous les champs sont requis"
                    }))
                    return
                
                # Test connexion rapide
                if not _mt5.initialize():
                    await websocket.send_text(json.dumps({
                        "login_error": "Impossible d'initialiser MT5"
                    }))
                    return
                
                try:
                    ok = _mt5.login(login=login, password=password, server=server)
                    if ok:
                        _cfg.MT5_LOGIN    = login
                        _cfg.MT5_PASSWORD = password
                        _cfg.MT5_SERVER   = server
                        _mt5.shutdown()
                        await websocket.send_text(json.dumps({"login_ok": True}))
                        logger.info(f"Login MT5 OK — compte {login} sur {server}")
                    else:
                        error_code = _mt5.last_error()
                        error_msg = f"Erreur MT5: {error_code[0]} - {error_code[1]}" if error_code else "Identifiants invalides"
                        _mt5.shutdown()
                        await websocket.send_text(json.dumps({"login_error": error_msg}))
                        logger.warning(f"Login MT5 échoué pour {login}: {error_msg}")
                except Exception as ex:
                    _mt5.shutdown()
                    await websocket.send_text(json.dumps({
                        "login_error": f"Erreur connexion: {str(ex)[:60]}"
                    }))
            except Exception as e:
                logger.error(f"mt5_login error: {e}")
                await websocket.send_text(json.dumps({
                    "login_error": f"Erreur: {str(e)[:60]}"
                }))

        elif cmd == "set_params":
            # Paramètres trading (RR, Lot, Protection)
            rr         = float(data.get("rr", 3.0))
            lot        = float(data.get("lot", 0.05))
            protection = float(data.get("protection", 5.0))
            try:
                import config as _cfg
                _cfg.TAKE_PROFIT_ATR_MULT = rr * _cfg.STOP_LOSS_ATR_MULT
                _cfg.DAILY_MAX_LOSS       = protection / 100.0
                # Lot size → stocké dans risk_manager si bot actif
                if bot and bot.risk_mgr:
                    bot.risk_mgr._manual_lot  = lot
                    bot.risk_mgr._use_manual  = True
                else:
                    _cfg.MANUAL_LOT_SIZE  = lot
                    _cfg.USE_MANUAL_LOT   = True
                await websocket.send_text(json.dumps({
                    "params_applied": True,
                    "rr": rr, "lot": lot, "protection": protection
                }))
                await self.broadcast({"log_message": f"Params: RR 1:{rr} | Lot {lot} | Protection {protection}%"})
                logger.info(f"Params mis a jour : RR=1:{rr} | Lot={lot} | Protection={protection}%")
            except Exception as e:
                logger.error(f"set_params erreur : {e}")

    async def broadcast(self, data: dict):
        """Envoie les données à tous les clients connectés."""
        if not self._clients:
            return
        msg = json.dumps(data)
        dead = set()
        for client in self._clients.copy():
            try:
                await client.send_text(msg)
            except Exception:
                dead.add(client)
        self._clients -= dead

    def broadcast_sync(self, data: dict):
        """Version synchrone fire-and-forget (depuis threads externes)."""
        if not self._loop:
            return
        try:
            # Fire and forget — ne bloque JAMAIS le thread appelant
            asyncio.run_coroutine_threadsafe(self.broadcast(data), self._loop)
        except Exception as e:
            logger.debug(f"broadcast_sync error: {e}")

    def start(self):
        """Démarre le serveur dans un thread séparé et ouvre le navigateur."""
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        # Ouvrir le navigateur après 1.5 secondes (laisse le serveur démarrer)
        threading.Timer(1.5, self._open_browser).start()
        logger.info(f"Dashboard web démarré sur http://localhost:{self.port}")

    def _run_server(self):
        """Lance uvicorn dans le thread."""
        try:
            import uvicorn
        except ImportError:
            logger.error("uvicorn non installé. pip install uvicorn")
            return

        app = self._build_app()
        if app is None:
            return

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.port,
            log_level="error",
            loop="asyncio",
        )
        server = uvicorn.Server(config)
        self._loop.run_until_complete(server.serve())

    def _open_browser(self):
        """Ouvre le dashboard dans le navigateur par défaut."""
        webbrowser.open(f"http://localhost:{self.port}")
        logger.info("Dashboard ouvert dans le navigateur.")