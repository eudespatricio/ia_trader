"""
scripts/maker_exec_test.py — O sinal de aprendizado mensal sobrevive a uma execução MAKER realista?

Modelo de execução (conservador, só com OHLC 1h; não há book/ticks):
- Entrada: ordem limite no preço de fechamento da barra de decisão (compra/venda passiva), válida por W=3 barras.
  Só preenche se o preço ATRAVESSAR o limite por >= 0,02% (trade-through) — sem fill por mero toque.
  Efeito de seleção adversa já embutido: só entramos nos casos em que o preço andou contra nós até o limite.
- Barreiras (TP/SL em ATR da decisão): TP = ordem limite (maker; exige trade-through de 0,02%);
  SL = stop a mercado (taker 0,06% + slippage 0,02%); timeout = mercado (taker + slippage).
- Na barra do fill só o SL pode disparar (a ordem intrabarra é desconhecida; TP ignorado nessa barra).
- Taxas: maker 0,02%, taker 0,06%, slippage de stop/mercado 0,02%.
- Rótulos de treino = resultados líquidos dessa execução (não preenchidos = fora do treino).
- Regra de entrada com custo estimado de 0,06% (fixo, pré-definido).
Uso: .venv/bin/python scripts/maker_exec_test.py --H 48 --tp 2.5 --sl 1.5
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import online_ml_search as ml  # noqa: E402

W = 3
TT = 0.0002
FEE_M, FEE_T, SLIP = 0.0002, 0.0006, 0.0002


def maker_outcomes(df, atr, H):
    n = len(df)
    c, hi, lo = df["close"].values, df["high"].values, df["low"].values
    tp_m, sl_m = ml.TP_M, ml.SL_M
    out = {}
    for side in ("long", "short"):
        res = np.full(n, np.nan)
        fill_w = np.zeros(n, dtype=int)
        for w in range(1, W + 1):
            b = np.full(n, np.nan)
            if side == "long":
                b[:n - w] = lo[w:]
                cond = (fill_w == 0) & ~np.isnan(b) & (b <= c * (1 - TT))
            else:
                b[:n - w] = hi[w:]
                cond = (fill_w == 0) & ~np.isnan(b) & (b >= c * (1 + TT))
            fill_w[cond] = w
        for w in range(1, W + 1):
            idx = np.where(fill_w == w)[0]
            if len(idx) == 0:
                continue
            e = c[idx]
            a = atr[idx]
            if side == "long":
                tp, sl = e + tp_m * a, e - sl_m * a
            else:
                tp, sl = e - tp_m * a, e + sl_m * a
            done = np.zeros(len(idx), dtype=bool)
            r = np.full(len(idx), np.nan)
            for k in range(0, H + 1):
                bi = idx + w + k
                valid = bi < n
                bi_c = np.where(valid, bi, 0)
                h_k, l_k = hi[bi_c], lo[bi_c]
                active = ~done & valid
                if side == "long":
                    sl_hit = active & (l_k <= sl)
                    tp_hit = active & (h_k >= tp * (1 + TT)) & ~sl_hit & (k > 0)
                    r[sl_hit] = (sl[sl_hit] * (1 - SLIP)) / e[sl_hit] - 1 - FEE_M - FEE_T
                    r[tp_hit] = tp[tp_hit] / e[tp_hit] - 1 - 2 * FEE_M
                else:
                    sl_hit = active & (h_k >= sl)
                    tp_hit = active & (l_k <= tp * (1 - TT)) & ~sl_hit & (k > 0)
                    r[sl_hit] = (e[sl_hit] - sl[sl_hit] * (1 + SLIP)) / e[sl_hit] - FEE_M - FEE_T
                    r[tp_hit] = (e[tp_hit] - tp[tp_hit]) / e[tp_hit] - 2 * FEE_M
                done |= sl_hit | tp_hit
                if k == H:
                    t_mask = ~done & valid
                    cb = c[bi_c]
                    if side == "long":
                        r[t_mask] = (cb[t_mask] * (1 - SLIP)) / e[t_mask] - 1 - FEE_M - FEE_T
                    else:
                        r[t_mask] = (e[t_mask] - cb[t_mask] * (1 + SLIP)) / e[t_mask] - FEE_M - FEE_T
            res[idx] = r
        out[side] = res
    return out


if __name__ == "__main__":
    ml.LABEL_FN = maker_outcomes
    ml.LABEL_COST = 0.0
    ml.EV_COST = 0.0006
    ml.HOLD_EXTRA = W
    ml.main()
    s = ml.SIGSTATS
    if s["signals"]:
        print(f"fill rate das ordens sinalizadas: {s['filled']}/{s['signals']} = {s['filled'] / s['signals'] * 100:.1f}%")
