"""Gera ``dados/vendas_exemplo.csv``: 24 meses de uma loja virtual fictícia.

Os dados seguem uma curva em S (mercado que satura) com leve sazonalidade e ruído.
Semente fixa: o arquivo gerado é sempre o mesmo.

    python scripts/gerar_dados_exemplo.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
meses = pd.period_range("2025-01", periods=24, freq="M")
t = np.arange(1, 25)

tendencia = 78000 / (1 + np.exp(-0.28 * (t - 11)))  # logística: L = 78 mil, ponto de inflexão ≈ mês 11
sazonal = 3500 * np.sin(2 * np.pi * t / 12 + 0.6)
ruido = rng.normal(0, 1200, size=t.size)
vendas = np.round(np.maximum(tendencia + sazonal + ruido, 1000), 2)

destino = Path(__file__).resolve().parent.parent / "dados" / "vendas_exemplo.csv"
pd.DataFrame({"mes": meses.astype(str), "vendas": vendas}).to_csv(destino, index=False)
print(f"Arquivo gerado: {destino}")
