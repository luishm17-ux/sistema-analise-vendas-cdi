"""Dados reais de vendas: leitura de CSV, variações e ajuste de modelo.

Fluxo da análise
----------------
1. ``carregar_csv`` lê e valida o arquivo (colunas ``mes`` e ``vendas``).
2. ``calcular_metricas`` calcula variações mês a mês, acumulado e médias móveis.
3. ``ajustar_modelos`` ajusta cada um dos 6 tipos de função (Aula 1) aos dados por mínimos
   quadrados e escolhe o melhor pelo R² ajustado.
4. O modelo escolhido alimenta os módulos de limites, derivadas e integrais.

O mês ``i`` (1, 2, …, n) é representado por ``t = i``.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import curve_fit

from .funcoes import TIPOS, para_numerica
from .validacao import ValidacaoErro

MINIMO_MESES = 6
MAXIMO_LINHAS = 5000
MAXIMO_BYTES = 2_000_000


# --------------------------------------------------------------------------- #
# Leitura e validação
# --------------------------------------------------------------------------- #
def carregar_csv(origem) -> pd.DataFrame:
    """Lê um CSV com a coluna ``vendas`` (obrigatória) e ``mes`` (opcional, só rótulo).

    ``origem`` pode ser caminho, bytes ou objeto de arquivo. Aceita vírgula, ponto e vírgula ou tab
    como separador e o padrão numérico brasileiro ("1.234,56", "1234,56") ou internacional ("1234.56").
    Um ponto seguido de exatamente três dígitos ("1.200") é lido como separador de milhar.
    """
    try:
        if isinstance(origem, (bytes, bytearray)):
            bruto = bytes(origem)
        elif hasattr(origem, "read"):
            bruto = origem.read()
            if isinstance(bruto, str):
                bruto = bruto.encode("utf-8")
        else:
            with open(origem, "rb") as arq:
                bruto = arq.read(MAXIMO_BYTES + 1)
    except OSError as erro:
        raise ValidacaoErro(f"Não consegui abrir o arquivo: {erro.strerror or erro}.") from None
    if len(bruto) > MAXIMO_BYTES:
        raise ValidacaoErro("Arquivo grande demais (máximo de 2 MB).")
    if not bruto.strip():
        raise ValidacaoErro("O arquivo está vazio.")
    try:
        texto = bruto.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = bruto.decode("latin-1")

    primeira_linha = texto.lstrip().splitlines()[0] if texto.strip() else ""
    separador = max([",", ";", "\t"], key=primeira_linha.count) if any(c in primeira_linha for c in ",;\t") else ","
    try:
        df = pd.read_csv(io.StringIO(texto), sep=separador, dtype=str, skip_blank_lines=True)
    except Exception:
        raise ValidacaoErro("Não consegui ler o arquivo como CSV. Use colunas 'mes' e 'vendas' separadas por vírgula ou ponto e vírgula.") from None

    df.columns = [str(c).strip().lower() for c in df.columns]
    if "vendas" not in df.columns:
        raise ValidacaoErro(f"O CSV precisa ter uma coluna chamada 'vendas'. Colunas encontradas: {', '.join(df.columns) or '(nenhuma)'}.")
    return validar_vendas(df)


def _para_numero(valor: str) -> float:
    s = str(valor).strip().replace("R$", "").replace(" ", "")
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return float("nan")
    if re.fullmatch(r"[1-9]\d{0,2}(\.\d{3})+", s):  # 1.200 ou 1.234.567 → ponto de milhar (padrão brasileiro)
        s = s.replace(".", "")
    elif "," in s and "." in s:  # 1.234,56
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def validar_vendas(df: pd.DataFrame) -> pd.DataFrame:
    """Confere o DataFrame e devolve colunas ``t``, ``mes`` e ``vendas`` prontas para uso."""
    if "vendas" not in df.columns:
        raise ValidacaoErro("Falta a coluna 'vendas'.")
    if len(df) > MAXIMO_LINHAS:
        raise ValidacaoErro(f"Linhas demais (máximo de {MAXIMO_LINHAS}).")
    valores = df["vendas"].map(_para_numero)
    invalidas = valores[~np.isfinite(valores)]
    if len(invalidas):
        linhas = ", ".join(str(i + 2) for i in invalidas.index[:5])  # +2: cabeçalho e base 1
        raise ValidacaoErro(f"Há valores de vendas vazios ou não numéricos (linhas do arquivo: {linhas}).")
    if (valores < 0).any():
        linhas = ", ".join(str(i + 2) for i in valores[valores < 0].index[:5])
        raise ValidacaoErro(f"Vendas não podem ser negativas (linhas do arquivo: {linhas}).")
    if len(valores) < MINIMO_MESES:
        raise ValidacaoErro(f"São necessários pelo menos {MINIMO_MESES} meses de dados (o arquivo tem {len(valores)}).")
    if float(valores.max()) <= 0:
        raise ValidacaoErro("Todas as vendas são zero: não há o que analisar.")

    rotulos = df["mes"].astype(str) if "mes" in df.columns else pd.Series([f"Mês {i}" for i in range(1, len(df) + 1)])
    return pd.DataFrame({
        "t": np.arange(1, len(valores) + 1, dtype=float),
        "mes": rotulos.values,
        "vendas": valores.astype(float).values,
    })


def dataframe_manual(texto: str) -> pd.DataFrame:
    """Aceita vendas digitadas (uma por linha ou separadas por vírgula/;)."""
    partes = [p for p in texto.replace(";", "\n").replace("\t", "\n").split("\n") if p.strip()]
    if len(partes) == 1 and partes[0].count(",") > 1:
        partes = [p for p in partes[0].split(",") if p.strip()]
    return validar_vendas(pd.DataFrame({"vendas": partes}))


# --------------------------------------------------------------------------- #
# Métricas descritivas
# --------------------------------------------------------------------------- #
def calcular_metricas(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta variação absoluta/percentual, acumulado e média móvel (3 meses)."""
    saida = df.copy()
    saida["variacao_abs"] = saida["vendas"].diff()
    anterior = saida["vendas"].shift(1)
    saida["variacao_pct"] = np.where(anterior > 0, saida["variacao_abs"] / anterior * 100, np.nan)
    saida["acumulado"] = saida["vendas"].cumsum()
    saida["media_movel_3"] = saida["vendas"].rolling(3).mean()
    return saida


def resumo(df: pd.DataFrame) -> dict:
    """Indicadores-chave (KPIs) de um histórico de vendas."""
    v = df["vendas"].to_numpy()
    n = len(v)
    cmgr = (v[-1] / v[0]) ** (1 / (n - 1)) - 1 if v[0] > 0 and n > 1 else float("nan")
    return {
        "meses": n,
        "total": float(v.sum()),
        "media": float(v.mean()),
        "melhor_mes": (str(df["mes"].iloc[int(v.argmax())]), float(v.max())),
        "pior_mes": (str(df["mes"].iloc[int(v.argmin())]), float(v.min())),
        "crescimento_total_pct": float((v[-1] / v[0] - 1) * 100) if v[0] > 0 else float("nan"),
        "taxa_media_mensal_pct": float(cmgr * 100),
    }


# --------------------------------------------------------------------------- #
# Ajuste de modelos
# --------------------------------------------------------------------------- #
@dataclass
class Ajuste:
    chave: str
    nome: str
    parametros: dict[str, float]
    expr: sp.Expr
    r2: float
    r2_ajustado: float
    rmse: float


def _metricas_ajuste(y: np.ndarray, previsto: np.ndarray, n_param: int) -> tuple[float, float, float]:
    residuo = y - previsto
    ss_res = float(np.sum(residuo**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n = len(y)
    r2_aj = 1 - (1 - r2) * (n - 1) / max(n - n_param - 1, 1)
    return r2, r2_aj, float(np.sqrt(ss_res / n))


def _ajustar_um(chave: str, t_: np.ndarray, y: np.ndarray) -> dict[str, float] | None:
    """Estima os parâmetros de um tipo por mínimos quadrados; ``None`` se não for aplicável."""
    n = len(t_)
    if chave == "linear":
        a, b = np.polyfit(t_, y, 1)
        return {"a": a, "b": b}
    if chave == "quadratica":
        a, b, c = np.polyfit(t_, y, 2)
        return {"a": a, "b": b, "c": c}
    if chave == "exponencial":
        if np.any(y <= 0):
            return None
        k, ln_a = np.polyfit(t_, np.log(y), 1)  # ln V = ln a + k t  (regressão linear em ln V)
        return {"a": float(np.exp(ln_a)), "k": float(k)}
    if chave == "logaritmica":
        a, b = np.polyfit(np.log(t_ + 1), y, 1)
        return {"a": a, "b": b}
    if chave == "logistica":
        def modelo(x, L, k, t0):
            return L / (1 + np.exp(-k * (x - t0)))
        try:
            p0 = (1.3 * y.max(), 4.0 / n, n / 2)
            (L, k, t0), _ = curve_fit(modelo, t_, y, p0=p0, bounds=([y.max() * 0.5, 1e-3, -n], [y.max() * 50, 5.0, 3 * n]), maxfev=20000)
            return {"L": L, "k": k, "t0": t0}
        except (RuntimeError, ValueError):
            return None
    if chave == "sazonal":
        P = 12.0
        if n < 12:  # sem um ciclo completo não dá para estimar sazonalidade
            return None
        X = np.column_stack([np.ones(n), t_, np.sin(2 * np.pi * t_ / P), np.cos(2 * np.pi * t_ / P)])
        (c, a, s, co), *_ = np.linalg.lstsq(X, y, rcond=None)
        amp = float(np.hypot(s, co))
        fase = float(np.arctan2(co, s))  # A·sin(x + φ) = A cosφ sin x + A sinφ cos x
        return {"c": c, "a": a, "A": amp, "P": P, "fase": fase}
    return None


def ajustar_modelos(df: pd.DataFrame) -> list[Ajuste]:
    """Ajusta todos os tipos aplicáveis e devolve a lista ordenada do melhor para o pior."""
    t_ = df["t"].to_numpy(float)
    y = df["vendas"].to_numpy(float)
    ajustes: list[Ajuste] = []
    for chave, tipo in TIPOS.items():
        try:
            params = _ajustar_um(chave, t_, y)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            params = None
        if params is None:
            continue
        params = {k: float(v) for k, v in params.items()}
        try:
            expr = tipo.construir(params, validar=False)
        except Exception:
            continue
        previsto = para_numerica(expr)(t_)
        if not np.all(np.isfinite(previsto)):
            continue
        r2, r2_aj, rmse = _metricas_ajuste(y, previsto, len(params))
        ajustes.append(Ajuste(chave, tipo.nome, params, expr, r2, r2_aj, rmse))
    if not ajustes:
        raise ValidacaoErro("Não foi possível ajustar nenhum modelo a esses dados.")
    return sorted(ajustes, key=lambda a: a.r2_ajustado, reverse=True)
