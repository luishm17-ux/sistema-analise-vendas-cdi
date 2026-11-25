"""Aula 1 — Funções.

Seis tipos de função modelam o comportamento das vendas ao longo do tempo.
``V(t)`` é a receita mensal (em R$) no mês ``t``.

=============  ==========================================  ===========================
Tipo           Fórmula                                     Situação de vendas
=============  ==========================================  ===========================
Afim (linear)  V(t) = a·t + b                              crescimento constante
Quadrática     V(t) = a·t² + b·t + c                       crescimento que acelera/freia
Exponencial    V(t) = a·e^(k·t)                            crescimento viral / composto
Logarítmica    V(t) = a·ln(t + 1) + b                      crescimento que desacelera
Logística      V(t) = L / (1 + e^(−k·(t − t0)))            saturação de mercado
Sazonal        V(t) = c + a·t + A·sin(2π·t/P + φ)          tendência + vendas sazonais
=============  ==========================================  ===========================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import sympy as sp

from .validacao import ValidacaoErro, exigir_intervalo, exigir_numero, t

# --------------------------------------------------------------------------- #
# Descrição dos tipos
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Parametro:
    nome: str
    rotulo: str
    padrao: float
    passo: float = 0.1
    minimo: float | None = None
    maximo: float | None = None
    nao_zero: bool = False
    estrito_positivo: bool = False
    ajuda: str = ""


@dataclass(frozen=True)
class TipoFuncao:
    chave: str
    nome: str
    classificacao: str
    formula_latex: str
    descricao: str
    parametros: tuple[Parametro, ...]
    construtor: Callable[[dict], sp.Expr]
    dominio_imagem: Callable[[dict], tuple[str, str]]

    def valores_padrao(self) -> dict[str, float]:
        return {p.nome: p.padrao for p in self.parametros}

    def validar(self, valores: dict) -> dict[str, float]:
        """Confere cada parâmetro e devolve um dicionário só com ``float`` válidos."""
        limpos: dict[str, float] = {}
        for p in self.parametros:
            if p.nome not in valores:
                raise ValidacaoErro(f"Falta o parâmetro '{p.nome}' ({p.rotulo}).")
            limpos[p.nome] = exigir_numero(
                valores[p.nome],
                f"{p.nome} ({p.rotulo})",
                minimo=p.minimo,
                maximo=p.maximo,
                nao_zero=p.nao_zero,
                estrito_positivo=p.estrito_positivo,
            )
        return limpos

    def construir(self, valores: dict, validar: bool = True) -> sp.Expr:
        """Monta a expressão simbólica V(t) a partir dos parâmetros."""
        v = self.validar(valores) if validar else {k: float(x) for k, x in valores.items()}
        return self.construtor({k: sp.nsimplify(x, rational=True, tolerance=1e-12) if _eh_simples(x) else sp.Float(x) for k, x in v.items()})


def _eh_simples(x: float) -> bool:
    """Valores 'redondos' viram racionais exatos (0.5 → 1/2), mantendo as fórmulas legíveis."""
    return abs(x) < 1e9 and abs(x * 1000 - round(x * 1000)) < 1e-9


def _intervalo_imagem(ext_inf: float | None, ext_sup: float | None) -> str:
    esquerda = "(-∞" if ext_inf is None else f"[{ext_inf:g}"
    direita = "+∞)" if ext_sup is None else f"{ext_sup:g}]"
    return f"{esquerda}, {direita}"


# --- domínio e imagem teóricos (dependem dos parâmetros) -------------------- #
def _di_linear(p):
    return "ℝ", ("ℝ" if p["a"] != 0 else f"{{{p['b']:g}}}")


def _di_quadratica(p):
    if p["a"] == 0:
        return "ℝ", ("ℝ" if p["b"] != 0 else f"{{{p['c']:g}}}")
    vertice = p["c"] - p["b"] ** 2 / (4 * p["a"])
    return "ℝ", (_intervalo_imagem(vertice, None) if p["a"] > 0 else _intervalo_imagem(None, vertice))


def _di_exponencial(p):
    if p["a"] == 0:
        return "ℝ", "{0}"
    return "ℝ", ("(0, +∞)" if p["a"] > 0 else "(-∞, 0)")


def _di_logaritmica(p):
    return "(-1, +∞)  → no sistema usamos t ≥ 0", ("ℝ" if p["a"] != 0 else f"{{{p['b']:g}}}")


def _di_logistica(p):
    return "ℝ", f"(0, {p['L']:g})"


def _di_sazonal(p):
    if p["a"] == 0:
        return "ℝ", f"[{p['c'] - abs(p['A']):g}, {p['c'] + abs(p['A']):g}]"
    return "ℝ", "ℝ"


TIPOS: dict[str, TipoFuncao] = {
    "linear": TipoFuncao(
        chave="linear",
        nome="Afim (linear)",
        classificacao="Função afim (1º grau)",
        formula_latex=r"V(t) = a\,t + b",
        descricao="Vendas crescem (ou caem) um valor fixo por mês. 'a' é o aumento mensal e 'b' as vendas em t = 0.",
        parametros=(
            Parametro("a", "aumento mensal (R$/mês)", 1500.0, 50.0),
            Parametro("b", "vendas iniciais (R$)", 20000.0, 500.0),
        ),
        construtor=lambda p: p["a"] * t + p["b"],
        dominio_imagem=_di_linear,
    ),
    "quadratica": TipoFuncao(
        chave="quadratica",
        nome="Quadrática",
        classificacao="Função quadrática (2º grau)",
        formula_latex=r"V(t) = a\,t^2 + b\,t + c",
        descricao="Crescimento que acelera (a > 0) ou perde fôlego e atinge um pico (a < 0).",
        parametros=(
            Parametro("a", "curvatura (R$/mês²)", -25.0, 1.0),
            Parametro("b", "tendência inicial (R$/mês)", 1500.0, 50.0),
            Parametro("c", "vendas iniciais (R$)", 20000.0, 500.0),
        ),
        construtor=lambda p: p["a"] * t**2 + p["b"] * t + p["c"],
        dominio_imagem=_di_quadratica,
    ),
    "exponencial": TipoFuncao(
        chave="exponencial",
        nome="Exponencial",
        classificacao="Função exponencial",
        formula_latex=r"V(t) = a\,e^{k\,t}",
        descricao="Crescimento composto: cada mês as vendas crescem uma porcentagem fixa (≈ k·100 % ao mês).",
        parametros=(
            Parametro("a", "vendas iniciais (R$)", 15000.0, 500.0, estrito_positivo=True),
            Parametro("k", "taxa de crescimento contínuo (por mês)", 0.06, 0.005, minimo=-1.0, maximo=1.0),
        ),
        construtor=lambda p: p["a"] * sp.exp(p["k"] * t),
        dominio_imagem=_di_exponencial,
    ),
    "logaritmica": TipoFuncao(
        chave="logaritmica",
        nome="Logarítmica",
        classificacao="Função logarítmica",
        formula_latex=r"V(t) = a\,\ln(t+1) + b",
        descricao="Crescimento rápido no começo que desacelera com o tempo (retornos decrescentes).",
        parametros=(
            Parametro("a", "intensidade do crescimento (R$)", 12000.0, 500.0),
            Parametro("b", "vendas iniciais (R$)", 10000.0, 500.0),
        ),
        construtor=lambda p: p["a"] * sp.log(t + 1) + p["b"],
        dominio_imagem=_di_logaritmica,
    ),
    "logistica": TipoFuncao(
        chave="logistica",
        nome="Logística (saturação)",
        classificacao="Função logística (exponencial limitada)",
        formula_latex=r"V(t) = \dfrac{L}{1 + e^{-k\,(t - t_0)}}",
        descricao="Curva em S: cresce devagar, acelera e satura em L (capacidade máxima do mercado).",
        parametros=(
            Parametro("L", "limite de mercado (R$/mês)", 80000.0, 1000.0, estrito_positivo=True),
            Parametro("k", "velocidade de crescimento", 0.30, 0.01, estrito_positivo=True, maximo=5.0),
            Parametro("t0", "mês do ponto de inflexão", 12.0, 0.5),
        ),
        construtor=lambda p: p["L"] / (1 + sp.exp(-p["k"] * (t - p["t0"]))),
        dominio_imagem=_di_logistica,
    ),
    "sazonal": TipoFuncao(
        chave="sazonal",
        nome="Sazonal (trigonométrica)",
        classificacao="Função trigonométrica somada a uma reta",
        formula_latex=r"V(t) = c + a\,t + A\,\sin\!\left(\dfrac{2\pi\,t}{P} + \varphi\right)",
        descricao="Tendência de crescimento com oscilação periódica (Natal, Dia das Mães, férias...).",
        parametros=(
            Parametro("c", "vendas-base (R$)", 20000.0, 500.0),
            Parametro("a", "tendência (R$/mês)", 800.0, 50.0),
            Parametro("A", "amplitude sazonal (R$)", 5000.0, 250.0),
            Parametro("P", "período da sazonalidade (meses)", 12.0, 1.0, estrito_positivo=True, minimo=2.0, maximo=60.0),
            Parametro("fase", "fase φ (radianos)", 0.0, 0.1, minimo=-7.0, maximo=7.0),
        ),
        construtor=lambda p: p["c"] + p["a"] * t + p["A"] * sp.sin(2 * sp.pi * t / p["P"] + p["fase"]),
        dominio_imagem=_di_sazonal,
    ),
}


def obter_tipo(chave: str) -> TipoFuncao:
    if chave not in TIPOS:
        raise ValidacaoErro(f"Tipo de função desconhecido: {chave!r}. Opções: {', '.join(TIPOS)}.")
    return TIPOS[chave]


def construir(chave: str, valores: dict | None = None) -> sp.Expr:
    """Atalho: ``construir('linear', {'a': 2, 'b': 5})`` → ``2*t + 5``."""
    tipo = obter_tipo(chave)
    return tipo.construir(valores if valores is not None else tipo.valores_padrao())


# --------------------------------------------------------------------------- #
# Avaliação numérica, tabela, domínio e imagem
# --------------------------------------------------------------------------- #
def para_numerica(expr: sp.Expr) -> Callable[[np.ndarray], np.ndarray]:
    """Converte uma expressão em função que aceita arrays NumPy (valores inválidos viram NaN)."""
    # "scipy" primeiro: funções especiais que aparecem em antiderivadas (erf, Ei, Si…) só aceitam arrays aí.
    f = sp.lambdify(t, expr, modules=["scipy", "numpy"])

    def avaliar(x):
        x = np.asarray(x, dtype=float)
        with np.errstate(all="ignore"):
            y = np.asarray(f(x), dtype=float)
        if y.shape != x.shape:  # expressão constante
            y = np.full_like(x, float(y))
        y = np.where(np.isfinite(y), y, np.nan)
        return y

    return avaliar


def simplificar_seguro(expr: sp.Expr, limite_ops: int = 250) -> sp.Expr:
    """``simplify`` que só vale se a forma nova continuar numericamente bem-comportada.

    O SymPy às vezes "simplifica" ``250000·ln(u)`` para ``ln(u**250000)``, que estoura (inf/NaN)
    para valores comuns de t. Se a versão simplificada perder pontos finitos em relação à original,
    ficamos com a original.
    """
    expr = sp.sympify(expr)
    if sp.count_ops(expr) > limite_ops:
        return expr
    try:
        nova = sp.expand_log(sp.simplify(expr), force=True)
    except Exception:
        return expr
    x = np.linspace(0.37, 40.0, 25)
    antes, depois = para_numerica(expr)(x), para_numerica(nova)(x)
    if np.isfinite(depois).sum() < np.isfinite(antes).sum():
        return expr
    return nova


def valor_em(expr: sp.Expr, ponto: float) -> float:
    """V(ponto) como float; levanta ``ValidacaoErro`` se a função não estiver definida ali."""
    ponto = exigir_numero(ponto, "t")
    y = float(para_numerica(expr)(np.array([ponto]))[0])
    if np.isnan(y):
        raise ValidacaoErro(f"A função não está definida (ou é infinita) em t = {ponto:g}.")
    return y


def tabela(expr: sp.Expr, t_ini: float, t_fim: float, passos: int = 13) -> pd.DataFrame:
    """Tabela de entradas (t) e saídas (V(t)) em ``passos`` pontos igualmente espaçados."""
    t_ini, t_fim = exigir_intervalo(t_ini, t_fim)
    if not (2 <= int(passos) <= 2000):
        raise ValidacaoErro("O número de pontos da tabela deve estar entre 2 e 2000.")
    x = np.linspace(t_ini, t_fim, int(passos))
    return pd.DataFrame({"t (mês)": x, "V(t) (R$)": para_numerica(expr)(x)})


def _formatar_conjunto(c) -> str:
    """Escreve um conjunto do SymPy em notação de intervalos, como (-1, ∞) ou ℝ."""
    if c == sp.S.Reals:
        return "ℝ"
    if c == sp.S.EmptySet:
        return "∅"
    if isinstance(c, sp.Interval):
        ini = "-∞" if c.start == -sp.oo else str(c.start)
        fim = "∞" if c.end == sp.oo else str(c.end)
        return f"{'(' if c.left_open else '['}{ini}, {fim}{')' if c.right_open else ']'}"
    if isinstance(c, sp.Union):
        return " ∪ ".join(_formatar_conjunto(a) for a in c.args)
    if isinstance(c, sp.FiniteSet):
        return str(c).replace("oo", "∞")
    return str(c).replace("oo", "∞")


def dominio_simbolico(expr: sp.Expr) -> str:
    """Domínio (em ℝ) calculado pelo SymPy para qualquer expressão em t."""
    from sympy.calculus.util import continuous_domain

    try:
        return _formatar_conjunto(continuous_domain(expr, t, sp.S.Reals))
    except Exception:
        return "não foi possível determinar simbolicamente"


def imagem_numerica(expr: sp.Expr, t_ini: float, t_fim: float, amostras: int = 2001) -> tuple[float, float]:
    """Menor e maior valor de V(t) em ``[t_ini, t_fim]`` (por amostragem)."""
    t_ini, t_fim = exigir_intervalo(t_ini, t_fim)
    y = para_numerica(expr)(np.linspace(t_ini, t_fim, amostras))
    y = y[np.isfinite(y)]
    if y.size == 0:
        raise ValidacaoErro("A função não está definida em nenhum ponto do intervalo escolhido.")
    return float(y.min()), float(y.max())


def classificar(expr: sp.Expr) -> str:
    """Descreve o tipo de uma expressão qualquer (usado nas funções personalizadas)."""
    e = sp.sympify(expr)
    if not e.has(t):
        return "Constante"
    if e.has(sp.sin, sp.cos, sp.tan):
        return "Trigonométrica"
    if e.has(sp.log):
        return "Logarítmica"
    if e.has(sp.exp) or any(isinstance(b, sp.Pow) and b.exp.has(t) for b in sp.preorder_traversal(e)):
        return "Exponencial"
    if e.has(sp.Abs):
        return "Modular"
    try:
        grau = sp.Poly(e, t).degree()
        return {0: "Constante", 1: "Afim (1º grau)", 2: "Quadrática (2º grau)", 3: "Cúbica (3º grau)"}.get(
            grau, f"Polinomial de grau {grau}"
        )
    except sp.PolynomialError:
        return "Racional ou composta"
