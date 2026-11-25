"""Componentes reutilizados pelas páginas: contexto, formatação, gráficos e tratamento de erros."""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sympy as sp

import streamlit as st

from vendas_cdi import analise
from vendas_cdi.dados import Ajuste
from vendas_cdi.derivadas import formatar_reais
from vendas_cdi.funcoes import TipoFuncao
from vendas_cdi.validacao import ValidacaoErro

COR_PRINCIPAL = "#0F766E"  # verde-azulado: modelo / função
COR_DESTAQUE = "#D97706"  # laranja: derivada / projeção
COR_DADOS = "#334155"  # cinza-azulado: dados observados
COR_AREA = "rgba(15, 118, 110, 0.18)"

# Streamlit mudou de use_container_width=True para width="stretch"; suportamos as duas.
_LARGURA = (
    {"width": "stretch"}
    if "width" in inspect.signature(st.plotly_chart).parameters
    else {"use_container_width": True}
)


@dataclass
class Contexto:
    """Modelo de vendas escolhido na barra lateral e usado por todas as páginas."""

    modo: str
    expr: sp.Expr
    rotulo: str
    n_meses: int
    horizonte: int
    df: pd.DataFrame | None = None  # t, mes, vendas (None na simulação)
    ajustes: list[Ajuste] | None = None
    escolhido: Ajuste | None = None


# --------------------------------------------------------------------------- #
# Erros
# --------------------------------------------------------------------------- #
@contextmanager
def tratar_erros(contexto: str = ""):
    """Mostra ``ValidacaoErro`` como aviso amigável e erros inesperados sem derrubar o app."""
    try:
        yield
    except ValidacaoErro as erro:
        st.error(f"⚠️ {erro}")
    except Exception as erro:  # noqa: BLE001 — a interface nunca deve mostrar stack trace
        st.error(f"Ocorreu um erro inesperado{' em ' + contexto if contexto else ''}: {type(erro).__name__}. "
                 "Tente outros valores ou reinicie a página.")


# --------------------------------------------------------------------------- #
# Formatação
# --------------------------------------------------------------------------- #
def para_exibicao(expr: sp.Expr, casas: int = 4) -> sp.Expr:
    """Troca frações e decimais longos por decimais curtos (só para *mostrar*; não afeta cálculos)."""
    expr = sp.sympify(expr)
    troca = {r: sp.Float(round(float(r), casas)) for r in expr.atoms(sp.Rational) if not r.is_Integer}
    troca.update({f: sp.Float(round(float(f), casas)) for f in expr.atoms(sp.Float)})
    return expr.xreplace(troca)


def latex(expr: sp.Expr, decimal: bool = False) -> str:
    e = para_exibicao(expr) if decimal else sp.sympify(expr)
    return sp.latex(e, ln_notation=True, mul_symbol="dot").replace(r"\log", r"\ln")


def reais(x: float) -> str:
    return "R$ " + formatar_reais(x)


def reais_compacto(x: float) -> str:
    """Para cartões estreitos: R$ 1,04 mi · R$ 450,6 mil · R$ 8.500,00."""
    a = abs(x)
    if a >= 1e9:
        return f"R$ {numero_br(x / 1e9, 2)} bi"
    if a >= 1e6:
        return f"R$ {numero_br(x / 1e6, 2)} mi"
    if a >= 1e5:
        return f"R$ {numero_br(x / 1e3, 1)} mil"
    return reais(x)


def numero_br(x: float, casas: int = 2) -> str:
    """1234.567 → '1.234,57' (padrão brasileiro)."""
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def mostrar_tabela(df: pd.DataFrame, **kwargs) -> None:
    st.dataframe(df, hide_index=True, **_largura_tabela(), **kwargs)


def _largura_tabela() -> dict:
    return {"width": "stretch"} if "width" in inspect.signature(st.dataframe).parameters else {"use_container_width": True}


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #
def figura(titulo: str, y: str, x: str = "Mês (t)", altura: int = 380) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=16)),
        xaxis_title=x,
        yaxis_title=y,
        height=altura,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        separators=",.",  # padrão brasileiro: vírgula decimal, ponto de milhar
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def mostrar_figura(fig: go.Figure, chave: str | None = None) -> None:
    st.plotly_chart(fig, key=chave, **_LARGURA)


def grade(a: float, b: float, n: int = 400) -> np.ndarray:
    return np.linspace(a, b, n)


# --------------------------------------------------------------------------- #
# Entrada de parâmetros de uma função
# --------------------------------------------------------------------------- #
def entrada_parametros(tipo: TipoFuncao, prefixo: str, container=None) -> dict[str, float]:
    """Cria um ``number_input`` por parâmetro e devolve o dicionário de valores digitados."""
    c = container or st
    valores: dict[str, float] = {}
    for p in tipo.parametros:
        kwargs = {}
        if p.minimo is not None:
            kwargs["min_value"] = float(p.minimo)
        elif p.estrito_positivo:
            kwargs["min_value"] = 0.0
        if p.maximo is not None:
            kwargs["max_value"] = float(p.maximo)
        valores[p.nome] = c.number_input(
            f"{p.nome} — {p.rotulo}",
            value=float(p.padrao),
            step=float(p.passo),
            key=f"{prefixo}_{tipo.chave}_{p.nome}",
            format="%g",
            **kwargs,
        )
    return valores


# --------------------------------------------------------------------------- #
# Relatório com cache (limites + derivadas + integrais são o trecho mais pesado)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Calculando limites, derivadas e integrais…", max_entries=64)
def relatorio_em_cache(expr_srepr: str, n_meses: int, horizonte: int, soma_observada: float | None):
    return analise.gerar_relatorio(sp.sympify(expr_srepr), n_meses, horizonte, soma_observada)


def gerar_relatorio(ctx: Contexto):
    soma = float(ctx.df["vendas"].sum()) if ctx.df is not None else None
    return relatorio_em_cache(sp.srepr(ctx.expr), ctx.n_meses, ctx.horizonte, soma)


def seletor_funcao(ctx: Contexto, chave: str, padrao_personalizada: str, exemplos: dict[str, str]) -> tuple[sp.Expr | None, str, bool]:
    """Escolha entre o modelo ativo e uma função digitada. Devolve (expr, rótulo, é_modelo)."""
    from vendas_cdi.validacao import interpretar_expressao

    origem = st.radio(
        "Função a analisar",
        ["Modelo de vendas ativo", "Função personalizada"],
        horizontal=True,
        key=f"{chave}_origem",
    )
    if origem == "Modelo de vendas ativo":
        st.caption(f"Modelo ativo: **{ctx.rotulo}**")
        st.latex("V(t) = " + latex(ctx.expr, decimal=True))
        return ctx.expr, ctx.rotulo, True

    col1, col2 = st.columns([2, 1])
    with col2:
        escolha = st.selectbox("Exemplos rápidos", ["(digitar)"] + list(exemplos), key=f"{chave}_exemplo")
    with col1:
        texto = st.text_input(
            "f(t) = ",
            value=exemplos[escolha] if escolha != "(digitar)" else padrao_personalizada,
            key=f"{chave}_texto_{escolha}",
            help="Use t como variável. Funções: sin, cos, tan, exp, ln, sqrt, abs. Potência: ^ (ex.: t^2).",
        )
    with tratar_erros("a função digitada"):
        expr = interpretar_expressao(texto)
        st.latex("f(t) = " + latex(expr))
        return expr, f"f(t) = {texto}", False
    return None, "", False
