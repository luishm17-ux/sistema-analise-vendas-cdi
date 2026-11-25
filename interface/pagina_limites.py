"""Aba "Limites" (Aula 2): limites laterais, no infinito, continuidade, inexistência e propriedades."""

from __future__ import annotations

import numpy as np
import sympy as sp
import streamlit as st

from vendas_cdi import limites
from vendas_cdi.funcoes import para_numerica
from vendas_cdi.validacao import interpretar_expressao, interpretar_ponto

from .componentes import (
    COR_DESTAQUE,
    COR_PRINCIPAL,
    Contexto,
    figura,
    latex,
    mostrar_figura,
    mostrar_tabela,
    numero_br,
    seletor_funcao,
    tratar_erros,
)

EXEMPLOS = {
    "Fatoração: (t²−4)/(t−2) em t = 2": "(t^2-4)/(t-2)",
    "Limite fundamental: sen(t)/t em t = 0": "sin(t)/t",
    "Não existe (salto): |t|/t em t = 0": "abs(t)/t",
    "Assíntota vertical: 1/t em t = 0": "1/t",
    "Racional no infinito: (3t+1)/(t+2)": "(3*t+1)/(t+2)",
    "Oscilação: sen(t) no infinito": "sin(t)",
}
PONTOS_SUGERIDOS = {"(t^2-4)/(t-2)": "2", "sin(t)/t": "0", "abs(t)/t": "0", "1/t": "0", "(3*t+1)/(t+2)": "oo", "sin(t)": "oo"}


def renderizar(ctx: Contexto) -> None:
    st.subheader("∞ Limites — Aula 2")
    st.markdown(
        "O limite descreve **para onde a função se aproxima**. Nas vendas, `lim t→∞ V(t)` é o "
        "**teto do mercado**; nos pontos de quebra, os **limites laterais** revelam saltos."
    )
    aba_calc, aba_negocio, aba_prop = st.tabs(["🧮 Calculadora de limites", "🏪 Limites do modelo de vendas", "➕ Propriedades dos limites"])
    with aba_calc:
        _calculadora(ctx)
    with aba_negocio:
        _negocio(ctx)
    with aba_prop:
        _propriedades()


# --------------------------------------------------------------------------- #
def _calculadora(ctx: Contexto) -> None:
    expr, rotulo, eh_modelo = seletor_funcao(ctx, "lim", "(t^2-4)/(t-2)", EXEMPLOS)
    if expr is None:
        return
    padrao = "oo" if eh_modelo else PONTOS_SUGERIDOS.get(str(rotulo).replace("f(t) = ", ""), "2")
    c1, c2 = st.columns([2, 1])
    ponto_txt = c1.text_input("t → (ponto do limite)", value=padrao, key=f"lim_ponto_{eh_modelo}_{padrao}",
                              help="Um número (2, 0.5, 1/3), 'oo' (+∞) ou '-oo' (−∞).")
    direcao = c2.selectbox("Direção", ["ambos os lados", "pela esquerda (t → a⁻)", "pela direita (t → a⁺)"], key="lim_dir")
    mapa = {"ambos os lados": "ambos", "pela esquerda (t → a⁻)": "-", "pela direita (t → a⁺)": "+"}

    with tratar_erros("o limite"):
        ponto = interpretar_ponto(ponto_txt)
        r = limites.calcular_limite(expr, ponto, mapa[direcao])
        _mostrar_resultado(r, ponto, eh_modelo)
        st.markdown("**Tabela de aproximação**")
        mostrar_tabela(limites.tabela_aproximacao(expr, ponto))
        _grafico_limite(expr, ponto, r)


def _tex_valor(r: limites.ResultadoLimite) -> str:
    """Valor do limite em LaTeX (∞ vira \\infty; inexistente vira texto)."""
    return sp.latex(r.valor) if r.existe and r.valor is not None else r"\text{não existe}"


def _mostrar_resultado(r: limites.ResultadoLimite, ponto: sp.Expr, decimal: bool = False) -> None:
    p_tex = sp.latex(ponto)
    st.latex(rf"\lim_{{t \to {p_tex}}} \; {latex(r.expr, decimal)} \;=\; {_tex_valor(r)}")
    if r.classificacao == "finito":
        st.success(r.motivo)
    elif r.classificacao == "infinito":
        st.warning(r.motivo)
    else:
        st.error(r.motivo)
    if r.esquerda is not None:
        c1, c2 = st.columns(2)
        c1.metric("Limite pela esquerda  (t → a⁻)", str(r.esquerda).replace("oo", "∞"))
        c2.metric("Limite pela direita  (t → a⁺)", str(r.direita).replace("oo", "∞"))
    if r.continuidade:
        st.info("**Continuidade:** " + r.continuidade)


def _grafico_limite(expr: sp.Expr, ponto: sp.Expr, r: limites.ResultadoLimite) -> None:
    f = para_numerica(expr)
    if ponto in (sp.oo, -sp.oo):
        x = np.linspace(1, 60, 600) if ponto == sp.oo else np.linspace(-60, -1, 600)
    else:
        p = float(ponto)
        x = np.linspace(p - 5, p + 5, 1001)
        x = x[np.abs(x - p) > 1e-9]
    y = f(x)
    # cortes verticais enormes (assíntotas) deixam o gráfico ilegível: limitamos o eixo y
    finitos = y[np.isfinite(y)]
    fig = figura("Gráfico perto do ponto", "f(t)")
    fig.add_scatter(x=x, y=y, mode="lines", line=dict(color=COR_PRINCIPAL, width=3), name="f(t)", connectgaps=False)
    if finitos.size:
        lo, hi = np.percentile(finitos, [2, 98])
        margem = max(abs(hi - lo) * 0.25, 1.0)
        fig.update_yaxes(range=[lo - margem, hi + margem])
    if r.existe and r.classificacao == "finito" and r.valor is not None:
        v = float(sp.N(r.valor))
        if ponto in (sp.oo, -sp.oo):
            fig.add_hline(y=v, line_dash="dot", line_color=COR_DESTAQUE, annotation_text=f"assíntota y = {numero_br(v, 3)}")
        else:
            fig.add_scatter(x=[float(ponto)], y=[v], mode="markers", name="valor do limite",
                            marker=dict(size=11, color="white", line=dict(color=COR_DESTAQUE, width=3)))
    if ponto not in (sp.oo, -sp.oo):
        fig.add_vline(x=float(ponto), line_dash="dot", line_color="#94A3B8")
    mostrar_figura(fig, "lim_grafico")


# --------------------------------------------------------------------------- #
def _negocio(ctx: Contexto) -> None:
    st.markdown(f"Modelo ativo: **{ctx.rotulo}**")
    st.latex("V(t) = " + latex(ctx.expr, decimal=True))
    with tratar_erros("os limites do modelo"):
        sat = limites.limite_saturacao(ctx.expr)
        st.markdown("##### 1. Saturação do mercado: lim t→∞ V(t)")
        st.latex(rf"\lim_{{t\to\infty}} V(t) = {_tex_valor(sat)}")
        st.markdown(limites.interpretar_saturacao(sat))

        st.markdown("##### 2. Crescimento relativo no longo prazo: lim t→∞ V′(t)/V(t)")
        rel = limites.limite_taxa_relativa(ctx.expr)
        st.latex(rf"\lim_{{t\to\infty}} \dfrac{{V'(t)}}{{V(t)}} = {_tex_valor(rel)}")
        if rel.classificacao == "finito":
            v = float(sp.N(rel.valor))
            if abs(v) < 1e-12:
                st.markdown("O crescimento **percentual** tende a zero: as vendas estabilizam (ou crescem só em valor absoluto, não em proporção).")
            else:
                st.markdown(f"No longo prazo as vendas crescem cerca de **{numero_br(v * 100, 2)}% ao mês** (taxa contínua).")

        st.markdown("##### 3. A derivada como limite")
        t0 = st.slider("Mês t₀", 1, ctx.n_meses + ctx.horizonte, ctx.n_meses, key="lim_t0")
        d = limites.limite_definicao_derivada(ctx.expr, t0)
        st.latex(r"V'(t_0) = \lim_{h\to 0} \dfrac{V(t_0+h) - V(t_0)}{h}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Pelo limite (definição)", f"{numero_br(d['valor_limite'], 4)}")
        c2.metric("Pelas regras de derivação", f"{numero_br(d['valor_regra'], 4)}")
        c3.metric("Coincidem?", "✔ sim" if d["coincidem"] else "✘ não")
        st.caption("Os dois caminhos devem dar o mesmo número: a derivada *é* um limite. Veja a aba Derivadas para as regras.")


def _propriedades() -> None:
    st.markdown("Confere as propriedades operatórias: limite da soma, diferença, produto, constante e quociente.")
    c1, c2, c3 = st.columns([2, 2, 1])
    f_txt = c1.text_input("f(t) =", value="t^2 + 1", key="prop_f")
    g_txt = c2.text_input("g(t) =", value="3*t - 1", key="prop_g")
    p_txt = c3.text_input("t →", value="2", key="prop_p")
    with tratar_erros("as propriedades"):
        f, g, p = interpretar_expressao(f_txt), interpretar_expressao(g_txt), interpretar_ponto(p_txt)
        lf, lg = limites.calcular_limite(f, p), limites.calcular_limite(g, p)
        st.latex(rf"\lim f = {lf.valor_texto},\qquad \lim g = {lg.valor_texto}")
        mostrar_tabela(limites.propriedades_dos_limites(f, g, p))
        st.caption("Cada linha compara o limite calculado diretamente na combinação com o resultado de combinar os limites de f e g.")
