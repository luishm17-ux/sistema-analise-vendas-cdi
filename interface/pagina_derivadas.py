"""Aba "Derivadas" (Aula 3): regras de derivação com passo a passo, verificação e leitura para vendas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from vendas_cdi import derivadas
from vendas_cdi.funcoes import para_numerica
from vendas_cdi.validacao import interpretar_expressao

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
    "Potência e soma: 3t² + 5t + 100": "3*t^2 + 5*t + 100",
    "Produto: t²·eᵗ": "t^2*exp(t)",
    "Quociente: t/(t+1)": "t/(t+1)",
    "Cadeia: (3t+1)⁵": "(3*t+1)^5",
    "Exponencial: 15000·e^(0,06t)": "15000*exp(0.06*t)",
    "Logaritmo: 12000·ln(t+1) + 10000": "12000*ln(t+1) + 10000",
    "Trigonométrica: 20000 + 800t + 5000·sen(πt/6)": "20000 + 800*t + 5000*sin(pi*t/6)",
}


def renderizar(ctx: Contexto) -> None:
    st.subheader("∂ Derivadas — Aula 3")
    st.markdown(
        "A derivada `V′(t)` é a **taxa de crescimento** das vendas (R$ por mês) e `V″(t)` diz se esse "
        "crescimento está **acelerando** ou **freando**. Cada regra é aplicada passo a passo e o resultado "
        "é conferido de duas formas independentes."
    )
    aba_calc, aba_regras = st.tabs(["🧮 Calculadora com passo a passo", "📚 Catálogo das regras"])
    with aba_calc:
        _calculadora(ctx)
    with aba_regras:
        _catalogo()


def _calculadora(ctx: Contexto) -> None:
    expr, rotulo, eh_modelo = seletor_funcao(ctx, "der", "3*t^2 + 5*t + 100", EXEMPLOS)
    if expr is None:
        return
    with tratar_erros("a derivada"):
        res = derivadas.derivar(expr)
        st.markdown("##### Resultado")
        st.latex(rf"\dfrac{{d}}{{dt}}\left[{latex(expr, decimal=eh_modelo)}\right] = {latex(res.derivada, decimal=eh_modelo)}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Regras usadas", len(res.regras_usadas))
        c2.metric("Confere com SymPy (simbólico)", "✔ sim" if res.confere_sympy else "✘ não")
        c3.metric("Confere com diferença finita", "✔ sim" if res.confere_numerico else "✘ não")
        st.markdown("**Regras aplicadas:** " + " · ".join(f"`{r}`" for r in res.regras_usadas))

        with st.expander("🪜 Passo a passo (cada aplicação de regra)", expanded=not eh_modelo):
            linhas = [{"Regra(s)": " + ".join(p.regras), "f(t)": str(p.f), "derivada desta etapa": str(p.df)} for p in res.passos]
            mostrar_tabela(pd.DataFrame(linhas))
            st.caption("As etapas aparecem na ordem em que o algoritmo desce pela expressão: primeiro a regra externa, depois as internas.")

        _grafico(expr, res)
        _leitura_vendas(ctx, expr, res, eh_modelo)


def _grafico(expr, res) -> None:
    c1, c2 = st.columns(2)
    xmin = c1.number_input("t inicial", value=0.0, step=1.0, key="der_xmin")
    xmax = c2.number_input("t final", value=36.0, step=1.0, key="der_xmax")
    if xmin >= xmax:
        st.error("⚠️ O início do intervalo deve ser menor que o fim.")
        return
    x = np.linspace(xmin, xmax, 500)
    fig = figura("f(t) e sua derivada f′(t)", "valor")
    fig.add_scatter(x=x, y=para_numerica(expr)(x), mode="lines", name="f(t)", line=dict(color=COR_PRINCIPAL, width=3))
    fig.add_scatter(x=x, y=para_numerica(res.derivada)(x), mode="lines", name="f′(t)", line=dict(color=COR_DESTAQUE, width=3, dash="dash"))
    fig.add_hline(y=0, line_color="#94A3B8", line_width=1)
    mostrar_figura(fig, "der_grafico")
    st.caption("Onde f′ > 0 a função sobe; onde f′ = 0 há pico, vale ou patamar; onde f′ < 0 a função desce.")


def _leitura_vendas(ctx: Contexto, expr, res, eh_modelo: bool) -> None:
    st.markdown("##### Leitura em vendas" if eh_modelo else "##### Leitura da derivada")
    t0 = st.slider("Mês t₀ para interpretar", 1, ctx.n_meses + ctx.horizonte, ctx.n_meses, key="der_t0")
    exato = float(para_numerica(res.derivada)(np.array([float(t0)]))[0])
    if eh_modelo:
        st.markdown("- " + derivadas.interpretar_taxa(expr, t0))
    elif np.isfinite(exato):
        sentido = "subindo" if exato > 0 else "descendo" if exato < 0 else "em um ponto estacionário"
        st.markdown(f"- Em t = {t0}, f′ = {numero_br(exato, 4)}: a função está {sentido}.")
    else:
        st.markdown(f"- A função não é derivável (ou não está definida) em t = {t0}.")
    with tratar_erros("a verificação numérica"):
        num = derivadas.derivada_numerica(expr, t0)
        st.markdown(f"- Verificação: f′({t0}) pela regra = **{numero_br(exato, 4)}**; pela diferença finita central = **{numero_br(num, 4)}**.")
    a, b = 1.0, float(ctx.n_meses + ctx.horizonte)
    an = derivadas.analisar_crescimento(expr, a, b)
    if an["criticos"]:
        for c in an["criticos"]:
            st.markdown(f"- f′(t) = 0 em **t ≈ {numero_br(c['t'], 2)}** → {c['tipo']}, f = {numero_br(c['V(t)'])}.")
    else:
        st.markdown(f"- Não há ponto com f′(t) = 0 em [{a:g}, {b:g}]: a função é monótona nesse intervalo.")
    mt = an["maior_taxa"]
    taxa_max = mt["V'(t)"]
    st.markdown(f"- Maior taxa de variação no intervalo: **{numero_br(taxa_max)}** em t ≈ {numero_br(mt['t'], 2)}.")


def _catalogo() -> None:
    st.markdown("Cada linha abaixo é **calculada ao vivo** pelo sistema (não é texto fixo).")
    linhas = []
    for regra, (formula, exemplo, esperado) in derivadas.CATALOGO_REGRAS.items():
        f = interpretar_expressao(exemplo) if "sign" not in esperado else interpretar_expressao("abs(t-2)")
        r = derivadas.derivar(f)
        linhas.append({"Regra": regra, "Exemplo f(t)": exemplo, "f′(t) calculada": str(r.derivada), "Verificada": "✔" if r.verificada else "✘"})
    mostrar_tabela(pd.DataFrame(linhas))
    st.markdown("**Fórmulas**")
    for regra, (formula, *_ ) in derivadas.CATALOGO_REGRAS.items():
        st.markdown(f"*{regra}*")
        st.latex(formula)
