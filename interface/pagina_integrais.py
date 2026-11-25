"""Aba "Integrais" (Aula 4): antiderivadas, integral definida (receita acumulada) e somas de Riemann."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from vendas_cdi import integrais
from vendas_cdi.funcoes import para_numerica
from vendas_cdi.validacao import exigir_intervalo, interpretar_expressao

from .componentes import (
    COR_AREA,
    COR_DESTAQUE,
    COR_PRINCIPAL,
    Contexto,
    figura,
    latex,
    mostrar_figura,
    mostrar_tabela,
    numero_br,
    reais,
    seletor_funcao,
    tratar_erros,
)

EXEMPLOS = {
    "Potência e soma: 3t² + 2t + 1": "3*t^2 + 2*t + 1",
    "Exponencial: e^(2t)": "exp(2*t)",
    "Trigonométrica: sen(3t)": "sin(3*t)",
    "Substituição: (2t+1)³": "(2*t+1)^3",
    "Partes: t·eᵗ": "t*exp(t)",
    "Partes: ln(t)": "ln(t)",
    "Inverso: 1/t": "1/t",
    "Vendas lineares: 1000t + 5000": "1000*t + 5000",
}


def renderizar(ctx: Contexto) -> None:
    st.subheader("∫ Integrais — Aula 4")
    st.markdown(
        "A integral definida `∫ V(t) dt` é a **receita acumulada** entre dois meses. "
        "A integral indefinida (antiderivada) é a função cuja derivada devolve `V(t)`."
    )
    aba_calc, aba_riemann, aba_regras = st.tabs(["🧮 Calculadora com passo a passo", "📐 Somas de Riemann (a integral como limite)", "📚 Catálogo das regras"])
    with aba_calc:
        _calculadora(ctx)
    with aba_riemann:
        _riemann(ctx)
    with aba_regras:
        _catalogo()


def _calculadora(ctx: Contexto) -> None:
    expr, rotulo, eh_modelo = seletor_funcao(ctx, "int", "3*t^2 + 2*t + 1", EXEMPLOS)
    if expr is None:
        return

    st.markdown("##### Integral indefinida")
    indef = None
    with tratar_erros("a integral indefinida"):
        try:
            indef = integrais.integrar(expr)
        except integrais.IntegracaoNaoSimbolica as erro:
            st.warning(str(erro))
        if indef is not None:
            st.latex(rf"\int {latex(expr, decimal=eh_modelo)}\,dt = {latex(indef.antiderivada, decimal=eh_modelo)} + C")
            c1, c2 = st.columns(2)
            c1.metric("Regras usadas", len(indef.regras_usadas))
            c2.metric("Verificação: d/dt F = f", "✔ confere" if indef.verificada else "✘ não confere")
            st.markdown("**Regras aplicadas:** " + " · ".join(f"`{r}`" for r in indef.regras_usadas))
            with st.expander("🪜 Passo a passo", expanded=not eh_modelo):
                mostrar_tabela(pd.DataFrame([
                    {"Regra(s)": " + ".join(p.regras), "integrando": str(p.f), "antiderivada desta etapa": str(p.F)} for p in indef.passos
                ]))

    st.markdown("##### Integral definida — receita acumulada")
    c1, c2 = st.columns(2)
    padrao_b = float(ctx.n_meses) + 0.5 if eh_modelo else 3.0
    a = c1.number_input("De t = a", value=0.5 if eh_modelo else 0.0, step=0.5, key=f"int_a_{eh_modelo}")
    b = c2.number_input("até t = b", value=padrao_b, step=0.5, key=f"int_b_{eh_modelo}")
    with tratar_erros("a integral definida"):
        a, b = exigir_intervalo(a, b, "intervalo de integração")
        r = integrais.integral_definida(expr, a, b)
        valor = r.valor_exato if r.valor_exato is not None else r.valor_numerico
        st.latex(rf"\int_{{{a:g}}}^{{{b:g}}} {latex(expr, decimal=eh_modelo)}\,dt = F({b:g}) - F({a:g}) = {numero_br(valor, 4)}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Valor exato F(b) − F(a)", numero_br(r.valor_exato, 4) if r.valor_exato is not None else "sem forma fechada")
        m2.metric("Valor numérico (scipy.quad)", numero_br(r.valor_numerico, 4))
        m3.metric("Valor médio no intervalo", numero_br(r.media, 4), help="(1/(b−a))·∫ f dt")
        if r.valor_exato is not None:
            st.caption("✔ Os dois métodos coincidem." if r.confere else "⚠️ Os métodos divergem: confira a função e o intervalo.")
        if eh_modelo:
            st.success(f"Receita acumulada pelo modelo entre os meses {a:g} e {b:g}: **{reais(valor)}** "
                       f"(média de {reais(r.media)} por mês).")
        _grafico_area(expr, a, b, valor)


def _grafico_area(expr, a: float, b: float, valor: float) -> None:
    x = np.linspace(a, b, 500)
    y = para_numerica(expr)(x)
    fig = figura("Área sob a curva", "f(t)")
    fig.add_scatter(x=x, y=y, mode="lines", fill="tozeroy", fillcolor=COR_AREA, line=dict(color=COR_PRINCIPAL, width=3),
                    name=f"área = {numero_br(valor, 2)}")
    mostrar_figura(fig, "int_area")


def _riemann(ctx: Contexto) -> None:
    st.markdown(
        "A integral é o **limite** das somas de Riemann quando o número de retângulos cresce: "
        r"$\int_a^b f\,dt = \lim_{n\to\infty}\sum_{i=1}^{n} f(t_i^*)\,\Delta t$. Aumente `n` e veja o erro cair."
    )
    expr, rotulo, eh_modelo = seletor_funcao(ctx, "rie", "t^2", {"t² de 0 a 3": "t^2", "sen(t)": "sin(t)", "eᵗ": "exp(t)"})
    if expr is None:
        return
    c1, c2, c3, c4 = st.columns(4)
    a = c1.number_input("a", value=0.5 if eh_modelo else 0.0, step=0.5, key=f"rie_a_{eh_modelo}")
    b = c2.number_input("b", value=float(ctx.n_meses) + 0.5 if eh_modelo else 3.0, step=0.5, key=f"rie_b_{eh_modelo}")
    n = c3.slider("n (retângulos)", 1, 200, 12, key="rie_n")
    metodo = c4.selectbox("Altura do retângulo", ["ponto_medio", "esquerda", "direita"],
                          format_func=lambda m: {"ponto_medio": "ponto médio", "esquerda": "extremo esquerdo", "direita": "extremo direito"}[m],
                          key="rie_metodo")
    with tratar_erros("a soma de Riemann"):
        a, b = exigir_intervalo(a, b, "intervalo")
        exata = integrais.integral_definida(expr, a, b)
        ref = exata.valor_exato if exata.valor_exato is not None else exata.valor_numerico
        soma = integrais.soma_de_riemann(expr, a, b, n, metodo)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Soma de Riemann (n = {n})", numero_br(soma, 4))
        m2.metric("Integral exata", numero_br(ref, 4))
        m3.metric("Erro absoluto", numero_br(abs(soma - ref), 6))

        dx = (b - a) / n
        desloc = {"esquerda": 0.0, "direita": 1.0, "ponto_medio": 0.5}[metodo]
        xs = a + (np.arange(n) + desloc) * dx
        ys = para_numerica(expr)(xs)
        xc = np.linspace(a, b, 400)
        fig = figura("Retângulos de Riemann", "f(t)")
        fig.add_bar(x=a + (np.arange(n) + 0.5) * dx, y=ys, width=dx, name="retângulos", marker_color=COR_AREA, marker_line_color=COR_PRINCIPAL, marker_line_width=1)
        fig.add_scatter(x=xc, y=para_numerica(expr)(xc), mode="lines", name="f(t)", line=dict(color=COR_DESTAQUE, width=3))
        mostrar_figura(fig, "rie_grafico")
        st.markdown("**Convergência (ponto médio)**")
        mostrar_tabela(integrais.tabela_riemann(expr, a, b), column_config={
            "Soma de Riemann": st.column_config.NumberColumn(format="%.6f"),
            "Erro absoluto": st.column_config.NumberColumn(format="%.2e"),
        })


def _catalogo() -> None:
    st.markdown("Cada linha abaixo é **calculada ao vivo** pelo sistema e conferida derivando o resultado.")
    linhas = []
    for regra, (formula, exemplo, esperado) in integrais.CATALOGO_REGRAS.items():
        r = integrais.integrar(interpretar_expressao(exemplo))
        linhas.append({"Regra": regra, "Integrando f(t)": exemplo, "F(t) calculada (+ C)": str(r.antiderivada), "d/dt F = f": "✔" if r.verificada else "✘"})
    mostrar_tabela(pd.DataFrame(linhas))
    st.markdown("**Fórmulas**")
    for regra, (formula, *_ ) in integrais.CATALOGO_REGRAS.items():
        st.markdown(f"*{regra}*")
        st.latex(formula)
