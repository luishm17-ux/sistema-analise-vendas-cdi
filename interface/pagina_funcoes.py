"""Aba "Funções" (Aula 1): seis tipos de função com domínio, imagem, tabela e gráfico."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from vendas_cdi import funcoes
from vendas_cdi.funcoes import TIPOS, para_numerica
from vendas_cdi.validacao import exigir_intervalo, interpretar_expressao

from .componentes import (
    COR_PRINCIPAL,
    Contexto,
    entrada_parametros,
    figura,
    latex,
    mostrar_figura,
    mostrar_tabela,
    numero_br,
    tratar_erros,
)

PALETA = ["#0F766E", "#D97706", "#334155", "#7C3AED", "#BE123C", "#0369A1"]


def renderizar(ctx: Contexto) -> None:
    st.subheader("📈 Funções — Aula 1")
    st.markdown(
        "Cada tipo de função descreve um **comportamento de vendas** diferente. "
        "**Entrada:** o mês `t`. **Saída:** a receita mensal `V(t)` em R$."
    )

    aba_explorar, aba_comparar, aba_livre = st.tabs(["🔎 Explorar um tipo", "🆚 Comparar os 6 tipos", "✍️ Função personalizada"])

    with aba_explorar:
        _explorar()
    with aba_comparar:
        _comparar()
    with aba_livre:
        _personalizada()


def _explorar() -> None:
    esquerda, direita = st.columns([1, 2])
    with esquerda:
        chave = st.selectbox("Tipo de função", list(TIPOS), format_func=lambda k: TIPOS[k].nome, key="fn_tipo")
        tipo = TIPOS[chave]
        valores = entrada_parametros(tipo, "fn")
        t_ini = st.number_input("t inicial (mês)", value=0.0, step=1.0, key="fn_ini")
        t_fim = st.number_input("t final (mês)", value=36.0, step=1.0, key="fn_fim")
        pontos = st.slider("Pontos na tabela", 3, 50, 13, key="fn_pontos")

    with direita:
        with tratar_erros("a função"):
            expr = tipo.construir(valores)
            a, b = exigir_intervalo(t_ini, t_fim, "intervalo de t")
            st.markdown(f"**{tipo.classificacao}** — {tipo.descricao}")
            st.latex(tipo.formula_latex)
            st.latex("V(t) = " + latex(expr, decimal=True))

            dominio, imagem = tipo.dominio_imagem(tipo.validar(valores))
            vmin, vmax = funcoes.imagem_numerica(expr, a, b)
            c1, c2, c3 = st.columns([1, 1.3, 2])
            c1.metric("Domínio", dominio.split("  →")[0])
            c2.metric("Imagem (teórica)", imagem)
            c3.metric(f"Imagem em [{a:g}, {b:g}] (R$)", f"{numero_br(vmin, 0)} a {numero_br(vmax, 0)}")
            if "→" in dominio:
                st.caption("Domínio: " + dominio)

            x = np.linspace(a, b, 500)
            fig = figura(f"{tipo.nome}", "V(t) (R$)")
            fig.add_scatter(x=x, y=para_numerica(expr)(x), mode="lines", line=dict(color=COR_PRINCIPAL, width=3), name="V(t)")
            mostrar_figura(fig, "fn_grafico")

            st.markdown("**Tabela de entradas e saídas**")
            tabela = funcoes.tabela(expr, a, b, pontos)
            mostrar_tabela(tabela, column_config={
                "t (mês)": st.column_config.NumberColumn(format="%.2f"),
                "V(t) (R$)": st.column_config.NumberColumn(format="%.2f"),
            })


def _comparar() -> None:
    st.caption("Todos os tipos com os parâmetros padrão. Repare nas formas: reta, parábola, curva que explode, curva que freia, curva em S e onda.")
    fig = figura("Seis comportamentos de vendas", "V(t) (R$)")
    x = np.linspace(0, 36, 400)
    linhas = []
    for (chave, tipo), cor in zip(TIPOS.items(), PALETA):
        expr = tipo.construir(tipo.valores_padrao())
        fig.add_scatter(x=x, y=para_numerica(expr)(x), mode="lines", name=tipo.nome, line=dict(color=cor, width=2.5))
        dominio, imagem = tipo.dominio_imagem(tipo.valores_padrao())
        linhas.append({"Tipo": tipo.nome, "Classificação": tipo.classificacao, "Situação de vendas": tipo.descricao.split(".")[0], "Imagem": imagem})
    mostrar_figura(fig, "fn_comparar")
    mostrar_tabela(pd.DataFrame(linhas))


def _personalizada() -> None:
    texto = st.text_input("f(t) =", value="2*t^2 - 12*t + 30", key="fn_livre",
                          help="Use t como variável. Funções: sin, cos, tan, exp, ln, sqrt, abs. Potência: ^")
    c1, c2 = st.columns(2)
    a = c1.number_input("t inicial", value=-2.0, step=1.0, key="fn_livre_a")
    b = c2.number_input("t final", value=10.0, step=1.0, key="fn_livre_b")
    with tratar_erros("a função personalizada"):
        expr = interpretar_expressao(texto)
        a, b = exigir_intervalo(a, b, "intervalo de t")
        st.latex("f(t) = " + latex(expr))
        vmin, vmax = funcoes.imagem_numerica(expr, a, b)
        c1, c2, c3 = st.columns([1, 1.5, 1.5])
        c1.metric("Tipo", funcoes.classificar(expr))
        c2.metric("Domínio", funcoes.dominio_simbolico(expr))
        c3.metric(f"Imagem em [{a:g}, {b:g}]", f"{numero_br(vmin)} a {numero_br(vmax)}")
        x = np.linspace(a, b, 500)
        fig = figura("f(t)", "f(t)")
        fig.add_scatter(x=x, y=para_numerica(expr)(x), mode="lines", line=dict(color=COR_PRINCIPAL, width=3), name="f(t)")
        mostrar_figura(fig, "fn_livre_grafico")
        if np.isnan(para_numerica(expr)(x)).any():
            st.info("A função não está definida em alguns pontos do intervalo (aparecem como falhas no gráfico).")
        mostrar_tabela(funcoes.tabela(expr, a, b, 11))
