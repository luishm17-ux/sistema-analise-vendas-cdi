"""Aba "Análise de vendas": o painel que junta funções, limites, derivadas e integrais."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from vendas_cdi import dados, integrais
from vendas_cdi.funcoes import para_numerica

from .componentes import (
    COR_AREA,
    COR_DADOS,
    COR_DESTAQUE,
    COR_PRINCIPAL,
    Contexto,
    figura,
    gerar_relatorio,
    latex,
    mostrar_figura,
    mostrar_tabela,
    numero_br,
    reais,
    reais_compacto,
    tratar_erros,
)


def renderizar(ctx: Contexto) -> None:
    st.subheader("📊 Análise de vendas")
    st.caption(
        "O modelo V(t) descreve a receita mensal. A **derivada** mostra a velocidade de crescimento, "
        "o **limite** mostra para onde as vendas tendem e a **integral** mostra a receita acumulada."
    )
    with tratar_erros("a análise de vendas"):
        rel = gerar_relatorio(ctx)
        _indicadores(ctx, rel)
        _graficos(ctx, rel)
        st.markdown("#### 🧭 Interpretação dos resultados")
        for texto in rel.textos:
            st.markdown(f"- {texto}")
        _tabelas(ctx)


# --------------------------------------------------------------------------- #
def _indicadores(ctx: Contexto, rel) -> None:
    if ctx.df is not None:
        r = dados.resumo(ctx.df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Receita total (real)", reais_compacto(r["total"]), help=f"Soma exata: {reais(r['total'])}")
        c2.metric("Média mensal", reais(r["media"]))
        c3.metric("Crescimento no período", f"{numero_br(r['crescimento_total_pct'], 1)}%",
                  help="Do primeiro ao último mês.")
        c4.metric("Taxa média mensal", f"{numero_br(r['taxa_media_mensal_pct'], 1)}% a.m.",
                  help="Taxa composta: (V_final / V_inicial)^(1/(n−1)) − 1.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Taxa de crescimento V′({ctx.n_meses})", reais(rel.taxa_atual) + "/mês", help="Derivada do modelo no último mês.")
    estado = "acelerando" if rel.aceleracao_atual > 0 else "desacelerando"
    c2.metric(f"Aceleração V″({ctx.n_meses}) · {estado}", numero_br(rel.aceleracao_atual, 1),
              help="Segunda derivada: sinal positivo = crescimento acelera; negativo = desacelera.")
    c3.metric("Limite (t → ∞)", reais(float(rel.saturacao.valor)) if rel.saturacao.classificacao == "finito" else rel.saturacao.valor_texto,
              help="Teto de vendas mensais que o modelo prevê no longo prazo.")
    proj = rel.receita_projecao.valor_exato if rel.receita_projecao.valor_exato is not None else rel.receita_projecao.valor_numerico
    c4.metric(f"Receita prevista ({ctx.horizonte} meses)", reais_compacto(proj), help=f"Integral definida do modelo no horizonte: {reais(proj)}.")


def _graficos(ctx: Contexto, rel) -> None:
    n, h = ctx.n_meses, ctx.horizonte
    x = np.linspace(1, n + h, 500)
    f = para_numerica(ctx.expr)
    d1 = para_numerica(rel.v1)

    tab1, tab2, tab3 = st.tabs(["📈 Vendas e modelo (Funções)", "⚡ Taxa de crescimento (Derivada)", "💰 Receita acumulada (Integral)"])

    # ---- 1. vendas -----------------------------------------------------------
    with tab1:
        fig = figura("Receita mensal: dados, modelo e projeção", "Receita mensal (R$)")
        fig.add_vrect(x0=n, x1=n + h, fillcolor=COR_AREA, line_width=0, annotation_text="projeção", annotation_position="top left")
        if ctx.df is not None:
            fig.add_trace(go.Scatter(x=ctx.df["t"], y=ctx.df["vendas"], mode="markers+lines", name="Vendas reais",
                                     line=dict(color=COR_DADOS, width=1), marker=dict(size=7), customdata=ctx.df["mes"],
                                     hovertemplate="%{customdata}: R$ %{y:,.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=f(x), mode="lines", name="Modelo V(t)", line=dict(color=COR_PRINCIPAL, width=3)))
        if rel.saturacao.classificacao == "finito":
            L = float(rel.saturacao.valor)
            fig.add_hline(y=L, line_dash="dot", line_color=COR_DESTAQUE, annotation_text=f"limite lim V(t) = {reais(L)}", annotation_position="bottom right")
        mostrar_figura(fig, "g_vendas")
        st.latex("V(t) = " + latex(ctx.expr, decimal=True))

    # ---- 2. derivada ---------------------------------------------------------
    with tab2:
        fig = figura("Velocidade de crescimento das vendas: V′(t)", "R$ por mês (a cada mês)")
        fig.add_hline(y=0, line_color="#94A3B8", line_width=1)
        fig.add_vrect(x0=n, x1=n + h, fillcolor=COR_AREA, line_width=0)
        if ctx.df is not None and len(ctx.df) > 1:
            variacao = ctx.df["vendas"].diff()
            fig.add_trace(go.Bar(x=ctx.df["t"], y=variacao, name="Variação mensal observada", marker_color=COR_DADOS, opacity=0.35))
        fig.add_trace(go.Scatter(x=x, y=d1(x), mode="lines", name="V′(t) do modelo", line=dict(color=COR_DESTAQUE, width=3)))
        for c in rel.crescimento["criticos"]:
            fig.add_vline(x=c["t"], line_dash="dash", line_color=COR_PRINCIPAL, annotation_text=f"V′=0 ({c['tipo'].split(' (')[0]})")
        for i in [i for i in rel.crescimento["inflexoes"] if "acelerava" in i["tipo"]][:2]:
            fig.add_vline(x=i["t"], line_dash="dot", line_color=COR_DESTAQUE, annotation_text="crescimento máximo")
        mostrar_figura(fig, "g_derivada")
        st.latex("V'(t) = " + latex(rel.v1, decimal=True))
        st.latex("V''(t) = " + latex(rel.v2, decimal=True))
        st.caption("Onde V′ > 0 as vendas sobem; onde V′ = 0 há pico ou vale; o topo da curva de V′ é o ponto de inflexão de V (V″ = 0).")

    # ---- 3. integral ---------------------------------------------------------
    with tab3:
        fig = figura("Receita acumulada: ∫ V(t) dt", "Receita acumulada (R$)")
        fig.add_vrect(x0=n, x1=n + h, fillcolor=COR_AREA, line_width=0)
        if ctx.df is not None:
            fig.add_trace(go.Scatter(x=ctx.df["t"], y=ctx.df["vendas"].cumsum(), mode="markers", name="Acumulado real (Σ vendas)",
                                     marker=dict(color=COR_DADOS, size=7)))
        acumulado = integrais.acumulada(ctx.expr, 0.5, x + 0.5)
        fig.add_trace(go.Scatter(x=x, y=acumulado, mode="lines", name="Acumulado do modelo (integral)", line=dict(color=COR_PRINCIPAL, width=3)))
        mostrar_figura(fig, "g_integral")
        hist = rel.receita_historico
        proj = rel.receita_projecao
        st.latex(
            rf"\int_{{0{{,}}5}}^{{{n}{{,}}5}} V(t)\,dt = {_num(hist)}\qquad\qquad "
            rf"\int_{{{n}{{,}}5}}^{{{n + h}{{,}}5}} V(t)\,dt = {_num(proj)}"
        )
        st.caption("Cada mês i ocupa o intervalo [i − 0,5; i + 0,5]; a soma das vendas reais dos meses 1…n aproxima a integral de 0,5 a n + 0,5.")


def _num(res) -> str:
    v = res.valor_exato if res.valor_exato is not None else res.valor_numerico
    return r"\text{R\$ }" + numero_br(v).replace(",", "{,}")


def _tabelas(ctx: Contexto) -> None:
    if ctx.df is not None:
        with st.expander("📋 Tabela mensal com variações (absoluta e percentual) e acumulado"):
            m = dados.calcular_metricas(ctx.df).rename(columns={
                "t": "t", "mes": "Mês", "vendas": "Vendas (R$)", "variacao_abs": "Variação (R$)",
                "variacao_pct": "Variação (%)", "acumulado": "Acumulado (R$)", "media_movel_3": "Média móvel 3m (R$)",
            })
            mostrar_tabela(m, column_config={c: st.column_config.NumberColumn(format="%.2f") for c in m.columns if c not in ("Mês", "t")})
    if ctx.ajustes:
        with st.expander("🔬 Qual modelo descreve melhor os dados? (ajuste por mínimos quadrados)"):
            import pandas as pd

            tabela = pd.DataFrame([
                {"Modelo": ("✅ " if a is ctx.escolhido else "") + a.nome, "R²": a.r2, "R² ajustado": a.r2_ajustado, "Erro médio (RMSE, R$)": a.rmse}
                for a in ctx.ajustes
            ])
            mostrar_tabela(tabela, column_config={
                "R²": st.column_config.NumberColumn(format="%.4f"),
                "R² ajustado": st.column_config.NumberColumn(format="%.4f"),
                "Erro médio (RMSE, R$)": st.column_config.NumberColumn(format="%.2f"),
            })
            st.caption("R² mede o quanto o modelo explica os dados (1 = perfeito). O R² ajustado penaliza modelos com mais parâmetros.")
