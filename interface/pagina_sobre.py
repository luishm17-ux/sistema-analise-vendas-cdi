"""Aba "Sobre": como usar e onde cada requisito do projeto está atendido."""

from __future__ import annotations

import streamlit as st

from vendas_cdi.derivadas import CATALOGO_REGRAS as REGRAS_DERIVACAO
from vendas_cdi.funcoes import TIPOS
from vendas_cdi.integrais import CATALOGO_REGRAS as REGRAS_INTEGRACAO

from .componentes import Contexto, mostrar_tabela


def renderizar(ctx: Contexto) -> None:
    import pandas as pd

    st.subheader("ℹ️ Sobre o sistema")
    st.markdown(
        """
**Sistema de Análise de Vendas** — Projeto Final de Cálculo Diferencial e Integral (CDI).

O problema: *uma empresa quer entender como suas vendas crescem, qual a velocidade e a tendência desse
crescimento, se o mercado tem um teto e quanto vai faturar nos próximos meses.* Cada módulo de CDI responde
uma parte da pergunta:
"""
    )
    mostrar_tabela(pd.DataFrame([
        {"Módulo": "Funções (Aula 1)", "Pergunta de negócio": "Que curva descreve as vendas?", "Ferramenta": f"{len(TIPOS)} tipos de função ajustados aos dados"},
        {"Módulo": "Limites (Aula 2)", "Pergunta de negócio": "O mercado tem um teto? Há quebras?", "Ferramenta": "lim t→∞ V(t); limites laterais; continuidade"},
        {"Módulo": "Derivadas (Aula 3)", "Pergunta de negócio": "Quão rápido (e acelerando?) crescemos?", "Ferramenta": f"V′(t), V″(t) com {len(REGRAS_DERIVACAO)} regras"},
        {"Módulo": "Integrais (Aula 4)", "Pergunta de negócio": "Quanto faturamos / vamos faturar?", "Ferramenta": f"∫ V(t) dt com {len(REGRAS_INTEGRACAO)} regras"},
    ]))

    st.markdown("#### Como usar")
    st.markdown(
        """
1. Na **barra lateral**, escolha a fonte dos dados: exemplo, seu CSV (colunas `mes` e `vendas`), vendas digitadas ou uma simulação.
2. Veja o painel **Análise de vendas**: gráficos, indicadores e a interpretação em português.
3. Explore cada módulo de CDI nas abas **Funções, Limites, Derivadas e Integrais** — todas aceitam o modelo ativo ou uma função digitada por você.
4. Confira a aba **Exemplos verificáveis**: contas com resposta conhecida, executadas ao vivo.
"""
    )

    st.markdown("#### Requisitos do projeto × onde estão atendidos")
    mostrar_tabela(pd.DataFrame([
        {"Requisito": "≥ 5 tipos de função, com domínio, imagem, entradas/saídas e gráfico/tabela", "Atendido por": f"{len(TIPOS)} tipos — aba Funções (`vendas_cdi/funcoes.py`)"},
        {"Requisito": "≥ 1 limite, laterais, infinitos, continuidade, inexistência, interpretação", "Atendido por": "aba Limites (`vendas_cdi/limites.py`)"},
        {"Requisito": "≥ 5 regras de derivação, com demonstração", "Atendido por": f"{len(REGRAS_DERIVACAO)} regras com passo a passo — aba Derivadas (`derivadas.py`)"},
        {"Requisito": "≥ 5 regras de integração (definidas ou indefinidas)", "Atendido por": f"{len(REGRAS_INTEGRACAO)} regras com passo a passo — aba Integrais (`integrais.py`)"},
        {"Requisito": "Resultados verificáveis, testes e entradas inválidas", "Atendido por": "aba Exemplos, `tests/` (pytest) e `validacao.py`"},
        {"Requisito": "GitHub e documentação", "Atendido por": "`README.md` (instalação, arquitetura, justificativas, registro de IA)"},
    ]))
    st.caption("Código-fonte, testes e README estão no repositório do projeto.")
