"""Sistema de Análise de Vendas — ponto de entrada do aplicativo Streamlit.

Execute com:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from interface import (
    pagina_derivadas,
    pagina_exemplos,
    pagina_funcoes,
    pagina_integrais,
    pagina_limites,
    pagina_sobre,
    pagina_vendas,
)
from interface.componentes import Contexto, entrada_parametros, tratar_erros
from vendas_cdi import dados
from vendas_cdi.funcoes import TIPOS
from vendas_cdi.validacao import ValidacaoErro

RAIZ = Path(__file__).resolve().parent
ARQUIVO_EXEMPLO = RAIZ / "dados" / "vendas_exemplo.csv"

st.set_page_config(page_title="Análise de Vendas · CDI", page_icon="📊", layout="wide")

MODOS = ["Dados de exemplo (24 meses)", "Enviar arquivo CSV", "Digitar vendas", "Simulação (modelo manual)"]


# --------------------------------------------------------------------------- #
# Barra lateral: de onde vêm os dados e qual modelo V(t) usar
# --------------------------------------------------------------------------- #
def construir_contexto() -> Contexto | None:
    sb = st.sidebar
    sb.title("📊 Análise de Vendas")
    sb.caption("Funções · Limites · Derivadas · Integrais")
    modo = sb.radio("Fonte dos dados", MODOS, key="modo")

    horizonte = sb.slider("Meses a projetar", 1, 24, 6, key="horizonte",
                          help="Quantos meses à frente o modelo vai prever (integral da projeção).")

    if modo == MODOS[3]:
        return _contexto_simulacao(sb, horizonte)

    df = _carregar_dados(sb, modo)
    if df is None:
        return None
    ajustes = dados.ajustar_modelos(df)
    opcoes = {f"{a.nome}  (R² = {a.r2:.3f})": a for a in ajustes}
    escolha = sb.selectbox("Modelo V(t)", list(opcoes), key="modelo_ajustado",
                           help="O primeiro é o de melhor R² ajustado. Você pode trocar para comparar.")
    escolhido = opcoes[escolha]
    sb.caption("O melhor ajuste vem primeiro na lista." if escolhido is ajustes[0] else "Você escolheu um modelo diferente do melhor ajuste.")
    return Contexto(modo, escolhido.expr, escolhido.nome, len(df), horizonte, df, ajustes, escolhido)


def _carregar_dados(sb, modo: str):
    try:
        if modo == MODOS[0]:
            return dados.carregar_csv(ARQUIVO_EXEMPLO)
        if modo == MODOS[1]:
            arquivo = sb.file_uploader("CSV com colunas `mes` e `vendas`", type=["csv", "txt"], key="csv")
            sb.download_button("⬇️ Baixar CSV de exemplo", ARQUIVO_EXEMPLO.read_bytes(), "vendas_exemplo.csv", "text/csv")
            if arquivo is None:
                st.info("👈 Envie um arquivo CSV na barra lateral (colunas `mes` e `vendas`) ou escolha os dados de exemplo.")
                return None
            return dados.carregar_csv(arquivo.getvalue())
        texto = sb.text_area("Vendas mensais (uma por linha ou separadas por vírgula)", "6000\n8500\n9000\n12000\n15500\n19000\n24000\n29500\n35000\n40000\n46000\n51000",
                             height=180, key="vendas_texto")
        return dados.dataframe_manual(texto)
    except ValidacaoErro as erro:
        st.error(f"⚠️ {erro}")
        return None


def _contexto_simulacao(sb, horizonte: int) -> Contexto | None:
    chave = sb.selectbox("Tipo de função", list(TIPOS), format_func=lambda k: TIPOS[k].nome, key="sim_tipo")
    tipo = TIPOS[chave]
    n_meses = sb.slider("Meses de histórico", 6, 60, 24, key="sim_n")
    valores = entrada_parametros(tipo, "sim", sb)
    try:
        expr = tipo.construir(valores)
    except ValidacaoErro as erro:
        st.error(f"⚠️ {erro}")
        return None
    return Contexto("simulação", expr, tipo.nome, n_meses, horizonte)


# --------------------------------------------------------------------------- #
def main() -> None:
    ctx = None
    with tratar_erros("a barra lateral"):
        ctx = construir_contexto()
    st.title("Sistema de Análise de Vendas")
    if ctx is None:
        st.stop()

    abas = st.tabs(["📊 Análise de vendas", "📈 Funções", "∞ Limites", "∂ Derivadas", "∫ Integrais", "✅ Exemplos", "ℹ️ Sobre"])
    paginas = [pagina_vendas, pagina_funcoes, pagina_limites, pagina_derivadas, pagina_integrais, pagina_exemplos, pagina_sobre]
    for aba, pagina in zip(abas, paginas):
        with aba:
            pagina.renderizar(ctx)


main()
