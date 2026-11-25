"""Aba "Exemplos verificáveis": roda, ao vivo, cálculos com resposta conhecida."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from vendas_cdi.exemplos import executar_exemplos

from .componentes import Contexto, mostrar_tabela, tratar_erros


@st.cache_data(show_spinner="Executando os exemplos…")
def _executar() -> pd.DataFrame:
    return executar_exemplos()


def renderizar(ctx: Contexto) -> None:
    st.subheader("✅ Exemplos verificáveis")
    st.markdown(
        "Cada linha compara o resultado do sistema com um valor **calculado à mão** (a conta aparece no nome do teste). "
        "Se alguma conta do sistema estivesse errada, ela apareceria aqui como ✘. As mesmas verificações rodam no "
        "`pytest` do repositório."
    )
    with tratar_erros("os exemplos"):
        df = _executar()
        resumo = df.groupby("Módulo", sort=False)["Confere"].agg(["sum", "count"])
        colunas = st.columns(len(resumo) + 1)
        for coluna, (modulo, linha) in zip(colunas, resumo.iterrows()):
            coluna.metric(modulo, f"{int(linha['sum'])}/{int(linha['count'])}")
        total_ok, total = int(df["Confere"].sum()), len(df)
        colunas[-1].metric("Total", f"{total_ok}/{total}")
        if total_ok == total:
            st.success(f"Todos os {total} exemplos conferem.")
        else:
            st.error(f"{total - total_ok} exemplo(s) não conferem — veja as linhas marcadas com ✘.")

        filtro = st.multiselect("Filtrar por módulo", list(resumo.index), default=list(resumo.index), key="ex_filtro")
        vis = df[df["Módulo"].isin(filtro)].copy()
        vis["Confere"] = vis["Confere"].map({True: "✔", False: "✘"})
        mostrar_tabela(vis)
        if st.button("🔄 Executar novamente", key="ex_rodar"):
            _executar.clear()
            st.rerun()
