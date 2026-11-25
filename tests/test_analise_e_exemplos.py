"""Relatório integrado (os 4 módulos juntos) e a lista de exemplos verificáveis."""

import pytest

from vendas_cdi import analise, dados, funcoes
from vendas_cdi.exemplos import executar_exemplos
from vendas_cdi.validacao import ValidacaoErro


def test_todos_os_exemplos_verificaveis_conferem():
    df = executar_exemplos()
    falhas = df[~df["Confere"]]
    assert falhas.empty, f"Exemplos que não conferem:\n{falhas[['Módulo', 'Teste', 'Esperado', 'Obtido']].to_string()}"
    assert set(df["Módulo"]) == {"Funções", "Limites", "Derivadas", "Integrais", "Validação"}
    assert len(df) >= 50


def test_relatorio_da_logistica_com_numeros_conhecidos():
    lo = funcoes.construir("logistica", {"L": 100000, "k": 0.4, "t0": 12})
    r = analise.gerar_relatorio(lo, 24, 6)
    assert float(r.saturacao.valor) == 100000
    assert r.crescimento["inflexoes"][0]["t"] == pytest.approx(12, abs=1e-6)
    assert r.crescimento["maior_taxa"]["V'(t)"] == pytest.approx(100000 * 0.4 / 4, rel=1e-4)    # L·k/4
    assert r.taxa_atual > 0 and r.aceleracao_atual < 0                                         # depois da inflexão: freia
    assert r.receita_projecao.valor_exato > 0
    texto = " ".join(r.textos)
    assert "satura" in texto and "desacelerando" in texto and "Projeção" in texto


def test_relatorio_com_soma_observada_mostra_diferenca():
    df = dados.carregar_csv("dados/vendas_exemplo.csv")
    melhor = dados.ajustar_modelos(df)[0]
    r = analise.gerar_relatorio(melhor.expr, len(df), 6, float(df["vendas"].sum()))
    assert any("receita real somada" in t for t in r.textos)


def test_integral_do_modelo_ajustado_se_aproxima_da_soma_real():
    df = dados.carregar_csv("dados/vendas_exemplo.csv")
    melhor = dados.ajustar_modelos(df)[0]
    r = analise.gerar_relatorio(melhor.expr, len(df), 6)
    assert r.receita_historico.valor_exato == pytest.approx(df["vendas"].sum(), rel=0.02)


def test_pico_aparece_no_relatorio_de_modelo_quadratico():
    v = funcoes.construir("quadratica", {"a": -25, "b": 1500, "c": 20000})
    r = analise.gerar_relatorio(v, 24, 12)
    assert any("Pico" in t and "42.500,00" in t for t in r.textos)


@pytest.mark.parametrize("n, h", [(1, 6), (24, 0), (24, 500), ("x", 6), (None, 6), (float("nan"), 6)])
def test_relatorio_argumentos_invalidos(n, h):
    with pytest.raises(ValidacaoErro):
        analise.gerar_relatorio(funcoes.construir("linear"), n, h)


def test_projecao():
    v = funcoes.construir("linear", {"a": 10, "b": 0})
    assert analise.projetar(v, 12, 3).tolist() == [130, 140, 150]
    with pytest.raises(ValidacaoErro):
        analise.projetar(v, 12, 0)


def test_antiderivada_da_logistica_e_numericamente_estavel():
    """Regressão: o SymPy simplificava para log(u**250000), que estoura. Deve haver forma fechada finita."""
    import numpy as np
    from vendas_cdi import integrais

    lo = funcoes.construir("logistica", {"L": 100000, "k": 0.4, "t0": 12})
    ind = integrais.integrar(lo)
    assert np.all(np.isfinite(funcoes.para_numerica(ind.antiderivada)(np.linspace(0, 60, 61))))
    r = integrais.integral_definida(lo, 0.5, 24.5)
    assert r.valor_exato is not None and r.valor_exato == pytest.approx(r.valor_numerico, rel=1e-9)
