"""Testes da interface Streamlit: o app abre, troca de modo e recusa entradas inválidas sem quebrar."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = 180


def novo() -> AppTest:
    return AppTest.from_file(APP, default_timeout=TIMEOUT).run()


def sem_excecoes(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def textos(elementos) -> str:
    return " ".join(str(e.value) for e in elementos)


def test_abre_com_dados_de_exemplo_sem_erros():
    at = novo()
    sem_excecoes(at)
    assert not at.error
    assert len(at.tabs) >= 7
    rotulos = {m.label for m in at.metric}
    assert "Receita total (real)" in rotulos and any(r.startswith("Taxa de crescimento") for r in rotulos)
    assert any("satura" in m.value for m in at.markdown)  # interpretação do limite aparece


def test_simulacao_manual_com_cada_tipo_de_funcao():
    at = novo()
    at.sidebar.radio(key="modo").set_value("Simulação (modelo manual)").run()
    sem_excecoes(at)
    for chave in ["linear", "quadratica", "exponencial", "logaritmica", "logistica", "sazonal"]:
        at.sidebar.selectbox(key="sim_tipo").set_value(chave).run()
        sem_excecoes(at)
        assert not at.error, f"{chave}: {textos(at.error)}"


def test_digitar_vendas_validas_e_invalidas():
    at = novo()
    at.sidebar.radio(key="modo").set_value("Digitar vendas").run()
    sem_excecoes(at)
    assert not at.error

    for ruim, trecho in [("abc\n1\n2\n3\n4\n5", "não numéricos"), ("1\n2\n3", "pelo menos 6"), ("-1\n2\n3\n4\n5\n6", "negativas")]:
        at.sidebar.text_area(key="vendas_texto").set_value(ruim).run()
        sem_excecoes(at)                       # nunca estoura exceção...
        assert at.error and trecho in textos(at.error)   # ...mostra mensagem clara


def test_csv_enviado_sem_arquivo_pede_upload():
    at = novo()
    at.sidebar.radio(key="modo").set_value("Enviar arquivo CSV").run()
    sem_excecoes(at)
    assert "Envie um arquivo CSV" in textos(at.info)


@pytest.mark.parametrize("prefixo, perigoso", [
    ("lim", "__import__('os').system('echo oi')"),
    ("der", "t.__class__"),
    ("int", "9^9^9"),
])
def test_funcao_personalizada_maliciosa_ou_invalida_e_recusada(prefixo, perigoso):
    at = novo()
    at.radio(key=f"{prefixo}_origem").set_value("Função personalizada").run()
    at.text_input(key=f"{prefixo}_texto_(digitar)").set_value(perigoso).run()
    sem_excecoes(at)
    assert at.error, "deveria exibir mensagem de erro de validação"


def test_funcao_personalizada_valida_calcula():
    at = novo()
    at.radio(key="der_origem").set_value("Função personalizada").run()
    at.text_input(key="der_texto_(digitar)").set_value("t^2*exp(t)").run()
    sem_excecoes(at)
    assert not at.error
    assert any(m.label == "Confere com SymPy (simbólico)" and "sim" in m.value for m in at.metric)


def test_limite_inexistente_e_explicado():
    at = novo()
    at.radio(key="lim_origem").set_value("Função personalizada").run()
    at.selectbox(key="lim_exemplo").set_value("Não existe (salto): |t|/t em t = 0").run()
    sem_excecoes(at)
    assert "NÃO existe" in textos(at.error)


def test_intervalo_invertido_na_integral_e_recusado():
    at = novo()
    at.number_input(key="int_a_True").set_value(10.0)
    at.number_input(key="int_b_True").set_value(2.0).run()
    sem_excecoes(at)
    assert "deve ser menor" in textos(at.error)


def test_aba_de_exemplos_mostra_tudo_conferido():
    at = novo()
    sem_excecoes(at)
    assert any("exemplos conferem" in str(s.value) for s in at.success)
