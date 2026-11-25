"""Aula 3 — derivadas: cada regra é conferida contra sympy.diff e contra diferença finita."""

import numpy as np
import pytest
import sympy as sp

from vendas_cdi import derivadas, funcoes
from vendas_cdi.derivadas import CATALOGO_REGRAS
from vendas_cdi.validacao import ValidacaoErro, interpretar_expressao as P, t


def test_ha_pelo_menos_cinco_regras():
    assert len(CATALOGO_REGRAS) >= 5


@pytest.mark.parametrize("regra", list(CATALOGO_REGRAS))
def test_cada_regra_do_catalogo(regra):
    _, exemplo, esperado = CATALOGO_REGRAS[regra]
    f = P(exemplo)
    r = derivadas.derivar(f)
    assert regra in r.regras_usadas, f"a regra {regra!r} não foi aplicada em {exemplo}"
    assert r.verificada
    esp = sp.sign(t - 2) if "sign" in esperado else P(esperado)
    x = np.array([0.37, 1.9, 2.6, 5.5, 11.2])
    from vendas_cdi.funcoes import para_numerica
    assert np.allclose(para_numerica(r.derivada)(x), para_numerica(esp)(x), rtol=1e-9)


@pytest.mark.parametrize("texto, esperada", [
    ("5000", "0"),
    ("t", "1"),
    ("t^3", "3*t^2"),
    ("4*t^2", "8*t"),
    ("t^2 + 3*t - 7", "2*t + 3"),
    ("t^2*exp(t)", "(t^2 + 2*t)*exp(t)"),
    ("t/(t+1)", "1/(t+1)^2"),
    ("(3*t+1)^5", "15*(3*t+1)^4"),
    ("exp(2*t)", "2*exp(2*t)"),
    ("ln(t^2+1)", "2*t/(t^2+1)"),
    ("sin(2*t)", "2*cos(2*t)"),
    ("cos(t)", "-sin(t)"),
    ("sqrt(t)", "1/(2*sqrt(t))"),
    ("1/t", "-1/t^2"),
    ("2^t", "2^t*ln(2)"),
    ("t*sin(t)", "sin(t) + t*cos(t)"),
    ("(t^2+1)/(t^2-1)", "-4*t/(t^2-1)^2"),
])
def test_derivadas_com_resposta_conhecida(texto, esperada):
    r = derivadas.derivar(P(texto))
    assert sp.simplify(r.derivada - P(esperada)) == 0
    assert r.verificada


def test_regra_especifica_aparece_nos_passos():
    assert {"Produto", "Exponencial"} <= set(derivadas.derivar(P("t^2*exp(t)")).regras_usadas)
    assert "Quociente" in derivadas.derivar(P("t/(t+1)")).regras_usadas
    assert {"Potência", "Cadeia"} <= set(derivadas.derivar(P("(3*t+1)^5")).regras_usadas)
    assert "Constante" in derivadas.derivar(P("7")).regras_usadas
    r = derivadas.derivar(P("3*t^2 + 5"))
    assert [p.regras[0] for p in r.passos][0] == "Soma/Subtração"


@pytest.mark.parametrize("chave", list(funcoes.TIPOS))
def test_derivada_dos_seis_modelos_e_verificada(chave):
    tp = funcoes.TIPOS[chave]
    assert derivadas.derivar(tp.construir(tp.valores_padrao())).verificada


def test_derivada_de_ordem_superior():
    assert derivadas.derivada_n(P("t^4"), 2) == 12 * t**2
    assert derivadas.derivada_n(P("t^4"), 4) == 24
    assert derivadas.derivada_n(P("t^4"), 1) == 4 * t**3
    for ordem in (0, 5, -1):
        with pytest.raises(ValidacaoErro):
            derivadas.derivada_n(P("t"), ordem)


def test_derivada_numerica_e_erros():
    assert derivadas.derivada_numerica(P("t^2"), 3) == pytest.approx(6, abs=1e-6)
    with pytest.raises(ValidacaoErro):
        derivadas.derivada_numerica(P("1/(t-2)"), 2)         # fora do domínio
    with pytest.raises(ValidacaoErro):
        derivadas.derivada_numerica(P("t"), "abc")
    with pytest.raises(ValidacaoErro):
        derivadas.derivada_numerica(P("t"), 1, h=0)


def test_analise_do_pico_de_vendas():
    v = funcoes.construir("quadratica", {"a": -25, "b": 1500, "c": 20000})   # V' = 1500 − 50t → pico em t = 30
    an = derivadas.analisar_crescimento(v, 0, 60)
    assert len(an["criticos"]) == 1
    c = an["criticos"][0]
    assert c["t"] == pytest.approx(30) and c["V(t)"] == pytest.approx(42500) and "máximo" in c["tipo"]
    assert an["inflexoes"] == []                                              # parábola não tem inflexão


def test_inflexao_da_logistica_e_em_t0():
    lo = funcoes.construir("logistica", {"L": 100, "k": 0.5, "t0": 10})
    an = derivadas.analisar_crescimento(lo, 1, 30)
    assert an["inflexoes"][0]["t"] == pytest.approx(10, abs=1e-6)
    assert "acelerava e passa a frear" in an["inflexoes"][0]["tipo"]
    assert an["maior_taxa"]["V'(t)"] == pytest.approx(100 * 0.5 / 4, rel=1e-4)   # L·k/4
    assert an["maior_taxa"]["t"] == pytest.approx(10, abs=0.05)


def test_interpretacao_em_texto():
    v = funcoes.construir("quadratica", {"a": -25, "b": 1500, "c": 20000})
    txt = derivadas.interpretar_taxa(v, 10)
    assert "1.000,00" in txt and "desacelerando" in txt
    assert "caindo" in derivadas.interpretar_taxa(v, 40)


def test_taxa_relativa_da_exponencial_e_constante():
    e = funcoes.construir("exponencial", {"a": 100, "k": 0.06})
    assert float(sp.N(derivadas.taxa_relativa(e).subs(t, 7))) == pytest.approx(0.06)


def test_intervalo_invalido_na_analise():
    with pytest.raises(ValidacaoErro):
        derivadas.analisar_crescimento(P("t"), 5, 1)
