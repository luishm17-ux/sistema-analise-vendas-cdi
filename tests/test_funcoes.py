"""Aula 1 — seis tipos de função: valores conferidos à mão, domínio, imagem e validação."""

import math

import pytest

from vendas_cdi import funcoes
from vendas_cdi.funcoes import TIPOS
from vendas_cdi.validacao import ValidacaoErro, interpretar_expressao as P


def test_ha_pelo_menos_cinco_tipos_distintos():
    assert len(TIPOS) >= 5
    assert len({tp.classificacao for tp in TIPOS.values()}) == len(TIPOS)


@pytest.mark.parametrize("chave, params, ponto, esperado", [
    ("linear", {"a": 2, "b": 5}, 10, 25),                                      # 2·10 + 5
    ("linear", {"a": 0, "b": 7}, 99, 7),                                       # constante
    ("quadratica", {"a": 1, "b": -2, "c": 3}, 4, 11),                          # 16 − 8 + 3
    ("exponencial", {"a": 100, "k": math.log(2)}, 3, 800),                     # 100·2³
    ("exponencial", {"a": 50, "k": 0}, 10, 50),                                # k = 0 → constante
    ("logaritmica", {"a": 10, "b": 5}, math.e - 1, 15),                        # 10·ln(e) + 5
    ("logistica", {"L": 100, "k": 0.5, "t0": 10}, 10, 50),                     # ponto de inflexão = L/2
    ("sazonal", {"c": 100, "a": 0, "A": 20, "P": 12, "fase": 0}, 3, 120),      # sen(π/2) = 1
    ("sazonal", {"c": 100, "a": 10, "A": 20, "P": 12, "fase": 0}, 0, 100),     # sen(0) = 0
])
def test_valores_conferidos_a_mao(chave, params, ponto, esperado):
    assert funcoes.valor_em(funcoes.construir(chave, params), ponto) == pytest.approx(esperado, rel=1e-9)


def test_logistica_tende_ao_limite_e_e_crescente():
    e = funcoes.construir("logistica", {"L": 100, "k": 0.5, "t0": 10})
    v = [funcoes.valor_em(e, x) for x in (0, 5, 10, 15, 30, 60)]
    assert v == sorted(v) and v[-1] == pytest.approx(100, abs=1e-6) and v[0] > 0


@pytest.mark.parametrize("chave, params, dominio, imagem", [
    ("linear", {"a": 2, "b": 1}, "ℝ", "ℝ"),
    ("linear", {"a": 0, "b": 4}, "ℝ", "{4}"),
    ("quadratica", {"a": -1, "b": 4, "c": 0}, "ℝ", "(-∞, 4]"),          # vértice (2, 4)
    ("quadratica", {"a": 1, "b": 0, "c": 3}, "ℝ", "[3, +∞)"),
    ("exponencial", {"a": 5, "k": 0.1}, "ℝ", "(0, +∞)"),
    ("logistica", {"L": 100, "k": 1, "t0": 0}, "ℝ", "(0, 100)"),
    ("sazonal", {"c": 10, "a": 0, "A": 3, "P": 12, "fase": 0}, "ℝ", "[7, 13]"),
])
def test_dominio_e_imagem_teoricos(chave, params, dominio, imagem):
    d, i = TIPOS[chave].dominio_imagem(TIPOS[chave].validar(params))
    assert d == dominio and i == imagem


def test_dominio_simbolico_e_classificacao():
    assert funcoes.dominio_simbolico(P("1/(t-2)")) == "(-∞, 2) ∪ (2, ∞)"
    assert funcoes.dominio_simbolico(P("sqrt(t)")) == "[0, ∞)"
    assert funcoes.dominio_simbolico(P("t^2")) == "ℝ"
    assert funcoes.classificar(P("5")) == "Constante"
    assert funcoes.classificar(P("2*t+1")).startswith("Afim")
    assert funcoes.classificar(P("t^2")).startswith("Quadr")
    assert funcoes.classificar(P("sin(t)")) == "Trigonométrica"
    assert funcoes.classificar(P("exp(t)")) == "Exponencial"
    assert funcoes.classificar(P("ln(t)")) == "Logarítmica"
    assert funcoes.classificar(P("abs(t)")) == "Modular"


def test_imagem_numerica_e_tabela():
    e = funcoes.construir("quadratica", {"a": -1, "b": 4, "c": 0})
    lo, hi = funcoes.imagem_numerica(e, 0, 4)
    assert hi == pytest.approx(4, abs=1e-4) and lo == pytest.approx(0, abs=1e-9)
    tab = funcoes.tabela(funcoes.construir("linear", {"a": 2, "b": 1}), 0, 4, 5)
    assert tab["t (mês)"].tolist() == [0, 1, 2, 3, 4]
    assert tab["V(t) (R$)"].tolist() == [1, 3, 5, 7, 9]


def test_valor_fora_do_dominio_gera_erro_claro():
    with pytest.raises(ValidacaoErro):
        funcoes.valor_em(P("1/(t-2)"), 2)
    with pytest.raises(ValidacaoErro):
        funcoes.valor_em(P("ln(t)"), -1)


@pytest.mark.parametrize("chave, params", [
    ("linear", {"a": "abc", "b": 1}),
    ("linear", {"a": float("nan"), "b": 1}),
    ("linear", {"a": 1}),                                   # falta parâmetro
    ("exponencial", {"a": -5, "k": 0.1}),                   # a deve ser > 0
    ("exponencial", {"a": 5, "k": 50}),                     # k fora do limite
    ("logistica", {"L": 0, "k": 1, "t0": 0}),               # L deve ser > 0
    ("logistica", {"L": 10, "k": -1, "t0": 0}),             # k deve ser > 0
    ("sazonal", {"c": 1, "a": 1, "A": 1, "P": 0, "fase": 0}),   # período zero
    ("sazonal", {"c": 1, "a": 1, "A": 1, "P": 1, "fase": 0}),   # período < 2
])
def test_parametros_invalidos_sao_recusados(chave, params):
    with pytest.raises(ValidacaoErro):
        funcoes.construir(chave, params)


def test_tipo_desconhecido_e_intervalo_invalido():
    with pytest.raises(ValidacaoErro):
        funcoes.obter_tipo("cubica")
    with pytest.raises(ValidacaoErro):
        funcoes.tabela(P("t"), 5, 1)
    with pytest.raises(ValidacaoErro):
        funcoes.tabela(P("t"), 0, 1, passos=1)
