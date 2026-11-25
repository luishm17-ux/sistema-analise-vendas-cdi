"""Aula 4 — integrais: antiderivadas conferidas por derivação e integrais definidas por scipy.quad."""

import math

import numpy as np
import pytest
import sympy as sp
from scipy.integrate import quad

from vendas_cdi import funcoes, integrais
from vendas_cdi.funcoes import para_numerica
from vendas_cdi.integrais import CATALOGO_REGRAS
from vendas_cdi.validacao import ValidacaoErro, interpretar_expressao as P, t


def test_ha_pelo_menos_cinco_regras():
    assert len(CATALOGO_REGRAS) >= 5


@pytest.mark.parametrize("regra", list(CATALOGO_REGRAS))
def test_cada_regra_do_catalogo(regra):
    _, exemplo, esperado = CATALOGO_REGRAS[regra]
    r = integrais.integrar(P(exemplo))
    assert regra in r.regras_usadas, f"a regra {regra!r} não foi aplicada em {exemplo}"
    assert r.verificada                                   # d/dt F == f
    x = np.linspace(0.4, 9, 12)                           # F e a resposta esperada diferem só por constante
    d = para_numerica(r.antiderivada)(x) - para_numerica(P(esperado))(x)
    assert np.allclose(d, d[0], atol=1e-9)


@pytest.mark.parametrize("texto, esperada", [
    ("5", "5*t"),
    ("t", "t^2/2"),
    ("t^3", "t^4/4"),
    ("3*t^2 + 2*t + 1", "t^3 + t^2 + t"),
    ("exp(2*t)", "exp(2*t)/2"),
    ("sin(3*t)", "-cos(3*t)/3"),
    ("cos(2*t)", "sin(2*t)/2"),
    ("(2*t+1)^3", "(2*t+1)^4/8"),
    ("1/(2*t+1)", "ln(abs(2*t+1))/2"),
    ("t*exp(t)", "(t-1)*exp(t)"),
    ("t^2*exp(t)", "(t^2-2*t+2)*exp(t)"),
    ("t*cos(t)", "cos(t) + t*sin(t)"),
    ("ln(t)", "t*ln(t) - t"),
    ("2^t", "2^t/ln(2)"),
])
def test_antiderivadas_conhecidas(texto, esperada):
    F = integrais.integrar(P(texto)).antiderivada
    x = np.linspace(0.5, 6, 10)
    d = para_numerica(F)(x) - para_numerica(P(esperada))(x)
    assert np.allclose(d, d[0], atol=1e-9)


def test_constante_aditiva_nao_importa_mas_derivada_sim():
    F = integrais.integrar(P("4*t^3")).antiderivada
    assert sp.simplify(sp.diff(F, t) - 4 * t**3) == 0


@pytest.mark.parametrize("texto, a, b, esperado", [
    ("t^2", 0, 3, 9),
    ("exp(t)", 0, 1, math.e - 1),
    ("sin(t)", 0, math.pi, 2),
    ("cos(t)", 0, math.pi / 2, 1),
    ("1/t", 1, math.e, 1),
    ("(2*t+1)^3", 0, 2, 78),
    ("t*exp(t)", 0, 1, 1),
    ("ln(1+t)", 0, 1, 2 * math.log(2) - 1),
    ("5", 2, 7, 25),
    ("t", -1, 1, 0),
])
def test_integrais_definidas_conhecidas(texto, a, b, esperado):
    r = integrais.integral_definida(P(texto), a, b)
    assert r.valor_exato == pytest.approx(esperado, abs=1e-9)
    assert r.valor_numerico == pytest.approx(esperado, abs=1e-8)
    assert r.confere


@pytest.mark.parametrize("chave", list(funcoes.TIPOS))
def test_receita_acumulada_dos_modelos_bate_com_quad(chave):
    tp = funcoes.TIPOS[chave]
    e = tp.construir(tp.valores_padrao())
    r = integrais.integral_definida(e, 0.5, 24.5)
    f = para_numerica(e)
    ref, _ = quad(lambda z: float(f(np.array([z]))[0]), 0.5, 24.5)
    assert r.valor_exato == pytest.approx(ref, rel=1e-7) and r.confere


def test_receita_acumulada_equivale_a_soma_mensal_no_modelo_linear():
    v = funcoes.construir("linear", {"a": 1000, "b": 5000})      # mês i vende 1000·i + 5000
    soma_discreta = sum(1000 * i + 5000 for i in range(1, 13))
    assert integrais.integral_definida(v, 0.5, 12.5).valor_exato == pytest.approx(soma_discreta)


def test_valor_medio():
    r = integrais.integral_definida(P("t^2"), 0, 3)
    assert r.media == pytest.approx(3)                            # 9 / 3


def test_aditividade_do_intervalo():
    f = P("t*exp(-t/5)")
    a = integrais.integral_definida(f, 0, 4).valor_exato
    b = integrais.integral_definida(f, 4, 10).valor_exato
    assert a + b == pytest.approx(integrais.integral_definida(f, 0, 10).valor_exato)


def test_soma_de_riemann_converge_e_limita():
    f = P("t^2")
    exatos = 9.0
    erros = [abs(integrais.soma_de_riemann(f, 0, 3, n) - exatos) for n in (4, 16, 64, 256)]
    assert erros == sorted(erros, reverse=True) and erros[-1] < 1e-4
    # t² é crescente em [0,3]: esquerda ≤ exata ≤ direita
    assert integrais.soma_de_riemann(f, 0, 3, 50, "esquerda") < exatos < integrais.soma_de_riemann(f, 0, 3, 50, "direita")
    tab = integrais.tabela_riemann(f, 0, 3)
    assert tab["Erro absoluto"].is_monotonic_decreasing


def test_acumulada_vetorial_para_grafico():
    v = funcoes.construir("linear", {"a": 0, "b": 10})            # V = 10 → R(t) = 10·(t − a)
    x = np.array([1.0, 2.0, 5.0])
    assert np.allclose(integrais.acumulada(v, 0.5, x + 0.5), 10 * x)


def test_sem_forma_fechada_ainda_calcula_numericamente():
    r = integrais.integral_definida(P("exp(-t^2)"), 0, 1)         # integral gaussiana: usa erf no SymPy
    assert r.valor_numerico == pytest.approx(0.7468241328, abs=1e-8)


@pytest.mark.parametrize("texto, a, b", [
    ("1/(t-5)", 0, 10),       # assíntota dentro do intervalo
    ("ln(t)", -1, 2),         # fora do domínio
    ("1/t", -1, 1),
    ("sqrt(t)", -4, -1),
])
def test_integral_fora_do_dominio_e_recusada(texto, a, b):
    with pytest.raises(ValidacaoErro):
        integrais.integral_definida(P(texto), a, b)


def test_argumentos_invalidos():
    with pytest.raises(ValidacaoErro):
        integrais.integral_definida(P("t"), 3, 1)
    with pytest.raises(ValidacaoErro):
        integrais.integral_definida(P("t"), "a", 1)
    with pytest.raises(ValidacaoErro):
        integrais.soma_de_riemann(P("t"), 0, 1, 0)
    with pytest.raises(ValidacaoErro):
        integrais.soma_de_riemann(P("t"), 0, 1, 10, "diagonal")
