"""Aula 2 — limites: finitos, laterais, no infinito, inexistência, continuidade e propriedades."""

import pytest
import sympy as sp

from vendas_cdi import funcoes, limites
from vendas_cdi.validacao import ValidacaoErro, interpretar_expressao as P


@pytest.mark.parametrize("expr, ponto, valor", [
    ("(t^2-4)/(t-2)", "2", "4"),          # indeterminação 0/0 resolvida por fatoração
    ("sin(t)/t", "0", "1"),               # limite trigonométrico fundamental
    ("(3*t+1)/(t+2)", "oo", "3"),         # racional: razão dos coeficientes líderes
    ("exp(-t)", "oo", "0"),
    ("1/t", "oo", "0"),
    ("t^2", "3", "9"),                    # função contínua: limite = valor
    ("(t^3-8)/(t-2)", "2", "12"),
])
def test_limites_que_existem(expr, ponto, valor):
    r = limites.calcular_limite(P(expr), ponto)
    assert r.existe and r.classificacao == "finito"
    assert sp.simplify(r.valor - sp.Rational(valor)) == 0


@pytest.mark.parametrize("expr, ponto, sinal", [
    ("1/t^2", "0", sp.oo),
    ("t^2", "oo", sp.oo),
    ("-t^2", "oo", -sp.oo),
    ("exp(t)", "oo", sp.oo),
])
def test_limites_infinitos(expr, ponto, sinal):
    r = limites.calcular_limite(P(expr), ponto)
    assert r.existe and r.classificacao == "infinito" and r.valor == sinal


@pytest.mark.parametrize("expr, ponto", [
    ("abs(t)/t", "0"),     # salto: laterais −1 e +1
    ("1/t", "0"),          # laterais −∞ e +∞
    ("1/(t-3)", "3"),
    ("sin(t)", "oo"),      # oscila
    ("cos(t)", "oo"),
])
def test_limites_que_nao_existem(expr, ponto):
    r = limites.calcular_limite(P(expr), ponto)
    assert not r.existe and r.classificacao == "nao_existe" and r.valor_texto == "não existe"


def test_limites_laterais_de_abs_t_sobre_t():
    r = limites.calcular_limite(P("abs(t)/t"), "0")
    assert (r.esquerda, r.direita) == (-1, 1)
    assert limites.calcular_limite(P("abs(t)/t"), "0", "+").valor == 1
    assert limites.calcular_limite(P("abs(t)/t"), "0", "-").valor == -1


def test_continuidade():
    assert limites.calcular_limite(P("t^2"), "3").continuidade.startswith("Contínua")
    assert "removível" in limites.calcular_limite(P("(t^2-4)/(t-2)"), "2").continuidade
    assert "salto" in limites.calcular_limite(P("abs(t)/t"), "0").continuidade
    assert "infinita" in limites.calcular_limite(P("1/t"), "0").continuidade


def test_saturacao_da_logistica_e_limite_zero_de_taxa_relativa():
    lo = funcoes.construir("logistica", {"L": 100, "k": 0.5, "t0": 10})
    assert limites.limite_saturacao(lo).valor == 100
    assert limites.limite_taxa_relativa(lo).valor == 0


def test_taxa_relativa_da_exponencial_e_k():
    e = funcoes.construir("exponencial", {"a": 100, "k": 0.06})
    r = limites.limite_taxa_relativa(e)
    assert float(r.valor) == pytest.approx(0.06)


def test_sazonal_sem_tendencia_oscila_e_com_tendencia_cresce():
    assert not limites.limite_saturacao(funcoes.construir("sazonal", {"c": 1, "a": 0, "A": 1, "P": 12, "fase": 0})).existe
    assert limites.limite_saturacao(funcoes.construir("sazonal", {"c": 1, "a": 5, "A": 1, "P": 12, "fase": 0})).valor == sp.oo


def test_definicao_de_derivada_como_limite():
    d = limites.limite_definicao_derivada(P("t^2"), 3)
    assert d["pelo_limite"] == 6 and d["pela_regra"] == 6 and d["coincidem"]
    d = limites.limite_definicao_derivada(funcoes.construir("quadratica", {"a": -25, "b": 1500, "c": 20000}), 10)
    assert d["valor_limite"] == pytest.approx(1000) and d["coincidem"]


def test_propriedades_dos_limites():
    tab = limites.propriedades_dos_limites(P("t^2+1"), P("3*t-1"), "2")   # lim f = 5, lim g = 5
    assert set(tab["Confere"]) == {"✔"} and len(tab) == 5
    assert tab.loc[tab["Propriedade"] == "Soma", "Limite da combinação"].iloc[0] == "10"
    assert tab.loc[tab["Propriedade"] == "Produto", "Limite da combinação"].iloc[0] == "25"


def test_propriedades_sem_quociente_quando_lim_g_zero():
    tab = limites.propriedades_dos_limites(P("t+1"), P("t-2"), "2")        # lim g = 0
    assert "Quociente" not in tab["Propriedade"].tolist()


def test_propriedades_exigem_limites_finitos():
    with pytest.raises(ValidacaoErro):
        limites.propriedades_dos_limites(P("1/t"), P("t"), "0")


def test_tabela_de_aproximacao():
    tab = limites.tabela_aproximacao(P("(t^2-4)/(t-2)"), "2", 5)
    assert tab["f(t) esq."].iloc[-1] == pytest.approx(4, abs=1e-3)
    assert tab["f(t) dir."].iloc[-1] == pytest.approx(4, abs=1e-3)
    inf = limites.tabela_aproximacao(P("exp(-t)"), "oo", 4)
    assert inf["f(t)"].iloc[-1] == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("ponto", ["banana", "", "1/0"])
def test_ponto_invalido(ponto):
    with pytest.raises(ValidacaoErro):
        limites.calcular_limite(P("t"), ponto)


def test_direcao_invalida():
    with pytest.raises(ValidacaoErro):
        limites.calcular_limite(P("t"), "0", "cima")
