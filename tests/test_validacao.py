"""Validação de entradas: números, intervalos, expressões e pontos de limite."""

import pytest
import sympy as sp

from vendas_cdi.validacao import (
    ValidacaoErro,
    exigir_intervalo,
    exigir_numero,
    interpretar_expressao,
    interpretar_ponto,
    t,
)


class TestNumeros:
    def test_aceita_numeros_e_strings_numericas(self):
        assert exigir_numero(3, "x") == 3.0
        assert exigir_numero("2,5", "x") == 2.5  # vírgula decimal
        assert exigir_numero(" 7 ", "x") == 7.0

    @pytest.mark.parametrize("ruim", ["abc", "", None, float("nan"), float("inf"), float("-inf"), True, [1]])
    def test_rejeita_nao_numericos_e_nao_finitos(self, ruim):
        with pytest.raises(ValidacaoErro):
            exigir_numero(ruim, "x")

    def test_restricoes(self):
        with pytest.raises(ValidacaoErro):
            exigir_numero(0, "x", estrito_positivo=True)
        with pytest.raises(ValidacaoErro):
            exigir_numero(0, "x", nao_zero=True)
        with pytest.raises(ValidacaoErro):
            exigir_numero(-1, "x", minimo=0)
        with pytest.raises(ValidacaoErro):
            exigir_numero(11, "x", maximo=10)
        assert exigir_numero(10, "x", minimo=0, maximo=10) == 10

    def test_intervalo(self):
        assert exigir_intervalo(1, 5) == (1.0, 5.0)
        for a, b in [(5, 5), (6, 5)]:
            with pytest.raises(ValidacaoErro):
                exigir_intervalo(a, b)
        with pytest.raises(ValidacaoErro):
            exigir_intervalo("x", 5)


class TestExpressoes:
    def test_interpreta_expressoes_validas(self):
        assert sp.simplify(interpretar_expressao("3*t^2 + 5*t") - (3 * t**2 + 5 * t)) == 0
        assert sp.simplify(interpretar_expressao("2t(t+1)") - 2 * t * (t + 1)) == 0  # multiplicação implícita
        assert interpretar_expressao("e^t") == sp.exp(t)
        assert interpretar_expressao("ln(t)") == sp.log(t)
        assert interpretar_expressao("50000*t").subs(t, 2) == 100000  # números de 5 dígitos são válidos

    @pytest.mark.parametrize("texto", [
        "", "   ", "3*t +", "((t)", "t t t +", "x", "foo(t)",
        "__import__('os').system('ls')", "t.__class__", "sin.free_symbols", "open('x')",
        "t; 1", "t = 2", "[t]", "{t}", "'a'", "lambda: 1", "t**t**t**t", "9^9^9", "9^(9^9)", "t^5000",
        "1/0", "a" * 300, "123456789012",
    ])
    def test_rejeita_entradas_perigosas_ou_invalidas(self, texto):
        with pytest.raises(ValidacaoErro):
            interpretar_expressao(texto)

    def test_none(self):
        with pytest.raises(ValidacaoErro):
            interpretar_expressao(None)


class TestPontos:
    def test_pontos_validos(self):
        assert interpretar_ponto("oo") == sp.oo
        assert interpretar_ponto("-inf") == -sp.oo
        assert interpretar_ponto("∞") == sp.oo
        assert interpretar_ponto("0,5") == sp.Rational(1, 2)
        assert interpretar_ponto("1/3") == sp.Rational(1, 3)
        assert interpretar_ponto(2) == 2
        assert interpretar_ponto(0.25) == sp.Rational(1, 4)

    @pytest.mark.parametrize("ruim", ["banana", "", "1/0", "2 3", "oo oo", None, True, float("nan")])
    def test_pontos_invalidos(self, ruim):
        with pytest.raises(ValidacaoErro):
            interpretar_ponto(ruim)
