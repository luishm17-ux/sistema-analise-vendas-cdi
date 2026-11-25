"""Exemplos executáveis com resposta conhecida (calculada à mão).

Cada linha compara o resultado do sistema com um valor esperado que pode ser conferido
no papel. A aba "Exemplos verificáveis" do aplicativo e os testes automatizados usam esta
mesma lista — se alguma conta do sistema estiver errada, ela aparece aqui.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
import sympy as sp

from . import dados, derivadas, funcoes, integrais, limites
from .funcoes import para_numerica
from .validacao import ValidacaoErro, exigir_intervalo, exigir_numero, interpretar_expressao as P, interpretar_ponto, t


def _mesma_funcao(f: sp.Expr, g: sp.Expr, a: float = 0.37, b: float = 17.3) -> bool:
    x = np.linspace(a, b, 15)
    u, v = para_numerica(f)(x), para_numerica(g)(x)
    ok = np.isfinite(u) & np.isfinite(v)
    return bool(ok.sum() >= 5 and np.allclose(u[ok], v[ok], rtol=1e-9, atol=1e-9))


def _difere_por_constante(F: sp.Expr, G: sp.Expr, a: float = 0.37, b: float = 17.3) -> bool:
    x = np.linspace(a, b, 15)
    d = para_numerica(F)(x) - para_numerica(G)(x)
    d = d[np.isfinite(d)]
    return bool(d.size >= 5 and np.allclose(d, d[0], rtol=1e-9, atol=1e-9))


def _linha(modulo: str, teste: str, esperado: str, obtido: str, ok: bool) -> dict:
    return {"Módulo": modulo, "Teste": teste, "Esperado": esperado, "Obtido": obtido, "Confere": bool(ok)}


def _num(x: float) -> str:
    return f"{x:.6g}"


# --------------------------------------------------------------------------- #
def exemplos_funcoes() -> list[dict]:
    casos = [
        # (tipo, parâmetros, t, esperado, descrição da conta)
        ("linear", {"a": 2, "b": 5}, 10, 25.0, "2·10 + 5"),
        ("quadratica", {"a": 1, "b": -2, "c": 3}, 4, 11.0, "4² − 2·4 + 3"),
        ("exponencial", {"a": 100, "k": math.log(2)}, 3, 800.0, "100·e^(3·ln2) = 100·2³"),
        ("logaritmica", {"a": 10, "b": 5}, math.e - 1, 15.0, "10·ln(e) + 5"),
        ("logistica", {"L": 100, "k": 0.5, "t0": 10}, 10, 50.0, "100/(1 + e⁰) = 50 (ponto de inflexão)"),
        ("sazonal", {"c": 100, "a": 0, "A": 20, "P": 12, "fase": 0}, 3, 120.0, "100 + 20·sin(π/2)"),
    ]
    linhas = []
    for chave, params, ponto, esperado, conta in casos:
        v = funcoes.valor_em(funcoes.construir(chave, params), ponto)
        linhas.append(_linha("Funções", f"{funcoes.TIPOS[chave].nome}: V({ponto:.4g}) = {conta}", _num(esperado), _num(v), abs(v - esperado) < 1e-9))

    dom, im = funcoes.TIPOS["quadratica"].dominio_imagem({"a": -1, "b": 4, "c": 0})
    linhas.append(_linha("Funções", "Quadrática −t²+4t: imagem (vértice em t=2, V=4)", "(-∞, 4]", im, im == "(-∞, 4]"))
    dom, im = funcoes.TIPOS["logistica"].dominio_imagem({"L": 100, "k": 1, "t0": 0})
    linhas.append(_linha("Funções", "Logística com L = 100: imagem", "(0, 100)", im, im == "(0, 100)"))
    d = funcoes.dominio_simbolico(P("1/(t-2)"))
    linhas.append(_linha("Funções", "Domínio de 1/(t−2)", "(-∞, 2) ∪ (2, ∞)", d, d == "(-∞, 2) ∪ (2, ∞)"))
    return linhas


def exemplos_limites() -> list[dict]:
    casos = [
        ("(t^2-4)/(t-2)", "2", "4", "fatora: (t+2)"),
        ("sin(t)/t", "0", "1", "limite fundamental"),
        ("(3*t+1)/(t+2)", "oo", "3", "divide por t: 3/1"),
        ("1/t^2", "0", "oo", "assíntota vertical"),
        ("abs(t)/t", "0", "não existe", "laterais: −1 e +1"),
        ("1/t", "0", "não existe", "laterais: −∞ e +∞"),
        ("sin(t)", "oo", "não existe", "oscila entre −1 e 1"),
        ("exp(-t)", "oo", "0", "decaimento"),
    ]
    linhas = []
    for expr_txt, ponto, esperado, nota in casos:
        r = limites.calcular_limite(P(expr_txt), ponto)
        ok = r.valor_texto == esperado.replace("oo", "∞")
        linhas.append(_linha("Limites", f"lim t→{ponto} de {expr_txt}  ({nota})", esperado.replace("oo", "∞"), r.valor_texto, ok))

    lo = funcoes.construir("logistica", {"L": 100, "k": 0.5, "t0": 10})
    r = limites.limite_saturacao(lo)
    linhas.append(_linha("Limites", "Saturação: lim t→∞ de 100/(1+e^(−0,5(t−10)))", "100", r.valor_texto, r.valor == 100))

    d = limites.limite_definicao_derivada(P("t^2"), 3)
    linhas.append(_linha("Limites", "Definição de derivada: lim h→0 [(3+h)² − 3²]/h", "6", str(d["pelo_limite"]), d["pelo_limite"] == 6 and d["coincidem"]))
    return linhas


def exemplos_derivadas() -> list[dict]:
    linhas = []
    for regra, (_, ex, esperado) in derivadas.CATALOGO_REGRAS.items():
        f, esp = P(ex), P(esperado) if "sign" not in esperado else sp.sign(t - 2)
        r = derivadas.derivar(f)
        usou = regra in r.regras_usadas
        ok = r.verificada and usou and _mesma_funcao(r.derivada, esp)
        linhas.append(_linha("Derivadas", f"Regra {regra}: d/dt [{ex}]", str(esp), str(r.derivada), ok))
    # aplicação em vendas
    v = funcoes.construir("quadratica", {"a": -25, "b": 1500, "c": 20000})
    taxa = derivadas.derivar(v).derivada.subs(t, 10)
    linhas.append(_linha("Derivadas", "Vendas V = −25t²+1500t+20000: V′(10) (taxa de crescimento)", "1000", str(taxa), taxa == 1000))
    pico = derivadas.analisar_crescimento(v, 0, 60)["criticos"]
    linhas.append(_linha("Derivadas", "Mesmo modelo: V′(t) = 0 no pico", "t = 30", f"t = {pico[0]['t']:.4g}" if pico else "—", bool(pico) and abs(pico[0]["t"] - 30) < 1e-6))
    return linhas


def exemplos_integrais() -> list[dict]:
    linhas = []
    for regra, (_, ex, esperado) in integrais.CATALOGO_REGRAS.items():
        f = P(ex)
        r = integrais.integrar(f)
        esp = P(esperado)
        ok = r.verificada and regra in r.regras_usadas and _difere_por_constante(r.antiderivada, esp)
        linhas.append(_linha("Integrais", f"Regra {regra}: ∫ {ex} dt", f"{esperado} + C", f"{r.antiderivada} + C", ok))

    definidas = [
        ("t^2", 0, 3, 9.0, "t³/3 de 0 a 3"),
        ("exp(t)", 0, 1, math.e - 1, "e − 1"),
        ("sin(t)", 0, math.pi, 2.0, "−cos(π) + cos(0)"),
        ("1/t", 1, math.e, 1.0, "ln(e) − ln(1)"),
        ("(2*t+1)^3", 0, 2, 78.0, "(5⁴ − 1)/8"),
        ("t*exp(t)", 0, 1, 1.0, "por partes: [(t−1)eᵗ] de 0 a 1"),
        ("ln(1+t)", 0, 1, 2 * math.log(2) - 1, "por partes: 2·ln2 − 1"),
    ]
    for ex, a, b, esperado, nota in definidas:
        r = integrais.integral_definida(P(ex), a, b)
        v = r.valor_exato if r.valor_exato is not None else r.valor_numerico
        linhas.append(_linha("Integrais", f"∫ de {a:.4g} a {b:.4g} de {ex} dt  ({nota})", _num(esperado), _num(v), abs(v - esperado) < 1e-8 and r.confere))

    v = funcoes.construir("linear", {"a": 1000, "b": 5000})
    r = integrais.integral_definida(v, 0.5, 12.5)
    esperado = 12 * 5000 + 1000 * (12 * 13 / 2)  # soma dos meses 1..12 de 1000·i + 5000
    linhas.append(_linha("Integrais", "Receita acumulada em 12 meses de V = 1000t + 5000 (= Σ 1000·i + 5000)", _num(esperado), _num(r.valor_exato), abs(r.valor_exato - esperado) < 1e-6))
    return linhas


def exemplos_validacao() -> list[dict]:
    """Entradas inválidas que devem ser *recusadas* com uma mensagem clara."""
    casos: list[tuple[str, Callable[[], object]]] = [
        ("Parâmetro não numérico: a = 'abc'", lambda: funcoes.construir("linear", {"a": "abc", "b": 1})),
        ("Parâmetro infinito: b = ∞", lambda: funcoes.construir("linear", {"a": 1, "b": float("inf")})),
        ("Exponencial com a ≤ 0 (vendas iniciais negativas)", lambda: funcoes.construir("exponencial", {"a": -5, "k": 0.1})),
        ("Logística com L = 0", lambda: funcoes.construir("logistica", {"L": 0, "k": 1, "t0": 0})),
        ("Intervalo invertido: [10, 2]", lambda: exigir_intervalo(10, 2)),
        ("Função com identificador desconhecido: os.system(t)", lambda: P("os.system(t)")),
        ("Código malicioso: __import__('os')", lambda: P("__import__('os')")),
        ("Função vazia", lambda: P("   ")),
        ("Sintaxe inválida: 3*t +", lambda: P("3*t +")),
        ("Torre de potências 9^9^9 (travaria o cálculo)", lambda: P("9^9^9")),
        ("Ponto de limite inválido: 'banana'", lambda: interpretar_ponto("banana")),
        ("Integral sobre intervalo com divisão por zero: ∫ 1/(t−5) em [0,10]", lambda: integrais.integral_definida(P("1/(t-5)"), 0, 10)),
        ("CSV sem a coluna 'vendas'", lambda: dados.carregar_csv(b"mes,valor\n1,2\n")),
        ("CSV com venda negativa", lambda: dados.carregar_csv(b"vendas\n1\n2\n-3\n4\n5\n6\n")),
        ("CSV com poucos meses (3)", lambda: dados.carregar_csv(b"vendas\n1\n2\n3\n")),
        ("CSV com texto no lugar de número", lambda: dados.carregar_csv(b"vendas\n1\n2\nabc\n4\n5\n6\n")),
    ]
    linhas = []
    for descricao, acao in casos:
        try:
            acao()
            linhas.append(_linha("Validação", descricao, "recusar (ValidacaoErro)", "ACEITOU (erro!)", False))
        except ValidacaoErro as erro:
            linhas.append(_linha("Validação", descricao, "recusar (ValidacaoErro)", str(erro), True))
    return linhas


def executar_exemplos() -> pd.DataFrame:
    """Roda todos os exemplos e devolve uma tabela com esperado × obtido."""
    linhas: list[dict] = []
    for grupo in (exemplos_funcoes, exemplos_limites, exemplos_derivadas, exemplos_integrais, exemplos_validacao):
        linhas.extend(grupo())
    return pd.DataFrame(linhas)
