"""Aula 4 — Integrais.

Em vendas, se ``V(t)`` é a receita mensal, então a integral definida
``∫ₐᵇ V(t) dt`` é a **receita acumulada** entre os meses ``a`` e ``b`` e
``(1/(b−a))·∫ₐᵇ V(t) dt`` é a **receita mensal média** nesse período.

Convenção do sistema: o mês ``i`` é representado por ``t = i`` e ocupa o intervalo
``[i − 0,5; i + 0,5]``; assim a soma das vendas dos meses 1…n corresponde (aproximadamente)
a ``∫ de 0,5 até n + 0,5``.

Como em ``derivadas.py``, as regras de integração são implementadas uma a uma
(``_integrar``) e registradas, para a interface mostrar o passo a passo. Quando
nenhuma regra própria se aplica, recorremos ao ``sympy.integrate`` (e dizemos isso).
Toda antiderivada é conferida derivando-a de volta e a integral definida é
conferida contra a integração numérica de ``scipy.integrate.quad``.

Regras implementadas (11): constante, potência, 1/t, soma/subtração, constante×função,
exponencial, seno, cosseno, substituição linear, substituição não linear (logística) e
integração por partes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import quad

from .derivadas import derivar
from .funcoes import para_numerica, simplificar_seguro
from .validacao import ValidacaoErro, exigir_intervalo, exigir_numero, t

R_CONSTANTE = "Constante"
R_POTENCIA = "Potência"
R_INVERSO = "1/t (logaritmo)"
R_SOMA = "Soma/Subtração"
R_CONST_FUNC = "Constante × função"
R_EXPONENCIAL = "Exponencial"
R_SENO = "Seno"
R_COSSENO = "Cosseno"
R_SUBST_LINEAR = "Substituição (u = at + b)"
R_SUBST_EXP = "Substituição (u = eᵍ)"
R_PARTES = "Integração por partes"
R_OUTRA = "Outra (SymPy)"

# Catálogo exibido na interface: regra -> (fórmula LaTeX, integrando, antiderivada esperada)
CATALOGO_REGRAS = {
    R_CONSTANTE: (r"\int c\,dt = c\,t + C", "5", "5*t"),
    R_POTENCIA: (r"\int t^n\,dt = \dfrac{t^{n+1}}{n+1} + C\quad(n\neq -1)", "t^3", "t^4/4"),
    R_INVERSO: (r"\int \dfrac{1}{t}\,dt = \ln|t| + C", "1/t", "ln(abs(t))"),
    R_SOMA: (r"\int (f\pm g)\,dt = \int f\,dt \pm \int g\,dt", "3*t^2 + 2*t + 1", "t^3 + t^2 + t"),
    R_CONST_FUNC: (r"\int c\,f\,dt = c\int f\,dt", "6*t^2", "2*t^3"),
    R_EXPONENCIAL: (r"\int e^{kt}\,dt = \dfrac{e^{kt}}{k} + C", "exp(2*t)", "exp(2*t)/2"),
    R_SENO: (r"\int \sin(kt)\,dt = -\dfrac{\cos(kt)}{k} + C", "sin(3*t)", "-cos(3*t)/3"),
    R_COSSENO: (r"\int \cos(kt)\,dt = \dfrac{\sin(kt)}{k} + C", "cos(2*t)", "sin(2*t)/2"),
    R_SUBST_LINEAR: (r"u = at+b,\ du = a\,dt \Rightarrow \int f(at+b)\,dt = \dfrac{1}{a}F(at+b)", "(2*t+1)^3", "(2*t+1)^4/8"),
    R_SUBST_EXP: (r"u = e^{g},\ \int \dfrac{dt}{1+e^{g}} = t - \dfrac{\ln(1+e^{g})}{g'}", "1/(1+exp(-t))", "ln(1+exp(t))"),
    R_PARTES: (r"\int u\,dv = u\,v - \int v\,du", "t*exp(t)", "(t-1)*exp(t)"),
}


class IntegracaoNaoSimbolica(ValidacaoErro):
    """Nenhuma antiderivada em forma fechada foi encontrada (a integral definida usa o método numérico)."""


@dataclass
class Passo:
    regras: tuple[str, ...]
    f: sp.Expr
    F: sp.Expr = sp.Integer(0)


@dataclass
class ResultadoIntegralIndefinida:
    entrada: sp.Expr
    antiderivada: sp.Expr
    passos: list[Passo] = field(default_factory=list)
    regras_usadas: list[str] = field(default_factory=list)
    verificada: bool = False  # d/dt F == f ?
    erro_maximo: float = 0.0


@dataclass
class ResultadoIntegralDefinida:
    entrada: sp.Expr
    a: float
    b: float
    indefinida: ResultadoIntegralIndefinida | None
    valor_exato: float | None  # F(b) − F(a)
    valor_numerico: float  # quad
    erro_numerico: float
    media: float
    confere: bool


# --------------------------------------------------------------------------- #
# Motor de regras
# --------------------------------------------------------------------------- #
def _linear(g: sp.Expr) -> tuple[sp.Expr, sp.Expr] | None:
    """Se g(t) = a·t + b com a ≠ 0 constante, devolve (a, b); caso contrário None."""
    try:
        poli = sp.Poly(g, t)
    except sp.PolynomialError:
        return None
    if poli.degree() != 1:
        return None
    a, b = poli.all_coeffs()
    return a, b


def _novo(passos: list[Passo], f: sp.Expr, *regras: str) -> Passo:
    p = Passo(tuple(regras), f)
    passos.append(p)
    return p


def _subst(a: sp.Expr, regra: str) -> tuple[str, ...]:
    return (regra,) if a == 1 else (regra, R_SUBST_LINEAR)


def _integrar(e: sp.Expr, passos: list[Passo]) -> sp.Expr:
    e = sp.sympify(e)

    if not e.has(t):
        p = _novo(passos, e, R_CONSTANTE)
        p.F = e * t
        return p.F

    if e == t:
        p = _novo(passos, e, R_POTENCIA)  # t = t¹  →  t²/2
        p.F = t**2 / 2
        return p.F

    if e.is_Add:
        p = _novo(passos, e, R_SOMA)
        p.F = sp.Add(*[_integrar(a, passos) for a in e.args])
        return p.F

    if e.is_Mul:
        constante, dependente = e.as_independent(t)
        if constante != 1:
            p = _novo(passos, e, R_CONST_FUNC)
            p.F = constante * _integrar(dependente, passos)
            return p.F
        r = _tentar_partes(e, passos)
        if r is not None:
            return r
        return _fallback(e, passos)

    if e.is_Pow:
        base, expoente = e.args
        if not expoente.has(t):
            lin = _linear(base)
            if lin is not None:
                a, b = lin
                if expoente == -1:
                    p = _novo(passos, e, *(((R_INVERSO,) if base == t else (R_INVERSO, R_SUBST_LINEAR))))
                    p.F = sp.log(sp.Abs(base)) / a
                else:
                    p = _novo(passos, e, *_subst(a, R_POTENCIA))
                    p.F = base ** (expoente + 1) / ((expoente + 1) * a)
                return p.F
            # 1 / (1 + e^g)
            if expoente == -1 and base.is_Add and len(base.args) == 2 and 1 in base.args:
                resto = base - 1
                if isinstance(resto, sp.exp):
                    lin = _linear(resto.args[0])
                    if lin is not None:
                        a, _ = lin
                        p = _novo(passos, e, R_SUBST_EXP)
                        p.F = t - sp.log(base) / a
                        return p.F
        elif not base.has(t):
            lin = _linear(expoente)
            if lin is not None:
                a, _ = lin
                p = _novo(passos, e, *_subst(a, R_EXPONENCIAL))
                p.F = e / (a * sp.log(base))
                return p.F
        return _fallback(e, passos)

    if isinstance(e, sp.exp):
        lin = _linear(e.args[0])
        if lin is not None:
            a, _ = lin
            p = _novo(passos, e, *_subst(a, R_EXPONENCIAL))
            p.F = e / a
            return p.F

    if isinstance(e, (sp.sin, sp.cos)):
        lin = _linear(e.args[0])
        if lin is not None:
            a, _ = lin
            u = e.args[0]
            if isinstance(e, sp.sin):
                p = _novo(passos, e, *_subst(a, R_SENO))
                p.F = -sp.cos(u) / a
            else:
                p = _novo(passos, e, *_subst(a, R_COSSENO))
                p.F = sp.sin(u) / a
            return p.F

    if isinstance(e, sp.log) and len(e.args) == 1:
        lin = _linear(e.args[0])
        if lin is not None:  # ∫ ln(u) dt  (u = at+b)  =  (u·ln u − u)/a   [por partes]
            a, _ = lin
            u = e.args[0]
            p = _novo(passos, e, R_PARTES, R_SUBST_LINEAR) if a != 1 else _novo(passos, e, R_PARTES)
            p.F = (u * sp.log(u) - u) / a
            return p.F

    return _fallback(e, passos)


def _tentar_partes(e: sp.Expr, passos: list[Passo]) -> sp.Expr | None:
    """∫ p(t)·q(t) dt, com p polinômio (grau ≥ 1) e q ∈ {exp, sin, cos} de argumento linear.

    Escolhemos u = p e dv = q dt. Então ∫ p q dt = p·Q − ∫ p'·Q dt, onde Q = ∫ q dt.
    Como o grau de p cai a cada passo, o processo termina.
    """
    for fator in e.args:
        if isinstance(fator, (sp.exp, sp.sin, sp.cos)) and _linear(fator.args[0]) is not None:
            resto = sp.simplify(e / fator)
            try:
                poli = sp.Poly(resto, t)
            except sp.PolynomialError:
                continue
            if poli.degree() >= 1:
                p = _novo(passos, e, R_PARTES)
                Q = _integrar(fator, passos)
                dp = sp.diff(resto, t)
                p.F = resto * Q - _integrar(sp.expand(dp * Q), passos)
                return p.F
    return None


def _fallback(e: sp.Expr, passos: list[Passo]) -> sp.Expr:
    try:
        F = sp.integrate(e, t)
    except Exception:
        F = None
    if F is None or F.has(sp.Integral):
        raise IntegracaoNaoSimbolica(
            "Não encontrei uma antiderivada em forma fechada para essa função. "
            "A integral definida ainda pode ser calculada numericamente."
        )
    p = _novo(passos, e, R_OUTRA)
    p.F = F
    return F


# --------------------------------------------------------------------------- #
# API pública
# --------------------------------------------------------------------------- #
def integrar(expr: sp.Expr) -> ResultadoIntegralIndefinida:
    """Antiderivada de ``expr`` em ``t`` (sem a constante C) com passo a passo e verificação."""
    expr = sp.sympify(expr)
    passos: list[Passo] = []
    F = _integrar(expr, passos)
    F = simplificar_seguro(F, limite_ops=200)
    usadas: list[str] = []
    for p in passos:
        for r in p.regras:
            if r not in usadas:
                usadas.append(r)
    ok, erro = _conferir_antiderivada(expr, F)
    return ResultadoIntegralIndefinida(expr, F, passos, usadas, ok, erro)


def _conferir_antiderivada(f: sp.Expr, F: sp.Expr) -> tuple[bool, float]:
    """Confere d/dt F = f em vários pontos (usa a regra de derivação do próprio sistema)."""
    dF = derivar(F).derivada
    x = np.linspace(0.37, 17.3, 11)
    a, b = para_numerica(dF)(x), para_numerica(f)(x)
    valido = np.isfinite(a) & np.isfinite(b)
    if valido.sum() < 3:
        return False, float("nan")
    erro = float(np.max(np.abs(a[valido] - b[valido]) / np.maximum(1.0, np.abs(b[valido]))))
    return bool(erro < 1e-8), erro


def _checar_definida_no_intervalo(expr: sp.Expr, a: float, b: float) -> None:
    y = para_numerica(expr)(np.linspace(a, b, 2001))
    if not np.all(np.isfinite(y)):
        raise ValidacaoErro(
            f"A função não está definida (ou é infinita) em parte do intervalo [{a:g}, {b:g}]; "
            "a integral não pode ser calculada aí. Escolha um intervalo dentro do domínio."
        )


def integral_definida(expr: sp.Expr, a: float, b: float) -> ResultadoIntegralDefinida:
    """∫ₐᵇ expr dt pelo Teorema Fundamental do Cálculo, conferida por integração numérica."""
    a, b = exigir_intervalo(a, b)
    expr = sp.sympify(expr)
    _checar_definida_no_intervalo(expr, a, b)

    f = para_numerica(expr)
    numerico, erro_num = quad(lambda z: float(f(np.array([z]))[0]), a, b, limit=200)

    indef: ResultadoIntegralIndefinida | None
    exato: float | None
    try:
        indef = integrar(expr)
        Fn = para_numerica(indef.antiderivada)
        exato = float(Fn(np.array([b]))[0] - Fn(np.array([a]))[0])
        if not np.isfinite(exato):
            exato = None
    except IntegracaoNaoSimbolica:
        indef, exato = None, None

    referencia = exato if exato is not None else numerico
    confere = exato is None or abs(exato - numerico) <= 1e-6 * max(1.0, abs(numerico))
    return ResultadoIntegralDefinida(
        expr, a, b, indef, exato, float(numerico), float(erro_num), referencia / (b - a), bool(confere)
    )


def soma_de_riemann(expr: sp.Expr, a: float, b: float, n: int = 10, metodo: str = "ponto_medio") -> float:
    """Aproxima ∫ₐᵇ f por soma de Riemann com ``n`` retângulos (esquerda, direita ou ponto médio)."""
    a, b = exigir_intervalo(a, b)
    if not (1 <= int(n) <= 1_000_000):
        raise ValidacaoErro("O número de retângulos deve estar entre 1 e 1.000.000.")
    n = int(n)
    if metodo not in {"esquerda", "direita", "ponto_medio"}:
        raise ValidacaoErro("Método inválido. Use 'esquerda', 'direita' ou 'ponto_medio'.")
    dx = (b - a) / n
    desloc = {"esquerda": 0.0, "direita": 1.0, "ponto_medio": 0.5}[metodo]
    xs = a + (np.arange(n) + desloc) * dx
    y = para_numerica(expr)(xs)
    if not np.all(np.isfinite(y)):
        raise ValidacaoErro("A função não está definida em todos os pontos do intervalo.")
    return float(np.sum(y) * dx)


def tabela_riemann(expr: sp.Expr, a: float, b: float, valores_n=(4, 8, 16, 32, 64, 128)) -> pd.DataFrame:
    """Mostra a soma de Riemann convergindo para a integral (a integral é um *limite*)."""
    exata = integral_definida(expr, a, b)
    ref = exata.valor_exato if exata.valor_exato is not None else exata.valor_numerico
    linhas = []
    for n in valores_n:
        s = soma_de_riemann(expr, a, b, n, "ponto_medio")
        linhas.append({"n (retângulos)": n, "Soma de Riemann": s, "Erro absoluto": abs(s - ref)})
    return pd.DataFrame(linhas)


def acumulada(expr: sp.Expr, a: float, t_final: np.ndarray) -> np.ndarray:
    """Receita acumulada R(t) = ∫ₐᵗ V(s) ds para um vetor de instantes (para o gráfico)."""
    a = exigir_numero(a, "início")
    t_final = np.asarray(t_final, dtype=float)
    try:
        Fn = para_numerica(integrar(expr).antiderivada)
        return Fn(t_final) - float(Fn(np.array([a]))[0])
    except IntegracaoNaoSimbolica:
        f = para_numerica(expr)
        xs = np.concatenate([[a], t_final])
        ys = f(xs)
        incrementos = (ys[1:] + ys[:-1]) / 2 * np.diff(xs)  # regra do trapézio
        return np.cumsum(incrementos)
