"""Aula 3 — Derivadas.

Em vendas, a derivada ``V'(t)`` é a **taxa de crescimento instantânea** (R$ por mês
a cada mês) e ``V''(t)`` diz se esse crescimento está **acelerando** ou **freando**.

Em vez de chamar ``sympy.diff`` diretamente, este módulo implementa as regras de
derivação uma a uma (``_derivar``) e registra cada regra aplicada. Assim a
interface consegue *mostrar o passo a passo* e o resultado é conferido de duas
maneiras independentes (``sympy.diff`` e diferença finita numérica).

Regras implementadas (13): constante, potência, constante×função, soma/subtração,
produto, quociente, cadeia, exponencial, logaritmo natural, seno, cosseno,
tangente e módulo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.optimize import brentq

from .funcoes import para_numerica, simplificar_seguro
from .validacao import ValidacaoErro, exigir_intervalo, exigir_numero, t

# Nomes das regras (usados na interface, nos testes e no README)
R_CONSTANTE = "Constante"
R_POTENCIA = "Potência"
R_CONST_FUNC = "Constante × função"
R_SOMA = "Soma/Subtração"
R_PRODUTO = "Produto"
R_QUOCIENTE = "Quociente"
R_CADEIA = "Cadeia"
R_EXPONENCIAL = "Exponencial"
R_LOGARITMO = "Logaritmo natural"
R_SENO = "Seno"
R_COSSENO = "Cosseno"
R_TANGENTE = "Tangente"
R_MODULO = "Módulo"
R_OUTRA = "Outra (SymPy)"

# Catálogo exibido na interface: regra -> (fórmula em LaTeX, exemplo, derivada esperada)
CATALOGO_REGRAS = {
    R_CONSTANTE: (r"(c)' = 0", "5000", "0"),
    R_POTENCIA: (r"(t^n)' = n\,t^{n-1}", "t^3", "3*t^2"),
    R_CONST_FUNC: (r"(c\cdot f)' = c\cdot f'", "4*t^2", "8*t"),
    R_SOMA: (r"(f \pm g)' = f' \pm g'", "t^2 + 3*t - 7", "2*t + 3"),
    R_PRODUTO: (r"(f\cdot g)' = f'g + fg'", "t^2*exp(t)", "(t^2 + 2*t)*exp(t)"),
    R_QUOCIENTE: (r"\left(\dfrac{f}{g}\right)' = \dfrac{f'g - fg'}{g^2}", "t/(t+1)", "1/(t+1)^2"),
    R_CADEIA: (r"\big(f(g(t))\big)' = f'(g(t))\cdot g'(t)", "(3*t+1)^5", "15*(3*t+1)^4"),
    R_EXPONENCIAL: (r"(e^{u})' = e^{u}\cdot u'", "exp(2*t)", "2*exp(2*t)"),
    R_LOGARITMO: (r"(\ln u)' = \dfrac{u'}{u}", "ln(t^2+1)", "2*t/(t^2+1)"),
    R_SENO: (r"(\sin u)' = \cos u\cdot u'", "sin(2*t)", "2*cos(2*t)"),
    R_COSSENO: (r"(\cos u)' = -\sin u\cdot u'", "cos(t)", "-sin(t)"),
    R_TANGENTE: (r"(\tan u)' = \sec^2 u\cdot u'", "tan(3*t)", "3/cos(3*t)^2"),
    R_MODULO: (r"(|u|)' = \operatorname{sinal}(u)\cdot u'", "abs(t-2)", "sign(t-2)"),
}


@dataclass
class Passo:
    """Uma aplicação de regra: ``f`` é a expressão tratada e ``df`` o resultado dessa etapa."""

    regras: tuple[str, ...]
    f: sp.Expr
    df: sp.Expr = sp.Integer(0)


@dataclass
class ResultadoDerivada:
    entrada: sp.Expr
    derivada: sp.Expr  # simplificada
    derivada_bruta: sp.Expr  # exatamente como as regras produziram
    passos: list[Passo] = field(default_factory=list)
    regras_usadas: list[str] = field(default_factory=list)
    confere_sympy: bool = False
    confere_numerico: bool = False
    erro_maximo: float = 0.0

    @property
    def verificada(self) -> bool:
        return self.confere_sympy and self.confere_numerico


# --------------------------------------------------------------------------- #
# Motor de regras
# --------------------------------------------------------------------------- #
def _novo(passos: list[Passo], f: sp.Expr, *regras: str) -> Passo:
    p = Passo(tuple(regras), f)
    passos.append(p)
    return p


def _com_cadeia(u: sp.Expr, regra: str) -> tuple[str, ...]:
    """Se o 'miolo' u não é simplesmente t, a regra da cadeia também entra em jogo."""
    return (regra,) if u == t else (regra, R_CADEIA)


def _derivar(e: sp.Expr, passos: list[Passo]) -> sp.Expr:
    e = sp.sympify(e)

    if not e.has(t):
        _novo(passos, e, R_CONSTANTE).df = sp.Integer(0)
        return sp.Integer(0)

    if e == t:
        p = _novo(passos, e, R_POTENCIA)
        p.df = sp.Integer(1)
        return p.df

    if e.is_Add:
        p = _novo(passos, e, R_SOMA)
        p.df = sp.Add(*[_derivar(a, passos) for a in e.args])
        return p.df

    if e.is_Mul:
        constante, dependente = e.as_independent(t)
        if constante != 1:
            p = _novo(passos, e, R_CONST_FUNC)
            p.df = constante * _derivar(dependente, passos)
            return p.df
        # fatores com expoente numérico negativo ficam no denominador
        numerador, denominador = [], []
        for fator in e.args:
            if fator.is_Pow and fator.exp.is_number and fator.exp.is_negative:
                denominador.append(fator.base ** (-fator.exp))
            else:
                numerador.append(fator)
        if denominador:
            num, den = sp.Mul(*numerador), sp.Mul(*denominador)
            p = _novo(passos, e, R_QUOCIENTE)
            dnum, dden = _derivar(num, passos), _derivar(den, passos)
            p.df = (dnum * den - num * dden) / den**2
            return p.df
        f, g = e.args[0], sp.Mul(*e.args[1:])
        p = _novo(passos, e, R_PRODUTO)
        df, dg = _derivar(f, passos), _derivar(g, passos)
        p.df = df * g + f * dg
        return p.df

    if e.is_Pow:
        base, expoente = e.args
        if not expoente.has(t):
            if base == t:
                p = _novo(passos, e, R_POTENCIA)
                p.df = expoente * t ** (expoente - 1)
                return p.df
            p = _novo(passos, e, R_POTENCIA, R_CADEIA)
            p.df = expoente * base ** (expoente - 1) * _derivar(base, passos)
            return p.df
        if not base.has(t):
            p = _novo(passos, e, *_com_cadeia(expoente, R_EXPONENCIAL))
            p.df = e * sp.log(base) * _derivar(expoente, passos)
            return p.df
        p = _novo(passos, e, R_OUTRA)  # t^t e similares: derivação logarítmica
        p.df = sp.diff(e, t)
        return p.df

    if isinstance(e, sp.exp):
        u = e.args[0]
        p = _novo(passos, e, *_com_cadeia(u, R_EXPONENCIAL))
        p.df = e * _derivar(u, passos)
        return p.df

    if isinstance(e, sp.log) and len(e.args) == 1:
        u = e.args[0]
        p = _novo(passos, e, *_com_cadeia(u, R_LOGARITMO))
        p.df = _derivar(u, passos) / u
        return p.df

    if isinstance(e, (sp.sin, sp.cos, sp.tan)):
        u = e.args[0]
        nome, externa = {
            sp.sin: (R_SENO, sp.cos(u)),
            sp.cos: (R_COSSENO, -sp.sin(u)),
            sp.tan: (R_TANGENTE, 1 / sp.cos(u) ** 2),
        }[type(e)]
        p = _novo(passos, e, *_com_cadeia(u, nome))
        p.df = externa * _derivar(u, passos)
        return p.df

    if isinstance(e, sp.Abs):
        u = e.args[0]
        p = _novo(passos, e, *_com_cadeia(u, R_MODULO))
        p.df = sp.sign(u) * _derivar(u, passos)
        return p.df

    p = _novo(passos, e, R_OUTRA)
    p.df = sp.diff(e, t)
    return p.df


def pontos_de_teste(a: float = 0.37, b: float = 17.3, n: int = 9) -> np.ndarray:
    """Pontos 'sem graça' (evitam inteiros) onde comparamos resultados numericamente."""
    return np.linspace(a, b, n)


def derivar(expr: sp.Expr) -> ResultadoDerivada:
    """Deriva ``expr`` em relação a ``t`` aplicando as regras e conferindo o resultado."""
    expr = sp.sympify(expr)
    passos: list[Passo] = []
    bruta = _derivar(expr, passos)
    simplificada = simplificar_seguro(bruta)

    usadas: list[str] = []
    for p in passos:
        for r in p.regras:
            if r not in usadas:
                usadas.append(r)

    res = ResultadoDerivada(expr, simplificada, bruta, passos, usadas)
    res.confere_sympy, res.confere_numerico, res.erro_maximo = _conferir(expr, simplificada)
    return res


def _conferir(expr: sp.Expr, derivada: sp.Expr) -> tuple[bool, bool, float]:
    """Confere V' (1) contra ``sympy.diff`` e (2) contra a diferença finita central."""
    x = pontos_de_teste()
    f = para_numerica(expr)
    d = para_numerica(derivada)
    ref = para_numerica(sp.diff(expr, t))
    h = 1e-5
    numerica = (f(x + h) - f(x - h)) / (2 * h)
    valido = np.isfinite(d(x)) & np.isfinite(ref(x)) & np.isfinite(numerica)
    if valido.any():
        # Perto de assíntotas (ex.: tan) a diferença finita perde precisão: descartamos valores
        # que são ordens de grandeza maiores que o típico. Crescimento legítimo (polinômios,
        # exponenciais) não é afetado, pois fica dentro dessa margem.
        tipico = np.median(np.abs(ref(x)[valido]))
        valido &= np.abs(ref(x)) <= 1e4 * tipico + 1.0
    if valido.sum() < 3:
        return False, False, float("nan")
    escala = np.maximum(1.0, np.abs(ref(x)[valido]))
    erro_sympy = np.max(np.abs(d(x)[valido] - ref(x)[valido]) / escala)
    erro_num = np.max(np.abs(d(x)[valido] - numerica[valido]) / escala)
    return bool(erro_sympy < 1e-8), bool(erro_num < 1e-4), float(max(erro_sympy, erro_num))


def derivada_numerica(expr: sp.Expr, ponto: float, h: float = 1e-5) -> float:
    """Diferença finita central (f(x+h) − f(x−h)) / 2h — verificação independente."""
    ponto = exigir_numero(ponto, "t")
    h = exigir_numero(h, "h", estrito_positivo=True)
    f = para_numerica(expr)
    no_ponto = f(np.array([ponto]))[0]
    y = (f(np.array([ponto + h])) - f(np.array([ponto - h]))) / (2 * h)
    if not (np.isfinite(no_ponto) and np.isfinite(y[0])):  # derivável ⇒ definida no ponto
        raise ValidacaoErro(f"A função não é derivável (ou não está definida) em t = {ponto:g}.")
    return float(y[0])


def derivada_n(expr: sp.Expr, ordem: int = 2) -> sp.Expr:
    """Derivada de ordem ``ordem`` (1 a 4), reaplicando as regras."""
    if not (1 <= int(ordem) <= 4):
        raise ValidacaoErro("A ordem da derivada deve ser um inteiro entre 1 e 4.")
    e = sp.sympify(expr)
    for _ in range(int(ordem)):
        e = derivar(e).derivada
    return e


# --------------------------------------------------------------------------- #
# Interpretação para vendas
# --------------------------------------------------------------------------- #
def taxa_relativa(expr: sp.Expr) -> sp.Expr:
    """Crescimento relativo V'(t)/V(t): quanto (em %) as vendas crescem por mês."""
    expr = sp.sympify(expr)
    return simplificar_seguro(derivar(expr).derivada / expr)


def _raizes(f, a: float, b: float, amostras: int = 2001) -> list[float]:
    x = np.linspace(a, b, amostras)
    y = f(x)
    raizes = []
    for i in range(len(x) - 1):
        if np.isfinite(y[i]) and np.isfinite(y[i + 1]):
            if y[i] == 0:
                raizes.append(float(x[i]))
            elif y[i] * y[i + 1] < 0:
                try:
                    raizes.append(float(brentq(lambda z: float(f(np.array([z]))[0]), x[i], x[i + 1])))
                except (ValueError, RuntimeError):
                    pass
    return raizes


def analisar_crescimento(expr: sp.Expr, t_ini: float, t_fim: float) -> dict:
    """Pontos críticos, inflexões e maior taxa de crescimento em ``[t_ini, t_fim]``.

    * ``V'(t) = 0``  → vendas no pico (V'' < 0) ou no vale (V'' > 0);
    * ``V''(t) = 0`` com troca de sinal → ponto de inflexão: o crescimento deixa de acelerar
      (ou de frear) — é um mês de **maior (ou menor) taxa de crescimento local**.
    """
    t_ini, t_fim = exigir_intervalo(t_ini, t_fim)
    d1 = derivar(expr).derivada
    d2 = derivar(d1).derivada
    f, f1, f2 = para_numerica(expr), para_numerica(d1), para_numerica(d2)

    criticos = []
    for r in _raizes(f1, t_ini, t_fim):
        segunda = float(f2(np.array([r]))[0])
        tipo = "máximo local (pico de vendas)" if segunda < 0 else "mínimo local (vale)" if segunda > 0 else "indeterminado"
        criticos.append({"t": r, "V(t)": float(f(np.array([r]))[0]), "tipo": tipo})

    inflexoes = []
    delta = (t_fim - t_ini) * 1e-4
    for r in _raizes(f2, t_ini, t_fim):
        antes = float(f2(np.array([r - delta]))[0])
        depois = float(f2(np.array([r + delta]))[0])
        if antes > 0 > depois:
            tipo = "acelerava e passa a frear (taxa de crescimento máxima)"
        elif antes < 0 < depois:
            tipo = "freava e passa a acelerar (taxa de crescimento mínima)"
        else:
            tipo = "sem mudança de concavidade"
        inflexoes.append({"t": r, "V(t)": float(f(np.array([r]))[0]), "V'(t)": float(f1(np.array([r]))[0]), "tipo": tipo})

    x = np.linspace(t_ini, t_fim, 2001)
    taxas = f1(x)
    if np.all(np.isnan(taxas)):
        raise ValidacaoErro("A taxa de crescimento não está definida no intervalo escolhido.")
    i_max = int(np.nanargmax(taxas))
    i_min = int(np.nanargmin(taxas))
    return {
        "derivada_1": d1,
        "derivada_2": d2,
        "criticos": criticos,
        "inflexoes": inflexoes,
        "maior_taxa": {"t": float(x[i_max]), "V'(t)": float(taxas[i_max])},
        "menor_taxa": {"t": float(x[i_min]), "V'(t)": float(taxas[i_min])},
    }


def formatar_reais(x: float) -> str:
    """1234.5 → '1.234,50' (padrão brasileiro)."""
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def interpretar_taxa(expr: sp.Expr, ponto: float) -> str:
    """Frase em português explicando V'(ponto) e V''(ponto)."""
    res = derivar(expr)
    f1 = para_numerica(res.derivada)
    f2 = para_numerica(derivar(res.derivada).derivada)
    v1 = float(f1(np.array([ponto]))[0])
    v2 = float(f2(np.array([ponto]))[0])
    if not np.isfinite(v1):
        return "A taxa de crescimento não está definida nesse mês."
    if abs(v1) < 1e-9:
        tendencia = "estão estáveis (taxa de crescimento ≈ 0)"
    elif v1 > 0:
        tendencia = f"estão crescendo cerca de R$ {formatar_reais(v1)} por mês"
    else:
        tendencia = f"estão caindo cerca de R$ {formatar_reais(-v1)} por mês"
    if not np.isfinite(v2) or abs(v2) < 1e-9:
        ritmo = "em ritmo constante"
    elif v2 > 0:
        ritmo = "e esse crescimento está acelerando (V'' > 0)"
    else:
        ritmo = "e esse crescimento está desacelerando (V'' < 0)"
    return f"No mês t = {ponto:g}, as vendas {tendencia}, {ritmo}."
