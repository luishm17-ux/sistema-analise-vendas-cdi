"""Aula 2 — Limites.

Usos no sistema de vendas:

* ``lim t→∞ V(t)``  → **saturação do mercado** (para onde as vendas tendem no longo prazo);
* ``lim t→∞ V'(t)/V(t)`` → taxa de crescimento relativo no longo prazo;
* ``lim h→0 [V(t0+h) − V(t0)]/h`` → a **definição de derivada**: liga limites às derivadas;
* limites laterais e continuidade → detectar quebras (ex.: promoção que muda o patamar de vendas).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import sympy as sp
from sympy.calculus.accumulationbounds import AccumBounds

from .derivadas import derivar, formatar_reais
from .funcoes import para_numerica, simplificar_seguro
from .validacao import ValidacaoErro, exigir_numero, interpretar_ponto, t

h = sp.Symbol("h", real=True)


@dataclass
class ResultadoLimite:
    expr: sp.Expr
    ponto: sp.Expr
    esquerda: sp.Expr | None  # None quando o ponto é ±∞ (só existe um lado)
    direita: sp.Expr | None
    valor: sp.Expr | None  # None quando o limite não existe
    existe: bool
    classificacao: str  # "finito" | "infinito" | "nao_existe"
    motivo: str
    continuidade: str = ""

    @property
    def valor_texto(self) -> str:
        if not self.existe or self.valor is None:
            return "não existe"
        return str(self.valor).replace("oo", "∞")


def _amigavel(v: sp.Expr) -> sp.Expr:
    """``sympy.limit`` transforma decimais em frações enormes (75660.57… → 7566…/10¹⁰).
    Aqui voltamos a decimais curtos; frações "de verdade" (1/3, 3/50) ficam como estão."""
    if not isinstance(v, sp.Basic):
        return v
    troca = {r: sp.Float(r, 12) for r in v.atoms(sp.Rational) if not r.is_Integer and r.q > 10**4}
    return v.xreplace(troca) if troca else v


def _limite_sympy(expr: sp.Expr, ponto: sp.Expr, direcao: str) -> sp.Expr:
    try:
        return _amigavel(sp.limit(expr, t, ponto, direcao))
    except (NotImplementedError, ValueError, TypeError, RecursionError) as erro:
        raise ValidacaoErro(f"Não consegui calcular esse limite simbolicamente ({type(erro).__name__}).") from None


def _eh_indefinido(v: sp.Expr) -> bool:
    return isinstance(v, AccumBounds) or v.has(sp.nan) or v.has(sp.zoo)


def _iguais(a: sp.Expr, b: sp.Expr) -> bool:
    if a == b:
        return True
    try:
        return bool(sp.simplify(a - b) == 0)
    except Exception:
        return False


def calcular_limite(expr: sp.Expr, ponto, direcao: str = "ambos") -> ResultadoLimite:
    """Calcula ``lim t→ponto expr``.

    ``direcao`` pode ser ``'ambos'`` (padrão), ``'+'`` (pela direita) ou ``'-'`` (pela esquerda).
    Em um ponto finito calculamos *sempre* os dois limites laterais; o limite bilateral só
    existe se eles forem iguais.
    """
    expr = sp.sympify(expr)
    if direcao not in {"ambos", "+", "-"}:
        raise ValidacaoErro("A direção do limite deve ser 'ambos', '+' ou '-'.")
    p = ponto if isinstance(ponto, sp.Basic) else interpretar_ponto(ponto)

    # ---- limites no infinito: só existe um lado ----
    if p in (sp.oo, -sp.oo):
        v = _limite_sympy(expr, p, "-" if p == sp.oo else "+")
        sinal = "+∞" if p == sp.oo else "-∞"
        if _eh_indefinido(v):
            return ResultadoLimite(
                expr, p, None, None, None, False, "nao_existe",
                f"Quando t → {sinal} a função oscila e não se aproxima de nenhum valor.",
            )
        if v in (sp.oo, -sp.oo):
            return ResultadoLimite(
                expr, p, None, None, v, True, "infinito",
                f"Quando t → {sinal} a função cresce sem limite ({'+' if v == sp.oo else '-'}∞).",
            )
        return ResultadoLimite(
            expr, p, None, None, v, True, "finito",
            f"Quando t → {sinal} a função se aproxima de {v} (assíntota horizontal y = {v}).",
        )

    # ---- ponto finito ----
    esq = _limite_sympy(expr, p, "-")
    dir_ = _limite_sympy(expr, p, "+")
    cont = _continuidade(expr, p, esq, dir_)

    if direcao == "-":
        return _unilateral(expr, p, esq, dir_, esq, "esquerda", cont)
    if direcao == "+":
        return _unilateral(expr, p, esq, dir_, dir_, "direita", cont)

    if _eh_indefinido(esq) or _eh_indefinido(dir_):
        return ResultadoLimite(expr, p, esq, dir_, None, False, "nao_existe",
                               "Um dos limites laterais não existe (a função oscila).", cont)
    if _iguais(esq, dir_):
        if esq in (sp.oo, -sp.oo):
            return ResultadoLimite(expr, p, esq, dir_, esq, True, "infinito",
                                   f"Os dois lados tendem a {str(esq).replace('oo', '∞')}: assíntota vertical em t = {p}.", cont)
        return ResultadoLimite(expr, p, esq, dir_, esq, True, "finito",
                               f"Os limites laterais coincidem ({esq}); logo o limite existe.", cont)
    return ResultadoLimite(
        expr, p, esq, dir_, None, False, "nao_existe",
        f"Os limites laterais são diferentes (esquerda = {str(esq).replace('oo', '∞')}, "
        f"direita = {str(dir_).replace('oo', '∞')}): o limite NÃO existe em t = {p}.", cont,
    )


def _unilateral(expr, p, esq, dir_, valor, lado, cont) -> ResultadoLimite:
    if _eh_indefinido(valor):
        return ResultadoLimite(expr, p, esq, dir_, None, False, "nao_existe",
                               f"O limite pela {lado} não existe.", cont)
    if valor in (sp.oo, -sp.oo):
        return ResultadoLimite(expr, p, esq, dir_, valor, True, "infinito",
                               f"Pela {lado}, a função tende a {str(valor).replace('oo', '∞')}.", cont)
    return ResultadoLimite(expr, p, esq, dir_, valor, True, "finito",
                           f"Pela {lado}, a função se aproxima de {valor}.", cont)


def _continuidade(expr: sp.Expr, p: sp.Expr, esq: sp.Expr, dir_: sp.Expr) -> str:
    """Classifica a continuidade de ``expr`` em ``t = p`` com base nas três condições clássicas."""
    try:
        f_p = expr.subs(t, p)
    except Exception:
        f_p = sp.nan
    definida = not (f_p.has(sp.nan) or f_p.has(sp.zoo) or f_p.has(sp.oo) or f_p.has(-sp.oo))
    lim_existe = (not _eh_indefinido(esq)) and (not _eh_indefinido(dir_)) and _iguais(esq, dir_) and esq not in (sp.oo, -sp.oo)
    if definida and lim_existe and _iguais(f_p, esq):
        return f"Contínua em t = {p}: f({p}) = {f_p} = lim."
    if not definida and lim_existe:
        return f"Descontinuidade removível em t = {p}: f não está definida aí, mas o limite existe ({esq})."
    if definida and lim_existe:
        return f"Descontinuidade removível em t = {p}: f({p}) = {f_p} difere do limite ({esq})."
    if not _eh_indefinido(esq) and not _eh_indefinido(dir_):
        if esq in (sp.oo, -sp.oo) or dir_ in (sp.oo, -sp.oo):
            return f"Descontinuidade infinita (assíntota vertical) em t = {p}."
        return f"Descontinuidade de salto em t = {p}: os limites laterais diferem."
    return f"Descontinuidade essencial (oscilante) em t = {p}."


# --------------------------------------------------------------------------- #
# Tabela de aproximação (visualização numérica do limite)
# --------------------------------------------------------------------------- #
def tabela_aproximacao(expr: sp.Expr, ponto, passos: int = 7) -> pd.DataFrame:
    """Valores de f perto do ponto — a ideia intuitiva de limite."""
    p = ponto if isinstance(ponto, sp.Basic) else interpretar_ponto(ponto)
    f = para_numerica(expr)
    if p in (sp.oo, -sp.oo):
        sinal = 1 if p == sp.oo else -1
        xs = np.array([sinal * 10.0**k for k in range(1, passos + 1)])
        return pd.DataFrame({"t": xs, "f(t)": f(xs)})
    x0 = float(p)
    hs = np.array([10.0 ** (-k) for k in range(0, passos)])
    return pd.DataFrame({
        "t (pela esquerda)": x0 - hs, "f(t) esq.": f(x0 - hs),
        "t (pela direita)": x0 + hs, "f(t) dir.": f(x0 + hs),
    })


# --------------------------------------------------------------------------- #
# Limites com significado no sistema de vendas
# --------------------------------------------------------------------------- #
def limite_saturacao(expr: sp.Expr) -> ResultadoLimite:
    """lim t→∞ V(t): para onde as vendas mensais tendem no longo prazo."""
    return calcular_limite(expr, sp.oo)


def limite_taxa_relativa(expr: sp.Expr) -> ResultadoLimite:
    """lim t→∞ V'(t)/V(t): crescimento percentual mensal no longo prazo."""
    razao = simplificar_seguro(derivar(expr).derivada / sp.sympify(expr))
    return calcular_limite(razao, sp.oo)


def limite_definicao_derivada(expr: sp.Expr, t0: float) -> dict:
    """Calcula V'(t0) como limite do quociente de Newton e compara com a regra de derivação."""
    t0 = exigir_numero(t0, "t0")
    expr = sp.sympify(expr)
    t0_exato = sp.nsimplify(t0, rational=True)
    quociente = (expr.subs(t, t0_exato + h) - expr.subs(t, t0_exato)) / h
    try:
        pelo_limite = sp.limit(quociente, h, 0, "+")
    except (NotImplementedError, ValueError, TypeError) as erro:
        raise ValidacaoErro(f"Não consegui calcular o limite da definição ({type(erro).__name__}).") from None
    pela_regra = derivar(expr).derivada.subs(t, t0_exato)
    a, b = float(sp.N(pelo_limite)), float(sp.N(pela_regra))
    return {
        "quociente": quociente,
        "pelo_limite": pelo_limite,
        "pela_regra": pela_regra,
        "valor_limite": a,
        "valor_regra": b,
        "coincidem": bool(abs(a - b) <= 1e-8 * max(1.0, abs(b))),
    }


def propriedades_dos_limites(f: sp.Expr, g: sp.Expr, ponto) -> pd.DataFrame:
    """Confere as propriedades operatórias dos limites para duas funções f e g.

    Só vale usar uma propriedade quando os limites de f e g existem e são finitos
    (e, no quociente, quando lim g ≠ 0). A tabela mostra isso explicitamente.
    """
    lf, lg = calcular_limite(f, ponto), calcular_limite(g, ponto)
    if not (lf.existe and lg.existe and lf.classificacao == "finito" and lg.classificacao == "finito"):
        raise ValidacaoErro(
            "As propriedades exigem que lim f e lim g existam e sejam finitos "
            f"(lim f = {lf.valor_texto}, lim g = {lg.valor_texto})."
        )
    a, b = lf.valor, lg.valor
    linhas = [
        ("Soma", f + g, a + b),
        ("Diferença", f - g, a - b),
        ("Produto", f * g, a * b),
        ("Constante × função (3·f)", 3 * f, 3 * a),
    ]
    if b != 0:
        linhas.append(("Quociente", f / g, a / b))
    saida = []
    for nome, combinada, esperado in linhas:
        direto = calcular_limite(combinada, ponto)
        ok = direto.existe and _iguais(direto.valor, esperado)
        saida.append({
            "Propriedade": nome,
            "Limite da combinação": direto.valor_texto,
            "Combinação dos limites": str(sp.simplify(esperado)),
            "Confere": "✔" if ok else "✘",
        })
    return pd.DataFrame(saida)


def interpretar_saturacao(res: ResultadoLimite) -> str:
    """Texto de negócio para ``lim t→∞ V(t)``."""
    if res.classificacao == "finito":
        v = float(sp.N(res.valor))
        return (f"O mercado **satura**: no longo prazo as vendas mensais se aproximam de "
                f"R$ {formatar_reais(v)} e não passam disso.")
    if res.classificacao == "infinito":
        return "Não há saturação no modelo: as vendas crescem sem limite (útil só para horizontes curtos)."
    return "As vendas não convergem para um valor: oscilam indefinidamente (sazonalidade sem tendência de equilíbrio)."
