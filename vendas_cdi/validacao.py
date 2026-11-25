"""Validação de entradas do usuário.

Toda entrada que vem da interface (números, intervalos, expressões de texto,
arquivos CSV) passa por aqui antes de chegar aos módulos de cálculo. Quando algo
é inválido, levantamos ``ValidacaoErro`` com uma mensagem em português que a
interface mostra diretamente ao usuário.
"""

from __future__ import annotations

import math
import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# Variável independente de todo o sistema: o tempo, em meses.
t = sp.Symbol("t", real=True)

TAMANHO_MAXIMO_EXPRESSAO = 200


class ValidacaoErro(ValueError):
    """Entrada inválida, com mensagem pensada para ser exibida ao usuário."""


# --------------------------------------------------------------------------- #
# Números e intervalos
# --------------------------------------------------------------------------- #
def exigir_numero(
    valor,
    nome: str,
    *,
    minimo: float | None = None,
    maximo: float | None = None,
    estrito_positivo: bool = False,
    nao_zero: bool = False,
) -> float:
    """Converte ``valor`` em ``float`` finito e confere as restrições pedidas."""
    if isinstance(valor, bool):
        raise ValidacaoErro(f"'{nome}' deve ser um número (recebido: {valor!r}).")
    if isinstance(valor, str):
        valor = valor.strip().replace(",", ".")
    try:
        x = float(valor)
    except (TypeError, ValueError):
        raise ValidacaoErro(f"'{nome}' deve ser um número (recebido: {valor!r}).") from None
    if not math.isfinite(x):
        raise ValidacaoErro(f"'{nome}' deve ser um número finito (recebido: {valor!r}).")
    if estrito_positivo and x <= 0:
        raise ValidacaoErro(f"'{nome}' deve ser maior que zero (recebido: {x:g}).")
    if nao_zero and x == 0:
        raise ValidacaoErro(f"'{nome}' não pode ser zero.")
    if minimo is not None and x < minimo:
        raise ValidacaoErro(f"'{nome}' deve ser maior ou igual a {minimo:g} (recebido: {x:g}).")
    if maximo is not None and x > maximo:
        raise ValidacaoErro(f"'{nome}' deve ser menor ou igual a {maximo:g} (recebido: {x:g}).")
    return x


def exigir_intervalo(a, b, nome: str = "intervalo") -> tuple[float, float]:
    """Garante que ``[a, b]`` é um intervalo finito com ``a < b``."""
    a = exigir_numero(a, f"início do {nome}")
    b = exigir_numero(b, f"fim do {nome}")
    if a >= b:
        raise ValidacaoErro(
            f"O início do {nome} ({a:g}) deve ser menor que o fim ({b:g})."
        )
    return a, b


# --------------------------------------------------------------------------- #
# Expressões digitadas pelo usuário
# --------------------------------------------------------------------------- #
_FUNCOES_PERMITIDAS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "ln": sp.log,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
}
_CONSTANTES_PERMITIDAS = {"pi": sp.pi, "e": sp.E}
_NOMES_PERMITIDOS = set(_FUNCOES_PERMITIDAS) | set(_CONSTANTES_PERMITIDAS) | {"t"}

# Construtores que as próprias transformações do SymPy inserem no código gerado.
# Nada além disso fica disponível ao interpretador (não há builtins do Python em uso útil,
# pois todos os identificadores do texto já foram conferidos contra a lista de permitidos).
_GLOBAIS_SEGUROS = {
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Symbol": sp.Symbol,
    "Function": sp.Function,
}

_CARACTERES_VALIDOS = re.compile(r"^[0-9A-Za-z_+\-*/^().,\s]*$")
_IDENTIFICADORES = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMERO_GRANDE = re.compile(r"\d{10,}")
_EXPOENTE_LITERAL = re.compile(r"(?:\^|\*\*)\s*\(?\s*[-+]?\s*(\d+)")
# Bloqueia torres de potências como 9^9^9 ou 9^(9^9), que travariam o cálculo.
_TORRE_DE_POTENCIAS = re.compile(r"(\^|\*\*)\s*\(*\s*[\w.]+\s*\)*\s*(\^|\*\*)")
EXPOENTE_MAXIMO = 1000


def interpretar_expressao(texto: str) -> sp.Expr:
    """Converte um texto como ``"3*t^2 + sin(2*t)"`` em expressão do SymPy.

    Apenas a variável ``t`` e algumas funções (sin, cos, tan, exp, ln, log, sqrt,
    abs) são aceitas. Qualquer outro identificador é rejeitado *antes* de o texto
    chegar ao interpretador, o que impede a execução de código arbitrário.
    """
    if texto is None or not str(texto).strip():
        raise ValidacaoErro("Digite uma função (por exemplo: 3*t^2 + 5*t + 100).")
    texto = str(texto).strip()
    if len(texto) > TAMANHO_MAXIMO_EXPRESSAO:
        raise ValidacaoErro(f"Expressão muito longa (máximo de {TAMANHO_MAXIMO_EXPRESSAO} caracteres).")
    if not _CARACTERES_VALIDOS.match(texto):
        raise ValidacaoErro(
            "A expressão contém caracteres não permitidos. "
            "Use apenas números, t, + - * / ^ ( ) e funções como sin, cos, exp, ln."
        )
    texto = texto.replace(",", ".")  # aceita vírgula decimal
    for nome in _IDENTIFICADORES.findall(texto):
        # "2t" e "3.5e" não são identificadores; só nomes que começam com letra chegam aqui.
        if nome not in _NOMES_PERMITIDOS:
            raise ValidacaoErro(
                f"'{nome}' não é reconhecido. Use a variável t e as funções "
                "sin, cos, tan, exp, ln, log, sqrt, abs (constantes: pi, e)."
            )
    if _NUMERO_GRANDE.search(texto):
        raise ValidacaoErro("Número grande demais na expressão (máximo de 9 dígitos por número).")
    for expoente in _EXPOENTE_LITERAL.findall(texto):
        if int(expoente) > EXPOENTE_MAXIMO:
            raise ValidacaoErro(f"Expoente grande demais (máximo permitido: {EXPOENTE_MAXIMO}).")
    if _TORRE_DE_POTENCIAS.search(texto):
        raise ValidacaoErro("Potências encadeadas (a^b^c) não são aceitas; use parênteses com valores pequenos.")

    local = {"t": t, **_FUNCOES_PERMITIDAS, **_CONSTANTES_PERMITIDAS}
    transformacoes = standard_transformations + (implicit_multiplication_application, convert_xor)
    try:
        expr = parse_expr(texto, local_dict=local, global_dict=dict(_GLOBAIS_SEGUROS), transformations=transformacoes)
    except Exception as erro:  # sintaxe inválida, parênteses desbalanceados, etc.
        raise ValidacaoErro(f"Não consegui interpretar a expressão ({type(erro).__name__}). Confira a sintaxe.") from None

    if not isinstance(expr, sp.Expr):
        raise ValidacaoErro("A expressão deve resultar em uma função de t.")
    if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ValidacaoErro("A expressão resulta em valor indefinido (divisão por zero?).")
    return expr


def interpretar_ponto(texto) -> sp.Expr:
    """Interpreta o ponto de um limite: um número, ``oo``/``inf`` ou ``-oo``."""
    if isinstance(texto, bool):
        raise ValidacaoErro(f"Ponto inválido: {texto!r}.")
    if isinstance(texto, (int, float)):
        return sp.nsimplify(exigir_numero(texto, "ponto do limite"), rational=True)
    s = str(texto).strip().lower().replace(",", ".")  # espaços internos NÃO são removidos: "2 3" é inválido
    if not s:
        raise ValidacaoErro("Informe o ponto do limite (um número, 'oo' ou '-oo').")
    if s in {"oo", "inf", "+oo", "+inf", "infinito", "+infinito", "∞", "+∞"}:
        return sp.oo
    if s in {"-oo", "-inf", "-infinito", "-∞"}:
        return -sp.oo
    if re.fullmatch(r"[-+]?\d{1,9}(\.\d{1,9})?(/\d{1,9})?", s):
        try:
            return sp.Rational(s)
        except (ValueError, ZeroDivisionError):
            pass
    raise ValidacaoErro(f"Ponto inválido: {texto!r}. Use um número (ex.: 2, 0.5, 1/3), 'oo' ou '-oo'.")
