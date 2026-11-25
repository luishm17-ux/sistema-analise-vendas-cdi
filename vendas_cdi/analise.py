"""Relatório de vendas: junta Funções, Limites, Derivadas e Integrais num único resultado.

``gerar_relatorio`` recebe o modelo V(t) (ajustado aos dados ou simulado) e responde, em
linguagem de negócio, as perguntas que a diretoria faria:

* Quanto as vendas estão crescendo agora?        → derivada V'(t0)
* O crescimento está acelerando ou freando?       → segunda derivada V''(t0)
* Quando o crescimento é máximo? Há pico?         → V'' = 0 e V' = 0
* O mercado tem limite?                           → limite quando t → ∞
* Quanto vamos faturar nos próximos meses?        → integral definida
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from . import derivadas, integrais, limites
from .derivadas import formatar_reais
from .funcoes import para_numerica
from .validacao import ValidacaoErro, exigir_numero


@dataclass
class Relatorio:
    expr: sp.Expr
    n_meses: int
    horizonte: int
    v1: sp.Expr
    v2: sp.Expr
    crescimento: dict
    saturacao: limites.ResultadoLimite
    receita_historico: integrais.ResultadoIntegralDefinida
    receita_projecao: integrais.ResultadoIntegralDefinida
    taxa_atual: float
    aceleracao_atual: float
    textos: list[str] = field(default_factory=list)


def gerar_relatorio(expr: sp.Expr, n_meses: int, horizonte: int = 6, soma_observada: float | None = None) -> Relatorio:
    """Analisa o modelo ``expr`` considerando ``n_meses`` meses de histórico e ``horizonte`` de projeção."""
    n_meses = int(exigir_numero(n_meses, "número de meses", minimo=2, maximo=600))
    horizonte = int(exigir_numero(horizonte, "horizonte de projeção", minimo=1, maximo=120))

    r1 = derivadas.derivar(expr)
    v1 = r1.derivada
    v2 = derivadas.derivar(v1).derivada
    crescimento = derivadas.analisar_crescimento(expr, 1, n_meses + horizonte)
    saturacao = limites.limite_saturacao(expr)

    # Cada mês i ocupa [i − 0,5; i + 0,5]
    hist = integrais.integral_definida(expr, 0.5, n_meses + 0.5)
    proj = integrais.integral_definida(expr, n_meses + 0.5, n_meses + horizonte + 0.5)

    f1 = para_numerica(v1)(np.array([float(n_meses)]))[0]
    f2 = para_numerica(v2)(np.array([float(n_meses)]))[0]

    rel = Relatorio(expr, n_meses, horizonte, v1, v2, crescimento, saturacao, hist, proj, float(f1), float(f2))
    rel.textos = _textos(rel, soma_observada)
    return rel


def _n1(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")


def _valor(res: integrais.ResultadoIntegralDefinida) -> float:
    return res.valor_exato if res.valor_exato is not None else res.valor_numerico


def _textos(r: Relatorio, soma_observada: float | None) -> list[str]:
    linhas: list[str] = []
    n = r.n_meses

    if np.isfinite(r.taxa_atual):
        sentido = "crescendo" if r.taxa_atual > 0 else "caindo" if r.taxa_atual < 0 else "estáveis"
        linhas.append(
            f"**Derivada (taxa de crescimento):** no mês {n}, V′({n}) = R$ {formatar_reais(r.taxa_atual)} por mês — "
            f"as vendas estão {sentido}."
        )
    if np.isfinite(r.aceleracao_atual):
        if abs(r.aceleracao_atual) < 1e-9:
            ritmo = "constante"
        elif r.aceleracao_atual > 0:
            ritmo = "**acelerando** (V″ > 0): cada mês cresce mais que o anterior"
        else:
            ritmo = "**desacelerando** (V″ < 0): o crescimento perde fôlego a cada mês"
        linhas.append(f"**Segunda derivada:** V″({n}) = {formatar_reais(r.aceleracao_atual)} — o crescimento está {ritmo}.")

    for c in [c for c in r.crescimento["criticos"] if "máximo" in c["tipo"]][:1]:
        linhas.append(
            f"**Pico:** V′(t) = 0 em t ≈ {_n1(c['t'])}: as vendas atingem R$ {formatar_reais(c['V(t)'])} "
            "e depois passam a cair."
        )
    for i in [i for i in r.crescimento["inflexoes"] if "acelerava" in i["tipo"]][:2]:
        taxa_na_inflexao = i["V'(t)"]
        linhas.append(
            f"**Inflexão:** V″(t) = 0 em t ≈ {_n1(i['t'])} — o crescimento {i['tipo'].split(' (')[0]} "
            f"(taxa de R$ {formatar_reais(taxa_na_inflexao)} por mês nesse ponto)."
        )

    linhas.append("**Limite:** " + limites.interpretar_saturacao(r.saturacao))

    hist, proj = _valor(r.receita_historico), _valor(r.receita_projecao)
    texto = f"**Integral (receita acumulada):** o modelo estima R$ {formatar_reais(hist)} faturados nos {n} meses de histórico"
    if soma_observada is not None and soma_observada > 0:
        dif = f"{(hist / soma_observada - 1) * 100:+.1f}".replace(".", ",")
        texto += f" (receita real somada: R$ {formatar_reais(soma_observada)}; diferença de {dif}%)"
    linhas.append(texto + ".")
    linhas.append(
        f"**Projeção:** nos próximos {r.horizonte} meses, ∫ V(t) dt ≈ R$ {formatar_reais(proj)} "
        f"(média de R$ {formatar_reais(proj / r.horizonte)} por mês)."
    )
    return linhas


def projetar(expr: sp.Expr, n_meses: int, horizonte: int) -> np.ndarray:
    """Vendas previstas pelo modelo para os meses n+1 … n+horizonte."""
    if horizonte < 1:
        raise ValidacaoErro("O horizonte deve ser de pelo menos 1 mês.")
    meses = np.arange(n_meses + 1, n_meses + horizonte + 1, dtype=float)
    return para_numerica(expr)(meses)
