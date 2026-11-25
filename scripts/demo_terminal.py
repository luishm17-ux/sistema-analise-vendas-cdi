"""Demonstração dos quatro módulos no terminal (sem a interface gráfica).

    python scripts/demo_terminal.py

Cada seção mostra: entrada → cálculo → resultado verificável → interpretação.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sympy as sp  # noqa: E402

from vendas_cdi import analise, dados, derivadas, funcoes, integrais, limites  # noqa: E402
from vendas_cdi.validacao import interpretar_expressao as P, t  # noqa: E402


def titulo(texto: str) -> None:
    print(f"\n{'=' * 78}\n{texto}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
titulo("AULA 1 — FUNÇÕES (6 tipos, V(12) com os parâmetros padrão)")
for chave, tipo in funcoes.TIPOS.items():
    p = tipo.valores_padrao()
    expr = tipo.construir(p)
    dom, im = tipo.dominio_imagem(p)
    v = funcoes.valor_em(expr, 12)
    print(f"{tipo.nome:26s} V(12) = R$ {v:>10,.2f}   Dom = {dom.split('  ')[0]:3s}  Im = {im}")

# --------------------------------------------------------------------------- #
titulo("AULA 2 — LIMITES")
for texto, ponto in [("(t^2-4)/(t-2)", "2"), ("sin(t)/t", "0"), ("(3*t+1)/(t+2)", "oo"), ("abs(t)/t", "0"), ("1/t^2", "0"), ("sin(t)", "oo")]:
    r = limites.calcular_limite(P(texto), ponto)
    print(f"lim t→{ponto:2s} de {texto:14s} = {r.valor_texto:10s} | {r.motivo}")
lo = funcoes.construir("logistica", {"L": 80000, "k": 0.3, "t0": 12})
sat = limites.limite_saturacao(lo)
print(f"\nSaturação do mercado (logística, L = 80000): lim t→∞ V(t) = {sat.valor_texto}")
print("→", limites.interpretar_saturacao(sat).replace("**", ""))
d = limites.limite_definicao_derivada(P("t^2"), 3)
print(f"Definição de derivada: lim h→0 [(3+h)²−3²]/h = {d['pelo_limite']}  (regra da potência: {d['pela_regra']})")

# --------------------------------------------------------------------------- #
titulo("AULA 3 — DERIVADAS (passo a passo)")
v = funcoes.construir("quadratica", {"a": -25, "b": 1500, "c": 20000})
r = derivadas.derivar(v)
print(f"V(t)  = {v}\nV'(t) = {r.derivada}")
print("Regras usadas:", ", ".join(r.regras_usadas))
print(f"Verificada com sympy.diff: {r.confere_sympy} | com diferença finita: {r.confere_numerico}")
print("→", derivadas.interpretar_taxa(v, 10))
pico = derivadas.analisar_crescimento(v, 0, 60)["criticos"][0]
print(f"→ V'(t) = 0 em t = {pico['t']:.1f}: {pico['tipo']}, V = R$ {pico['V(t)']:,.2f}")
print("\nCada regra (derivada calculada pelo sistema × verificação):")
for regra, (_, ex, _) in derivadas.CATALOGO_REGRAS.items():
    rr = derivadas.derivar(P(ex))
    print(f"  {regra:20s} d/dt[{ex:14s}] = {str(rr.derivada):28s} {'✔' if rr.verificada else '✘'}")

# --------------------------------------------------------------------------- #
titulo("AULA 4 — INTEGRAIS")
for regra, (_, ex, _) in integrais.CATALOGO_REGRAS.items():
    ri = integrais.integrar(P(ex))
    print(f"  {regra:26s} ∫ {ex:16s} dt = {str(ri.antiderivada):28s} + C   {'✔' if ri.verificada else '✘'}")
lin = funcoes.construir("linear", {"a": 1000, "b": 5000})
ri = integrais.integral_definida(lin, 0.5, 12.5)
soma = sum(1000 * i + 5000 for i in range(1, 13))
print(f"\nReceita acumulada em 12 meses, V = 1000t + 5000: ∫ de 0,5 a 12,5 = R$ {ri.valor_exato:,.2f}")
print(f"Conferência (soma mês a mês):               R$ {soma:,.2f}   | scipy.quad: R$ {ri.valor_numerico:,.2f}")

# --------------------------------------------------------------------------- #
titulo("SISTEMA COMPLETO — dados de exemplo (24 meses)")
df = dados.carregar_csv(Path(__file__).resolve().parent.parent / "dados" / "vendas_exemplo.csv")
melhor = dados.ajustar_modelos(df)[0]
print(f"Melhor modelo: {melhor.nome} (R² = {melhor.r2:.4f})")
rel = analise.gerar_relatorio(melhor.expr, len(df), 6, float(df["vendas"].sum()))
for linha in rel.textos:
    print("•", linha.replace("**", ""))
