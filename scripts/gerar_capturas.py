"""Tira capturas de tela de cada aba do aplicativo (para o README).

Requer: ``pip install playwright`` e um Chrome/Chromium instalado, com o app rodando em
``http://localhost:8501`` (``streamlit run app.py``).

    python scripts/gerar_capturas.py [URL] [CAMINHO_DO_CHROME]
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
CHROME = sys.argv[2] if len(sys.argv) > 2 else None
SAIDA = Path(__file__).resolve().parent.parent / "docs" / "capturas"
SAIDA.mkdir(parents=True, exist_ok=True)

# (arquivo, nome da aba principal, sub-aba opcional)
CAPTURAS = [
    ("01_analise_vendas.png", "📊 Análise de vendas", None),
    ("02_taxa_crescimento.png", "📊 Análise de vendas", "⚡ Taxa de crescimento (Derivada)"),
    ("03_receita_acumulada.png", "📊 Análise de vendas", "💰 Receita acumulada (Integral)"),
    ("04_funcoes.png", "📈 Funções", None),
    ("05_limites.png", "∞ Limites", None),
    ("06_derivadas.png", "∂ Derivadas", None),
    ("07_integrais.png", "∫ Integrais", None),
    ("08_exemplos.png", "✅ Exemplos", None),
]


def main() -> None:
    with sync_playwright() as p:
        opcoes = {"headless": True, "args": ["--no-sandbox"]}
        if CHROME:
            opcoes["executable_path"] = CHROME
        navegador = p.chromium.launch(**opcoes)
        pagina = navegador.new_page(viewport={"width": 1440, "height": 2000})
        pagina.goto(URL)
        pagina.wait_for_selector("text=Sistema de Análise de Vendas", timeout=60000)
        pagina.wait_for_timeout(4000)
        for arquivo, aba, sub in CAPTURAS:
            pagina.get_by_role("tab", name=aba).first.click()
            pagina.wait_for_timeout(2500)
            if sub:
                pagina.get_by_role("tab", name=sub).first.click()
                pagina.wait_for_timeout(2000)
            pagina.screenshot(path=str(SAIDA / arquivo), full_page=True)
            print("ok", arquivo)
        navegador.close()


if __name__ == "__main__":
    main()
