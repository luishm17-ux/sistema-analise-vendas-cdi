"""Leitura de CSV, métricas mensais e ajuste de modelos aos dados."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vendas_cdi import dados, funcoes
from vendas_cdi.validacao import ValidacaoErro

EXEMPLO = Path(__file__).resolve().parent.parent / "dados" / "vendas_exemplo.csv"


def csv(texto: str) -> bytes:
    return texto.encode("utf-8")


class TestLeitura:
    def test_arquivo_de_exemplo(self):
        df = dados.carregar_csv(EXEMPLO)
        assert len(df) == 24 and list(df.columns) == ["t", "mes", "vendas"]
        assert df["t"].tolist() == list(range(1, 25))

    def test_so_a_coluna_vendas(self):
        df = dados.carregar_csv(csv("vendas\n1\n2\n3\n4\n5\n6\n"))
        assert df["vendas"].tolist() == [1, 2, 3, 4, 5, 6] and df["mes"].iloc[0] == "Mês 1"

    def test_separador_ponto_e_virgula_e_formato_brasileiro(self):
        df = dados.carregar_csv(csv("mes;vendas\na;1.000,50\nb;R$ 1.100,00\nc;1.200\nd;1300\ne;1400,5\nf;1500\n"))
        assert df["vendas"].tolist() == [1000.5, 1100.0, 1200.0, 1300.0, 1400.5, 1500.0]

    def test_formato_internacional_e_bom_e_latin1(self):
        assert dados.carregar_csv(csv("vendas\n10.5\n20.25\n30\n40\n50\n60\n"))["vendas"].iloc[0] == 10.5
        assert len(dados.carregar_csv("﻿mes,vendas\n1,1\n2,2\n3,3\n4,4\n5,5\n6,6\n".encode("utf-8"))) == 6
        assert len(dados.carregar_csv("mês,vendas\njan,1\nfev,2\nmar,3\nabr,4\nmai,5\njun,6\n".encode("latin-1"))) == 6

    def test_aceita_objeto_de_arquivo_e_caminho_str(self):
        assert len(dados.carregar_csv(io.BytesIO(EXEMPLO.read_bytes()))) == 24
        assert len(dados.carregar_csv(str(EXEMPLO))) == 24

    def test_ponto_com_tres_digitos_e_separador_de_milhar(self):
        """Convenção documentada: '1.200' (ponto + exatamente 3 dígitos) é 1200; '1.25' e '0.500' são decimais."""
        df = dados.carregar_csv(csv("vendas\n1.200\n1.25\n0.500\n12.345.678\n2.5\n1500\n"))
        assert df["vendas"].tolist() == [1200, 1.25, 0.5, 12345678, 2.5, 1500]

    def test_digitacao_manual(self):
        assert dados.dataframe_manual("100, 120, 130, 160, 170, 190")["vendas"].tolist() == [100, 120, 130, 160, 170, 190]
        assert len(dados.dataframe_manual("1\n2\n3\n4\n5\n6")) == 6
        assert len(dados.dataframe_manual("1;2;3;4;5;6")) == 6

    @pytest.mark.parametrize("conteudo, trecho", [
        (b"", "vazio"),
        (b"   \n  ", "vazio"),
        (csv("mes,valor\n1,2\n"), "vendas"),
        (csv("vendas\n1\n2\nabc\n4\n5\n6\n"), "não numéricos"),
        (csv("vendas\n1\n2\n\n4\n5\n6\n7\n"), None),                       # linha em branco é ignorada, não é erro
        (csv("mes,vendas\na,1\nb,\nc,3\nd,4\ne,5\nf,6\n"), "vazios"),
        (csv("vendas\n1\n2\n-3\n4\n5\n6\n"), "negativas"),
        (csv("vendas\n1\n2\n3\n"), "pelo menos 6"),
        (csv("vendas\n0\n0\n0\n0\n0\n0\n"), "zero"),
        (csv("vendas\n1\n2\n3\n4\n5\nnan\n"), "não numéricos"),
        (csv("vendas\n1\n2\n3\n4\n5\ninf\n"), "não numéricos"),
        (b"x" * 3_000_000, "grande demais"),
    ])
    def test_entradas_invalidas(self, conteudo, trecho):
        if trecho is None:
            assert len(dados.carregar_csv(conteudo)) >= 6
            return
        with pytest.raises(ValidacaoErro, match=trecho):
            dados.carregar_csv(conteudo)

    def test_arquivo_inexistente(self):
        with pytest.raises(ValidacaoErro):
            dados.carregar_csv("/nao/existe.csv")

    def test_linhas_no_erro_sao_do_arquivo(self):
        with pytest.raises(ValidacaoErro, match="4"):   # cabeçalho = linha 1; o 'abc' está na linha 4
            dados.carregar_csv(csv("vendas\n1\n2\nabc\n4\n5\n6\n"))


class TestMetricas:
    def test_variacoes_e_acumulado(self):
        df = dados.validar_vendas(pd.DataFrame({"vendas": ["100", "110", "99", "198", "200", "220"]}))
        m = dados.calcular_metricas(df)
        assert m["variacao_abs"].tolist()[1:] == [10, -11, 99, 2, 20]
        assert m["variacao_pct"].iloc[1] == pytest.approx(10.0)
        assert m["variacao_pct"].iloc[2] == pytest.approx(-10.0)
        assert m["variacao_pct"].iloc[3] == pytest.approx(100.0)
        assert m["acumulado"].iloc[-1] == 927
        assert m["media_movel_3"].iloc[2] == pytest.approx(103.0)

    def test_resumo(self):
        df = dados.validar_vendas(pd.DataFrame({"mes": list("abcdef"), "vendas": ["100", "110", "121", "133.1", "146.41", "161.05"]}))
        r = dados.resumo(df)
        assert r["taxa_media_mensal_pct"] == pytest.approx(10.0, abs=1e-3)       # crescimento composto de 10 %
        assert r["crescimento_total_pct"] == pytest.approx(61.05)
        assert r["melhor_mes"][0] == "f" and r["pior_mes"][0] == "a" and r["meses"] == 6

    def test_variacao_percentual_com_mes_anterior_zero_e_nan(self):
        m = dados.calcular_metricas(dados.validar_vendas(pd.DataFrame({"vendas": ["0", "5", "6", "7", "8", "9"]})))
        assert np.isnan(m["variacao_pct"].iloc[1])


class TestAjuste:
    T = np.arange(1, 25, dtype=float)

    @staticmethod
    def sintetico(chave, params, n=24):
        t_ = np.arange(1, n + 1, dtype=float)
        expr = funcoes.construir(chave, params)
        return dados.validar_vendas(pd.DataFrame({"vendas": funcoes.para_numerica(expr)(t_)}))

    @pytest.mark.parametrize("chave, params", [
        ("linear", {"a": 1500, "b": 20000}),
        ("quadratica", {"a": -25, "b": 1500, "c": 20000}),
        ("exponencial", {"a": 15000, "k": 0.06}),
        ("logaritmica", {"a": 12000, "b": 10000}),
        ("logistica", {"L": 80000, "k": 0.3, "t0": 12}),
        ("sazonal", {"c": 20000, "a": 800, "A": 5000, "P": 12, "fase": 0.4}),
    ])
    def test_recupera_parametros_de_dados_sem_ruido(self, chave, params):
        ajustes = {a.chave: a for a in dados.ajustar_modelos(self.sintetico(chave, params))}
        a = ajustes[chave]
        assert a.r2 > 0.9999
        for nome, valor in params.items():
            assert a.parametros[nome] == pytest.approx(valor, rel=1e-3, abs=1e-3), nome

    def test_escolhe_logistica_para_curva_em_s(self):
        melhor = dados.ajustar_modelos(self.sintetico("logistica", {"L": 80000, "k": 0.3, "t0": 12}))[0]
        assert melhor.chave == "logistica"

    def test_exemplo_tem_logistica_como_melhor(self):
        ajustes = dados.ajustar_modelos(dados.carregar_csv(EXEMPLO))
        assert ajustes[0].chave == "logistica" and ajustes[0].r2 > 0.98
        assert [a.r2_ajustado for a in ajustes] == sorted([a.r2_ajustado for a in ajustes], reverse=True)

    def test_exponencial_e_pulada_quando_ha_vendas_zero(self):
        df = dados.validar_vendas(pd.DataFrame({"vendas": ["0", "10", "20", "30", "45", "50", "60"]}))
        assert "exponencial" not in {a.chave for a in dados.ajustar_modelos(df)}

    def test_sazonal_exige_um_ciclo_completo(self):
        df = dados.validar_vendas(pd.DataFrame({"vendas": [str(100 + 7 * i) for i in range(8)]}))
        assert "sazonal" not in {a.chave for a in dados.ajustar_modelos(df)}

    def test_dados_constantes_nao_quebram(self):
        df = dados.validar_vendas(pd.DataFrame({"vendas": ["500"] * 12}))
        ajustes = dados.ajustar_modelos(df)
        assert ajustes and ajustes[0].rmse == pytest.approx(0, abs=1e-6)
