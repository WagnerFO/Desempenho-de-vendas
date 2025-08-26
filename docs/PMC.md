# Project Model Canvas — Kaggle Chatbot MVP (BigMart Sales)

## Contexto

Grandes redes varejistas possuem diferentes **tipos de loja**, que influenciam diretamente no **desempenho de vendas**.
O **BigMart Sales Dataset (Kaggle)** contém informações sobre **produtos, visibilidade dos itens, características das lojas e vendas totais**.
O objetivo educacional é usar esse conjunto para treinar modelos simples e construir um **chatbot interativo** que auxilie na análise de vendas e estratégias de marketing.

---

## Problema a ser Respondido

Qual perfil de loja mais vende e por quais razões?
Um item tem **alta ou baixa visibilidade** dependendo do tipo de loja?

---

## Pergunta Norteadora

* Existe relação entre **tipo de loja** e **volume de vendas**?
* Como a **visibilidade dos itens** impacta as vendas em diferentes formatos de loja?
* Quais variáveis (tamanho da loja, localização, tipo de produto) mais influenciam no desempenho?

---

## Solução Proposta

Desenvolver um **chatbot educacional em Streamlit** que:

1. Permita upload do dataset do **BigMart Sales**.
2. Treine modelos de:

   * **Regressão linear** (predição de vendas).
   * **Regressão logística** (probabilidade de um item ter alta ou baixa visibilidade).
3. Mostre métricas de avaliação (RMSE, R², acurácia, f1-score).
4. Explique a importância das variáveis por meio de coeficientes, gráficos e tabelas.
5. Responda perguntas do usuário sobre o dataset via chatbot regrado.

---

## Desenho de Arquitetura

O sistema será estruturado em camadas:

* **Interface (app/):** Streamlit como front-end para upload, treino e perguntas.
* **Core (core/):** módulos para dados, features, modelos, explicabilidade e chatbot.
* **Dados (data/):** pastas para armazenar arquivos brutos, tratados e modelos treinados.
* **Documentação (docs/):** PMC, arquitetura, governança e testes.

---

## Resultados Esperados

* Modelo de regressão que explique **fatores determinantes das vendas** por tipo de loja.
* Classificação simples para **visibilidade dos itens** (alta/baixa) com boa acurácia.
* Relatório de métricas e importâncias de variáveis.
* Deploy em **Streamlit Cloud** com documentação completa no GitHub.

---

## Observação Didática

O **PMC** é o mapa inicial do projeto, conectando **contexto, problema e solução** a uma implementação prática.
Ele orienta as decisões de ciência de dados e permite alinhar objetivos estratégicos (estoque e marketing) com a parte técnica (modelos e chatbot).

---