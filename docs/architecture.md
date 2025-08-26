# Arquitetura de Software — Kaggle Chatbot MVP (BigMart Sales)

> Este documento descreve a arquitetura lógica, a organização em camadas e o fluxo de dados do MVP, além de instruções detalhadas para desenhar o diagrama no **draw\.io** (diagrams.net) e exportar para o repositório.

---

## 1. Objetivos de Arquitetura

* **Separação de responsabilidades**: UI (Streamlit) isolada da lógica de negócio (`core/`) e dos artefatos (`data/`).
* **Simplicidade (MVP)**: poucos componentes, baixo acoplamento e convenções claras.
* **Reprodutibilidade**: pipelines de ML encapsulados; execução determinística.
* **Evolução**: fácil trocar o dataset Kaggle, incluir novos modelos ou expandir análises.
* **Governança**: espaço próprio para configs (`configs/`) e documentação (`docs/`).

---

## 2. Visão em Camadas

```
Usuário (navegador)
   ↓
[ app/ ]  → Streamlit UI
   ↓
[ core/ ]
   ├─ data/       → leitura/validação de CSV BigMart
   ├─ features/   → pipelines de preprocessamento
   ├─ models/     → treino, avaliação, predição
   ├─ explain/    → coeficientes/importâncias
   └─ chatbot/    → regras de resposta sobre vendas e visibilidade
   ↓
[ data/ ]
   ├─ raw/        → dados brutos
   ├─ processed/  → dados tratados
   └─ models/     → artefatos .pkl
```

---

## 3. Componentes e Responsabilidades

### app/

* Camada de apresentação (UI em Streamlit).
* Upload do dataset BigMart.
* Escolha da tarefa (regressão ou classificação).
* Exibição de métricas, gráficos de vendas, importâncias e respostas do chatbot.

### core/data/

* Funções de I/O para CSVs BigMart.
* Validação de schema (colunas obrigatórias: tipo de loja, visibilidade do item, vendas, etc.).

### core/features/

* Pipelines de preprocessamento: imputação de valores faltantes, one-hot encoding, escala.
* Seleção de features relevantes para vendas e visibilidade.

### core/models/

* Treinamento de modelos:

  * **Regressão Linear** → previsão de vendas.
  * **Regressão Logística** → classificação da visibilidade do item (alta/baixa).
* Avaliação:

  * Regressão → RMSE, R².
  * Classificação → acurácia, f1-score, precisão, recall.

### core/explain/

* Extração de coeficientes e importâncias de variáveis.
* Interpretação sobre **quais fatores impactam vendas e visibilidade** (ex.: tipo de loja, tamanho, localização).

### core/chatbot/

* Regras para perguntas frequentes:

  * “Qual tipo de loja mais vende?”
  * “A visibilidade influencia mais em supermercados ou lojas pequenas?”
* Respostas sobre métricas, variáveis e pipeline.

### data/

* `raw/`: datasets brutos (não alterar).
* `processed/`: datasets tratados.
* `models/`: modelos salvos (.pkl).

### configs/

* Arquivos de configuração (paths, parâmetros de treino, logging).

### docs/

* Documentação integrada: PMC, arquitetura, modelagem de dados, governança LGPD, testes, deploy.

---

## 4. Fluxo de Execução

1. Usuário acessa aplicação pelo navegador.
2. Faz upload do dataset **BigMart Sales**.
3. A camada `app/` envia os dados para `core/data/` → validação.
4. `core/features/` monta o pipeline de preprocessamento.
5. `core/models/` treina e avalia o modelo escolhido.
6. `core/explain/` gera coeficientes, gráficos e importâncias.
7. `core/chatbot/` interpreta perguntas e gera respostas sobre vendas e visibilidade.
8. Resultados (métricas, gráficos, tabelas, explicações) são exibidos em `app/`.
9. Artefatos (datasets tratados, modelos) são salvos em `data/`.

---

## 5. Como desenhar no draw\.io

1. Acesse [draw.io](https://app.diagrams.net/).
2. Crie caixas para cada camada:

   * **Usuário (Navegador)** no topo.
   * **app/** logo abaixo, identificado como “Streamlit UI”.
   * **core/** ao centro, subdividido em: data, features, models, explain, chatbot.
   * **data/** na base, com subpastas raw, processed, models.
   * **configs/** e **docs/** ao lado, conectados como apoio.
3. Conecte com setas verticais (usuário → app → core → data).
4. Adicione rótulos nas setas, por exemplo: “upload dataset BigMart”, “pré-processamento”, “treino/predição”, “resultados de vendas”.
5. Salve o diagrama em formato `.drawio` e exporte também como `.png`.
6. No repositório, coloque em `docs/images/architecture.png` e referencie neste arquivo com:

```markdown
![Arquitetura](./images/architecture.png)
```

---

Essa arquitetura é suficiente para um **MVP educacional BigMart**, mas já prepara terreno para futuras extensões (ex.: outros algoritmos de ML, integração com APIs externas de estoque, dashboards avançados).

---