# Arquitetura de Software — Kaggle Chatbot MVP (BigMart Sales)

> Documento atualizado para refletir o uso de camadas **SOR, SOT, SPEC** com MySQL e o salvamento de modelos em `.pickle`.

---

## 1. Objetivos de Arquitetura

* **Separação de responsabilidades**: UI (Streamlit) isolada da lógica (`core/`) e dos dados (`SOR/SOT/SPEC`).
* **Governança de dados**: dados brutos preservados em **SOR**, tratados em **SOT** e especializados em **SPEC**.
* **Reprodutibilidade**: tabelas SQL versionadas em `core/data/`.
* **Evolução**: fácil trocar dataset Kaggle, adicionar novos modelos ou expandir análises.
* **Portabilidade**: modelos treinados salvos em `.pickle`.

---

## 2. Visão em Camadas

```

Usuário (navegador)
↓
\[ app/ ]  → Streamlit UI
↓
\[ core/ ]
├─ db\_utils.py     → criação/drop DB, execução SQL
├─ data\_loader.py  → carga CSV → SOR
├─ model\_utils.py  → treino e salvamento dos modelos
└─ data/           → scripts .sql (SOR, SOT, SPEC)
↓
\[ MySQL Database ]
├─ SOR   → tabelas brutas (espelho CSV)
├─ SOT   → tabelas tratadas (imputação, padronização)
└─ SPEC  → tabelas finais (features + labels para ML)
↓
\[ ML Models ]
├─ Regressão Linear → previsão de vendas
├─ Regressão Logística → classificação de visibilidade
└─ Arquivos salvos em /model/\*.pickle

```

---

## 3. Componentes e Responsabilidades

### app/
* Camada de apresentação (Streamlit).
* Upload do dataset BigMart.
* Seleção de tarefa (regressão ou classificação).
* Exibição de métricas, gráficos e respostas do chatbot.

### core/db_utils.py
* Criar e dropar o banco MySQL.
* Executar arquivos `.sql` de `core/data`.

### core/data_loader.py
* Inserir dados do CSV (`Train.csv`) nas tabelas SOR.
* Preparar transformações para SOT e SPEC.

### core/model_utils.py
* Treinar modelos:
  * **LinearRegression** → prever `Item_Outlet_Sales`.
  * **LogisticRegression** → prever visibilidade (alta/baixa).
* Avaliar rapidamente (RMSE, acurácia).
* Salvar modelos em `.pickle` na pasta `model/`.

### core/data/
* Scripts `.sql` separados por camada e tabela:
  * `sor_*.sql` → tabelas brutas.
  * `sot_*.sql` → tabelas tratadas.
  * `spec_*.sql` → tabelas finais para treino.

### MySQL (camada de dados)
* **SOR** → cópia bruta do CSV.  
* **SOT** → dados limpos (sem nulos, categorias padronizadas).  
* **SPEC** → features e labels prontos para treino.  

### model/
* Artefatos finais salvos como `.pickle`.

---

## 4. Fluxo de Execução

1. Usuário acessa a aplicação pelo navegador.  
2. Upload do dataset **BigMart Sales**.  
3. `app/` envia CSV para `core/data_loader`.  
4. Dados inseridos em **SOR** no MySQL.  
5. `db_utils` aplica scripts para transformar SOR → SOT → SPEC.  
6. `model_utils` lê SPEC e treina modelos:
   - Linear → prever vendas.  
   - Logístico → classificar visibilidade.  
7. Modelos são salvos em `/model/*.pickle`.  
8. `app/` exibe métricas, gráficos e chatbot com respostas baseadas nos modelos.  
9. Ao final, o DB pode ser **dropado** para liberar espaço.  

---

## 5. Como desenhar no draw.io

1. Caixa **Usuário (Navegador)** no topo.  
2. Abaixo, caixa **app/** (Streamlit).  
3. Caixa **core/** subdividida em:
   - db_utils
   - data_loader
   - model_utils
   - data (SQL scripts)
4. Caixa **MySQL Database** com três sub-blocos: SOR, SOT, SPEC.  
5. Caixa **ML Models** à direita, conectada ao SPEC.  
6. Fluxo: Usuário → app → core → DB (SOR → SOT → SPEC) → ML Models → app.  
7. Salvar `.drawio` e exportar `.png` em `docs/images/architecture.png`.

---