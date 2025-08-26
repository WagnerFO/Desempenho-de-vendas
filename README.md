# 📊 Desempenho de Vendas por Tipo de Loja

**Base:** BigMart Sales Dataset (Kaggle)
**Contexto:** Grandes redes varejistas possuem diferentes tipos de loja, com padrões de vendas distintos.
**Motivação:** Apoiar decisões estratégicas de **estoque** e **marketing** com base no perfil de cada loja.
**Problema:** Identificar quais perfis de loja mais vendem e por quais razões.
**Pergunta Norteadora:** Um item tem alta ou baixa visibilidade dependendo do tipo de loja?

---

# 🛠️ Kaggle Chatbot MVP

Um **MVP educacional** construído em **Streamlit**, que responde perguntas sobre o **BigMart Sales Dataset**, possibilitando:

* Exploração interativa do dataset.
* Treino de modelos simples (**Regressão Linear** e **Logística**).
* Insights sobre **tipo de loja**, **visibilidade dos produtos** e **impacto nas vendas**.
* Documentação clara e organizada.

---

## 📖 Documentação

A documentação completa está na pasta [`docs/`](./docs):

* [PMC](./docs/pmc.md)
* [Arquitetura](./docs/architecture.md)
* [Modelagem de Dados](./docs/data_model.md)
* [Governança LGPD/DAMA](./docs/governance_lgpd.md)
* [Testes](./docs/testing.md)
* [Deploy](./docs/deployment.md)

---

## 🖥️ Como rodar o projeto no Visual Studio Code

### 1. Abrir o projeto

* Abra o **VS Code**.
* Vá em **File → Open Folder** e escolha a pasta do seu projeto, exemplo, `bigmart_streamlit/`.

### 2. Criar e ativar ambiente virtual

No terminal integrado do VS Code (`Ctrl+`):

```bash
# Criar ambiente virtual
python3 -m venv .venv

# Ativar no Linux/Mac
source .venv/bin/activate

# Ativar no Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

> ⚠️ No canto inferior direito do VS Code, selecione o interpretador **Python da pasta `.venv`**.

### 3. Instalar dependências

Com o ambiente ativo:

```bash
pip install -r requirements.txt
```

### 4. Rodar o Streamlit

No terminal do VS Code:

```bash
streamlit run app/main_app.py
```

O app abrirá no navegador em [http://localhost:8501](http://localhost:8501).

### 5. Trabalhar com o código

* **Front-end**: `app/main_app.py` (UI em Streamlit).
* **Back-end**: `core/` (dados, features, modelos, explicabilidade e chatbot).
* **Notebooks**: `notebooks/01_eda_bigmart.ipynb` (exploração inicial).

### 6. Rodar testes

```bash
pytest tests/
```

---

## 📂 Estrutura de pastas

```bash
kaggle-chatbot-mvp/
├─ app/            # Interface com o usuário (Streamlit)
├─ core/           # Lógica de negócio (dados, modelos, chatbot)
├─ configs/        # Arquivos de configuração
├─ data/           # Dados brutos, processados e modelos
├─ notebooks/      # Notebooks de exploração (EDA BigMart)
├─ tests/          # Testes unitários e de integração
├─ docs/           # Documentação (PMC, arquitetura, dados, LGPD, etc.)
├─ requirements.txt
└─ README.md
```

---

## 🚀 Deploy

Para publicar rapidamente, veja [docs/deployment.md](./docs/deployment.md).

---