# Governança e LGPD

## 1. Contexto

Este documento descreve como o projeto **BigMart Sales Chatbot MVP** atende aos requisitos da **Lei Geral de Proteção de Dados Pessoais (LGPD — Lei nº 13.709/2018)**.
Embora o dataset atual (`Train.csv`) **não contenha dados pessoais identificáveis (PII)**, as práticas aqui definidas visam preparar o sistema para cenários de uso com dados reais de clientes.

---

## 2. Princípios da LGPD aplicados

| Princípio               | Aplicação no Projeto                                                                                                                         |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Finalidade**          | Dados são usados exclusivamente para treino e avaliação de modelos de regressão/classificação em vendas e visibilidade de produtos.          |
| **Necessidade**         | Apenas atributos essenciais para análise de vendas e características de itens/outlets são mantidos (ex.: preço, tipo de loja, tipo de item). |
| **Adequação**           | As transformações (SOR → SOT → SPEC) seguem propósito educacional e analítico, sem coleta ou compartilhamento externo.                       |
| **Livre Acesso**        | Logs de execução e tabelas intermediárias permitem auditoria e rastreabilidade.                                                              |
| **Qualidade dos Dados** | Valores ausentes são tratados na camada **SOT**, garantindo integridade e consistência.                                                      |
| **Segurança**           | O banco de dados é criado e destruído em cada execução do pipeline, evitando persistência indevida.                                          |
| **Prevenção**           | Não há armazenamento de dados pessoais; mesmo assim, recomenda-se anonimização caso dados de clientes sejam usados futuramente.              |
| **Responsabilização**   | Toda a pipeline é documentada (`data_model.md`, `architecture.md`), garantindo prestação de contas.                                          |

---

## 3. Estrutura de Dados

* **SOR (System of Record):** armazena dados brutos do CSV sem alteração.
* **SOT (System of Truth):** dados tratados, com imputação e padronização.
* **SPEC (Specific Dataset):** dataset temporário usado para treino de ML.
* 🔐 **Dados são descartados (DROP DATABASE)** ao final da execução, garantindo não retenção indevida.

---

## 4. Segurança

* Conexão local ao **MySQL** (sem exposição a rede externa).
* Senhas de acesso ao banco não são versionadas em código-fonte público (devem ser configuradas via `.env` em produção).
* Logs de execução não armazenam dados sensíveis.
* Recomendação futura: **criptografia em repouso (at rest)** e **criptografia em trânsito (SSL/TLS)**.

---

## 5. Tratamento de Dados Sensíveis

* O dataset BigMart **não contém dados pessoais** (apenas itens, vendas e outlets).
* Caso futuramente sejam integrados dados de clientes (ex.: perfil de compras, CPF, endereço), será necessário:

  * **Anonimização** (hash de identificadores pessoais);
  * **Minimização** (coletar apenas atributos estritamente necessários);
  * **Consentimento** explícito do titular dos dados.

---

## 6. Ciclo de Vida dos Dados

1. **Ingestão**: Leitura do `Train.csv`.
2. **Persistência temporária**: Criação do banco `bigmart_db` (SOR → SOT → SPEC).
3. **Treinamento**: Geração de modelos de ML (`.pickle`).
4. **Descarte**: `DROP DATABASE bigmart_db;` ao final do processo.
5. **Retenção**: Apenas modelos salvos em `data/models/`, que **não contêm dados pessoais**.

---

## 7. Próximos Passos (quando houver PII)

* Adotar **Data Protection by Design** (proteção desde a concepção).
* Implementar controles de acesso por perfil de usuário.
* Disponibilizar **relatório de impacto de proteção de dados (RIPD)**.
* Estabelecer processos de **direito do titular** (acesso, correção, exclusão de dados).

---