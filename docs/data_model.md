# Data Model

## 1. SOR (System of Record)
Tabelas que armazenam os dados brutos vindos do arquivo `Train.csv`.

### Tabela: sor_items
- `Item_Identifier` (VARCHAR(10), PK)
- `Item_Weight` (FLOAT, NULLABLE)
- `Item_Fat_Content` (VARCHAR(20))
- `Item_Visibility` (FLOAT)
- `Item_Type` (VARCHAR(50))
- `Item_MRP` (DECIMAL(10,2))

### Tabela: sor_outlets
- `Outlet_Identifier` (VARCHAR(10), PK)
- `Outlet_Establishment_Year` (INT)
- `Outlet_Size` (VARCHAR(20), NULLABLE)
- `Outlet_Location_Type` (VARCHAR(20))
- `Outlet_Type` (VARCHAR(50))

### Tabela: sor_sales
- `Item_Identifier` (VARCHAR(10), FK → sor_items.Item_Identifier)
- `Outlet_Identifier` (VARCHAR(10), FK → sor_outlets.Outlet_Identifier)
- `Item_Outlet_Sales` (DECIMAL(12,2))

---

## 2. SOT (System of Transformation)
Tabelas com dados tratados (após imputações, ajustes de categorias, etc).

### Tabela: sot_items
- `Item_Identifier` (VARCHAR(10), PK)
- `Item_Weight` (FLOAT, imputado quando NULL)
- `Item_Fat_Content` (ENUM('Low Fat', 'Regular', 'Other'))
- `Item_Visibility` (FLOAT)
- `Item_Type` (VARCHAR(50))
- `Item_MRP` (DECIMAL(10,2))

### Tabela: sot_outlets
- `Outlet_Identifier` (VARCHAR(10), PK)
- `Outlet_Establishment_Year` (INT)
- `Outlet_Size` (ENUM('Small', 'Medium', 'High'), imputado quando NULL)
- `Outlet_Location_Type` (VARCHAR(20))
- `Outlet_Type` (VARCHAR(50))

### Tabela: sot_sales
- `Item_Identifier` (VARCHAR(10), FK → sot_items.Item_Identifier)
- `Outlet_Identifier` (VARCHAR(10), FK → sot_outlets.Outlet_Identifier)
- `Item_Outlet_Sales` (DECIMAL(12,2))

---

## 3. SPEC (Specialized / Prediction)
Tabelas finais usadas para alimentar os modelos de ML.

### Tabela: spec_features
- `Item_Identifier` (VARCHAR(10))
- `Outlet_Identifier` (VARCHAR(10))
- `Item_Weight` (FLOAT)
- `Item_Fat_Content_Bin` (TINYINT) → 0 = Low Fat, 1 = Regular
- `Item_Visibility_Bin` (TINYINT) → 0 = Baixa, 1 = Alta
- `Item_MRP` (DECIMAL(10,2))
- `Outlet_Size` (TINYINT) → 0 = Small, 1 = Medium, 2 = High
- `Outlet_Location_Type` (VARCHAR(20))
- `Outlet_Type` (VARCHAR(50))

### Tabela: spec_labels
- `Item_Identifier` (VARCHAR(10))
- `Outlet_Identifier` (VARCHAR(10))
- `Target_Sales` (DECIMAL(12,2))  → variável alvo
