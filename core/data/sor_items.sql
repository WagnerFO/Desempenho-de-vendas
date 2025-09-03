CREATE TABLE IF NOT EXISTS sor_items (
    Item_Identifier VARCHAR(10) PRIMARY KEY,
    Item_Weight FLOAT,
    Item_Fat_Content VARCHAR(20),
    Item_Visibility FLOAT,
    Item_Type VARCHAR(50),
    Item_MRP DECIMAL(10,2)
);
