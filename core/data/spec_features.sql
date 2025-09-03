CREATE TABLE IF NOT EXISTS spec_features (
    Item_Identifier VARCHAR(10),
    Outlet_Identifier VARCHAR(10),
    Item_Weight FLOAT,
    Item_Fat_Content_Bin TINYINT, -- 0 = Low Fat, 1 = Regular
    Item_Visibility_Bin TINYINT, -- 0 = Baixa, 1 = Alta
    Item_MRP DECIMAL(10,2),
    Outlet_Size TINYINT, -- 0 = Small, 1 = Medium, 2 = High
    Outlet_Location_Type VARCHAR(20),
    Outlet_Type VARCHAR(50)
);
