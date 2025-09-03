CREATE TABLE IF NOT EXISTS sor_sales (
    Item_Identifier VARCHAR(10),
    Outlet_Identifier VARCHAR(10),
    Item_Outlet_Sales DECIMAL(12,2),
    FOREIGN KEY (Item_Identifier) REFERENCES sor_items(Item_Identifier),
    FOREIGN KEY (Outlet_Identifier) REFERENCES sor_outlets(Outlet_Identifier)
);
