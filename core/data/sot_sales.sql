CREATE TABLE IF NOT EXISTS sot_sales (
    Item_Identifier VARCHAR(10),
    Outlet_Identifier VARCHAR(10),
    Item_Outlet_Sales DECIMAL(12,2) NOT NULL,
    FOREIGN KEY (Item_Identifier) REFERENCES sot_items(Item_Identifier),
    FOREIGN KEY (Outlet_Identifier) REFERENCES sot_outlets(Outlet_Identifier)
);
