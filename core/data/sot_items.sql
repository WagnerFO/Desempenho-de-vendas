CREATE TABLE IF NOT EXISTS sot_items (
    Item_Identifier VARCHAR(10) PRIMARY KEY,
    Item_Weight FLOAT NOT NULL,
    Item_Fat_Content ENUM('Low Fat','Regular','Other') NOT NULL,
    Item_Visibility FLOAT NOT NULL,
    Item_Type VARCHAR(50) NOT NULL,
    Item_MRP DECIMAL(10,2) NOT NULL
);
