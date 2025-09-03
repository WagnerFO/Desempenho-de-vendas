CREATE TABLE IF NOT EXISTS sot_outlets (
    Outlet_Identifier VARCHAR(10) PRIMARY KEY,
    Outlet_Establishment_Year INT NOT NULL,
    Outlet_Size ENUM('Small','Medium','High') NOT NULL,
    Outlet_Location_Type VARCHAR(20) NOT NULL,
    Outlet_Type VARCHAR(50) NOT NULL
);
