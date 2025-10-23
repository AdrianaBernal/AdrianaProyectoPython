import pandas as pd
from pathlib import Path

class ManagedDataset:
    """
    Clase para gestionar datasets de manera centralizada.
    """
    def __init__(self, df: pd.DataFrame, name: str = "Dataset", description: str = ""):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("El argumento df debe ser un DataFrame")
        self.data = df.copy()
        self.name = name
        self.description = description
        self.transformations = []

    def __str__(self):
        info = f"ManagedDataset: {self.name}\n"
        info += f"Descripción: {self.description}\n"
        info += f"Dimensiones: {self.data.shape[0]} filas x {self.data.shape[1]} columnas\n"
        info += f"Columnas: {', '.join(self.data.columns)}\n"
        if self.transformations:
            info += f"Transformaciones aplicadas: {' | '.join(self.transformations)}\n"
        return info

    # Obtener y modificar datos
    def get_data(self) -> pd.DataFrame:
        return self.data.copy()

    def set_data(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Debe ser un DataFrame")
        self.data = df.copy()

    # Registrar transformaciones
    def add_transformation(self, description: str):
        self.transformations.append(description)

    # --------------------------------------
    # Lectura de dataset desde varios formatos
    # --------------------------------------
    def read_dataset(file: str, format: str = "csv", name: str = "Dataset", 
                     description: str = "", sheet = 0):
        """
        Lee un dataset desde un archivo en distintos formatos y lo devuelve como un objeto ManagedDataset.
        
        Args:
            file (str): Ruta al archivo de datos.
            format (str, opcional): Formato del archivo. Valores permitidos: 'csv', 'tsv', 'excel', 'pickle'. 
                Por defecto es 'csv'.
            name (str, opcional): Nombre identificativo del dataset. Por defecto es 'Dataset'.
            description (str, opcional): Descripción breve del dataset. Por defecto es cadena vacía.
            sheet (int, opcional): Índice o nombre de la hoja a leer en archivos Excel. Por defecto es 0.
        
        Returns:
            ManagedDataset: Objeto que contiene el DataFrame leído junto con su nombre y descripción.
        
        Raises:
            ValueError: Si el formato especificado no está soportado.
        """
        file_path = Path(file)
        format = format.lower()
        
        if format == "csv":
            df = pd.read_csv(file_path)
        elif format == "tsv":
            df = pd.read_csv(file_path, sep="\t")
        elif format == "excel":
            df = pd.read_excel(file_path, sheet_name=sheet)
        elif format == "pickle":
            df = pd.read_pickle(file_path)
        else:
            raise ValueError("Formato no soportado. Usar: csv, tsv, excel, pickle")
        
        return ManagedDataset(df, name=name, description=description)

    # --------------------------------------
    # Guardar dataset en varios formatos
    # --------------------------------------
    def save_dataset(self, file: str, format: str = "csv", sheet: str = "Sheet1"):
        """
        Guarda el dataset actual en un archivo en distintos formatos.
    
        Args:
            file (str): Ruta donde se guardará el archivo.
            format (str, opcional): Formato de salida. Valores permitidos: 'csv', 'tsv', 'excel', 'pickle'.
                Por defecto es 'csv'.
            sheet (str, opcional): Nombre de la hoja al exportar a Excel. Por defecto es 'Sheet1'.
    
        Returns:
            None
    
        Raises:
            ValueError: Si el formato especificado no está soportado.
        """
        format = format.lower()
        file_path = Path(file)

        if format == "csv":
            self.data.to_csv(file_path, index=False)
        elif format == "tsv":
            self.data.to_csv(file_path, sep="\t", index=False)
        elif format == "excel":
            self.data.to_excel(file_path, sheet_name=sheet, index=False)
        elif format == "pickle":
            self.data.to_pickle(file_path)
        else:
            raise ValueError("Formato no soportado. Usar: csv, tsv, excel, pickle")
