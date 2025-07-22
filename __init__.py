"""
Arquivo de inicialização para facilitar imports no projeto
"""

# Facilitar imports dos módulos principais
from src.exploratory_analysis import ExploratoryDataAnalysis
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer

# Configurações principais
__version__ = "0.0.1"
__author__ = "Thalles Felipe, Leonardo de Souza, Ian Martins"
__description__ = "Sistema de Predição de Readmissão Hospitalar Diabética"

# Funções de conveniência
def run_complete_pipeline():
    """Executa o pipeline completo de processamento"""
    from main_pipeline import main
    return main()

def load_processed_data():
    """Carrega dados já processados"""
    import pandas as pd
    import os
    
    data_dir = "data"
    try:
        X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))["target"]
        y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))["target"]
        
        print(f"✅ Dados carregados com sucesso!")
        print(f"Treino: {X_train.shape[0]} amostras, {X_train.shape[1]} features")
        print(f"Teste: {X_test.shape[0]} amostras, {X_test.shape[1]} features")
        
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print("❌ Dados processados não encontrados. Execute o pipeline primeiro:")
        print("python main_pipeline.py")
        return None

# Informações do projeto
def project_info():
    """Exibe informações sobre o projeto"""
    print("="*60)
    print("🏥 SISTEMA DE PREDIÇÃO DE READMISSÃO HOSPITALAR DIABÉTICA")
    print("="*60)
    print(f"Versão: {__version__}")
    print(f"Descrição: {__description__}")
    print()
    print("📋 Módulos Disponíveis:")
    print("• ExploratoryDataAnalysis - Análise exploratória completa")
    print("• DataCleaner - Pipeline de limpeza de dados")
    print("• FeatureEngineer - Engenharia de features para ML")
    print()
    print("🚀 Funções de Conveniência:")
    print("• run_complete_pipeline() - Executa pipeline completo")
    print("• load_processed_data() - Carrega dados processados")
    print("• project_info() - Mostra estas informações")
    print()
    print("📁 Arquivos Principais:")
    print("• main_pipeline.py - Script principal")
    print("• notebooks/diabetic_readmission_analysis.ipynb - Análise interativa")
    print("• README.md - Documentação completa")
    print("="*60)

if __name__ == "__main__":
    project_info()
