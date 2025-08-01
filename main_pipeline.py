"""
Script principal para execução do pipeline completo de análise de readmissão hospitalar diabética

Este script executa todas as etapas do projeto:
1. Análise exploratória dos dados
2. Limpeza dos dados
3. Engenharia de features
4. Preparação para modelagem
"""

import os
import sys
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.exploratory_analysis import ExploratoryDataAnalysis
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.logistic_regression_model import LogisticRegressionModel
from src.config import RAW_DATA_FILE, CLEAN_DATA_FILE, PROCESSED_DATA_FILE


def run_exploratory_analysis():
    """Executa análise exploratória dos dados"""
    print("\nETAPA 1: ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("="*50)
    
    eda = ExploratoryDataAnalysis(RAW_DATA_FILE)
    df_analyzed = eda.run_complete_analysis()
    
    return df_analyzed


def run_data_cleaning():
    """Executa limpeza dos dados"""
    print("\nETAPA 2: LIMPEZA DOS DADOS")
    print("="*50)
    
    cleaner = DataCleaner(RAW_DATA_FILE)
    df_clean = cleaner.clean_data(CLEAN_DATA_FILE)
    
    return df_clean


def run_feature_engineering():
    """Executa engenharia de features"""
    print("\nETAPA 3: ENGENHARIA DE FEATURES")
    print("="*50)
    
    engineer = FeatureEngineer(CLEAN_DATA_FILE)
    X_train, X_test, y_train, y_test = engineer.process_features(PROCESSED_DATA_FILE)
    engineer.save_processed_datasets()
    
    return X_train, X_test, y_train, y_test


def run_machine_learning_models():
    """Executa modelos de Machine Learning"""
    print("\nETAPA 4: MODELOS DE MACHINE LEARNING")
    print("="*50)
    
    # Regressão Logística
    print("\nExecutando Regressão Logística...")
    lr_model = LogisticRegressionModel()
    lr_success = lr_model.run_complete_pipeline()
    
    if lr_success:
        print("✅ Regressão Logística executada com sucesso!")
        return lr_model
    else:
        print("❌ Erro na execução da Regressão Logística")
        return None


def main():
    """Função principal que executa todo o pipeline"""
    print("PIPELINE COMPLETO DE ANÁLISE DE READMISSÃO HOSPITALAR DIABÉTICA")
    print("="*70)
    
    try:
        # Etapa 1: Análise Exploratória
        df_analyzed = run_exploratory_analysis()
        
        # Etapa 2: Limpeza dos Dados
        df_clean = run_data_cleaning()
        
        # Etapa 3: Engenharia de Features
        X_train, X_test, y_train, y_test = run_feature_engineering()
        
        # Etapa 4: Modelos de Machine Learning
        lr_model = run_machine_learning_models()
        
        print("\n" + "="*70)
        print("PIPELINE EXECUTADO COM SUCESSO!")
        print("="*70)
        
        print(f"\nResultados finais:")
        print(f"- Conjunto de treino: {X_train.shape[0]} amostras, {X_train.shape[1]} features")
        print(f"- Conjunto de teste: {X_test.shape[0]} amostras, {X_test.shape[1]} features")
        print(f"- Taxa de readmissão no treino: {y_train.mean():.3f}")
        print(f"- Taxa de readmissão no teste: {y_test.mean():.3f}")
        
        if lr_model:
            print(f"\nModelos treinados:")
            print(f"- ✅ Regressão Logística")
            print(f"- 🔜 Random Forest (próximo)")
            print(f"- 🔜 XGBoost (próximo)")
        
        print(f"\nArquivos gerados:")
        print(f"- Dados limpos: {CLEAN_DATA_FILE}")
        print(f"- Dados processados: {PROCESSED_DATA_FILE}")
        print(f"- Conjuntos de treino e teste na pasta data/")
        print(f"- Modelo e resultados na pasta models/ e results/")
        
        print(f"\nPróximos passos:")
        print(f"- ✅ Regressão Logística implementada")
        print(f"- 🔜 Implementar Random Forest")
        print(f"- 🔜 Implementar XGBoost")
        print(f"- 🔜 Comparar performance dos modelos")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"\nERRO durante execução do pipeline: {e}")
        print(f"Tipo do erro: {type(e).__name__}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
