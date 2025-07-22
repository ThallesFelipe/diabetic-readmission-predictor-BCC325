"""
Script principal para execução do pipeline completo de análise de readmissão hospitalar diabética

Este script executa todas as etapas do projeto:
1. Análise exploratória dos dados
2. Limpeza dos dados
3. Engenharia de features
4. Preparação para modelagem
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.exploratory_analysis import ExploratoryDataAnalysis
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.config import RAW_DATA_FILE, CLEAN_DATA_FILE, PROCESSED_DATA_FILE


def run_exploratory_analysis():
    """Executa análise exploratória dos dados"""
    print("\n" + "="*80)
    print("ETAPA 1: ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("="*80)
    
    eda = ExploratoryDataAnalysis(RAW_DATA_FILE)
    df_analyzed = eda.run_complete_analysis()
    
    return df_analyzed


def run_data_cleaning():
    """Executa limpeza dos dados"""
    print("\n" + "="*80)
    print("ETAPA 2: LIMPEZA DOS DADOS")
    print("="*80)
    
    cleaner = DataCleaner(RAW_DATA_FILE)
    df_clean = cleaner.clean_data(CLEAN_DATA_FILE)
    
    return df_clean


def run_feature_engineering():
    """Executa engenharia de features"""
    print("\n" + "="*80)
    print("ETAPA 3: ENGENHARIA DE FEATURES")
    print("="*80)
    
    engineer = FeatureEngineer(CLEAN_DATA_FILE)
    X_train, X_test, y_train, y_test = engineer.process_features(PROCESSED_DATA_FILE)
    engineer.save_processed_datasets()
    
    return X_train, X_test, y_train, y_test


def main():
    """Função principal que executa todo o pipeline"""
    print("="*100)
    print("PIPELINE COMPLETO DE ANÁLISE DE READMISSÃO HOSPITALAR DIABÉTICA")
    print("="*100)
    
    try:
        # Etapa 1: Análise Exploratória
        df_analyzed = run_exploratory_analysis()
        
        # Etapa 2: Limpeza dos Dados
        df_clean = run_data_cleaning()
        
        # Etapa 3: Engenharia de Features
        X_train, X_test, y_train, y_test = run_feature_engineering()
        
        print("\n" + "="*100)
        print("PIPELINE EXECUTADO COM SUCESSO!")
        print("="*100)
        print(f"\nResultados finais:")
        print(f"- Conjunto de treino: {X_train.shape[0]} amostras, {X_train.shape[1]} features")
        print(f"- Conjunto de teste: {X_test.shape[0]} amostras, {X_test.shape[1]} features")
        print(f"- Taxa de readmissão no treino: {y_train.mean():.3f}")
        print(f"- Taxa de readmissão no teste: {y_test.mean():.3f}")
        
        print(f"\nArquivos gerados:")
        print(f"- Dados limpos: {CLEAN_DATA_FILE}")
        print(f"- Dados processados: {PROCESSED_DATA_FILE}")
        print(f"- Conjuntos de treino e teste na pasta data/")
        
        print(f"\nPróximos passos:")
        print(f"- Implementar modelos de Machine Learning")
        print(f"- Avaliar performance dos modelos")
        print(f"- Fazer tuning de hiperparâmetros")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"\nERRO durante execução do pipeline: {e}")
        print(f"Tipo do erro: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
