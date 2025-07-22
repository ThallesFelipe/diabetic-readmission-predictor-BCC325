"""
Módulo para engenharia de features e preparação dos dados para modelagem
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from src.config import (
    CLEAN_DATA_FILE, PROCESSED_DATA_FILE, COLUMNS_TO_DROP_MODELING,
    TEST_SIZE, RANDOM_STATE
)


class FeatureEngineer:
    """Classe responsável pela engenharia de features e preparação dos dados"""
    
    def __init__(self, clean_data_path=None):
        """
        Inicializa o engenheiro de features
        
        Args:
            clean_data_path (str): Caminho para o arquivo de dados limpos
        """
        self.clean_data_path = clean_data_path or CLEAN_DATA_FILE
        self.df_clean = None
        self.df_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_clean_data(self):
        """Carrega os dados limpos do arquivo CSV"""
        print(f"Carregando dados limpos de: {self.clean_data_path}")
        self.df_clean = pd.read_csv(self.clean_data_path)
        print(f"Dados carregados. Shape: {self.df_clean.shape}")
        return self.df_clean
    
    def remove_unnecessary_columns(self):
        """Remove colunas desnecessárias para modelagem"""
        print("\n1. REMOVENDO COLUNAS DESNECESSÁRIAS...")
        print(f"Colunas a serem removidas: {COLUMNS_TO_DROP_MODELING}")
        
        self.df_processed = self.df_clean.drop(COLUMNS_TO_DROP_MODELING, axis=1)
        print(f"Dimensões após remoção: {self.df_processed.shape}")
    
    def identify_column_types(self):
        """Identifica e categoriza os tipos de colunas"""
        print("\n2. IDENTIFICANDO TIPOS DE COLUNAS...")
        
        numeric_cols = self.df_processed.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = self.df_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Remove o target da lista de features numéricas
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        print(f"Colunas numéricas ({len(numeric_cols)}): {numeric_cols}")
        print(f"Colunas categóricas ({len(categorical_cols)}): {categorical_cols}")
        
        return numeric_cols, categorical_cols
    
    def apply_one_hot_encoding(self, categorical_cols):
        """
        Aplica One-Hot Encoding nas colunas categóricas
        
        Args:
            categorical_cols (list): Lista de colunas categóricas
            
        Returns:
            pd.DataFrame: DataFrame com encoding aplicado
        """
        print("\n3. APLICANDO ONE-HOT ENCODING...")
        print(f"Aplicando encoding em {len(categorical_cols)} colunas categóricas")
        
        df_encoded = pd.get_dummies(
            self.df_processed, 
            columns=categorical_cols, 
            drop_first=True
        )
        
        print(f"Dimensões após encoding: {df_encoded.shape}")
        print(f"Novas features criadas: {df_encoded.shape[1] - self.df_processed.shape[1]}")
        
        return df_encoded
    
    def prepare_features_and_target(self, df_encoded):
        """
        Separa features (X) e variável alvo (y)
        
        Args:
            df_encoded (pd.DataFrame): DataFrame com encoding aplicado
            
        Returns:
            tuple: (X, y) - features e target
        """
        print("\n4. DEFININDO FEATURES (X) E ALVO (y)...")
        
        X = df_encoded.drop('target', axis=1)
        y = df_encoded['target']
        
        print(f"Formato de X (features): {X.shape}")
        print(f"Formato de y (target): {y.shape}")
        print(f"Número total de features: {X.shape[1]}")
        
        return X, y
    
    def split_train_test(self, X, y):
        """
        Divide os dados em conjuntos de treino e teste
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("\n5. DIVIDINDO DADOS EM TREINO E TESTE...")
        print(f"Proporção de teste: {TEST_SIZE}")
        print(f"Random state: {RANDOM_STATE}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y  # Importante para datasets desbalanceados
        )
        
        print(f"Tamanho de X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Tamanho de X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Verificar distribuição da classe nos conjuntos
        print(f"\nDistribuição da classe no treino: {y_train.value_counts().to_dict()}")
        print(f"Distribuição da classe no teste: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_summary(self, X):
        """
        Gera resumo das features criadas
        
        Args:
            X (pd.DataFrame): DataFrame de features
        """
        print("\n6. RESUMO DAS FEATURES:")
        
        # Contar tipos de features
        medication_cols = [col for col in X.columns if any(med in col.lower() for med in 
                          ['metformin', 'insulin', 'glyburide', 'glipizide', 'glimepiride'])]
        demographic_cols = [col for col in X.columns if any(demo in col.lower() for demo in 
                           ['race', 'gender', 'age'])]
        
        print(f"Features de medicamentos: {len(medication_cols)}")
        print(f"Features demográficas: {len(demographic_cols)}")
        print(f"Outras features: {X.shape[1] - len(medication_cols) - len(demographic_cols)}")
    
    def process_features(self, save_path=None):
        """
        Executa todo o pipeline de engenharia de features
        
        Args:
            save_path (str): Caminho para salvar os dados processados
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("\n" + "="*60)
        print("INICIANDO ENGENHARIA DE FEATURES")
        print("="*60)
        
        # Executar todas as etapas
        if self.df_clean is None:
            self.load_clean_data()
            
        self.remove_unnecessary_columns()
        numeric_cols, categorical_cols = self.identify_column_types()
        df_encoded = self.apply_one_hot_encoding(categorical_cols)
        X, y = self.prepare_features_and_target(df_encoded)
        self.get_feature_summary(X)
        
        # Dividir dados
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test(X, y)
        
        # Salvar dados processados
        if save_path:
            df_encoded.to_csv(save_path, index=False)
            print(f"\nDataset processado salvo em: {save_path}")
        
        print("\n" + "="*60)
        print("PREPARAÇÃO DOS DADOS CONCLUÍDA!")
        print("="*60)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_processed_datasets(self, output_dir=None):
        """
        Salva os conjuntos de treino e teste separadamente
        
        Args:
            output_dir (str): Diretório para salvar os arquivos
        """
        if output_dir is None:
            output_dir = os.path.dirname(PROCESSED_DATA_FILE)
            
        # Salvar conjuntos de treino e teste
        train_features_path = os.path.join(output_dir, 'X_train.csv')
        test_features_path = os.path.join(output_dir, 'X_test.csv')
        train_target_path = os.path.join(output_dir, 'y_train.csv')
        test_target_path = os.path.join(output_dir, 'y_test.csv')
        
        self.X_train.to_csv(train_features_path, index=False)
        self.X_test.to_csv(test_features_path, index=False)
        self.y_train.to_csv(train_target_path, index=False)
        self.y_test.to_csv(test_target_path, index=False)
        
        print(f"\nConjuntos salvos em:")
        print(f"- Features de treino: {train_features_path}")
        print(f"- Features de teste: {test_features_path}")
        print(f"- Target de treino: {train_target_path}")
        print(f"- Target de teste: {test_target_path}")


def main():
    """Função principal para executar a engenharia de features"""
    engineer = FeatureEngineer()
    X_train, X_test, y_train, y_test = engineer.process_features(PROCESSED_DATA_FILE)
    engineer.save_processed_datasets()
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()
