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
        """Inicializa o engenheiro de features"""
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
        print("\nRemovendo colunas desnecessárias...")
        
        # Identificar colunas descritivas (criadas pelos mapeamentos)
        desc_cols = [col for col in self.df_clean.columns if col.endswith('_desc')]
        id_cols = [col for col in self.df_clean.columns if col.endswith('_id') and 
                  col.replace('_id', '_desc') in desc_cols]
        
        # Atualizar colunas a serem removidas para incluir IDs que têm descrições
        cols_to_remove = list(COLUMNS_TO_DROP_MODELING) + id_cols
        
        print(f"Colunas a serem removidas: {COLUMNS_TO_DROP_MODELING}")
        if id_cols:
            print(f"Colunas de ID com mapeamentos removidas: {id_cols}")
            print(f"Colunas descritivas mantidas: {desc_cols}")
        
        # Remover apenas as colunas que existem no DataFrame
        existing_cols_to_remove = [col for col in cols_to_remove if col in self.df_clean.columns]
        self.df_processed = self.df_clean.drop(existing_cols_to_remove, axis=1)
        
        print(f"Total de colunas removidas: {len(existing_cols_to_remove)}")
        print(f"Dimensões após remoção: {self.df_processed.shape}")
    
    def identify_column_types(self):
        """Identifica tipos de colunas para processamento"""
        print("\nIdentificando tipos de colunas")
        
        numeric_cols = self.df_processed.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = self.df_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Remover target da lista de features numéricas
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        print(f"Colunas numéricas ({len(numeric_cols)}): {numeric_cols[:5]}{'...' if len(numeric_cols) > 5 else ''}")
        print(f"Colunas categóricas ({len(categorical_cols)}): {categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''}")
        
        return numeric_cols, categorical_cols
    
    def apply_one_hot_encoding(self, categorical_cols):
        """Aplica One-Hot Encoding nas colunas categóricas"""
        print(f"\nAplicando One-Hot Encoding")
        print(f"Aplicando encoding em {len(categorical_cols)} colunas categóricas")
        print(f"Dimensões antes do encoding: {self.df_processed.shape}")
        
        df_encoded = pd.get_dummies(self.df_processed, columns=categorical_cols, drop_first=True)
        
        print(f"Dimensões após encoding: {df_encoded.shape}")
        print(f"Novas features criadas: {df_encoded.shape[1] - self.df_processed.shape[1]}")
        
        return df_encoded
    
    def prepare_features_and_target(self, df_encoded):
        """Separa features (X) e variável alvo (y)"""
        print(f"\nSeparando features (X) e variável alvo (y)")
        
        X = df_encoded.drop('target', axis=1)
        y = df_encoded['target']
        
        print(f"Features (X): {X.shape}")
        print(f"Target (y): {y.shape}")
        print(f"Distribuição da variável alvo: {y.value_counts().to_dict()}")
        
        return X, y
    
    def split_train_test(self, X, y):
        """Divide os dados em conjuntos de treino e teste"""
        print(f"\nDividindo dados em treino e teste")
        print(f"Proporção de teste: {TEST_SIZE} ({TEST_SIZE*100}%)")
        print(f"Random state: {RANDOM_STATE}")
        print(f"Estratificação: Sim")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        # Verificar distribuição após divisão
        print(f"\nDistribuição após divisão:")
        print(f"  Treino: Classe 0: {(y_train == 0).sum():,}, Classe 1: {(y_train == 1).sum():,}")
        print(f"  Teste:  Classe 0: {(y_test == 0).sum():,}, Classe 1: {(y_test == 1).sum():,}")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_summary(self, X):
        """Exibe resumo das features finais"""
        print(f"\nResumo das features finais:")
        print(f"  Total de features: {X.shape[1]}")
        print(f"  Tipos de dados: {X.dtypes.value_counts().to_dict()}")
        
        # Mostrar algumas features de exemplo
        print(f"  Exemplos de features: {list(X.columns[:10])}")
        if X.shape[1] > 10:
            print(f"  ... e mais {X.shape[1] - 10} features")
    
    def process_features(self, save_path=None):
        """Executa todo o pipeline de engenharia de features"""
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
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test(X, y)
        
        self.get_feature_summary(X)
        
        # Salvar dados processados
        if save_path:
            df_encoded.to_csv(save_path, index=False)
            print(f"\nDados processados salvos em: {save_path}")
        
        print("\n" + "="*60)
        print("ENGENHARIA DE FEATURES CONCLUÍDA")
        print("="*60)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_processed_datasets(self, output_dir=None):
        """Salva os conjuntos de treino e teste separadamente"""
        from src.config import DATA_DIR
        
        output_dir = output_dir or DATA_DIR
        
        # Caminhos dos arquivos
        train_features_path = os.path.join(output_dir, 'X_train.csv')
        test_features_path = os.path.join(output_dir, 'X_test.csv')
        train_target_path = os.path.join(output_dir, 'y_train.csv')
        test_target_path = os.path.join(output_dir, 'y_test.csv')
        
        # Salvar conjuntos
        self.X_train.to_csv(train_features_path, index=False)
        self.X_test.to_csv(test_features_path, index=False)
        self.y_train.to_csv(train_target_path, index=False, header=['target'])
        self.y_test.to_csv(test_target_path, index=False, header=['target'])
        
        print(f"\nConjuntos salvos:")
        print(f"  Features de treino: {train_features_path} {self.X_train.shape}")
        print(f"  Features de teste: {test_features_path} {self.X_test.shape}")
        print(f"  Target de treino: {train_target_path} {self.y_train.shape}")
        print(f"  Target de teste: {test_target_path} {self.y_test.shape}")


def main():
    """Função principal para executar a engenharia de features"""
    engineer = FeatureEngineer()
    X_train, X_test, y_train, y_test = engineer.process_features(PROCESSED_DATA_FILE)
    engineer.save_processed_datasets()
    return


if __name__ == "__main__":
    main()
