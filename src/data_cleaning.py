"""
Módulo para limpeza e pré-processamento dos dados de readmissão hospitalar diabética
"""

import pandas as pd
import numpy as np
from src.config import (
    RAW_DATA_FILE, CLEAN_DATA_FILE, EXPIRED_DISCHARGE_CODES,
    COLUMNS_TO_DROP_INITIAL
)


class DataCleaner:
    """Classe responsável pela limpeza dos dados do dataset diabético"""
    
    def __init__(self, raw_data_path=None):
        """
        Inicializa o limpador de dados
        
        Args:
            raw_data_path (str): Caminho para o arquivo de dados brutos
        """
        self.raw_data_path = raw_data_path or RAW_DATA_FILE
        self.df_raw = None
        self.df_clean = None
        
    def load_raw_data(self):
        """Carrega os dados brutos do arquivo CSV"""
        print(f"Carregando dados de: {self.raw_data_path}")
        self.df_raw = pd.read_csv(self.raw_data_path)
        print(f"Dados carregados. Shape: {self.df_raw.shape}")
        return self.df_raw
    
    def create_target_variable(self):
        """
        Cria a variável alvo binária para predição de readmissão
        
        Returns:
            pd.Series: Série com a variável target (1 para readmissão <30 dias, 0 caso contrário)
        """
        print("Criando variável alvo binária...")
        
        # Criar variável target: 1 se readmitido em <30 dias, 0 caso contrário
        self.df_raw['target'] = self.df_raw['readmitted'].apply(
            lambda x: 1 if x == '<30' else 0
        )
        
        print("Distribuição da variável 'readmitted':")
        print(self.df_raw['readmitted'].value_counts())
        print(f"\nDistribuição da variável 'target':")
        print(self.df_raw['target'].value_counts())
        print(f"Proporção de readmissões em <30 dias: {self.df_raw['target'].mean():.3f}")
        
        return self.df_raw['target']
    
    def remove_expired_patients(self):
        """Remove pacientes que faleceram (códigos de discharge específicos)"""
        print(f"\n1. REMOVENDO PACIENTES EXPIRADOS")
        print(f"Registros antes da remoção: {len(self.df_raw)}")
        
        expired_count = self.df_raw['discharge_disposition_id'].isin(EXPIRED_DISCHARGE_CODES).sum()
        print(f"Registros com discharge_disposition_id em {EXPIRED_DISCHARGE_CODES}: {expired_count}")
        
        self.df_clean = self.df_raw[~self.df_raw['discharge_disposition_id'].isin(EXPIRED_DISCHARGE_CODES)].copy()
        
        print(f"Registros após remoção: {len(self.df_clean)}")
        print(f"Registros removidos: {len(self.df_raw) - len(self.df_clean)}")
    
    def handle_missing_data(self):
        """Trata dados faltantes no dataset"""
        print(f"\n2. ANÁLISE DE DADOS FALTANTES")
        
        # Verificar percentual de dados faltantes
        missing_weight = (self.df_clean['weight'] == '?').sum() / len(self.df_clean) * 100
        missing_payer = (self.df_clean['payer_code'] == '?').sum() / len(self.df_clean) * 100
        missing_specialty = (self.df_clean['medical_specialty'] == '?').sum() / len(self.df_clean) * 100
        
        print(f"Dados faltantes em 'weight': {missing_weight:.1f}%")
        print(f"Dados faltantes em 'payer_code': {missing_payer:.1f}%")
        print(f"Dados faltantes em 'medical_specialty': {missing_specialty:.1f}%")
        
        # Remover colunas com muitos dados faltantes
        print(f"\nRemovendo colunas: {COLUMNS_TO_DROP_INITIAL}")
        self.df_clean = self.df_clean.drop(COLUMNS_TO_DROP_INITIAL, axis=1)
        
        # Tratar medical_specialty
        print(f"Tratando valores faltantes em 'medical_specialty'...")
        self.df_clean['medical_specialty'] = self.df_clean['medical_specialty'].replace('?', 'missing')
        missing_count = (self.df_clean['medical_specialty'] == 'missing').sum()
        print(f"Valores 'missing' em medical_specialty: {missing_count}")
    
    def remove_duplicate_patients(self):
        """Remove registros duplicados, mantendo apenas a primeira internação de cada paciente"""
        print(f"\n3. REMOVENDO DADOS DUPLICADOS DE PACIENTES")
        print(f"Registros antes da remoção de duplicatas: {len(self.df_clean)}")
        print(f"Pacientes únicos: {self.df_clean['patient_nbr'].nunique()}")
        
        # Manter apenas a primeira internação de cada paciente
        self.df_clean = self.df_clean.drop_duplicates(subset=['patient_nbr'], keep='first')
        
        print(f"Registros após remoção de duplicatas: {len(self.df_clean)}")
        print(f"Pacientes únicos após limpeza: {self.df_clean['patient_nbr'].nunique()}")
    
    def get_cleaning_summary(self):
        """Exibe resumo da limpeza dos dados"""
        print(f"\n4. DISTRIBUIÇÃO DA VARIÁVEL ALVO APÓS LIMPEZA")
        print("Distribuição da variável 'readmitted' após limpeza:")
        print(self.df_clean['readmitted'].value_counts())
        print(f"\nDistribuição da variável 'target' após limpeza:")
        print(self.df_clean['target'].value_counts())
        print(f"Proporção de readmissões em <30 dias após limpeza: {self.df_clean['target'].mean():.3f}")
    
    def clean_data(self, save_path=None):
        """
        Executa todo o pipeline de limpeza dos dados
        
        Args:
            save_path (str): Caminho para salvar os dados limpos
            
        Returns:
            pd.DataFrame: DataFrame com dados limpos
        """
        print("\n" + "="*60)
        print("INICIANDO LIMPEZA DOS DADOS")
        print("="*60)
        
        # Executar todas as etapas de limpeza
        if self.df_raw is None:
            self.load_raw_data()
            
        self.create_target_variable()
        self.remove_expired_patients()
        self.handle_missing_data()
        self.remove_duplicate_patients()
        self.get_cleaning_summary()
        
        print("\n" + "="*60)
        print("LIMPEZA DOS DADOS CONCLUÍDA")
        print("="*60)
        
        # Salvar dados limpos
        save_path = save_path or CLEAN_DATA_FILE
        self.df_clean.to_csv(save_path, index=False)
        print(f"\nDataset limpo salvo em: {save_path}")
        print(f"Dimensões finais: {self.df_clean.shape}")
        
        return self.df_clean


def main():
    """Função principal para executar a limpeza dos dados"""
    cleaner = DataCleaner()
    df_clean = cleaner.clean_data()
    return df_clean


if __name__ == "__main__":
    main()
