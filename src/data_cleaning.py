"""
Módulo para limpeza e pré-processamento dos dados de readmissão hospitalar diabética
"""

import pandas as pd
import numpy as np
import os

from src.config import (
    RAW_DATA_FILE, CLEAN_DATA_FILE, EXPIRED_DISCHARGE_CODES,
    COLUMNS_TO_DROP_INITIAL
)
from src.id_mapping_utils import IDMappingUtils


class DataCleaner:
    """Classe responsável pela limpeza dos dados do dataset diabético"""
    
    def __init__(self, raw_data_path=None):
        """Inicializa o limpador de dados"""
        self.raw_data_path = raw_data_path or RAW_DATA_FILE
        self.df_raw = None
        self.df_clean = None
        self.id_mapper = IDMappingUtils()
        
        # Carregar mapeamentos de IDs
        try:
            self.id_mapper.load_mappings()
        except Exception as e:
            print(f"Aviso: Não foi possível carregar mapeamentos de IDs: {e}")
        
    def load_raw_data(self):
        """Carrega os dados brutos do arquivo CSV"""
        print(f"Carregando dados de: {self.raw_data_path}")
        self.df_raw = pd.read_csv(self.raw_data_path)
        print(f"Dados carregados. Shape: {self.df_raw.shape}")
        return self.df_raw
    
    def create_target_variable(self):
        """Cria a variável alvo binária para predição de readmissão"""
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
        print(f"\nRemovendo pacientes expirados")
        print(f"Registros antes da remoção: {len(self.df_raw)}")
        
        expired_count = self.df_raw['discharge_disposition_id'].isin(EXPIRED_DISCHARGE_CODES).sum()
        print(f"Registros com discharge_disposition_id expirado: {expired_count}")
        
        self.df_clean = self.df_raw[~self.df_raw['discharge_disposition_id'].isin(EXPIRED_DISCHARGE_CODES)].copy()
        
        print(f"Registros após remoção: {len(self.df_clean)}")
        print(f"Registros removidos: {len(self.df_raw) - len(self.df_clean)}")
    
    def handle_missing_data(self):
        """Trata dados faltantes no dataset"""
        print(f"\nTratando dados faltantes")
        
        # Remover colunas com muitos dados faltantes
        existing_cols_to_drop = [col for col in COLUMNS_TO_DROP_INITIAL if col in self.df_clean.columns]
        if existing_cols_to_drop:
            print(f"Removendo colunas com muitos dados faltantes: {existing_cols_to_drop}")
            self.df_clean = self.df_clean.drop(existing_cols_to_drop, axis=1)
        
        # Tratar medical_specialty: substituir '?' por 'missing'
        if 'medical_specialty' in self.df_clean.columns:
            missing_count = (self.df_clean['medical_specialty'] == '?').sum()
            self.df_clean['medical_specialty'] = self.df_clean['medical_specialty'].replace('?', 'missing')
            print(f"Valores '?' em medical_specialty substituídos por 'missing': {missing_count}")
    
    def remove_duplicate_patients(self):
        """Remove registros duplicados, mantendo apenas a primeira internação de cada paciente"""
        print(f"\nRemovendo dados duplicados de pacientes")
        print(f"Registros antes da remoção de duplicatas: {len(self.df_clean)}")
        print(f"Pacientes únicos: {self.df_clean['patient_nbr'].nunique()}")
        
        # Manter apenas a primeira internação de cada paciente
        self.df_clean = self.df_clean.drop_duplicates(subset=['patient_nbr'], keep='first')
        
        print(f"Registros após remoção de duplicatas: {len(self.df_clean)}")
        print(f"Pacientes únicos após limpeza: {self.df_clean['patient_nbr'].nunique()}")
    
    def apply_id_mappings(self):
        """Aplica mapeamentos de IDs para enriquecer dados com descrições legíveis"""
        print(f"\nAplicando mapeamentos de IDs")
        
        if self.id_mapper is None:
            print("Mapeamentos de IDs não disponíveis. Pulando esta etapa.")
            return
        
        print("Aplicando mapeamentos de IDs aos dados...")
        
        # Validar mapeamentos antes de aplicar
        validation_report = self.id_mapper.validate_mappings(self.df_clean)
        print("\nRelatório de validação dos mapeamentos:")
        for mapping_type, report in validation_report.items():
            coverage = report['coverage_rate'] * 100
            print(f"  {mapping_type}: {report['mapped_ids']}/{report['total_unique_ids']} IDs mapeados ({coverage:.1f}%)")
            if report['unmapped_ids']:
                print(f"    IDs não mapeados: {report['unmapped_ids']}")
        
        # Aplicar mapeamentos
        df_before = self.df_clean.shape[1]
        self.df_clean = self.id_mapper.apply_mappings_to_dataframe(self.df_clean)
        df_after = self.df_clean.shape[1]
        
        print(f"\nColunas descritivas adicionadas: {df_after - df_before}")
        print(f"Dimensões finais após mapeamentos: {self.df_clean.shape}")
    
    def get_cleaning_summary(self):
        """Exibe resumo da limpeza dos dados"""
        print(f"\nDistribuição da variável alvo após limpeza")
        print("Distribuição da variável 'readmitted' após limpeza:")
        print(self.df_clean['readmitted'].value_counts())
        print(f"\nDistribuição da variável 'target' após limpeza:")
        print(self.df_clean['target'].value_counts())
        print(f"Proporção de readmissões em <30 dias após limpeza: {self.df_clean['target'].mean():.3f}")
    
    def clean_data(self, save_path=None):
        """Executa todo o pipeline de limpeza dos dados"""
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
        self.apply_id_mappings()
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
