"""
Módulo para análise exploratória dos dados de readmissão hospitalar diabética
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import RAW_DATA_FILE, CLEAN_DATA_FILE
from src.id_mapping_utils import IDMappingUtils


class ExploratoryDataAnalysis:
    """Classe para análise exploratória dos dados"""
    
    def __init__(self, data_path=None):
        """
        Inicializa a análise exploratória
        
        Args:
            data_path (str): Caminho para o arquivo de dados
        """
        self.data_path = data_path or RAW_DATA_FILE
        self.df = None
        self.id_mapper = IDMappingUtils()
        
        # Carregar mapeamentos de IDs
        try:
            self.id_mapper.load_mappings()
            print("Mapeamentos de IDs carregados para análise exploratória!")
        except Exception as e:
            print(f"Aviso: Não foi possível carregar mapeamentos de IDs: {e}")
            self.id_mapper = None
        
    def load_data(self):
        """Carrega os dados do arquivo CSV"""
        print(f"Carregando dados de: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"Dados carregados. Shape: {self.df.shape}")
        return self.df
    
    def basic_info(self):
        """Exibe informações básicas sobre o dataset"""
        print("\n" + "="*60)
        print("INFORMAÇÕES BÁSICAS DO DATASET")
        print("="*60)
        
        print(f"\nTamanho do dataset: {self.df.shape}")
        print(f"Número de colunas: {self.df.shape[1]}")
        print(f"Número de registros: {self.df.shape[0]}")
        print(f"Número de pacientes únicos: {self.df['patient_nbr'].nunique()}")
        print(f"Número de encontros: {len(self.df)}")
        
        print(f"\nInformações dos tipos de dados:")
        print(self.df.info())
        
        print(f"\nPrimeiras 5 linhas:")
        print(self.df.head())
        
    def analyze_target_variable(self):
        """Analisa a variável alvo 'readmitted'"""
        print("\n" + "="*60)
        print("ANÁLISE DA VARIÁVEL ALVO")
        print("="*60)
        
        print("\nDistribuição da variável 'readmitted':")
        readmitted_counts = self.df['readmitted'].value_counts()
        print(readmitted_counts)
        
        # Criar variável alvo binária
        self.df['target'] = self.df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
        
        print(f"\nDistribuição da variável alvo binária:")
        target_counts = self.df['target'].value_counts()
        print(target_counts)
        print(f"Proporção de readmissões em <30 dias: {self.df['target'].mean():.3f}")
        
        # Verificar valores nulos
        print(f"\nValores nulos em 'readmitted': {self.df['readmitted'].isnull().sum()}")
        print(f"Valores nulos em 'target': {self.df['target'].isnull().sum()}")
        
        return self.df['target']
    
    def analyze_missing_data(self):
        """Analisa dados faltantes no dataset"""
        print("\n" + "="*60)
        print("ANÁLISE DE DADOS FALTANTES")
        print("="*60)
        
        # Contar valores '?' (que representam dados faltantes)
        missing_summary = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                missing_count = (self.df[col] == '?').sum()
                missing_pct = (missing_count / len(self.df)) * 100
                if missing_count > 0:
                    missing_summary.append({
                        'Coluna': col,
                        'Valores Faltantes': missing_count,
                        'Percentual': f"{missing_pct:.1f}%"
                    })
        
        if missing_summary:
            missing_df = pd.DataFrame(missing_summary)
            missing_df = missing_df.sort_values('Valores Faltantes', ascending=False)
            print("\nColunas com dados faltantes ('?'):")
            print(missing_df.to_string(index=False))
        else:
            print("\nNenhum dado faltante encontrado.")
        
        # Verificar valores nulos tradicionais
        null_counts = self.df.isnull().sum()
        if null_counts.sum() > 0:
            print(f"\nValores nulos (NaN) por coluna:")
            print(null_counts[null_counts > 0])
        else:
            print(f"\nNenhum valor nulo (NaN) encontrado.")
    
    def analyze_demographic_data(self):
        """Analisa dados demográficos"""
        print("\n" + "="*60)
        print("ANÁLISE DEMOGRÁFICA")
        print("="*60)
        
        # Distribuição por raça
        print("\nDistribuição por raça:")
        race_counts = self.df['race'].value_counts()
        print(race_counts)
        
        # Distribuição por gênero
        print(f"\nDistribuição por gênero:")
        gender_counts = self.df['gender'].value_counts()
        print(gender_counts)
        
        # Distribuição por faixa etária
        print(f"\nDistribuição por faixa etária:")
        age_counts = self.df['age'].value_counts()
        print(age_counts)
        
        # Readmissão por demografia
        if 'target' in self.df.columns:
            print(f"\nTaxa de readmissão por raça:")
            race_readmission = self.df.groupby('race')['target'].agg(['count', 'mean'])
            race_readmission.columns = ['Total_Pacientes', 'Taxa_Readmissao']
            race_readmission['Taxa_Readmissao'] = race_readmission['Taxa_Readmissao'].round(3)
            print(race_readmission)
            
            print(f"\nTaxa de readmissão por gênero:")
            gender_readmission = self.df.groupby('gender')['target'].agg(['count', 'mean'])
            gender_readmission.columns = ['Total_Pacientes', 'Taxa_Readmissao']
            gender_readmission['Taxa_Readmissao'] = gender_readmission['Taxa_Readmissao'].round(3)
            print(gender_readmission)
    
    def analyze_medical_data(self):
        """Analisa dados médicos"""
        print("\n" + "="*60)
        print("ANÁLISE DE DADOS MÉDICOS")
        print("="*60)
        
        # Estatísticas de internação
        print(f"\nEstatísticas de tempo de internação:")
        print(f"Média: {self.df['time_in_hospital'].mean():.2f} dias")
        print(f"Mediana: {self.df['time_in_hospital'].median():.2f} dias")
        print(f"Mínimo: {self.df['time_in_hospital'].min()} dias")
        print(f"Máximo: {self.df['time_in_hospital'].max()} dias")
        
        # Procedimentos e medicamentos
        print(f"\nEstatísticas de procedimentos:")
        print(f"Procedimentos laboratoriais - Média: {self.df['num_lab_procedures'].mean():.2f}")
        print(f"Procedimentos - Média: {self.df['num_procedures'].mean():.2f}")
        print(f"Medicamentos - Média: {self.df['num_medications'].mean():.2f}")
        
        # Diagnósticos
        print(f"\nNúmero de diagnósticos:")
        print(f"Média: {self.df['number_diagnoses'].mean():.2f}")
        print(f"Distribuição:")
        print(self.df['number_diagnoses'].value_counts().sort_index())
        
        # Especialidades médicas mais comuns
        print(f"\nEspecialidades médicas mais comuns:")
        specialty_counts = self.df['medical_specialty'].value_counts().head(10)
        print(specialty_counts)
    
    def analyze_id_mappings(self):
        """Analisa dados com mapeamentos de IDs aplicados"""
        print("\n" + "="*60)
        print("ANÁLISE DE MAPEAMENTOS DE IDs")
        print("="*60)
        
        if self.id_mapper is None:
            print("Mapeamentos de IDs não disponíveis.")
            return
        
        # Aplicar mapeamentos temporariamente para análise
        df_with_mappings = self.id_mapper.apply_mappings_to_dataframe(self.df)
        
        # Analisar admission_type
        if 'admission_type_desc' in df_with_mappings.columns:
            print("\nTIPOS DE ADMISSÃO:")
            admission_analysis = df_with_mappings.groupby('admission_type_desc').agg({
                'target': ['count', 'mean']
            }).round(3)
            admission_analysis.columns = ['Total_Casos', 'Taxa_Readmissao']
            print(admission_analysis.sort_values('Taxa_Readmissao', ascending=False))
        
        # Analisar discharge_disposition
        if 'discharge_disposition_desc' in df_with_mappings.columns:
            print("\nDISPOSIÇÃO DE ALTA:")
            discharge_analysis = df_with_mappings.groupby('discharge_disposition_desc').agg({
                'target': ['count', 'mean']
            }).round(3)
            discharge_analysis.columns = ['Total_Casos', 'Taxa_Readmissao']
            # Mostrar apenas os 10 mais comuns
            top_discharges = discharge_analysis.sort_values('Total_Casos', ascending=False).head(10)
            print(top_discharges)
        
        # Analisar admission_source
        if 'admission_source_desc' in df_with_mappings.columns:
            print("\nFONTE DE ADMISSÃO:")
            source_analysis = df_with_mappings.groupby('admission_source_desc').agg({
                'target': ['count', 'mean']
            }).round(3)
            source_analysis.columns = ['Total_Casos', 'Taxa_Readmissao']
            # Mostrar apenas os 10 mais comuns
            top_sources = source_analysis.sort_values('Total_Casos', ascending=False).head(10)
            print(top_sources)
        
        # Análise cruzada dos mapeamentos
        if all(col in df_with_mappings.columns for col in ['admission_type_desc', 'discharge_disposition_desc']):
            print("\nANÁLISE CRUZADA - TIPO DE ADMISSÃO x DISPOSIÇÃO DE ALTA:")
            cross_analysis = pd.crosstab(
                df_with_mappings['admission_type_desc'],
                df_with_mappings['discharge_disposition_desc'],
                margins=True
            )
            print(cross_analysis.head())
    
    def analyze_medication_data(self):
        """Analisa dados de medicamentos"""
        print("\n" + "="*60)
        print("ANÁLISE DE MEDICAMENTOS")
        print("="*60)
        
        # Identificar colunas de medicamentos
        medication_cols = [col for col in self.df.columns if col not in [
            'encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'weight',
            'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
            'time_in_hospital', 'payer_code', 'medical_specialty', 'num_lab_procedures',
            'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency',
            'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
            'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed', 'readmitted', 'target'
        ]]
        
        print(f"Medicamentos analisados: {len(medication_cols)}")
        
        # Contar uso de medicamentos
        medication_usage = {}
        for med in medication_cols:
            usage_count = (self.df[med] != 'No').sum()
            medication_usage[med] = usage_count
        
        # Top medicamentos mais prescritos
        sorted_meds = sorted(medication_usage.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop 10 medicamentos mais prescritos:")
        for i, (med, count) in enumerate(sorted_meds[:10], 1):
            percentage = (count / len(self.df)) * 100
            print(f"{i:2d}. {med:20s}: {count:5d} ({percentage:5.1f}%)")
        
        # Análise de mudança de medicação
        print(f"\nMudança de medicação:")
        change_counts = self.df['change'].value_counts()
        print(change_counts)
        
        # Uso de medicação diabética
        print(f"\nUso de medicação diabética:")
        diabetes_med_counts = self.df['diabetesMed'].value_counts()
        print(diabetes_med_counts)
    
    def run_complete_analysis(self):
        """Executa análise exploratória completa"""
        print("\n" + "="*80)
        print("ANÁLISE EXPLORATÓRIA DE DADOS - DATASET DIABÉTICO")
        print("="*80)
        
        if self.df is None:
            self.load_data()
        
        self.basic_info()
        self.analyze_target_variable()
        self.analyze_missing_data()
        self.analyze_demographic_data()
        self.analyze_medical_data()
        self.analyze_id_mappings()  # Nova análise de mapeamentos
        self.analyze_medication_data()
        
        print("\n" + "="*80)
        print("ANÁLISE EXPLORATÓRIA CONCLUÍDA")
        print("="*80)
        
        return self.df


def main():
    """Função principal para executar a análise exploratória"""
    eda = ExploratoryDataAnalysis()
    df_analyzed = eda.run_complete_analysis()
    return df_analyzed


if __name__ == "__main__":
    main()
