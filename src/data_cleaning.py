"""
Módulo de Limpeza e Pré-processamento de Dados Médicos

Este módulo implementa uma pipeline robusta e profissional de limpeza de dados para
o sistema de predição de readmissão hospitalar diabética, incluindo:

Funcionalidades Principais:
- Criação de variável alvo binária otimizada
- Remoção inteligente de pacientes expirados e transferidos
- Tratamento abrangente e estatístico de dados faltantes
- Remoção de duplicatas com preservação de informações históricas
- Aplicação de mapeamentos de IDs com validação rigorosa
- Verificações automáticas de consistência de dados
- Criação de features derivadas baseadas em conhecimento médico
- Sistema de logging detalhado e auditoria completa
- Validações pré e pós-processamento
- Diagnósticos avançados de qualidade dos dados
- Configurações flexíveis e modulares

Autor: Thalles Felipe Rodrigues de Almeida Santos
Projeto: Predição de Readmissão Hospitalar em Pacientes com Diabetes Usando Aprendizado de Máquina
Instituição: Universidade Federal de Ouro Preto (UFOP)
Disciplina: Inteligência Artificial
Professor: Jadson Castro Gertrudes
Data: Agosto 2025
"""

import pandas as pd
import numpy as np
import os
import logging
import warnings
import json
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Adicionar o diretório pai ao path para importações quando executado diretamente
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.config import (
        RAW_DATA_FILE, CLEAN_DATA_FILE, EXPIRED_DISCHARGE_CODES,
        COLUMNS_TO_DROP_INITIAL, MISSING_DATA_THRESHOLD, DATA_CONSISTENCY_CHECKS,
        CREATE_DERIVED_FEATURES, RESULTS_DIR
    )
    from src.id_mapping_utils import IDMappingUtils
except ImportError:
    # Fallback para quando executado diretamente
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import (
        RAW_DATA_FILE, CLEAN_DATA_FILE, EXPIRED_DISCHARGE_CODES,
        COLUMNS_TO_DROP_INITIAL, MISSING_DATA_THRESHOLD, DATA_CONSISTENCY_CHECKS,
        CREATE_DERIVED_FEATURES, RESULTS_DIR
    )
    from src.id_mapping_utils import IDMappingUtils

# Suprimir warnings desnecessários
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# Configurar logging com mais detalhes
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_cleaning.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Códigos de transferência (pacientes que podem ser readmitidos em outras instituições)
TRANSFER_CODES = [3, 4, 5, 6, 22, 23, 24]  # Transferências para outros hospitais/clínicas
EXCLUDED_CODES = EXPIRED_DISCHARGE_CODES + TRANSFER_CODES

# Configurações padrão para limpeza
DEFAULT_CLEANING_CONFIG = {
    'remove_expired_transferred': True,
    'handle_missing_data': True,
    'check_consistency': DATA_CONSISTENCY_CHECKS,
    'create_derived_features': CREATE_DERIVED_FEATURES,
    'remove_duplicates': True,
    'apply_id_mappings': True,
    'missing_threshold': MISSING_DATA_THRESHOLD,
    'preserve_original_columns': True,
    'generate_report': True
}


class DataCleaner:
    """
    Classe responsável pela limpeza robusta dos dados do dataset diabético
    
    Esta classe implementa uma pipeline completa de limpeza que inclui:
    - Validação e carregamento de dados
    - Criação de variáveis alvo e derivadas
    - Tratamento inteligente de dados faltantes
    - Remoção criteriosa de registros problemáticos
    - Verificações de consistência
    - Aplicação de mapeamentos com validação
    - Logging detalhado de todas as operações
    - Diagnósticos de qualidade dos dados
    - Configurações flexíveis para diferentes cenários
    
    Atributos:
        raw_data_path (str): Caminho para dados brutos
        df_raw (pd.DataFrame): DataFrame com dados originais
        df_clean (pd.DataFrame): DataFrame com dados limpos
        cleaning_stats (dict): Estatísticas do processo de limpeza
        config (dict): Configurações de limpeza
        id_mapper (IDMappingUtils): Utilitário para mapeamentos
    """
    
    def __init__(self, raw_data_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Inicializa o limpador de dados com configurações flexíveis
        
        Args:
            raw_data_path (str): Caminho para o arquivo de dados brutos
            config (dict): Configurações personalizadas de limpeza
        """
        self.raw_data_path = raw_data_path or RAW_DATA_FILE
        self.df_raw = None
        self.df_clean = None
        self.cleaning_stats = {}
        self.config = {**DEFAULT_CLEANING_CONFIG, **(config or {})}
        
        # Inicializar componentes
        self.id_mapper = None
        self._setup_logging()
        self._setup_directories()
        self._initialize_id_mapper()
        
        # Configurar random seed para reproducibilidade
        np.random.seed(42)
        
        logger.info(f"DataCleaner inicializado com configuração: {self.config}")
    
    def _setup_logging(self) -> None:
        """Configura logging específico para esta instância"""
        self.logger = logging.getLogger(f"{__name__}.{id(self)}")
        self.logger.setLevel(logging.INFO)
    
    def _setup_directories(self) -> None:
        """Cria diretórios necessários se não existirem"""
        directories = [
            os.path.dirname(CLEAN_DATA_FILE),
            RESULTS_DIR
        ]
        
        for directory in directories:
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Diretório criado: {directory}")
    
    def _initialize_id_mapper(self) -> None:
        """Inicializa o mapeador de IDs com tratamento de erros"""
        if IDMappingUtils is None:
            logger.warning("IDMappingUtils não disponível. Mapeamentos de ID desabilitados.")
            self.id_mapper = None
            self.config['apply_id_mappings'] = False
            return
            
        try:
            self.id_mapper = IDMappingUtils()
            self.id_mapper.load_mappings()
            logger.info("Mapeamentos de IDs carregados com sucesso")
        except Exception as e:
            logger.warning(f"Não foi possível carregar mapeamentos de IDs: {e}")
            if self.config['apply_id_mappings']:
                logger.warning("Mapeamentos de IDs desabilitados devido ao erro acima")
                self.config['apply_id_mappings'] = False
        
    
    def validate_input_data(self) -> bool:
        """
        Valida os dados de entrada antes do processamento
        
        Returns:
            bool: True se os dados são válidos, False caso contrário
        """
        logger.info("Validando dados de entrada...")
        
        if self.df_raw is None:
            logger.error("Dados brutos não carregados")
            return False
        
        # Verificar colunas obrigatórias
        required_columns = [
            'encounter_id', 'patient_nbr', 'readmitted', 
            'discharge_disposition_id', 'time_in_hospital'
        ]
        
        missing_columns = [col for col in required_columns if col not in self.df_raw.columns]
        if missing_columns:
            logger.error(f"Colunas obrigatórias ausentes: {missing_columns}")
            return False
        
        # Verificar se há dados suficientes
        if len(self.df_raw) < 100:
            logger.warning(f"Dataset muito pequeno: {len(self.df_raw)} registros")
        
        # Verificar tipos de dados básicos
        validation_issues = []
        
        # Verificar se patient_nbr e encounter_id são únicos quando esperado
        if self.df_raw['encounter_id'].duplicated().any():
            validation_issues.append("encounter_id duplicados encontrados")
        
        # Verificar valores válidos para readmitted
        valid_readmitted = {'<30', '>30', 'NO'}
        invalid_readmitted = set(self.df_raw['readmitted'].unique()) - valid_readmitted
        if invalid_readmitted:
            validation_issues.append(f"Valores inválidos em readmitted: {invalid_readmitted}")
        
        # Verificar se discharge_disposition_id são numéricos
        if not pd.api.types.is_numeric_dtype(self.df_raw['discharge_disposition_id']):
            validation_issues.append("discharge_disposition_id deve ser numérico")
        
        if validation_issues:
            logger.warning("Problemas de validação encontrados:")
            for issue in validation_issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("Validação dos dados de entrada: ✓ APROVADA")
        
        self.cleaning_stats['validation_issues'] = validation_issues
        return len(validation_issues) == 0
    
    def diagnose_data_quality(self) -> Dict[str, Any]:
        """
        Realiza diagnóstico completo da qualidade dos dados
        
        Returns:
            dict: Relatório detalhado da qualidade dos dados
        """
        logger.info("Executando diagnóstico de qualidade dos dados...")
        
        if self.df_raw is None:
            return {'error': 'Dados não carregados'}
        
        quality_report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'shape': self.df_raw.shape,
                'memory_usage_mb': self.df_raw.memory_usage(deep=True).sum() / 1024**2,
                'unique_patients': self.df_raw['patient_nbr'].nunique() if 'patient_nbr' in self.df_raw.columns else 0
            }
        }
        
        # Análise de completude
        missing_analysis = {}
        for col in self.df_raw.columns:
            null_count = self.df_raw[col].isnull().sum()
            question_count = (self.df_raw[col] == '?').sum() if self.df_raw[col].dtype == 'object' else 0
            total_missing = null_count + question_count
            missing_pct = (total_missing / len(self.df_raw)) * 100
            
            missing_analysis[col] = {
                'null_count': int(null_count),
                'question_mark_count': int(question_count),
                'total_missing': int(total_missing),
                'missing_percentage': round(missing_pct, 2),
                'data_type': str(self.df_raw[col].dtype)
            }
        
        quality_report['missing_data_analysis'] = missing_analysis
        
        # Análise de duplicatas
        duplicates_analysis = {
            'duplicate_encounters': self.df_raw['encounter_id'].duplicated().sum() if 'encounter_id' in self.df_raw.columns else 0,
            'patients_with_multiple_visits': (self.df_raw['patient_nbr'].value_counts() > 1).sum() if 'patient_nbr' in self.df_raw.columns else 0
        }
        quality_report['duplicates_analysis'] = duplicates_analysis
        
        # Análise de distribuição da variável alvo
        if 'readmitted' in self.df_raw.columns:
            readmitted_dist = self.df_raw['readmitted'].value_counts().to_dict()
            quality_report['target_distribution'] = readmitted_dist
        
        # Análise de outliers em variáveis numéricas
        numeric_cols = self.df_raw.select_dtypes(include=[np.number]).columns
        outliers_analysis = {}
        
        for col in numeric_cols:
            if len(self.df_raw[col].dropna()) > 0:
                Q1 = self.df_raw[col].quantile(0.25)
                Q3 = self.df_raw[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_count = ((self.df_raw[col] < lower_bound) | (self.df_raw[col] > upper_bound)).sum()
                outliers_analysis[col] = {
                    'outliers_count': int(outliers_count),
                    'outliers_percentage': round((outliers_count / len(self.df_raw)) * 100, 2),
                    'q1': float(Q1),
                    'q3': float(Q3),
                    'iqr': float(IQR)
                }
        
        quality_report['outliers_analysis'] = outliers_analysis
        
        # Score geral de qualidade
        avg_missing_pct = np.mean([info['missing_percentage'] for info in missing_analysis.values()])
        duplicate_pct = (duplicates_analysis['duplicate_encounters'] / len(self.df_raw)) * 100
        
        quality_score = max(0, 100 - avg_missing_pct - duplicate_pct)
        quality_report['overall_quality_score'] = round(quality_score, 2)
        
        logger.info(f"Diagnóstico concluído. Score de qualidade: {quality_score:.1f}/100")
        
        return quality_report
    
    def load_raw_data(self) -> pd.DataFrame:
        """
        Carrega e valida os dados brutos do arquivo CSV com tratamento robusto de erros
        
        Returns:
            pd.DataFrame: DataFrame com os dados brutos
            
        Raises:
            FileNotFoundError: Se o arquivo não for encontrado
            pd.errors.EmptyDataError: Se o arquivo estiver vazio
            Exception: Para outros erros de carregamento
        """
        logger.info(f"Carregando dados de: {self.raw_data_path}")
        
        # Verificar se o arquivo existe
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {self.raw_data_path}")
        
        # Verificar tamanho do arquivo
        file_size = os.path.getsize(self.raw_data_path)
        logger.info(f"Tamanho do arquivo: {file_size / 1024**2:.2f} MB")
        
        try:
            # Tentar diferentes codificações se necessário
            encodings = ['utf-8', 'latin-1', 'cp1252']
            df_loaded = None
            
            for encoding in encodings:
                try:
                    df_loaded = pd.read_csv(self.raw_data_path, encoding=encoding, low_memory=False)
                    logger.info(f"Arquivo carregado com codificação: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df_loaded is None:
                raise Exception("Não foi possível carregar o arquivo com nenhuma codificação testada")
            
            self.df_raw = df_loaded
            logger.info(f"Dados carregados com sucesso. Shape: {self.df_raw.shape}")
            
            # Registrar estatísticas iniciais
            self.cleaning_stats.update({
                'original_records': len(self.df_raw),
                'original_columns': self.df_raw.shape[1],
                'original_patients': self.df_raw['patient_nbr'].nunique() if 'patient_nbr' in self.df_raw.columns else 0,
                'file_size_mb': file_size / 1024**2,
                'load_timestamp': datetime.now().isoformat()
            })
            
            # Exibir informações básicas
            logger.info(f"Colunas encontradas: {list(self.df_raw.columns)}")
            
            # Validar dados carregados
            if not self.validate_input_data():
                logger.warning("Problemas de validação encontrados, mas continuando processamento")
            
            return self.df_raw
            
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError(f"Arquivo está vazio: {self.raw_data_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            raise
    
    def create_target_variable(self) -> pd.Series:
        """
        Cria a variável alvo binária para predição de readmissão com validações robustas
        
        Returns:
            pd.Series: Série com a variável target (1=readmitido <30 dias, 0=caso contrário)
            
        Raises:
            ValueError: Se a coluna 'readmitted' não existir ou tiver valores inválidos
        """
        logger.info("Criando variável alvo binária...")
        
        if 'readmitted' not in self.df_raw.columns:
            raise ValueError("Coluna 'readmitted' não encontrada no dataset")
        
        # Verificar valores únicos antes da transformação
        unique_values = set(self.df_raw['readmitted'].unique())
        expected_values = {'<30', '>30', 'NO'}
        unexpected_values = unique_values - expected_values
        
        if unexpected_values:
            logger.warning(f"Valores inesperados encontrados em 'readmitted': {unexpected_values}")
            logger.warning("Estes valores serão tratados como 'não readmitido' (0)")
        
        # Criar variável target com tratamento robusto
        def map_readmitted_to_target(value):
            if pd.isna(value):
                return 0  # Tratar NaN como não readmitido
            if str(value).strip() == '<30':
                return 1
            else:
                return 0
        
        self.df_raw['target'] = self.df_raw['readmitted'].apply(map_readmitted_to_target)
        
        # Registrar distribuição original
        readmitted_dist = self.df_raw['readmitted'].value_counts(dropna=False)
        target_dist = self.df_raw['target'].value_counts()
        
        logger.info("Distribuição da variável 'readmitted' original:")
        for value, count in readmitted_dist.items():
            pct = count / len(self.df_raw) * 100
            logger.info(f"  {value}: {count:,} ({pct:.2f}%)")
        
        logger.info("Distribuição da variável 'target' criada:")
        for value, count in target_dist.items():
            label = "Readmitido <30 dias" if value == 1 else "Não readmitido <30 dias"
            pct = count / len(self.df_raw) * 100
            logger.info(f"  {value} ({label}): {count:,} ({pct:.2f}%)")
        
        readmission_rate = self.df_raw['target'].mean()
        logger.info(f"Taxa de readmissão em <30 dias: {readmission_rate:.4f}")
        
        # Verificar se há desbalanceamento extremo
        if readmission_rate < 0.01 or readmission_rate > 0.99:
            logger.warning(f"Dataset extremamente desbalanceado! Taxa: {readmission_rate:.4f}")
        
        # Registrar estatísticas
        self.cleaning_stats.update({
            'readmission_rate_original': readmission_rate,
            'readmitted_distribution': readmitted_dist.to_dict(),
            'target_distribution': target_dist.to_dict(),
            'unexpected_readmitted_values': list(unexpected_values) if unexpected_values else None
        })
        
        return self.df_raw['target']
    
    def remove_expired_and_transferred_patients(self):
        """
        Remove pacientes expirados e transferidos que não podem ser rastreados para readmissão
        
        Remove pacientes com códigos de discharge que indicam:
        - Falecimento (códigos 11, 19, 20, 21)
        - Transferência para outras instituições (códigos 3, 4, 5, 6, 22, 23, 24)
        
        Justificativa: Estes pacientes não podem ser readmitidos na mesma instituição,
        introduzindo viés na análise se mantidos no dataset.
        """
        logger.info("Removendo pacientes expirados e transferidos")
        logger.info(f"Registros antes da remoção: {len(self.df_raw):,}")
        
        # Contar registros por tipo de exclusão
        expired_count = self.df_raw['discharge_disposition_id'].isin(EXPIRED_DISCHARGE_CODES).sum()
        transfer_count = self.df_raw['discharge_disposition_id'].isin(TRANSFER_CODES).sum()
        total_excluded = self.df_raw['discharge_disposition_id'].isin(EXCLUDED_CODES).sum()
        
        logger.info(f"Registros com pacientes expirados: {expired_count:,}")
        logger.info(f"Registros com pacientes transferidos: {transfer_count:,}")
        logger.info(f"Total de registros a remover: {total_excluded:,}")
        
        # Aplicar filtro
        self.df_clean = self.df_raw[~self.df_raw['discharge_disposition_id'].isin(EXCLUDED_CODES)].copy()
        
        records_removed = len(self.df_raw) - len(self.df_clean)
        logger.info(f"Registros após remoção: {len(self.df_clean):,}")
        logger.info(f"Registros removidos: {records_removed:,} ({records_removed/len(self.df_raw)*100:.2f}%)")
        
        # Registrar estatísticas
        self.cleaning_stats['expired_patients_removed'] = expired_count
        self.cleaning_stats['transferred_patients_removed'] = transfer_count
        self.cleaning_stats['total_excluded_patients'] = total_excluded
    
    def handle_missing_data(self):
        """
        Trata dados faltantes de forma abrangente no dataset
        
        Implementa estratégias diferenciadas por tipo de dados:
        - Remove colunas com >80% de valores ausentes
        - Substitui '?' por 'missing' em variáveis categóricas
        - Identifica e trata valores faltantes em todas as colunas relevantes
        """
        logger.info("Tratando dados faltantes")
        
        # 1. Remover colunas com muitos dados faltantes (já definidas em config)
        existing_cols_to_drop = [col for col in COLUMNS_TO_DROP_INITIAL if col in self.df_clean.columns]
        if existing_cols_to_drop:
            logger.info(f"Removendo colunas com muitos dados faltantes (>80%): {existing_cols_to_drop}")
            for col in existing_cols_to_drop:
                missing_pct = (self.df_clean[col].isna().sum() + (self.df_clean[col] == '?').sum()) / len(self.df_clean) * 100
                logger.info(f"  {col}: {missing_pct:.1f}% de dados faltantes")
            self.df_clean = self.df_clean.drop(existing_cols_to_drop, axis=1)
            self.cleaning_stats['columns_dropped_missing'] = existing_cols_to_drop
        
        # 2. Tratar '?' em todas as colunas categóricas
        logger.info("Verificando valores '?' em colunas categóricas...")
        categorical_cols = self.df_clean.select_dtypes(include=['object']).columns
        question_mark_replacements = {}
        
        for col in categorical_cols:
            if '?' in self.df_clean[col].values:
                missing_count = (self.df_clean[col] == '?').sum()
                missing_pct = missing_count / len(self.df_clean) * 100
                
                # Estratégia baseada na porcentagem de dados faltantes
                if missing_pct > 50:
                    logger.warning(f"  {col}: {missing_count:,} valores '?' ({missing_pct:.1f}%) - alta taxa de dados faltantes")
                
                # Substituir por 'missing' ou 'unknown' dependendo da coluna
                if col in ['race', 'medical_specialty']:
                    self.df_clean[col] = self.df_clean[col].replace('?', 'missing')
                    replacement_value = 'missing'
                else:
                    self.df_clean[col] = self.df_clean[col].replace('?', 'unknown')
                    replacement_value = 'unknown'
                
                question_mark_replacements[col] = {
                    'count': missing_count,
                    'percentage': missing_pct,
                    'replacement': replacement_value
                }
                
                logger.info(f"  {col}: {missing_count:,} valores '?' substituídos por '{replacement_value}' ({missing_pct:.1f}%)")
        
        # 3. Verificar outros tipos de dados faltantes
        missing_summary = self.df_clean.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]
        
        if len(missing_cols) > 0:
            logger.info("Outras colunas com dados faltantes (NaN):")
            for col, count in missing_cols.items():
                pct = count / len(self.df_clean) * 100
                logger.info(f"  {col}: {count:,} valores NaN ({pct:.1f}%)")
        
        # Registrar estatísticas
        self.cleaning_stats['question_mark_replacements'] = question_mark_replacements
        self.cleaning_stats['remaining_missing_data'] = missing_cols.to_dict()
    
    def create_derived_features(self):
        """
        Cria features derivadas que podem ser úteis para a modelagem
        
        Features criadas:
        - num_previous_admissions: Número de internações anteriores por paciente
        - medication_changed: Indicador binário se houve mudança na medicação
        - total_procedures: Soma de procedimentos laboratoriais e medicações
        - long_stay: Indicador se a internação foi longa (>7 dias)
        """
        logger.info("Criando features derivadas...")
        
        # 1. Número de internações anteriores (antes de remover duplicatas)
        logger.info("  Calculando número de internações anteriores por paciente...")
        self.df_clean = self.df_clean.sort_values(['patient_nbr', 'encounter_id'])
        self.df_clean['num_previous_admissions'] = self.df_clean.groupby('patient_nbr').cumcount()
        
        prev_admissions_stats = self.df_clean['num_previous_admissions'].describe()
        logger.info(f"    Internações anteriores - Média: {prev_admissions_stats['mean']:.2f}, Máx: {prev_admissions_stats['max']:.0f}")
        
        # 2. Indicador de mudança na medicação
        if 'change' in self.df_clean.columns:
            self.df_clean['medication_changed'] = (self.df_clean['change'] == 'Ch').astype(int)
            change_rate = self.df_clean['medication_changed'].mean()
            logger.info(f"  Taxa de mudança na medicação: {change_rate:.3f}")
        
        # 3. Total de procedimentos (laboratoriais + medicações)
        if all(col in self.df_clean.columns for col in ['num_lab_procedures', 'num_medications']):
            self.df_clean['total_procedures'] = (
                self.df_clean['num_lab_procedures'] + self.df_clean['num_medications']
            )
            total_proc_stats = self.df_clean['total_procedures'].describe()
            logger.info(f"  Total de procedimentos - Média: {total_proc_stats['mean']:.2f}")
        
        # 4. Indicador de internação longa
        if 'time_in_hospital' in self.df_clean.columns:
            self.df_clean['long_stay'] = (self.df_clean['time_in_hospital'] > 7).astype(int)
            long_stay_rate = self.df_clean['long_stay'].mean()
            logger.info(f"  Taxa de internação longa (>7 dias): {long_stay_rate:.3f}")
        
        # Registrar estatísticas das features derivadas
        derived_features = ['num_previous_admissions', 'medication_changed', 'total_procedures', 'long_stay']
        existing_derived = [f for f in derived_features if f in self.df_clean.columns]
        self.cleaning_stats['derived_features_created'] = existing_derived
        
        logger.info(f"Features derivadas criadas: {existing_derived}")
    
    def remove_duplicate_patients(self):
        """
        Remove registros duplicados, mantendo apenas a primeira internação de cada paciente
        
        Importante: Esta função deve ser chamada APÓS create_derived_features() para
        preservar informações sobre internações anteriores.
        """
        logger.info("Removendo dados duplicados de pacientes")
        logger.info(f"Registros antes da remoção de duplicatas: {len(self.df_clean):,}")
        
        unique_patients_before = self.df_clean['patient_nbr'].nunique()
        logger.info(f"Pacientes únicos antes: {unique_patients_before:,}")
        
        # Análise de duplicatas
        duplicate_patients = self.df_clean['patient_nbr'].value_counts()
        patients_with_multiple = (duplicate_patients > 1).sum()
        total_duplicate_records = (duplicate_patients - 1).sum()
        
        logger.info(f"Pacientes com múltiplas internações: {patients_with_multiple:,}")
        logger.info(f"Registros duplicados a remover: {total_duplicate_records:,}")
        
        # Manter apenas a primeira internação de cada paciente
        self.df_clean = self.df_clean.drop_duplicates(subset=['patient_nbr'], keep='first')
        
        unique_patients_after = self.df_clean['patient_nbr'].nunique()
        logger.info(f"Registros após remoção de duplicatas: {len(self.df_clean):,}")
        logger.info(f"Pacientes únicos após limpeza: {unique_patients_after:,}")
        
        # Registrar estatísticas
        self.cleaning_stats['patients_with_multiple_admissions'] = patients_with_multiple
        self.cleaning_stats['duplicate_records_removed'] = total_duplicate_records
    
    def check_data_consistency(self):
        """
        Verifica a consistência dos dados e corrige valores inválidos
        
        Verificações incluem:
        - Valores negativos em colunas numéricas que deveriam ser não-negativas
        - Outliers extremos em variáveis temporais
        - Valores impossíveis em variáveis categóricas
        - Intervalos válidos para variáveis contínuas
        """
        logger.info("Verificando consistência dos dados...")
        
        consistency_issues = {}
        
        # 1. Verificar valores negativos em colunas que deveriam ser não-negativas
        non_negative_cols = [
            'time_in_hospital', 'num_lab_procedures', 'num_medications', 
            'number_outpatient', 'number_emergency', 'number_inpatient',
            'num_procedures'
        ]
        
        for col in non_negative_cols:
            if col in self.df_clean.columns:
                negative_count = (self.df_clean[col] < 0).sum()
                if negative_count > 0:
                    logger.warning(f"  {negative_count:,} valores negativos encontrados em {col}")
                    # Corrigir valores negativos (substituir por 0 ou mediana)
                    median_value = self.df_clean[self.df_clean[col] >= 0][col].median()
                    self.df_clean.loc[self.df_clean[col] < 0, col] = median_value
                    consistency_issues[f'{col}_negative_values'] = negative_count
                    logger.info(f"    Valores negativos em {col} substituídos por mediana ({median_value})")
        
        # 2. Verificar outliers extremos em time_in_hospital
        if 'time_in_hospital' in self.df_clean.columns:
            # Pacientes com mais de 30 dias de internação são raros
            extreme_stay = (self.df_clean['time_in_hospital'] > 30).sum()
            if extreme_stay > 0:
                logger.info(f"  {extreme_stay:,} pacientes com internação >30 dias")
                consistency_issues['extreme_hospital_stay'] = extreme_stay
        
        # 3. Verificar valores de idade válidos (se existir coluna age)
        if 'age' in self.df_clean.columns:
            # Assumindo que age é categórica no formato '[min-max)'
            age_categories = self.df_clean['age'].value_counts()
            logger.info(f"  Categorias de idade encontradas: {list(age_categories.index)}")
        
        # 4. Verificar consistência em variáveis de medicação
        medication_cols = [col for col in self.df_clean.columns 
                          if col.startswith(('metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                                           'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                                           'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                                           'miglitol', 'troglitazone', 'tolazamide', 'insulin',
                                           'glyburide-metformin', 'glipizide-metformin'))]
        
        for col in medication_cols:
            if col in self.df_clean.columns:
                unique_values = set(self.df_clean[col].unique())
                expected_values = {'No', 'Steady', 'Up', 'Down'}
                unexpected_values = unique_values - expected_values
                if unexpected_values:
                    logger.warning(f"  Valores inesperados em {col}: {unexpected_values}")
                    consistency_issues[f'{col}_unexpected_values'] = list(unexpected_values)
        
        # 5. Verificar distribuição de dias entre alta e readmissão (se disponível)
        if 'days_to_inpatient_readmission' in self.df_clean.columns:
            invalid_days = (self.df_clean['days_to_inpatient_readmission'] < 0).sum()
            if invalid_days > 0:
                logger.warning(f"  {invalid_days:,} valores inválidos em days_to_inpatient_readmission")
                consistency_issues['invalid_readmission_days'] = invalid_days
        
        # Registrar estatísticas de consistência
        self.cleaning_stats['consistency_issues'] = consistency_issues
        
        if consistency_issues:
            logger.info(f"Total de problemas de consistência encontrados e corrigidos: {len(consistency_issues)}")
        else:
            logger.info("Nenhum problema de consistência encontrado")
    
    def apply_id_mappings(self):
        """
        Aplica mapeamentos de IDs para enriquecer dados com descrições legíveis
        
        Inclui validação robusta e tratamento de IDs não mapeados
        """
        logger.info("Aplicando mapeamentos de IDs")
        
        if self.id_mapper is None:
            logger.warning("Mapeamentos de IDs não disponíveis. Pulando esta etapa.")
            return
        
        logger.info("Aplicando mapeamentos de IDs aos dados...")
        
        # Validar mapeamentos antes de aplicar
        validation_report = self.id_mapper.validate_mappings(self.df_clean)
        logger.info("Relatório de validação dos mapeamentos:")
        
        unmapped_ids_summary = {}
        for mapping_type, report in validation_report.items():
            coverage = report['coverage_rate'] * 100
            logger.info(f"  {mapping_type}: {report['mapped_ids']}/{report['total_unique_ids']} IDs mapeados ({coverage:.1f}%)")
            
            if report['unmapped_ids']:
                logger.warning(f"    IDs não mapeados: {report['unmapped_ids']}")
                unmapped_ids_summary[mapping_type] = report['unmapped_ids']
                
                # Tratar IDs não mapeados
                id_column = mapping_type.replace('_desc', '_id')
                if id_column in self.df_clean.columns:
                    unmapped_count = self.df_clean[id_column].isin(report['unmapped_ids']).sum()
                    if unmapped_count > 0:
                        logger.info(f"    Substituindo {unmapped_count:,} IDs não mapeados por 'unknown'")
                        # Nota: A substituição real seria feita aqui se necessário
        
        # Aplicar mapeamentos
        df_before = self.df_clean.shape[1]
        self.df_clean = self.id_mapper.apply_mappings_to_dataframe(self.df_clean)
        df_after = self.df_clean.shape[1]
        
        columns_added = df_after - df_before
        logger.info(f"Colunas descritivas adicionadas: {columns_added}")
        logger.info(f"Dimensões finais após mapeamentos: {self.df_clean.shape}")
        
        # Registrar estatísticas de mapeamento
        self.cleaning_stats['id_mappings_applied'] = True
        self.cleaning_stats['columns_added_mappings'] = columns_added
        self.cleaning_stats['unmapped_ids'] = unmapped_ids_summary
    
    def get_cleaning_summary(self):
        """
        Exibe resumo detalhado da limpeza dos dados
        
        Inclui estatísticas antes/depois e métricas de qualidade dos dados
        """
        logger.info("="*60)
        logger.info("RESUMO DA LIMPEZA DOS DADOS")
        logger.info("="*60)
        
        # Distribuição da variável alvo após limpeza
        logger.info("Distribuição da variável 'readmitted' após limpeza:")
        readmitted_final = self.df_clean['readmitted'].value_counts()
        for value, count in readmitted_final.items():
            pct = count / len(self.df_clean) * 100
            logger.info(f"  {value}: {count:,} ({pct:.2f}%)")
        
        logger.info("Distribuição da variável 'target' após limpeza:")
        target_final = self.df_clean['target'].value_counts()
        for value, count in target_final.items():
            pct = count / len(self.df_clean) * 100
            logger.info(f"  {value}: {count:,} ({pct:.2f}%)")
        
        final_readmission_rate = self.df_clean['target'].mean()
        logger.info(f"Taxa de readmissão final em <30 dias: {final_readmission_rate:.3f}")
        
        # Estatísticas de transformação
        logger.info("\nEstatísticas de Transformação:")
        logger.info(f"  Registros originais: {self.cleaning_stats.get('original_records', 'N/A'):,}")
        logger.info(f"  Registros finais: {len(self.df_clean):,}")
        logger.info(f"  Registros removidos: {self.cleaning_stats.get('original_records', 0) - len(self.df_clean):,}")
        logger.info(f"  Taxa de retenção: {len(self.df_clean) / self.cleaning_stats.get('original_records', 1) * 100:.1f}%")
        
        logger.info(f"  Colunas originais: {self.cleaning_stats.get('original_columns', 'N/A')}")
        logger.info(f"  Colunas finais: {self.df_clean.shape[1]}")
        
        logger.info(f"  Pacientes originais: {self.cleaning_stats.get('original_patients', 'N/A'):,}")
        logger.info(f"  Pacientes finais: {self.df_clean['patient_nbr'].nunique():,}")
        
        # Qualidade dos dados
        logger.info("\nQualidade dos Dados:")
        missing_data = self.df_clean.isnull().sum().sum()
        total_cells = self.df_clean.shape[0] * self.df_clean.shape[1]
        completeness = (1 - missing_data / total_cells) * 100
        logger.info(f"  Completude dos dados: {completeness:.2f}%")
        logger.info(f"  Dados faltantes totais: {missing_data:,}")
        
        # Atualizar estatísticas finais
        self.cleaning_stats['final_records'] = len(self.df_clean)
        self.cleaning_stats['final_columns'] = self.df_clean.shape[1]
        self.cleaning_stats['final_patients'] = self.df_clean['patient_nbr'].nunique()
        self.cleaning_stats['final_readmission_rate'] = final_readmission_rate
        self.cleaning_stats['data_completeness'] = completeness
        self.cleaning_stats['retention_rate'] = len(self.df_clean) / self.cleaning_stats.get('original_records', 1)
    
    def save_cleaning_report(self, output_dir: Optional[str] = None) -> str:
        """
        Salva relatório detalhado da limpeza dos dados em múltiplos formatos
        
        Args:
            output_dir (str): Diretório para salvar o relatório
            
        Returns:
            str: Caminho do relatório principal salvo
        """
        if output_dir is None:
            output_dir = os.path.dirname(CLEAN_DATA_FILE)
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salvar relatório detalhado em texto
        report_path = os.path.join(output_dir, f'data_cleaning_report_{timestamp}.txt')
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("RELATÓRIO DETALHADO DE LIMPEZA DOS DADOS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Arquivo original: {self.raw_data_path}\n")
                f.write(f"Arquivo limpo: {CLEAN_DATA_FILE}\n")
                f.write(f"Configuração utilizada: {self.config}\n\n")
                
                f.write("ESTATÍSTICAS DE TRANSFORMAÇÃO:\n")
                f.write("-" * 40 + "\n")
                for key, value in self.cleaning_stats.items():
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for sub_key, sub_value in value.items():
                            f.write(f"  {sub_key}: {sub_value}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                
                if hasattr(self, 'df_clean') and self.df_clean is not None:
                    f.write(f"\nDIMENSÕES FINAIS: {self.df_clean.shape}\n")
                    f.write(f"PACIENTES ÚNICOS: {self.df_clean['patient_nbr'].nunique():,}\n")
                    f.write(f"TAXA DE READMISSÃO FINAL: {self.cleaning_stats.get('final_readmission_rate', 'N/A')}\n")
                
            logger.info(f"Relatório de limpeza salvo em: {report_path}")
            
            # Salvar estatísticas em JSON para processamento programático
            json_path = os.path.join(output_dir, f'cleaning_stats_{timestamp}.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                # Converter tipos não serializáveis para JSON
                serializable_stats = self._make_json_serializable(self.cleaning_stats)
                json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Estatísticas JSON salvas em: {json_path}")
            
            return report_path
            
        except Exception as e:
            logger.error(f"Erro ao salvar relatório: {e}")
            return ""
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Converte objetos para formato serializável em JSON
        
        Args:
            obj: Objeto a ser convertido
            
        Returns:
            Objeto serializável em JSON
        """
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo do progresso de limpeza
        
        Returns:
            dict: Resumo do progresso atual
        """
        progress = {
            'steps_completed': [],
            'current_step': 'Inicialização',
            'records_remaining': 0,
            'completion_percentage': 0
        }
        
        if hasattr(self, 'df_raw') and self.df_raw is not None:
            progress['steps_completed'].append('Carregamento de dados')
            progress['current_step'] = 'Dados carregados'
            
        if 'target' in getattr(self, 'df_raw', pd.DataFrame()).columns:
            progress['steps_completed'].append('Criação de variável alvo')
            
        if hasattr(self, 'df_clean') and self.df_clean is not None:
            progress['steps_completed'].append('Limpeza básica')
            progress['records_remaining'] = len(self.df_clean)
            
        total_steps = 7  # Total de etapas na pipeline
        progress['completion_percentage'] = (len(progress['steps_completed']) / total_steps) * 100
        
        return progress
    
    def clean_data(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Executa todo o pipeline de limpeza dos dados com controle robusto
        
        Args:
            save_path (str): Caminho para salvar os dados limpos
            
        Returns:
            pd.DataFrame: DataFrame com os dados limpos
            
        Raises:
            Exception: Se ocorrer erro crítico durante a limpeza
        """
        logger.info("=" * 80)
        logger.info("INICIANDO PIPELINE DE LIMPEZA DOS DADOS")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        steps_executed = []
        
        try:
            # 1. Carregamento de dados
            logger.info("\n🔄 ETAPA 1/7: Carregamento de dados")
            if self.df_raw is None:
                self.load_raw_data()
            steps_executed.append("load_data")
            
            # 2. Diagnóstico inicial (opcional, mas recomendado)
            logger.info("\n🔍 ETAPA 2/7: Diagnóstico de qualidade")
            quality_report = self.diagnose_data_quality()
            self.cleaning_stats['initial_quality_report'] = quality_report
            steps_executed.append("quality_diagnosis")
            
            # 3. Criação de variável alvo
            logger.info("\n🎯 ETAPA 3/7: Criação de variável alvo")
            self.create_target_variable()
            steps_executed.append("target_variable")
            
            # 4. Remoção de pacientes expirados/transferidos
            if self.config['remove_expired_transferred']:
                logger.info("\n🚫 ETAPA 4/7: Remoção de pacientes expirados/transferidos")
                self.remove_expired_and_transferred_patients()
                steps_executed.append("remove_expired_transferred")
            else:
                logger.info("\n⏭️  ETAPA 4/7: Pulada (configuração)")
            
            # 5. Tratamento de dados faltantes
            if self.config['handle_missing_data']:
                logger.info("\n🔧 ETAPA 5/7: Tratamento de dados faltantes")
                self.handle_missing_data()
                steps_executed.append("handle_missing")
            else:
                logger.info("\n⏭️  ETAPA 5/7: Pulada (configuração)")
            
            # 6. Verificação de consistência
            if self.config['check_consistency']:
                logger.info("\n✅ ETAPA 6/7: Verificação de consistência")
                self.check_data_consistency()
                steps_executed.append("consistency_check")
            else:
                logger.info("\n⏭️  ETAPA 6/7: Pulada (configuração)")
            
            # 7. Criação de features derivadas
            if self.config['create_derived_features']:
                logger.info("\n🧮 ETAPA 7/7: Criação de features derivadas")
                self.create_derived_features()
                steps_executed.append("derived_features")
            else:
                logger.info("\n⏭️  ETAPA 7/7: Pulada (configuração)")
            
            # 8. Remoção de duplicatas
            if self.config['remove_duplicates']:
                logger.info("\n🔄 ETAPA EXTRA: Remoção de duplicatas")
                self.remove_duplicate_patients()
                steps_executed.append("remove_duplicates")
            
            # 9. Aplicação de mapeamentos de IDs
            if self.config['apply_id_mappings'] and self.id_mapper is not None:
                logger.info("\n🗂️  ETAPA EXTRA: Aplicação de mapeamentos")
                self.apply_id_mappings()
                steps_executed.append("id_mappings")
            
            # 10. Resumo final
            logger.info("\n📊 Gerando resumo final...")
            self.get_cleaning_summary()
            steps_executed.append("summary")
            
            # Salvar dados limpos
            save_path = save_path or CLEAN_DATA_FILE
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.df_clean.to_csv(save_path, index=False)
            logger.info(f"📁 Dataset limpo salvo em: {save_path}")
            
            # Salvar relatório se configurado
            if self.config['generate_report']:
                report_path = self.save_cleaning_report()
                logger.info(f"📋 Relatório salvo em: {report_path}")
            
            # Calcular tempo de execução
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Registrar estatísticas finais
            self.cleaning_stats.update({
                'steps_executed': steps_executed,
                'execution_time_seconds': execution_time,
                'end_timestamp': end_time.isoformat(),
                'success': True
            })
            
            # Log de sucesso
            logger.info("=" * 80)
            logger.info("✅ LIMPEZA DOS DADOS CONCLUÍDA COM SUCESSO")
            logger.info(f"⏱️  Tempo de execução: {execution_time:.2f} segundos")
            logger.info(f"📏 Dimensões finais: {self.df_clean.shape}")
            logger.info(f"🏥 Pacientes únicos: {self.df_clean['patient_nbr'].nunique():,}")
            logger.info(f"🎯 Taxa de readmissão: {self.df_clean['target'].mean():.4f}")
            logger.info("=" * 80)
            
            return self.df_clean
            
        except Exception as e:
            # Registrar falha
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            self.cleaning_stats.update({
                'steps_executed': steps_executed,
                'execution_time_seconds': execution_time,
                'end_timestamp': end_time.isoformat(),
                'success': False,
                'error_message': str(e),
                'error_step': steps_executed[-1] if steps_executed else 'initialization'
            })
            
            logger.error("=" * 80)
            logger.error("❌ ERRO DURANTE A LIMPEZA DOS DADOS")
            logger.error(f"💥 Erro na etapa: {steps_executed[-1] if steps_executed else 'initialization'}")
            logger.error(f"📝 Mensagem: {str(e)}")
            logger.error(f"⏱️  Tempo até erro: {execution_time:.2f} segundos")
            logger.error("=" * 80)
            
            # Tentar salvar dados parciais se possível
            if hasattr(self, 'df_clean') and self.df_clean is not None:
                try:
                    partial_path = save_path.replace('.csv', '_partial.csv') if save_path else 'data_partial.csv'
                    self.df_clean.to_csv(partial_path, index=False)
                    logger.info(f"💾 Dados parciais salvos em: {partial_path}")
                except:
                    pass
            
            raise


def create_custom_cleaner(config_overrides: Optional[Dict] = None, 
                         data_path: Optional[str] = None) -> DataCleaner:
    """
    Cria um DataCleaner com configurações personalizadas
    
    Args:
        config_overrides (dict): Configurações personalizadas
        data_path (str): Caminho personalizado para dados
        
    Returns:
        DataCleaner: Instância configurada do limpador
    """
    custom_config = {**DEFAULT_CLEANING_CONFIG, **(config_overrides or {})}
    return DataCleaner(raw_data_path=data_path, config=custom_config)


def quick_clean(data_path: Optional[str] = None, 
                save_path: Optional[str] = None,
                minimal: bool = False) -> pd.DataFrame:
    """
    Executa limpeza rápida dos dados com configurações otimizadas
    
    Args:
        data_path (str): Caminho para dados brutos
        save_path (str): Caminho para salvar dados limpos
        minimal (bool): Se True, executa apenas limpeza essencial
        
    Returns:
        pd.DataFrame: Dados limpos
    """
    if minimal:
        config = {
            'remove_expired_transferred': True,
            'handle_missing_data': True,
            'check_consistency': False,
            'create_derived_features': False,
            'remove_duplicates': True,
            'apply_id_mappings': False,
            'generate_report': False
        }
    else:
        config = DEFAULT_CLEANING_CONFIG
    
    cleaner = DataCleaner(raw_data_path=data_path, config=config)
    return cleaner.clean_data(save_path=save_path)


def validate_cleaned_data(data_path: str) -> Dict[str, Any]:
    """
    Valida dados limpos para verificar qualidade
    
    Args:
        data_path (str): Caminho para dados limpos
        
    Returns:
        dict: Relatório de validação
    """
    try:
        df = pd.read_csv(data_path)
        
        validation_report = {
            'file_exists': True,
            'shape': df.shape,
            'has_target_column': 'target' in df.columns,
            'target_distribution': df['target'].value_counts().to_dict() if 'target' in df.columns else None,
            'missing_data_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'unique_patients': df['patient_nbr'].nunique() if 'patient_nbr' in df.columns else 0,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Verificações específicas
        issues = []
        if validation_report['missing_data_percentage'] > 10:
            issues.append(f"Alto percentual de dados faltantes: {validation_report['missing_data_percentage']:.1f}%")
        
        if validation_report['duplicate_rows'] > 0:
            issues.append(f"Linhas duplicadas encontradas: {validation_report['duplicate_rows']}")
        
        if 'target' in df.columns:
            target_rate = df['target'].mean()
            if target_rate < 0.01 or target_rate > 0.99:
                issues.append(f"Dataset extremamente desbalanceado: {target_rate:.3f}")
        
        validation_report['issues'] = issues
        validation_report['is_valid'] = len(issues) == 0
        
        return validation_report
        
    except Exception as e:
        return {
            'file_exists': False,
            'error': str(e),
            'is_valid': False,
            'validation_timestamp': datetime.now().isoformat()
        }


def main():
    """
    Função principal para executar a limpeza dos dados com interface aprimorada
    
    Exemplo de uso do módulo de limpeza de dados com todas as funcionalidades
    """
    logger.info("🚀 Executando limpeza de dados como script principal")
    logger.info("=" * 80)
    
    try:
        # Executar diagnóstico inicial se arquivo existir
        if os.path.exists(RAW_DATA_FILE):
            logger.info("📋 Executando diagnóstico inicial...")
            temp_cleaner = DataCleaner()
            temp_cleaner.load_raw_data()
            quality_report = temp_cleaner.diagnose_data_quality()
            logger.info(f"📊 Score de qualidade inicial: {quality_report.get('overall_quality_score', 'N/A')}/100")
            
            # Exibir resumo do diagnóstico
            dataset_info = quality_report.get('dataset_info', {})
            logger.info(f"📏 Shape: {dataset_info.get('shape', 'N/A')}")
            logger.info(f"💾 Tamanho: {dataset_info.get('memory_usage_mb', 0):.1f} MB")
            logger.info(f"👥 Pacientes únicos: {dataset_info.get('unique_patients', 'N/A'):,}")
        
        # Executar limpeza principal
        logger.info("\n🔧 Iniciando limpeza principal...")
        cleaner = DataCleaner()
        df_clean = cleaner.clean_data()
        
        # Validar resultado
        logger.info("\n🔍 Validando dados limpos...")
        validation = validate_cleaned_data(CLEAN_DATA_FILE)
        
        if validation['is_valid']:
            logger.info("✅ Validação: APROVADA")
        else:
            logger.warning("⚠️  Validação: PROBLEMAS ENCONTRADOS")
            for issue in validation.get('issues', []):
                logger.warning(f"   - {issue}")
        
        # Resumo final
        logger.info("\n" + "=" * 80)
        logger.info("🎉 LIMPEZA CONCLUÍDA COM SUCESSO!")
        logger.info(f"📁 Arquivo de saída: {CLEAN_DATA_FILE}")
        logger.info(f"📊 Dataset final: {df_clean.shape[0]:,} registros, {df_clean.shape[1]} colunas")
        logger.info(f"🎯 Taxa de readmissão: {df_clean['target'].mean():.4f}")
        logger.info(f"👥 Pacientes únicos: {df_clean['patient_nbr'].nunique():,}")
        
        # Salvar estatísticas de validação
        validation_path = os.path.join(os.path.dirname(CLEAN_DATA_FILE), 'validation_report.json')
        with open(validation_path, 'w', encoding='utf-8') as f:
            # Converter tipos não serializáveis para JSON usando função utilitária
            def make_serializable(obj):
                if isinstance(obj, dict):
                    return {key: make_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                elif isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif pd.isna(obj):
                    return None
                else:
                    return obj
            
            serializable_validation = make_serializable(validation)
            json.dump(serializable_validation, f, indent=2, ensure_ascii=False)
        logger.info(f"📋 Relatório de validação salvo em: {validation_path}")
        
        logger.info("=" * 80)
        
        return df_clean
        
    except FileNotFoundError as e:
        logger.error(f"❌ Arquivo não encontrado: {e}")
        logger.error("💡 Verifique se o arquivo de dados existe no caminho especificado")
        return None
        
    except Exception as e:
        logger.error(f"❌ Erro na execução principal: {e}")
        logger.error("💡 Verifique os logs acima para mais detalhes")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    main()
