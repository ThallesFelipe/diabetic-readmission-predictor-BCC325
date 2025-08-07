"""
Configurações Globais do Sistema de Predição de Readmissão Hospitalar Diabética

Este módulo centraliza todas as configurações críticas do projeto, garantindo:
- Padronização de caminhos de arquivos e diretórios
- Parâmetros otimizados de processamento de dados
- Configurações de modelos de Machine Learning
- Parâmetros de visualização e geração de relatórios
- Constantes do sistema e valores padrão

Autor: Thalles Felipe Rodrigues de Almeida Santos
Projeto: Predição de Readmissão Hospitalar em Pacientes com Diabetes Usando Aprendizado de Máquina
Instituição: Universidade Federal de Ouro Preto (UFOP)
Disciplina: Inteligência Artificial
Professor: Jadson Castro Gertrudes
Data: Agosto 2025
"""

import os

# ============================================================================
# CONFIGURAÇÕES DE SISTEMA E CAMINHOS
# ============================================================================

# Caminhos fundamentais do projeto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Arquivos de dados principais
RAW_DATA_FILE = os.path.join(DATA_DIR, 'diabetic_data.csv')
CLEAN_DATA_FILE = os.path.join(DATA_DIR, 'diabetic_data_clean.csv')
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'diabetic_data_processed.csv')
MAPPING_FILE = os.path.join(DATA_DIR, 'IDS_mapping.csv')

# Arquivos de dados para modelagem
X_TRAIN_FILE = os.path.join(DATA_DIR, 'X_train.csv')
X_TEST_FILE = os.path.join(DATA_DIR, 'X_test.csv')
Y_TRAIN_FILE = os.path.join(DATA_DIR, 'y_train.csv')
Y_TEST_FILE = os.path.join(DATA_DIR, 'y_test.csv')

# ============================================================================
# CONFIGURAÇÕES DE PROCESSAMENTO DE DADOS
# ============================================================================

# Códigos críticos para filtragem de dados
EXPIRED_DISCHARGE_CODES = [11, 19, 20, 21]  # Códigos de falecimento
TRANSFER_CODES = [3, 4, 5, 6, 22, 23, 24]   # Códigos de transferência

# Colunas para remoção baseada em qualidade dos dados
COLUMNS_TO_DROP_INITIAL = ['weight', 'payer_code']  # Colunas com >80% dados faltantes
COLUMNS_TO_DROP_MODELING = ['encounter_id', 'patient_nbr', 'readmitted', 'diag_1', 'diag_2', 'diag_3']

# Parâmetros de limpeza e qualidade
MISSING_DATA_THRESHOLD = 0.8  # 80% de dados faltantes para remoção de coluna
DATA_CONSISTENCY_CHECKS = True
CREATE_DERIVED_FEATURES = True

# ============================================================================
# CONFIGURAÇÕES DE MODELAGEM E MACHINE LEARNING
# ============================================================================

# Parâmetros globais de modelagem
TEST_SIZE = 0.2
RANDOM_STATE = 42
CLASSIFICATION_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Configurações de Regressão Logística
LOGISTIC_REGRESSION_CONFIG = {
    'random_state': RANDOM_STATE,
    'max_iter': 1000,
    'solver': 'liblinear',
    'class_weight': 'balanced'  # Para lidar com desbalanceamento
}

# Configurações de Random Forest
RANDOM_FOREST_CONFIG = {
    'random_state': RANDOM_STATE,
    'n_estimators': 300,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'n_jobs': -1,
    'oob_score': True,
    'bootstrap': True,
    'warm_start': False,
    'verbose': 0
}

# Grid de parâmetros para Random Forest
RANDOM_FOREST_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [2, 4, 6],
    'max_features': ['sqrt', 'log2', None]
}

# ============================================================================
# CONFIGURAÇÕES DE OTIMIZAÇÃO E VALIDAÇÃO
# ============================================================================

# Configurações de Otimização de Hiperparâmetros
HYPERPARAMETER_TUNING_CONFIG = {
    'cv_folds': 5,
    'scoring': 'roc_auc',
    'n_jobs': -1,
    'verbose': 1,
    'default_param_grid': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
}

# Configurações de Otimização de Limiar
THRESHOLD_OPTIMIZATION_CONFIG = {
    'default_metric': 'f1',  # Métrica a ser otimizada
    'available_metrics': ['f1', 'precision', 'recall'],
    'min_recall_for_precision': 0.3,  # Recall mínimo ao otimizar precision
    'min_precision_for_recall': 0.3   # Precision mínima ao otimizar recall
}

# ============================================================================
# CONFIGURAÇÕES DE VISUALIZAÇÃO E RELATÓRIOS
# ============================================================================

# Configurações de Visualização
VISUALIZATION_CONFIG = {
    'figure_size': (18, 10),
    'dpi': 300,
    'color_palette': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
    'save_format': 'png',
    'show_plots': True
}

# ============================================================================
# CONSTANTES DO DOMÍNIO MÉDICO
# ============================================================================

# Faixas etárias do dataset
AGE_RANGES = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
              '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']

# Medicações diabéticas disponíveis no dataset
DIABETES_MEDICATIONS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
    'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone'
]

# Estados de mudança de medicação
MEDICATION_CHANGE_STATES = ['No', 'Ch', 'Up', 'Down', 'Steady']

# Resultados de exames laboratoriais
LAB_RESULT_STATES = ['None', 'Norm', '>200', '>300', 'Norm', '>7', '>8']

# ============================================================================
# CONFIGURAÇÕES DE LOGGING E DEBUG
# ============================================================================

# Configurações de logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': 'diabetic_readmission_pipeline.log',
    'filemode': 'a',
    'encoding': 'utf-8'
}

# Configurações de debug e desenvolvimento
DEBUG_CONFIG = {
    'verbose_mode': False,
    'save_intermediate_files': True,
    'generate_debug_plots': False,
    'max_debug_samples': 1000,
    'enable_data_profiling': False
}

# ============================================================================
# METADADOS DO PROJETO
# ============================================================================

PROJECT_METADATA = {
    'name': 'Diabetic Readmission Predictor',
    'version': '2.0',
    'author': 'Thalles Felipe Rodrigues de Almeida Santos',
    'institution': 'Universidade Federal de Ouro Preto (UFOP)',
    'course': 'BCC325 - Inteligência Artificial',
    'professor': 'Jadson Castro Gertrudes',
    'dataset_source': 'UCI Machine Learning Repository',
    'dataset_name': 'Diabetes 130-US hospitals for years 1999-2008',
    'target_variable': 'readmitted',
    'prediction_objective': 'Predict hospital readmission within 30 days for diabetic patients'
}
