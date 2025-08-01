"""
Configurações globais do projeto de predição de readmissão hospitalar diabética
"""

import os

# Caminhos dos arquivos
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Arquivos de dados
RAW_DATA_FILE = os.path.join(DATA_DIR, 'diabetic_data.csv')
CLEAN_DATA_FILE = os.path.join(DATA_DIR, 'diabetic_data_clean.csv')
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'diabetic_data_processed.csv')
MAPPING_FILE = os.path.join(DATA_DIR, 'IDS_mapping.csv')

# Configurações de processamento
EXPIRED_DISCHARGE_CODES = [11, 19, 20, 21]
COLUMNS_TO_DROP_INITIAL = ['weight', 'payer_code']
COLUMNS_TO_DROP_MODELING = ['encounter_id', 'patient_nbr', 'readmitted', 'diag_1', 'diag_2', 'diag_3']

# Configurações de modelagem
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Configurações de avaliação
CLASSIFICATION_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Arquivos de modelos
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Arquivos de dados para modelagem
X_TRAIN_FILE = os.path.join(DATA_DIR, 'X_train.csv')
X_TEST_FILE = os.path.join(DATA_DIR, 'X_test.csv')
Y_TRAIN_FILE = os.path.join(DATA_DIR, 'y_train.csv')
Y_TEST_FILE = os.path.join(DATA_DIR, 'y_test.csv')

# Configurações de Regressão Logística
LOGISTIC_REGRESSION_CONFIG = {
    'random_state': RANDOM_STATE,
    'max_iter': 1000,
    'solver': 'liblinear',
    'class_weight': 'balanced'  # Para lidar com desbalanceamento
}
