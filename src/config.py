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
EXPIRED_DISCHARGE_CODES = [11, 19, 20, 21]  # Códigos de falecimento
TRANSFER_CODES = [3, 4, 5, 6, 22, 23, 24]   # Códigos de transferência
COLUMNS_TO_DROP_INITIAL = ['weight', 'payer_code']  # Colunas com >80% dados faltantes
COLUMNS_TO_DROP_MODELING = ['encounter_id', 'patient_nbr', 'readmitted', 'diag_1', 'diag_2', 'diag_3']

# Configurações de limpeza de dados
DATA_CONSISTENCY_CHECKS = True
CREATE_DERIVED_FEATURES = True
MISSING_DATA_THRESHOLD = 0.8  # 80% de dados faltantes para remoção de coluna

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

# Configurações de Visualização
VISUALIZATION_CONFIG = {
    'figure_size': (18, 10),
    'dpi': 300,
    'color_palette': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
    'save_format': 'png',
    'show_plots': True
}
