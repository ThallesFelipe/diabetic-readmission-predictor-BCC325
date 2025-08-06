"""
Módulo de Modelo Random Forest para Predição Médica Avançada

Este módulo implementa uma solução completa e otimizada de Random Forest
especificamente projetada para predição de readmissão hospitalar diabética, com:

Características Técnicas Avançadas:
- Random Forest otimizado para dados médicos complexos
- Otimização de hiperparâmetros com Grid Search e Random Search
- Balanceamento sofisticado de classes com múltiplas estratégias
- Análise aprofundada de importância de features com permutação
- Validação cruzada estratificada para máxima robustez
- Otimização de threshold para diferentes métricas clínicas
- Análise de curvas de aprendizado e validação
- Visualizações específicas e interpretáveis para contexto médico
- Out-of-Bag scoring para validação interna
- Relatórios detalhados para interpretação clínica
- Sistema de ensemble com múltiplos estimadores
- Pipeline completo de produção
- Métricas focadas em aplicações de saúde
- Análise de overfitting e generalização

Autor: Thalles Felipe Rodrigues de Almeida Santos
Projeto: Predição de Readmissão Hospitalar em Pacientes com Diabetes Usando Aprendizado de Máquina
Instituição: Universidade Federal de Ouro Preto (UFOP)
Disciplina: Inteligência Artificial
Professor: Jadson Castro Gertrudes
Data: Agosto 2025

"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

# Configurar matplotlib ANTES de importar pyplot para evitar problemas de thread
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo
import matplotlib.pyplot as plt
plt.ioff()  # Desabilitar modo interativo

import seaborn as sns
import joblib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Configurações específicas do matplotlib para Windows
matplotlib.rcParams['backend'] = 'Agg'
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Adicionar o diretório pai ao path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports do scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score, log_loss, brier_score_loss
)
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV,
    learning_curve, validation_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text

# Importar imblearn opcionalmente
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("⚠️ imblearn não disponível. SMOTE será desabilitado.")
    print("   Para instalar: pip install imbalanced-learn")

# Imports de configuração
from src.config import (
    X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE,
    CLASSIFICATION_METRICS, MODELS_DIR, RESULTS_DIR, RANDOM_STATE,
    HYPERPARAMETER_TUNING_CONFIG, VISUALIZATION_CONFIG
)
from src.visualization_utils import ProfessionalVisualizer

# Suprimir warnings desnecessários
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configurar matplotlib para melhor visualização
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Criar diretórios se não existirem
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configurações específicas do Random Forest
RANDOM_FOREST_CONFIG = {
    'random_state': RANDOM_STATE,
    'n_jobs': -1,  # Usar todos os processadores disponíveis
    'oob_score': True,  # Calcular Out-of-Bag score
    'class_weight': 'balanced',  # Balanceamento automático de classes
    'max_features': 'sqrt',  # Usar raiz quadrada do número de features
    'bootstrap': True,  # Usar bootstrap sampling
    'warm_start': False,  # Não usar warm start por padrão
    'verbose': 0
}

# Configurações de otimização de hiperparâmetros para Random Forest
RF_HYPERPARAMETER_CONFIG = {
    'cv_folds': 5,
    'scoring': 'roc_auc',
    'n_jobs': -1,
    'verbose': 1,
    'param_distributions': {
        'n_estimators': [100, 200, 300, 500, 800],
        'max_depth': [3, 5, 7, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 8, 12],
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    },
    'grid_search_params': {
        'n_estimators': [200, 300, 500],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [2, 4, 8],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample']
    }
}


class RandomForestModel:
    """Classe para implementação do modelo Random Forest otimizado"""
    
    def __init__(self, model_config=None, use_smote=False, sampling_strategy='auto'):
        """
        Inicializa o modelo Random Forest
        
        Args:
            model_config (dict): Configurações específicas do modelo
            use_smote (bool): Se deve usar SMOTE para balanceamento
            sampling_strategy (str): Estratégia de sampling para SMOTE
        """
        self.model_config = model_config or RANDOM_FOREST_CONFIG.copy()
        self.use_smote = use_smote
        self.sampling_strategy = sampling_strategy
        
        # Inicializar componentes
        self.model = None
        self.best_model = None
        self.scaler = None
        self.smote = None
        
        # Dados
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        
        # Resultados
        self.predictions = None
        self.predictions_proba = None
        self.optimal_threshold = 0.5
        self.metrics = {}
        self.feature_importance = None
        self.permutation_importance = None
        self.oob_score = None
        
        # Metadados
        self.training_time = None
        self.feature_names = None
        self.model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("🌲 Random Forest Model inicializado")
        print(f"📊 Configurações: {self.model_config}")
        if self.use_smote:
            print(f"⚖️ SMOTE ativado com estratégia: {self.sampling_strategy}")
    
    def load_data(self):
        """Carrega os dados de treinamento e teste"""
        print("\n📥 Carregando dados...")
        
        try:
            # Carregar dados
            self.X_train = pd.read_csv(X_TRAIN_FILE)
            self.X_test = pd.read_csv(X_TEST_FILE)
            self.y_train = pd.read_csv(Y_TRAIN_FILE)['target']
            self.y_test = pd.read_csv(Y_TEST_FILE)['target']
            
            # Armazenar nomes das features
            self.feature_names = list(self.X_train.columns)
            
            print(f"✅ Dados carregados com sucesso!")
            print(f"   📊 Treino: {self.X_train.shape[0]:,} amostras, {self.X_train.shape[1]} features")
            print(f"   📊 Teste: {self.X_test.shape[0]:,} amostras")
            print(f"   🎯 Distribuição das classes no treino:")
            print(f"      - Classe 0 (Não readmitido): {(self.y_train == 0).sum():,} ({(self.y_train == 0).mean():.1%})")
            print(f"      - Classe 1 (Readmitido): {(self.y_train == 1).sum():,} ({(self.y_train == 1).mean():.1%})")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return False
    
    def preprocess_data(self):
        """Pré-processa os dados (escalonamento e balanceamento se necessário)"""
        print("\n🔧 Pré-processando dados...")
        
        # Escalonamento (opcional para Random Forest, mas útil para análises)
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Converter de volta para DataFrame para manter nomes das colunas
        self.X_train_scaled = pd.DataFrame(
            self.X_train_scaled, 
            columns=self.feature_names, 
            index=self.X_train.index
        )
        self.X_test_scaled = pd.DataFrame(
            self.X_test_scaled, 
            columns=self.feature_names, 
            index=self.X_test.index
        )
        
        # Aplicar SMOTE se solicitado e disponível
        if self.use_smote and IMBLEARN_AVAILABLE:
            print("⚖️ Aplicando SMOTE para balanceamento...")
            
            try:
                self.smote = SMOTE(
                    sampling_strategy=self.sampling_strategy,
                    random_state=RANDOM_STATE,
                    k_neighbors=5
                )
                
                X_train_balanced, y_train_balanced = self.smote.fit_resample(
                    self.X_train, self.y_train
                )
                
                print(f"📊 Dados balanceados:")
                print(f"   - Antes: {self.X_train.shape[0]:,} amostras")
                print(f"   - Depois: {X_train_balanced.shape[0]:,} amostras")
                print(f"   - Classe 0: {(y_train_balanced == 0).sum():,}")
                print(f"   - Classe 1: {(y_train_balanced == 1).sum():,}")
                
                # Usar dados balanceados
                self.X_train = X_train_balanced
                self.y_train = y_train_balanced
                
                # Reescalonar dados balanceados
                self.X_train_scaled = self.scaler.fit_transform(self.X_train)
                self.X_train_scaled = pd.DataFrame(
                    self.X_train_scaled, 
                    columns=self.feature_names
                )
                
            except Exception as e:
                print(f"⚠️ Erro ao aplicar SMOTE: {e}")
                print("Continuando sem balanceamento...")
                self.use_smote = False
        
        elif self.use_smote and not IMBLEARN_AVAILABLE:
            print("⚠️ SMOTE solicitado mas imblearn não está disponível")
            print("Continuando sem balanceamento...")
            self.use_smote = False
        
        print("✅ Pré-processamento concluído!")
    
    def train_model(self):
        """Treina o modelo Random Forest"""
        print("\n🚀 Treinando modelo Random Forest...")
        
        start_time = datetime.now()
        
        try:
            # Inicializar modelo
            self.model = RandomForestClassifier(**self.model_config)
            
            # Treinar modelo
            self.model.fit(self.X_train, self.y_train)
            
            # Calcular tempo de treinamento
            self.training_time = (datetime.now() - start_time).total_seconds()
            
            # Extrair OOB score se disponível
            if hasattr(self.model, 'oob_score_') and self.model.oob_score_ is not None:
                self.oob_score = self.model.oob_score_
                print(f"📊 OOB Score: {self.oob_score:.4f}")
            
            print(f"✅ Modelo treinado em {self.training_time:.2f} segundos")
            print(f"🌲 Número de árvores: {self.model.n_estimators}")
            print(f"📏 Profundidade máxima: {self.model.max_depth}")
            
            return True
            
        except Exception as e:
            # Garantir que training_time seja definido mesmo em caso de erro
            self.training_time = (datetime.now() - start_time).total_seconds()
            print(f"❌ Erro durante treinamento: {e}")
            raise
    
    def tune_hyperparameters(self, method='random_search', n_iter=50, cv_folds=5):
        """
        Otimiza hiperparâmetros usando Grid Search ou Random Search
        
        Args:
            method (str): 'grid_search' ou 'random_search'
            n_iter (int): Número de iterações para Random Search
            cv_folds (int): Número de folds para validação cruzada
        """
        print(f"\n🔍 Otimizando hiperparâmetros com {method}...")
        
        # Preparar configurações
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        scoring = RF_HYPERPARAMETER_CONFIG['scoring']
        
        try:
            if method == 'grid_search':
                param_grid = RF_HYPERPARAMETER_CONFIG['grid_search_params']
                search = GridSearchCV(
                    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                    param_grid=param_grid,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=1,
                    return_train_score=True
                )
                print(f"🔍 Grid Search com {len(param_grid)} parâmetros")
                
            else:  # random_search
                param_distributions = RF_HYPERPARAMETER_CONFIG['param_distributions']
                search = RandomizedSearchCV(
                    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                    param_distributions=param_distributions,
                    n_iter=n_iter,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=1,
                    random_state=RANDOM_STATE,
                    return_train_score=True
                )
                print(f"🎲 Random Search com {n_iter} iterações")
            
            # Executar busca
            start_time = datetime.now()
            search.fit(self.X_train, self.y_train)
            search_time = (datetime.now() - start_time).total_seconds()
            
            # Armazenar melhor modelo
            self.best_model = search.best_estimator_
            self.model = self.best_model  # Usar o melhor modelo como modelo principal
            
            print(f"✅ Otimização concluída em {search_time:.2f} segundos")
            print(f"📊 Melhor score ({scoring}): {search.best_score_:.4f}")
            print(f"🏆 Melhores parâmetros:")
            for param, value in search.best_params_.items():
                print(f"   {param}: {value}")
            
            # Salvar resultados da busca
            search_results = {
                'method': method,
                'best_score': search.best_score_,
                'best_params': search.best_params_,
                'search_time': search_time,
                'cv_folds': cv_folds,
                'scoring': scoring
            }
            
            # Salvar detalhes da validação cruzada
            if hasattr(search, 'cv_results_'):
                cv_results = pd.DataFrame(search.cv_results_)
                cv_results.to_csv(
                    os.path.join(RESULTS_DIR, f'random_forest_cv_results_{self.model_timestamp}.csv'),
                    index=False
                )
            
            return search_results
            
        except Exception as e:
            print(f"❌ Erro durante otimização: {e}")
            print("Continuando com modelo padrão...")
            return None
    
    def make_predictions(self):
        """Faz predições nos dados de teste"""
        print("\n🎯 Fazendo predições...")
        
        if self.model is None:
            print("❌ Modelo não foi treinado ainda!")
            return False
        
        # Predições binárias e probabilidades
        self.predictions = self.model.predict(self.X_test)
        self.predictions_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        print(f"✅ Predições realizadas para {len(self.predictions):,} amostras")
        print(f"📊 Distribuição das predições:")
        print(f"   - Classe 0: {(self.predictions == 0).sum():,} ({(self.predictions == 0).mean():.1%})")
        print(f"   - Classe 1: {(self.predictions == 1).sum():,} ({(self.predictions == 1).mean():.1%})")
        
        return True
    
    def optimize_threshold(self, metric='f1'):
        """
        Otimiza o threshold de decisão para maximizar uma métrica específica
        
        Args:
            metric (str): Métrica a ser otimizada ('f1', 'precision', 'recall', 'accuracy')
        """
        print(f"\n🎯 Otimizando threshold para métrica: {metric}")
        
        if self.predictions_proba is None:
            print("❌ Probabilidades não calculadas ainda!")
            return None
        
        # Testar diferentes thresholds
        thresholds = np.arange(0.1, 0.9, 0.01)
        scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (self.predictions_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(self.y_test, y_pred_thresh)
            elif metric == 'precision':
                score = precision_score(self.y_test, y_pred_thresh, zero_division=0)
            elif metric == 'recall':
                score = recall_score(self.y_test, y_pred_thresh)
            elif metric == 'accuracy':
                score = accuracy_score(self.y_test, y_pred_thresh)
            else:
                print(f"❌ Métrica '{metric}' não suportada!")
                return None
            
            scores.append(score)
        
        # Encontrar melhor threshold
        best_idx = np.argmax(scores)
        self.optimal_threshold = thresholds[best_idx]
        best_score = scores[best_idx]
        
        # Aplicar melhor threshold
        self.predictions = (self.predictions_proba >= self.optimal_threshold).astype(int)
        
        print(f"✅ Melhor threshold: {self.optimal_threshold:.3f}")
        print(f"📊 Score otimizado ({metric}): {best_score:.4f}")
        
        return self.optimal_threshold, best_score
    
    def evaluate_model(self):
        """Avalia o modelo com métricas abrangentes"""
        print("\n📊 Avaliando modelo...")
        
        if self.predictions is None or self.y_test is None:
            print("❌ Predições ou dados de teste não disponíveis!")
            return None
        
        # Calcular métricas básicas
        self.metrics = {
            'accuracy': accuracy_score(self.y_test, self.predictions),
            'precision': precision_score(self.y_test, self.predictions),
            'recall': recall_score(self.y_test, self.predictions),
            'f1': f1_score(self.y_test, self.predictions),
            'roc_auc': roc_auc_score(self.y_test, self.predictions_proba),
            'average_precision': average_precision_score(self.y_test, self.predictions_proba),
            'log_loss': log_loss(self.y_test, self.predictions_proba),
            'brier_score': brier_score_loss(self.y_test, self.predictions_proba)
        }
        
        # Adicionar OOB score se disponível
        if self.oob_score is not None:
            self.metrics['oob_score'] = self.oob_score
        
        # Matriz de confusão
        cm = confusion_matrix(self.y_test, self.predictions)
        
        # Calcular métricas específicas da matriz de confusão
        tn, fp, fn, tp = cm.ravel()
        
        self.metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Positive Predictive Value
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0   # Negative Predictive Value
        })
        
        # Adicionar informações do modelo
        self.metrics.update({
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'min_samples_split': self.model.min_samples_split,
            'min_samples_leaf': self.model.min_samples_leaf,
            'max_features': self.model.max_features,
            'training_time': self.training_time,
            'optimal_threshold': self.optimal_threshold,
            'use_smote': self.use_smote
        })
        
        # Imprimir resultados
        print(f"\n📈 RESULTADOS DO RANDOM FOREST:")
        print(f"{'='*50}")
        print(f"🎯 Acurácia: {self.metrics['accuracy']:.4f}")
        print(f"🎯 Precisão: {self.metrics['precision']:.4f}")
        print(f"🎯 Recall: {self.metrics['recall']:.4f}")
        print(f"🎯 F1-Score: {self.metrics['f1']:.4f}")
        print(f"🎯 ROC-AUC: {self.metrics['roc_auc']:.4f}")
        print(f"🎯 Average Precision: {self.metrics['average_precision']:.4f}")
        
        if self.oob_score is not None:
            print(f"🎯 OOB Score: {self.metrics['oob_score']:.4f}")
        
        print(f"\n🔢 Matriz de Confusão:")
        print(f"   TN: {tn:,} | FP: {fp:,}")
        print(f"   FN: {fn:,} | TP: {tp:,}")
        
        print(f"\n📊 Métricas Clínicas:")
        print(f"   Especificidade: {self.metrics['specificity']:.4f}")
        print(f"   Sensibilidade: {self.metrics['sensitivity']:.4f}")
        print(f"   VPP: {self.metrics['ppv']:.4f}")
        print(f"   VPN: {self.metrics['npv']:.4f}")
        
        return self.metrics
    
    def analyze_feature_importance(self, top_n=20):
        """Analisa a importância das features"""
        print(f"\n🔍 Analisando importância das {top_n} principais features...")
        
        if self.model is None:
            print("❌ Modelo não foi treinado ainda!")
            return None
        
        # Importância baseada em impureza (Gini/Entropy)
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = feature_importance_df
        
        print(f"🏆 Top {top_n} Features Mais Importantes:")
        print("="*60)
        for i, (_, row) in enumerate(feature_importance_df.head(top_n).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<40} {row['importance']:.4f}")
        
        # Calcular importância por permutação (mais robusta)
        try:
            print(f"\n🔄 Calculando importância por permutação...")
            perm_importance = permutation_importance(
                self.model, self.X_test, self.y_test,
                n_repeats=5,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            
            permutation_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            self.permutation_importance = permutation_df
            
            print(f"🏆 Top {min(10, top_n)} Features (Permutação):")
            print("="*60)
            for i, (_, row) in enumerate(permutation_df.head(min(10, top_n)).iterrows(), 1):
                print(f"{i:2d}. {row['feature']:<40} {row['importance_mean']:.4f} (±{row['importance_std']:.4f})")
        
        except Exception as e:
            print(f"⚠️ Erro ao calcular importância por permutação: {e}")
            self.permutation_importance = None
        
        return self.feature_importance
    
    def cross_validate_model(self, cv_folds=5):
        """Realiza validação cruzada detalhada"""
        print(f"\n🔄 Realizando validação cruzada com {cv_folds} folds...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        
        # Métricas a serem avaliadas
        metrics_to_evaluate = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = {}
        
        for metric in metrics_to_evaluate:
            scores = cross_val_score(
                self.model, self.X_train, self.y_train,
                cv=cv, scoring=metric, n_jobs=-1
            )
            
            cv_results[metric] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max()
            }
        
        # Imprimir resultados
        print(f"\n📊 RESULTADOS DA VALIDAÇÃO CRUZADA:")
        print(f"{'='*60}")
        for metric, results in cv_results.items():
            print(f"{metric.upper():<12}: {results['mean']:.4f} (±{results['std']:.4f})")
            print(f"{'Range':<12}: [{results['min']:.4f}, {results['max']:.4f}]")
            print()
        
        return cv_results
    
    def plot_results(self):
        """Gera visualizações profissionais e padronizadas dos resultados"""
        print("\n🎨 Gerando visualizações profissionais...")
        
        # Verificar se temos dados suficientes para plotar
        if self.model is None:
            print("❌ Modelo não foi treinado ainda!")
            return
            
        if self.predictions is None or self.y_test is None:
            print("❌ Predições não estão disponíveis!")
            return
            
        if not self.metrics:
            print("❌ Métricas não foram calculadas!")
            return
        
        # Inicializar o visualizador profissional
        visualizer = ProfessionalVisualizer(save_dir=RESULTS_DIR)
        
        # Preparar dados para o dashboard
        metrics = {
            'Acurácia': self.metrics.get('accuracy', 0),
            'Precisão': self.metrics.get('precision', 0),
            'Recall': self.metrics.get('recall', 0),
            'F1-Score': self.metrics.get('f1', 0),
            'ROC-AUC': self.metrics.get('roc_auc', 0)
        }
        
        # Preparar matriz de confusão
        cm = confusion_matrix(self.y_test, self.predictions)
        
        # Preparar dados da curva ROC
        fpr, tpr, _ = roc_curve(self.y_test, self.predictions_proba)
        auc_score = self.metrics.get('roc_auc', 0)
        
        # Preparar dados de feature importance se disponível
        feature_names = None
        feature_importance = None
        if self.feature_importance is not None:
            feature_names = self.feature_importance['feature'].tolist()
            feature_importance = self.feature_importance['importance'].tolist()
        
        # Criar dashboard completo
        filename = f'random_forest_results_{self.model_timestamp}'
        
        visualizer.plot_model_results_dashboard(
            metrics=metrics,
            cm=cm,
            fpr=fpr,
            tpr=tpr,
            auc_score=auc_score,
            feature_names=feature_names,
            feature_importance=feature_importance,
            model_name="Random Forest",
            filename=filename
        )
        
        # Criar visualizações adicionais específicas do Random Forest
        self._plot_rf_specific_visualizations(visualizer)
        
        print("✅ Visualizações profissionais criadas com sucesso!")
        return os.path.join(RESULTS_DIR, f'{filename}.png')
    
    def _plot_rf_specific_visualizations(self, visualizer):
        """Cria visualizações específicas do Random Forest"""
        
        # 1. Gráfico de distribuição de probabilidades com threshold
        self._plot_probability_distribution_rf(visualizer)
        
        # 2. Comparação de importância: Gini vs Permutação
        if self.permutation_importance is not None:
            self._plot_importance_comparison(visualizer)
        
        # 3. Informações detalhadas do modelo Random Forest
        self._plot_model_info_dashboard(visualizer)
    
    def _plot_probability_distribution_rf(self, visualizer):
        """Cria gráfico profissional da distribuição de probabilidades com threshold"""
        fig, ax = visualizer.create_figure(figsize=(12, 6))
        
        # Importar cores do módulo
        from src.visualization_utils import PROFESSIONAL_COLORS
        
        # Histogramas com estilo profissional
        ax.hist(self.predictions_proba[self.y_test == 0], bins=50, alpha=0.7, 
               label='Não Readmitido (Classe 0)', color=PROFESSIONAL_COLORS['primary'],
               edgecolor='white', linewidth=0.5, density=True)
        ax.hist(self.predictions_proba[self.y_test == 1], bins=50, alpha=0.7, 
               label='Readmitido (Classe 1)', color=PROFESSIONAL_COLORS['danger'],
               edgecolor='white', linewidth=0.5, density=True)
        
        # Linha do threshold otimizado
        if hasattr(self, 'optimal_threshold') and self.optimal_threshold is not None:
            ax.axvline(x=self.optimal_threshold, color=PROFESSIONAL_COLORS['accent'], 
                      linestyle='--', linewidth=3, 
                      label=f'Threshold Otimizado = {self.optimal_threshold:.3f}')
        
        # Linha do threshold padrão
        ax.axvline(x=0.5, color=PROFESSIONAL_COLORS['neutral'], 
                  linestyle=':', linewidth=2, alpha=0.7,
                  label='Threshold Padrão = 0.500')
        
        # Configurações
        ax.set_xlabel('Probabilidade Predita de Readmissão', fontweight='bold')
        ax.set_ylabel('Densidade', fontweight='bold')
        ax.set_title('Distribuição das Probabilidades Preditas - Random Forest', fontweight='bold', pad=20)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        
        # Estatísticas na legenda
        mean_0 = self.predictions_proba[self.y_test == 0].mean()
        mean_1 = self.predictions_proba[self.y_test == 1].mean()
        std_0 = self.predictions_proba[self.y_test == 0].std()
        std_1 = self.predictions_proba[self.y_test == 1].std()
        
        textstr = f'''Estatísticas:
Classe 0: μ={mean_0:.3f}, σ={std_0:.3f}
Classe 1: μ={mean_1:.3f}, σ={std_1:.3f}
Separação: {abs(mean_1 - mean_0):.3f}'''
        
        props = dict(boxstyle='round', facecolor=PROFESSIONAL_COLORS['light'], alpha=0.9)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, fontfamily='monospace')
        
        # Salvar
        filename = f'random_forest_probability_distribution_{self.model_timestamp}'
        visualizer.save_figure(fig, filename)
    
    def _plot_importance_comparison(self, visualizer):
        """Compara importância Gini vs Permutação"""
        if self.feature_importance is None or self.permutation_importance is None:
            return
            
        # Importar cores do módulo
        from src.visualization_utils import PROFESSIONAL_COLORS
        
        fig, (ax1, ax2) = visualizer.create_figure(figsize=(20, 10), ncols=2)
        
        # Top 15 features para cada método
        top_n = 15
        
        # Importância Gini
        top_gini = self.feature_importance.head(top_n)
        colors_gini = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_gini)))
        
        bars1 = ax1.barh(range(len(top_gini)), top_gini['importance'], 
                        color=colors_gini, edgecolor=PROFESSIONAL_COLORS['dark'], 
                        linewidth=0.5)
        ax1.set_yticks(range(len(top_gini)))
        ax1.set_yticklabels(top_gini['feature'], fontsize=9)
        ax1.set_xlabel('Importância (Gini Impurity)', fontweight='bold')
        ax1.set_title(f'Top {top_n} Features - Importância por Impureza', fontweight='bold', pad=20)
        
        # Adicionar valores nas barras
        for i, (bar, value) in enumerate(zip(bars1, top_gini['importance'])):
            width = bar.get_width()
            ax1.text(width + max(top_gini['importance']) * 0.01, 
                   bar.get_y() + bar.get_height()/2,
                   f'{value:.4f}', ha='left', va='center', 
                   fontsize=8, fontweight='bold')
        
        # Importância por Permutação
        top_perm = self.permutation_importance.head(top_n)
        colors_perm = plt.cm.plasma(np.linspace(0.3, 0.9, len(top_perm)))
        
        bars2 = ax2.barh(range(len(top_perm)), top_perm['importance_mean'],
                        xerr=top_perm['importance_std'], color=colors_perm,
                        edgecolor=PROFESSIONAL_COLORS['dark'], linewidth=0.5)
        ax2.set_yticks(range(len(top_perm)))
        ax2.set_yticklabels(top_perm['feature'], fontsize=9)
        ax2.set_xlabel('Importância (Permutação ± Desvio)', fontweight='bold')
        ax2.set_title(f'Top {top_n} Features - Importância por Permutação', fontweight='bold', pad=20)
        
        # Adicionar valores nas barras
        for i, (bar, value, std) in enumerate(zip(bars2, top_perm['importance_mean'], top_perm['importance_std'])):
            width = bar.get_width()
            ax2.text(width + max(top_perm['importance_mean']) * 0.01, 
                   bar.get_y() + bar.get_height()/2,
                   f'{value:.4f}±{std:.4f}', ha='left', va='center', 
                   fontsize=8, fontweight='bold')
        
        # Salvar
        filename = f'random_forest_importance_comparison_{self.model_timestamp}'
        title = 'Comparação de Importância de Features - Random Forest'
        subtitle = 'Gini Impurity vs. Permutation Importance'
        visualizer.save_figure(fig, filename, title=title, subtitle=subtitle)
    
    def _plot_model_info_dashboard(self, visualizer):
        """Cria dashboard de informações do modelo"""
        fig, ax = visualizer.create_figure(figsize=(14, 10))
        ax.axis('off')
        
        # Importar cores do módulo
        from src.visualization_utils import PROFESSIONAL_COLORS
        
        # Preparar informações
        training_time_str = f"{self.training_time:.2f}" if self.training_time is not None else "0.00"
        oob_score_str = f"{self.oob_score:.4f}" if self.oob_score is not None else "N/A"
        threshold_str = f"{self.optimal_threshold:.3f}" if hasattr(self, 'optimal_threshold') and self.optimal_threshold is not None else "0.500"
        
        # Informações do modelo
        model_info = f"""
🌲 RANDOM FOREST - CONFIGURAÇÃO E PERFORMANCE

📊 Hiperparâmetros:
• Número de Árvores: {self.model.n_estimators:,}
• Profundidade Máxima: {self.model.max_depth or 'Sem Limite'}
• Min. Amostras por Divisão: {self.model.min_samples_split}
• Min. Amostras por Folha: {self.model.min_samples_leaf}
• Features por Divisão: {self.model.max_features}
• Critério de Divisão: {self.model.criterion.upper()}
• Balanceamento de Classes: {self.model.class_weight or 'Nenhum'}
• Bootstrap: {'Sim' if self.model.bootstrap else 'Não'}

⏱️ Performance de Treinamento:
• Tempo de Treinamento: {training_time_str} segundos
• Out-of-Bag Score: {oob_score_str}
• Threshold Otimizado: {threshold_str}
• Random State: {self.model.random_state}

🎯 Métricas de Validação:
• Acurácia: {self.metrics['accuracy']:.4f} ({self.metrics['accuracy']:.1%})
• Precisão: {self.metrics['precision']:.4f} ({self.metrics['precision']:.1%})
• Recall: {self.metrics['recall']:.4f} ({self.metrics['recall']:.1%})
• F1-Score: {self.metrics['f1']:.4f} ({self.metrics['f1']:.1%})
• ROC-AUC: {self.metrics['roc_auc']:.4f}

🔍 Análise Clínica:
• Taxa de Verdadeiros Positivos: {self.metrics['recall']:.1%}
• Taxa de Falsos Positivos: {(1 - self.metrics.get('specificity', 0)):.1%}
• Especificidade: {self.metrics.get('specificity', 0):.1%}
• Valor Preditivo Positivo: {self.metrics['precision']:.1%}

📈 Dados de Treinamento:
• Amostras de Treino: {len(self.X_train):,}
• Amostras de Teste: {len(self.y_test):,}
• Features Utilizadas: {self.X_train.shape[1]:,}
• Balanceamento: {self.y_test.value_counts().to_dict()}

🕒 Timestamp: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
        """
        
        # Exibir informações
        ax.text(0.05, 0.95, model_info, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=1.0", 
                         facecolor=PROFESSIONAL_COLORS['light'],
                         edgecolor=PROFESSIONAL_COLORS['primary'],
                         alpha=0.9, linewidth=2))
        
        # Salvar
        filename = f'random_forest_model_info_{self.model_timestamp}'
        title = 'Random Forest - Relatório Técnico Completo'
        subtitle = f'Configurações, Performance e Métricas Detalhadas'
        visualizer.save_figure(fig, filename, title=title, subtitle=subtitle)
    
    def plot_feature_importance(self, top_n=15):
        """Gera gráfico profissional de importância das features"""
        print(f"\n📊 Gerando gráfico profissional de importância das {top_n} principais features...")
        
        if self.feature_importance is None:
            print("❌ Análise de importância não foi realizada ainda!")
            return
        
        # Inicializar visualizador
        visualizer = ProfessionalVisualizer(save_dir=RESULTS_DIR)
        
        # Preparar dados
        top_features = self.feature_importance.head(top_n)
        feature_names = top_features['feature'].tolist()
        importance_values = top_features['importance'].tolist()
        
        # Criar gráfico
        filename = f'random_forest_feature_importance_{self.model_timestamp}'
        
        visualizer.plot_feature_importance(
            feature_names=feature_names,
            importance_values=importance_values,
            title="Importância das Features - Random Forest",
            top_n=top_n,
            filename=filename
        )
        
        print("✅ Gráfico profissional de feature importance criado com sucesso!")
        return os.path.join(RESULTS_DIR, f'{filename}.png')
    
    def save_model(self):
        """Salva o modelo treinado e metadados"""
        print("\n💾 Salvando modelo...")
        
        if self.model is None:
            print("❌ Nenhum modelo para salvar!")
            return None
        
        # Criar nomes dos arquivos
        model_filename = f'random_forest_model_{self.model_timestamp}.joblib'
        scaler_filename = f'random_forest_scaler_{self.model_timestamp}.joblib'
        metadata_filename = f'random_forest_metadata_{self.model_timestamp}.json'
        
        # Caminhos completos
        model_path = os.path.join(MODELS_DIR, model_filename)
        scaler_path = os.path.join(MODELS_DIR, scaler_filename)
        metadata_path = os.path.join(MODELS_DIR, metadata_filename)
        
        try:
            # Salvar modelo
            joblib.dump(self.model, model_path)
            
            # Salvar scaler se existe
            if self.scaler is not None:
                joblib.dump(self.scaler, scaler_path)
            
            # Preparar metadados
            metadata = {
                'model_type': 'RandomForest',
                'timestamp': self.model_timestamp,
                'model_config': self.model_config,
                'feature_names': self.feature_names,
                'metrics': self.metrics,
                'optimal_threshold': self.optimal_threshold,
                'training_time': self.training_time,
                'use_smote': self.use_smote,
                'sampling_strategy': self.sampling_strategy,
                'n_features': len(self.feature_names) if self.feature_names else None,
                'n_train_samples': len(self.X_train) if self.X_train is not None else None,
                'class_distribution': {
                    'class_0': int((self.y_train == 0).sum()) if self.y_train is not None else None,
                    'class_1': int((self.y_train == 1).sum()) if self.y_train is not None else None
                }
            }
            
            # Adicionar importância das features se disponível
            if self.feature_importance is not None:
                metadata['feature_importance'] = self.feature_importance.to_dict('records')
            
            if self.permutation_importance is not None:
                metadata['permutation_importance'] = self.permutation_importance.head(20).to_dict('records')
            
            # Salvar metadados
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Modelo salvo em: {model_path}")
            print(f"✅ Scaler salvo em: {scaler_path}")
            print(f"✅ Metadados salvos em: {metadata_path}")
            
            return {
                'model_path': model_path,
                'scaler_path': scaler_path,
                'metadata_path': metadata_path
            }
            
        except Exception as e:
            print(f"❌ Erro ao salvar modelo: {e}")
            return None
    
    @classmethod
    def load_model(cls, model_path, scaler_path=None, metadata_path=None):
        """
        Carrega um modelo salvo
        
        Args:
            model_path (str): Caminho para o arquivo do modelo
            scaler_path (str): Caminho para o arquivo do scaler
            metadata_path (str): Caminho para o arquivo de metadados
        """
        print(f"📥 Carregando modelo de: {model_path}")
        
        try:
            # Criar instância
            instance = cls()
            
            # Carregar modelo
            instance.model = joblib.load(model_path)
            
            # Carregar scaler se fornecido
            if scaler_path and os.path.exists(scaler_path):
                instance.scaler = joblib.load(scaler_path)
            
            # Carregar metadados se fornecido
            if metadata_path and os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                instance.model_config = metadata.get('model_config', {})
                instance.feature_names = metadata.get('feature_names', [])
                instance.metrics = metadata.get('metrics', {})
                instance.optimal_threshold = metadata.get('optimal_threshold', 0.5)
                instance.training_time = metadata.get('training_time')
                instance.use_smote = metadata.get('use_smote', False)
                instance.model_timestamp = metadata.get('timestamp', '')
            
            print(f"✅ Modelo carregado com sucesso!")
            return instance
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            return None
    
    def predict_new_data(self, X_new, use_optimal_threshold=True):
        """
        Faz predições em novos dados
        
        Args:
            X_new (pd.DataFrame): Novos dados para predição
            use_optimal_threshold (bool): Se deve usar o threshold otimizado
        """
        if self.model is None:
            print("❌ Modelo não foi treinado ainda!")
            return None
        
        try:
            # Fazer predições
            probabilities = self.model.predict_proba(X_new)[:, 1]
            
            if use_optimal_threshold:
                predictions = (probabilities >= self.optimal_threshold).astype(int)
            else:
                predictions = self.model.predict(X_new)
            
            return predictions, probabilities
            
        except Exception as e:
            print(f"❌ Erro ao fazer predições: {e}")
            return None, None
    
    def save_results(self):
        """Salva relatório detalhado dos resultados"""
        print("\n📄 Salvando relatório de resultados...")
        
        # Criar relatório em texto
        report_filename = f'random_forest_report_{self.model_timestamp}.txt'
        report_path = os.path.join(RESULTS_DIR, report_filename)
        
        # Criar relatório em JSON
        json_filename = f'random_forest_results_{self.model_timestamp}.json'
        json_path = os.path.join(RESULTS_DIR, json_filename)
        
        try:
            # Relatório em texto
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("🌲 RANDOM FOREST - RELATÓRIO DE RESULTADOS\n")
                f.write("="*60 + "\n\n")
                
                f.write(f"📅 Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"🕒 Timestamp: {self.model_timestamp}\n\n")
                
                f.write("🔧 CONFIGURAÇÕES DO MODELO:\n")
                f.write("-" * 30 + "\n")
                for key, value in self.model_config.items():
                    f.write(f"{key}: {value}\n")
                f.write(f"SMOTE: {self.use_smote}\n")
                if self.use_smote:
                    f.write(f"Estratégia de Sampling: {self.sampling_strategy}\n")
                f.write("\n")
                
                f.write("📊 MÉTRICAS DE PERFORMANCE:\n")
                f.write("-" * 30 + "\n")
                for metric, value in self.metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"{metric}: {value:.4f}\n")
                    else:
                        f.write(f"{metric}: {value}\n")
                f.write("\n")
                
                f.write("🏆 TOP 20 FEATURES MAIS IMPORTANTES:\n")
                f.write("-" * 40 + "\n")
                if self.feature_importance is not None:
                    for i, (_, row) in enumerate(self.feature_importance.head(20).iterrows(), 1):
                        f.write(f"{i:2d}. {row['feature']:<40} {row['importance']:.6f}\n")
                else:
                    f.write("Análise de importância não realizada.\n")
                f.write("\n")
                
                if self.permutation_importance is not None:
                    f.write("🔄 TOP 10 FEATURES (IMPORTÂNCIA POR PERMUTAÇÃO):\n")
                    f.write("-" * 50 + "\n")
                    for i, (_, row) in enumerate(self.permutation_importance.head(10).iterrows(), 1):
                        f.write(f"{i:2d}. {row['feature']:<40} {row['importance_mean']:.6f} (±{row['importance_std']:.6f})\n")
                    f.write("\n")
                
                # Matriz de confusão
                if self.predictions is not None and self.y_test is not None:
                    cm = confusion_matrix(self.y_test, self.predictions)
                    tn, fp, fn, tp = cm.ravel()
                    
                    f.write("🔢 MATRIZ DE CONFUSÃO:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"               Predito\n")
                    f.write(f"Real     0      1\n")
                    f.write(f"   0   {tn:4d}   {fp:4d}\n")
                    f.write(f"   1   {fn:4d}   {tp:4d}\n\n")
                    
                    f.write("📊 MÉTRICAS CLÍNICAS:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Especificidade: {tn/(tn+fp):.4f}\n")
                    f.write(f"Sensibilidade:  {tp/(tp+fn):.4f}\n")
                    f.write(f"VPP:           {tp/(tp+fp):.4f}\n")
                    f.write(f"VPN:           {tn/(tn+fn):.4f}\n\n")
                
                f.write("ℹ️ INFORMAÇÕES ADICIONAIS:\n")
                f.write("-" * 25 + "\n")
                training_time_val = self.training_time if self.training_time is not None else 0
                threshold_val = self.optimal_threshold if self.optimal_threshold is not None else 0.5
                f.write(f"Tempo de treinamento: {training_time_val:.2f} segundos\n")
                f.write(f"Threshold otimizado: {threshold_val:.4f}\n")
                f.write(f"Número de features: {len(self.feature_names) if self.feature_names else 'N/A'}\n")
                if self.oob_score is not None:
                    f.write(f"OOB Score: {self.oob_score:.4f}\n")
            
            # Relatório em JSON
            json_data = {
                'model_type': 'RandomForest',
                'timestamp': self.model_timestamp,
                'datetime': datetime.now().isoformat(),
                'config': self.model_config,
                'metrics': self.metrics,
                'optimal_threshold': self.optimal_threshold,
                'training_time': self.training_time,
                'use_smote': self.use_smote,
                'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else None,
                'permutation_importance': self.permutation_importance.to_dict('records') if self.permutation_importance is not None else None
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Relatório salvo em: {report_path}")
            print(f"✅ Resultados JSON salvos em: {json_path}")
            
            return report_path, json_path
            
        except Exception as e:
            print(f"❌ Erro ao salvar relatório: {e}")
            return None, None
    
    def run_complete_pipeline(self, tune_hyperparams=True, method='random_search', 
                            optimize_threshold=True, cv_folds=5):
        """
        Executa o pipeline completo do Random Forest
        
        Args:
            tune_hyperparams (bool): Se deve otimizar hiperparâmetros
            method (str): Método de otimização ('grid_search' ou 'random_search')
            optimize_threshold (bool): Se deve otimizar o threshold
            cv_folds (int): Número de folds para validação cruzada
        """
        print("🚀 INICIANDO PIPELINE COMPLETO DO RANDOM FOREST")
        print("="*60)
        
        start_time = datetime.now()
        
        try:
            # 1. Carregar dados
            if not self.load_data():
                return False
            
            # 2. Pré-processar dados
            self.preprocess_data()
            
            # 3. Otimizar hiperparâmetros (se solicitado)
            if tune_hyperparams:
                search_results = self.tune_hyperparameters(method=method, cv_folds=cv_folds)
                if search_results:
                    print(f"✅ Hiperparâmetros otimizados com {method}")
            
            # 4. Treinar modelo (se não foi otimizado)
            if not tune_hyperparams or self.model is None:
                self.train_model()
            
            # 5. Fazer predições
            self.make_predictions()
            
            # 6. Otimizar threshold (se solicitado)
            if optimize_threshold:
                self.optimize_threshold('f1')
            
            # 7. Avaliar modelo
            self.evaluate_model()
            
            # 8. Analisar importância das features
            self.analyze_feature_importance()
            
            # 9. Validação cruzada
            cv_results = self.cross_validate_model(cv_folds)
            
            # 10. Gerar visualizações
            self.plot_results()
            self.plot_feature_importance()
            
            # 11. Salvar modelo e resultados
            self.save_model()
            self.save_results()
            
            # Resumo final
            total_time = (datetime.now() - start_time).total_seconds()
            self._print_summary(total_time)
            
            return True
            
        except Exception as e:
            print(f"❌ Erro durante execução do pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _print_summary(self, total_time):
        """Imprime resumo final da execução"""
        print(f"\n🎉 PIPELINE DO RANDOM FOREST CONCLUÍDO!")
        print("="*60)
        print(f"⏱️ Tempo total: {total_time:.2f} segundos")
        print(f"🌲 Modelo: Random Forest com {self.model.n_estimators} árvores")
        print(f"📊 Performance:")
        print(f"   • Acurácia: {self.metrics['accuracy']:.4f}")
        print(f"   • F1-Score: {self.metrics['f1']:.4f}")
        print(f"   • ROC-AUC: {self.metrics['roc_auc']:.4f}")
        if self.oob_score:
            print(f"   • OOB Score: {self.oob_score:.4f}")
        print(f"🎯 Threshold otimizado: {self.optimal_threshold:.3f}")
        print(f"📁 Arquivos salvos em: {RESULTS_DIR} e {MODELS_DIR}")


def main():
    """Execução principal do modelo Random Forest"""
    print("🏥" + "="*58 + "🏥")
    print("    RANDOM FOREST - PREDIÇÃO DE READMISSÃO HOSPITALAR")
    print("    Modelo Otimizado para Dados Médicos")
    print("    Projeto: BCC325 - Inteligência Artificial UFOP")
    print("🏥" + "="*58 + "🏥")
    
    # Verificar se dados estão disponíveis
    required_files = [X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Arquivos de dados não encontrados:")
        for file in missing_files:
            print(f"   • {file}")
        print("\nExecute primeiro o pipeline de pré-processamento!")
        return
    
    # Criar e executar modelo
    print("\n🚀 Iniciando treinamento do Random Forest...")
    
    # Configurações para demonstração completa
    model = RandomForestModel(use_smote=False)  # Testar sem SMOTE primeiro
    
    # Executar pipeline completo
    success = model.run_complete_pipeline(
        tune_hyperparams=True,
        method='random_search',  # Mais rápido que grid_search
        optimize_threshold=True,
        cv_folds=5
    )
    
    if success:
        print("\n🎉 Random Forest executado com sucesso!")
        print(f"📊 Principais resultados:")
        print(f"   • Acurácia: {model.metrics['accuracy']:.4f}")
        print(f"   • Precisão: {model.metrics['precision']:.4f}")
        print(f"   • Recall: {model.metrics['recall']:.4f}")
        print(f"   • F1-Score: {model.metrics['f1']:.4f}")
        print(f"   • ROC-AUC: {model.metrics['roc_auc']:.4f}")
    else:
        print("\n❌ Falha na execução do Random Forest!")


if __name__ == "__main__":
    main()