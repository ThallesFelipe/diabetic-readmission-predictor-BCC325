"""
Módulo de Modelo de Regressão Logística para Predição Médica

Este módulo implementa uma solução completa e profissional de Regressão Logística
otimizada para predição de readmissão hospitalar diabética, oferecendo:

Funcionalidades Avançadas:
- Implementação robusta de Regressão Logística com regularização
- Otimização automática de hiperparâmetros com Grid Search
- Balanceamento inteligente de classes para dados médicos
- Validação cruzada estratificada para máxima confiabilidade
- Otimização de threshold baseada em métricas clínicas
- Análise detalhada de importância de features
- Curvas ROC e Precision-Recall otimizadas
- Relatórios clínicos interpretáveis
- Visualizações específicas para contexto médico
- Sistema de salvamento e carregamento de modelos
- Métricas de avaliação focadas em aplicações médicas
- Pipeline completo de treinamento e validação

Autor: Thalles Felipe Rodrigues de Almeida Santos
Projeto: Predição de Readmissão Hospitalar em Pacientes com Diabetes Usando Aprendizado de Máquina
Instituição: Universidade Federal de Ouro Preto (UFOP)
Disciplina: Inteligência Artificial
Professor: Jadson Castro Gertrudes
Data: Agosto 2025
"""

import pandas as pd
import numpy as np

# Configurar matplotlib ANTES de importar pyplot para evitar problemas de thread
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo
import matplotlib.pyplot as plt
plt.ioff()  # Desabilitar modo interativo

import seaborn as sns
import joblib
import os
import sys
import json
from datetime import datetime

# Configurações específicas do matplotlib para Windows
matplotlib.rcParams['backend'] = 'Agg'

# Adicionar o diretório pai ao path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from src.config import (
    X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE,
    LOGISTIC_REGRESSION_CONFIG, CLASSIFICATION_METRICS,
    MODELS_DIR, RESULTS_DIR, RANDOM_STATE,
    HYPERPARAMETER_TUNING_CONFIG, THRESHOLD_OPTIMIZATION_CONFIG,
    VISUALIZATION_CONFIG
)
from src.visualization_utils import ProfessionalVisualizer


class LogisticRegressionModel:
    """Classe para implementação do modelo de Regressão Logística"""
    
    def __init__(self, model_config=None):
        """Inicializa o modelo de Regressão Logística"""
        self.config = model_config or LOGISTIC_REGRESSION_CONFIG
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        self.evaluation_results = {}
        self.feature_importance = None
        self.best_threshold = 0.5
        self.use_preprocessed_data = True  # Flag para usar dados já pré-processados
        
    def load_data(self):
        """Carrega os dados de treino e teste"""
        print("Carregando dados...")
        
        # Primeiro, tenta carregar dados já pré-processados (escalonados)
        try:
            scaled_train_path = X_TRAIN_FILE.replace('.csv', '_scaled.csv')
            scaled_test_path = X_TEST_FILE.replace('.csv', '_scaled.csv')
            
            if os.path.exists(scaled_train_path) and os.path.exists(scaled_test_path):
                print("Carregando dados já pré-processados...")
                self.X_train = pd.read_csv(scaled_train_path)
                self.X_test = pd.read_csv(scaled_test_path)
                self.use_preprocessed_data = True
                
                # Tentar carregar o scaler se disponível
                scaler_path = os.path.join(os.path.dirname(X_TRAIN_FILE), 'scaler.joblib')
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    print("Scaler carregado com sucesso.")
            else:
                raise FileNotFoundError("Dados pré-processados não encontrados")
                
        except (FileNotFoundError, Exception):
            print("Carregando dados brutos (será necessário pré-processar)...")
            self.X_train = pd.read_csv(X_TRAIN_FILE)
            self.X_test = pd.read_csv(X_TEST_FILE)
            self.use_preprocessed_data = False
        
        # Carregar variáveis target
        self.y_train = pd.read_csv(Y_TRAIN_FILE).iloc[:, 0]
        self.y_test = pd.read_csv(Y_TEST_FILE).iloc[:, 0]
        
        print(f"Dados carregados: {self.X_train.shape[0]} treino, {self.X_test.shape[0]} teste")
        return True
    
    def preprocess_data(self):
        """Pré-processa os dados (normalização) apenas se necessário"""
        if not self.use_preprocessed_data:
            print("Aplicando pré-processamento aos dados...")
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
        else:
            print("Usando dados já pré-processados.")
            self.X_train_scaled = self.X_train.values
            self.X_test_scaled = self.X_test.values
        return True
    
    def train_model(self):
        """Treina o modelo de Regressão Logística"""
        self.model = LogisticRegression(**self.config)
        self.model.fit(self.X_train_scaled, self.y_train)
        return True
    
    def tune_hyperparameters(self, param_grid=None, cv_folds=5, scoring='roc_auc'):
        """Otimiza hiperparâmetros usando GridSearchCV"""
        print("Otimizando hiperparâmetros...")
        
        if param_grid is None:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']  # liblinear é compatível com L1 e L2
            }
        
        # Usar GridSearchCV com validação cruzada
        grid_search = GridSearchCV(
            LogisticRegression(
                random_state=self.config.get('random_state', RANDOM_STATE),
                max_iter=self.config.get('max_iter', 1000),
                class_weight=self.config.get('class_weight', 'balanced')
            ), 
            param_grid, 
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE),
            scoring=scoring, 
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        print(f"Melhores hiperparâmetros encontrados: {grid_search.best_params_}")
        print(f"Melhor score CV ({scoring}): {grid_search.best_score_:.4f}")
        
        # Atualiza a configuração do modelo com os melhores parâmetros
        self.config.update(grid_search.best_params_)
        self.evaluation_results['best_cv_score'] = grid_search.best_score_
        self.evaluation_results['best_params'] = grid_search.best_params_
        
        # Treina o modelo final com os melhores parâmetros
        self.model = grid_search.best_estimator_
        
        return grid_search
    
    def make_predictions(self):
        """Faz predições no conjunto de teste"""
        self.y_pred = self.model.predict(self.X_test_scaled)
        self.y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        return True
    
    def optimize_threshold(self, metric='f1'):
        """Otimiza o limiar de decisão baseado em uma métrica específica"""
        print(f"Otimizando limiar de decisão para maximizar {metric.upper()}...")
        
        # Calcular curva precision-recall
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_pred_proba)
        
        if metric.lower() == 'f1':
            # Calcular F1-Score para cada limiar
            with np.errstate(divide='ignore', invalid='ignore'):
                f1_scores = 2 * (precision * recall) / (precision + recall)
                f1_scores = np.nan_to_num(f1_scores, nan=0.0)
            
            # Encontrar o limiar que maximiza F1-Score
            best_idx = np.argmax(f1_scores)
            self.best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            best_score = f1_scores[best_idx]
            
        elif metric.lower() == 'precision':
            # Encontrar limiar que maximiza precision (com recall mínimo)
            min_recall = 0.3  # Recall mínimo aceitável
            valid_indices = recall >= min_recall
            if np.any(valid_indices):
                valid_precision = precision[valid_indices]
                valid_thresholds = thresholds[:len(precision)][valid_indices]
                best_idx = np.argmax(valid_precision)
                self.best_threshold = valid_thresholds[best_idx] if len(valid_thresholds) > best_idx else 0.5
                best_score = valid_precision[best_idx]
            else:
                self.best_threshold = 0.5
                best_score = 0.0
                
        elif metric.lower() == 'recall':
            # Encontrar limiar que maximiza recall (com precision mínima)
            min_precision = 0.3  # Precision mínima aceitável
            valid_indices = precision >= min_precision
            if np.any(valid_indices):
                valid_recall = recall[valid_indices]
                valid_thresholds = thresholds[:len(precision)][valid_indices]
                best_idx = np.argmax(valid_recall)
                self.best_threshold = valid_thresholds[best_idx] if len(valid_thresholds) > best_idx else 0.5
                best_score = valid_recall[best_idx]
            else:
                self.best_threshold = 0.5
                best_score = 0.0
        
        print(f"Limiar ótimo encontrado: {self.best_threshold:.4f}")
        print(f"Melhor {metric.upper()}: {best_score:.4f}")
        
        # Fazer predições com o novo limiar
        self.y_pred_optimized = (self.y_pred_proba >= self.best_threshold).astype(int)
        
        # Calcular métricas com o limiar otimizado
        optimized_metrics = {
            'accuracy_optimized': accuracy_score(self.y_test, self.y_pred_optimized),
            'precision_optimized': precision_score(self.y_test, self.y_pred_optimized),
            'recall_optimized': recall_score(self.y_test, self.y_pred_optimized),
            'f1_optimized': f1_score(self.y_test, self.y_pred_optimized),
            'optimal_threshold': self.best_threshold
        }
        
        self.evaluation_results.update(optimized_metrics)
        
        return self.best_threshold, optimized_metrics
    
    def evaluate_model(self):
        """Avalia o modelo usando métricas de classificação"""
        # Calcular matriz de confusão
        cm = confusion_matrix(self.y_test, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calcular métricas clínicas adicionais
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # mesmo que recall
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # mesmo que precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        self.evaluation_results = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred),
            'recall': recall_score(self.y_test, self.y_pred),
            'f1': f1_score(self.y_test, self.y_pred),
            'roc_auc': roc_auc_score(self.y_test, self.y_pred_proba),
            'specificity': specificity,
            'sensitivity': sensitivity,
            'ppv': ppv,
            'npv': npv
        }
        return self.evaluation_results
    
    def analyze_feature_importance(self, top_n=20):
        """Analisa importância das features"""
        feature_names = self.X_train.columns
        coefficients = self.model.coef_[0]
        
        # Criar DataFrame com importância das features
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        self.feature_importance = importance_df.head(top_n)
        return self.feature_importance
    
    def cross_validate_model(self, cv_folds=5):
        """Realiza validação cruzada"""
        cv_results = {}
        
        for metric in CLASSIFICATION_METRICS:
            if metric == 'roc_auc':
                scoring = 'roc_auc'
            else:
                scoring = metric
                
            scores = cross_val_score(
                self.model, self.X_train_scaled, self.y_train,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE),
                scoring=scoring
            )
            
            cv_results[metric] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std()
            }
        
        return cv_results
    
    def plot_results(self):
        """Cria visualizações profissionais dos resultados usando o sistema padronizado"""
        print("\n🎨 Criando visualizações profissionais dos resultados...")
        
        # Inicializar o visualizador profissional
        visualizer = ProfessionalVisualizer(save_dir=RESULTS_DIR)
        
        # Preparar dados para o dashboard
        metrics = {
            'Acurácia': self.evaluation_results.get('accuracy', 0),
            'Precisão': self.evaluation_results.get('precision', 0),
            'Recall': self.evaluation_results.get('recall', 0),
            'F1-Score': self.evaluation_results.get('f1', 0),
            'ROC-AUC': self.evaluation_results.get('roc_auc', 0)
        }
        
        # Preparar matriz de confusão
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Preparar dados da curva ROC
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        auc_score = self.evaluation_results.get('roc_auc', 0)
        
        # Preparar dados de feature importance se disponível
        feature_names = None
        feature_importance = None
        if hasattr(self, 'feature_importance') and self.feature_importance is not None:
            feature_names = self.feature_importance['feature'].tolist()
            feature_importance = self.feature_importance['abs_coefficient'].tolist()
        
        # Criar dashboard completo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'logistic_regression_results_{timestamp}'
        
        visualizer.plot_model_results_dashboard(
            metrics=metrics,
            cm=cm,
            fpr=fpr,
            tpr=tpr,
            auc_score=auc_score,
            feature_names=feature_names,
            feature_importance=feature_importance,
            model_name="Regressão Logística",
            filename=filename
        )
        
        # Criar visualizações individuais adicionais se há threshold otimizado
        if hasattr(self, 'y_pred_optimized') and hasattr(self, 'best_threshold'):
            print(f"📊 Criando visualizações para threshold otimizado ({self.best_threshold:.3f})...")
            
            # Métricas otimizadas
            optimized_metrics = {
                'Acurácia': self.evaluation_results.get('accuracy_optimized', 0),
                'Precisão': self.evaluation_results.get('precision_optimized', 0),
                'Recall': self.evaluation_results.get('recall_optimized', 0),
                'F1-Score': self.evaluation_results.get('f1_optimized', 0),
                'ROC-AUC': self.evaluation_results.get('roc_auc', 0)
            }
            
            # Matriz de confusão otimizada
            cm_opt = confusion_matrix(self.y_test, self.y_pred_optimized)
            
            # Dashboard 
            filename_opt = f'logistic_regression_optimized_{timestamp}'
            
            visualizer.plot_model_results_dashboard(
                metrics=optimized_metrics,
                cm=cm_opt,
                fpr=fpr,
                tpr=tpr,
                auc_score=auc_score,
                feature_names=feature_names,
                feature_importance=feature_importance,
                model_name=f"Regressão Logística (Threshold = {self.best_threshold:.3f})",
                filename=filename_opt
            )
        
        # Criar gráfico de distribuição de probabilidades
        self._plot_probability_distribution(visualizer, timestamp)
        
        # Criar relatório técnico completo
        self._plot_model_info_dashboard(visualizer, timestamp)
        
        print("✅ Visualizações profissionais criadas com sucesso!")
        return os.path.join(RESULTS_DIR, f'{filename}.png')
    
    def _plot_probability_distribution(self, visualizer, timestamp):
        """Cria gráfico profissional da distribuição de probabilidades"""
        fig, ax = visualizer.create_figure(figsize=(10, 6))
        
        # Importar cores do módulo
        from src.visualization_utils import PROFESSIONAL_COLORS
        
        # Histogramas com estilo profissional
        ax.hist(self.y_pred_proba[self.y_test == 0], bins=40, alpha=0.7, 
               label='Não Readmitido (Classe 0)', color=PROFESSIONAL_COLORS['primary'],
               edgecolor='white', linewidth=0.5)
        ax.hist(self.y_pred_proba[self.y_test == 1], bins=40, alpha=0.7, 
               label='Readmitido (Classe 1)', color=PROFESSIONAL_COLORS['danger'],
               edgecolor='white', linewidth=0.5)
        
        # Linha do threshold otimizado se disponível
        if hasattr(self, 'best_threshold'):
            ax.axvline(x=self.best_threshold, color=PROFESSIONAL_COLORS['accent'], 
                      linestyle='--', linewidth=3, 
                      label=f'Threshold Otimizado = {self.best_threshold:.3f}')
        
        # Linha do threshold padrão
        ax.axvline(x=0.5, color=PROFESSIONAL_COLORS['neutral'], 
                  linestyle=':', linewidth=2, alpha=0.7,
                  label='Threshold Padrão = 0.500')
        
        # Configurações
        ax.set_xlabel('Probabilidade Predita de Readmissão', fontweight='bold')
        ax.set_ylabel('Frequência', fontweight='bold')
        ax.set_title('Distribuição das Probabilidades Preditas', fontweight='bold', pad=20)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        
        # Estatísticas na legenda
        mean_0 = self.y_pred_proba[self.y_test == 0].mean()
        mean_1 = self.y_pred_proba[self.y_test == 1].mean()
        
        textstr = f'Média Classe 0: {mean_0:.3f}\nMédia Classe 1: {mean_1:.3f}'
        props = dict(boxstyle='round', facecolor=PROFESSIONAL_COLORS['light'], alpha=0.8)
        ax.text(0.75, 0.8, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Salvar
        filename = f'logistic_regression_probability_distribution_{timestamp}'
        visualizer.save_figure(fig, filename)
    
    def _plot_model_info_dashboard(self, visualizer, timestamp):
        """Cria dashboard de informações técnicas do modelo (Relatório Técnico Completo)"""
        fig, ax = visualizer.create_figure(figsize=(14, 10))
        ax.axis('off')
        
        # Importar cores do módulo
        from src.visualization_utils import PROFESSIONAL_COLORS
        
        # Preparar informações
        training_time_str = "0.00"  # Será implementado no futuro
        threshold_str = f"{self.best_threshold:.3f}" if hasattr(self, 'best_threshold') and self.best_threshold is not None else "0.500"
        
        # Obter parâmetros do modelo
        penalty = getattr(self.model, 'penalty', 'l2')
        C_value = getattr(self.model, 'C', 1.0)
        solver = getattr(self.model, 'solver', 'lbfgs')
        max_iter = getattr(self.model, 'max_iter', 100)
        class_weight = getattr(self.model, 'class_weight', None)
        
        # Informações do modelo
        model_info = f"""
⚖️ REGRESSÃO LOGÍSTICA - CONFIGURAÇÃO E PERFORMANCE

📊 Hiperparâmetros:
• Regularização (Penalty): {penalty.upper()}
• Força de Regularização (C): {C_value}
• Algoritmo de Otimização: {solver.upper()}
• Iterações Máximas: {max_iter:,}
• Balanceamento de Classes: {class_weight or 'Nenhum'}
• Random State: {getattr(self.model, 'random_state', 'N/A')}

⏱️ Performance de Treinamento:
• Tempo de Treinamento: {training_time_str} segundos
• Threshold Otimizado: {threshold_str}
• Convergência: {'Sim' if getattr(self.model, 'n_iter_', [0])[0] < max_iter else 'Não'}
• Número de Iterações: {getattr(self.model, 'n_iter_', ['N/A'])[0]}

🎯 Métricas de Validação:
• Acurácia: {self.evaluation_results.get('accuracy', 0):.4f} ({self.evaluation_results.get('accuracy', 0):.1%})
• Precisão: {self.evaluation_results.get('precision', 0):.4f} ({self.evaluation_results.get('precision', 0):.1%})
• Recall: {self.evaluation_results.get('recall', 0):.4f} ({self.evaluation_results.get('recall', 0):.1%})
• F1-Score: {self.evaluation_results.get('f1', 0):.4f} ({self.evaluation_results.get('f1', 0):.1%})
• ROC-AUC: {self.evaluation_results.get('roc_auc', 0):.4f}

🔍 Análise Clínica:
• Taxa de Verdadeiros Positivos: {self.evaluation_results.get('recall', 0):.1%}
• Taxa de Falsos Positivos: {(1 - self.evaluation_results.get('specificity', 0)):.1%}
• Especificidade: {self.evaluation_results.get('specificity', 0):.1%}
• Valor Preditivo Positivo: {self.evaluation_results.get('precision', 0):.1%}

📈 Dados de Treinamento:
• Amostras de Treino: {len(self.X_train):,}
• Amostras de Teste: {len(self.X_test):,}
• Features Utilizadas: {self.X_train.shape[1]:,}
• Balanceamento: {dict(self.y_test.value_counts())}

📋 Métricas Otimizadas (se disponível):
• Acurácia Otimizada: {self.evaluation_results.get('accuracy_optimized', 'N/A')}
• Precisão Otimizada: {self.evaluation_results.get('precision_optimized', 'N/A')}
• Recall Otimizado: {self.evaluation_results.get('recall_optimized', 'N/A')}
• F1-Score Otimizado: {self.evaluation_results.get('f1_optimized', 'N/A')}

🔬 Validação Cruzada:
• CV Score Médio: {self.evaluation_results.get('best_cv_score', 'N/A')}
• Melhores Parâmetros: {str(self.evaluation_results.get('best_params', 'N/A'))}

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
        filename = f'logistic_regression_model_info_{timestamp}'
        title = 'Regressão Logística - Relatório Técnico Completo'
        subtitle = f'Configurações, Performance e Métricas Detalhadas'
        visualizer.save_figure(fig, filename, title=title, subtitle=subtitle)
    
    def plot_feature_importance(self, top_n=15):
        """Cria visualização profissional da importância das features"""
        if self.feature_importance is None:
            self.analyze_feature_importance(top_n)
        
        print(f"\n📊 Criando gráfico de importância das top {top_n} features...")
        
        # Inicializar visualizador
        visualizer = ProfessionalVisualizer(save_dir=RESULTS_DIR)
        
        # Preparar dados
        top_features = self.feature_importance.head(top_n)
        feature_names = top_features['feature'].tolist()
        importance_values = top_features['abs_coefficient'].tolist()
        
        # Criar gráfico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'logistic_regression_feature_importance_{timestamp}'
        
        visualizer.plot_feature_importance(
            feature_names=feature_names,
            importance_values=importance_values,
            title="Importância das Features - Regressão Logística",
            top_n=top_n,
            filename=filename
        )
        
        print("✅ Gráfico de feature importance criado com sucesso!")
        return os.path.join(RESULTS_DIR, f'{filename}.png')
    
    def save_model(self):
        """Salva o modelo e scaler treinados"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Criar diretório se não existir
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Salvar modelo
        model_path = os.path.join(MODELS_DIR, f'logistic_regression_model_{timestamp}.joblib')
        joblib.dump(self.model, model_path)
        
        # Salvar scaler
        scaler_path = os.path.join(MODELS_DIR, f'logistic_regression_scaler_{timestamp}.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        return model_path, scaler_path
    
    @classmethod
    def load_trained_model(cls, model_path, scaler_path=None):
        """Carrega um modelo treinado e scaler salvos"""
        print(f"Carregando modelo de: {model_path}")
        
        # Criar instância da classe
        instance = cls()
        
        # Carregar modelo
        instance.model = joblib.load(model_path)
        
        # Carregar scaler se fornecido
        if scaler_path and os.path.exists(scaler_path):
            instance.scaler = joblib.load(scaler_path)
            print(f"Scaler carregado de: {scaler_path}")
        else:
            print("Aviso: Scaler não carregado. Certifique-se de usar dados já pré-processados.")
        
        return instance
    
    def predict_new_data(self, X_new, use_optimal_threshold=True):
        """Faz predições em novos dados"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Execute run_complete_pipeline() primeiro.")
        
        # Pré-processar se scaler estiver disponível
        if self.scaler is not None:
            X_new_scaled = self.scaler.transform(X_new)
        else:
            print("Aviso: Usando dados sem pré-processamento. Certifique-se de que estão escalonados.")
            X_new_scaled = X_new
        
        # Fazer predições
        probabilities = self.model.predict_proba(X_new_scaled)[:, 1]
        
        # Usar limiar otimizado se disponível e solicitado
        if use_optimal_threshold and hasattr(self, 'best_threshold'):
            predictions = (probabilities >= self.best_threshold).astype(int)
            print(f"Usando limiar otimizado: {self.best_threshold:.4f}")
        else:
            predictions = self.model.predict(X_new_scaled)
            print("Usando limiar padrão: 0.5")
        
        return predictions, probabilities
    
    def save_results(self):
        """Salva relatório técnico detalhado dos resultados (similar ao Random Forest)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n📄 Salvando relatório técnico de resultados...")
        
        # Criar diretório se não existir
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Criar relatório em texto
        report_filename = f'logistic_regression_report_{timestamp}.txt'
        report_path = os.path.join(RESULTS_DIR, report_filename)
        
        # Criar relatório em JSON
        json_filename = f'logistic_regression_results_{timestamp}.json'
        json_path = os.path.join(RESULTS_DIR, json_filename)
        
        try:
            # Relatório técnico em texto (similar ao Random Forest)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("🔬 REGRESSÃO LOGÍSTICA - RELATÓRIO TÉCNICO COMPLETO\n")
                f.write("="*60 + "\n\n")
                
                f.write(f"📅 Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"🕒 Timestamp: {timestamp}\n\n")
                
                f.write("🔧 CONFIGURAÇÕES DO MODELO:\n")
                f.write("-" * 30 + "\n")
                for key, value in self.config.items():
                    f.write(f"{key}: {value}\n")
                
                # Adicionar informações do modelo atual se disponíveis
                if self.model is not None:
                    f.write(f"solver: {self.model.solver}\n")
                    f.write(f"C: {self.model.C}\n")
                    f.write(f"penalty: {self.model.penalty}\n")
                    f.write(f"class_weight: {self.model.class_weight}\n")
                    f.write(f"max_iter: {self.model.max_iter}\n")
                    f.write(f"random_state: {self.model.random_state}\n")
                    f.write(f"n_jobs: {self.model.n_jobs}\n")
                f.write("\n")
                
                f.write("📊 MÉTRICAS DE PERFORMANCE:\n")
                f.write("-" * 30 + "\n")
                
                # Métricas principais organizadas
                metric_order = [
                    'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision',
                    'log_loss', 'brier_score', 'true_negatives', 'false_positives', 
                    'false_negatives', 'true_positives', 'specificity', 'sensitivity',
                    'ppv', 'npv', 'training_time', 'optimal_threshold'
                ]
                
                for metric in metric_order:
                    if metric in self.evaluation_results:
                        value = self.evaluation_results[metric]
                        if isinstance(value, (int, float)):
                            f.write(f"{metric}: {value:.4f}\n")
                        else:
                            f.write(f"{metric}: {value}\n")
                
                # Adicionar hiperparâmetros otimizado se disponíveis
                if 'best_params' in self.evaluation_results:
                    for param, value in self.evaluation_results['best_params'].items():
                        f.write(f"{param}: {value}\n")
                
                f.write("\n")
                
                # Feature Importance
                if self.feature_importance is not None and len(self.feature_importance) > 0:
                    f.write("🏆 TOP 20 FEATURES MAIS IMPORTANTES (COEFICIENTES):\n")
                    f.write("-" * 50 + "\n")
                    
                    # Ordenar por valor absoluto do coeficiente
                    top_features = self.feature_importance.copy()
                    top_features['abs_coefficient'] = top_features['coefficient'].abs()
                    top_features = top_features.sort_values('abs_coefficient', ascending=False)
                    
                    for i, (_, row) in enumerate(top_features.head(20).iterrows(), 1):
                        f.write(f"{i:2d}. {row['feature']:<40} {row['coefficient']:>12.6f}\n")
                    f.write("\n")
                    
                    # Separar coeficientes positivos e negativos
                    positive_coefs = top_features[top_features['coefficient'] > 0].head(10)
                    negative_coefs = top_features[top_features['coefficient'] < 0].head(10)
                    
                    if len(positive_coefs) > 0:
                        f.write("📈 TOP 10 FEATURES COM COEFICIENTES POSITIVOS (aumentam risco):\n")
                        f.write("-" * 55 + "\n")
                        for i, (_, row) in enumerate(positive_coefs.iterrows(), 1):
                            f.write(f"{i:2d}. {row['feature']:<40} {row['coefficient']:>12.6f}\n")
                        f.write("\n")
                    
                    if len(negative_coefs) > 0:
                        f.write("📉 TOP 10 FEATURES COM COEFICIENTES NEGATIVOS (diminuem risco):\n")
                        f.write("-" * 55 + "\n")
                        for i, (_, row) in enumerate(negative_coefs.iterrows(), 1):
                            f.write(f"{i:2d}. {row['feature']:<40} {row['coefficient']:>12.6f}\n")
                        f.write("\n")
                
                # Matriz de confusão
                if self.y_pred is not None and self.y_test is not None:
                    cm = confusion_matrix(self.y_test, self.y_pred)
                    tn, fp, fn, tp = cm.ravel()
                    
                    f.write("🔢 MATRIZ DE CONFUSÃO:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"               Predito\n")
                    f.write(f"Real     0      1\n")
                    f.write(f"   0   {tn:4d}   {fp:4d}\n")
                    f.write(f"   1   {fn:4d}   {tp:4d}\n\n")
                    
                    f.write("📊 MÉTRICAS CLÍNICAS:\n")
                    f.write("-" * 20 + "\n")
                    specificity = tn/(tn+fp) if (tn+fp) > 0 else 0
                    sensitivity = tp/(tp+fn) if (tp+fn) > 0 else 0
                    ppv = tp/(tp+fp) if (tp+fp) > 0 else 0
                    npv = tn/(tn+fn) if (tn+fn) > 0 else 0
                    
                    f.write(f"Especificidade: {specificity:.4f}\n")
                    f.write(f"Sensibilidade:  {sensitivity:.4f}\n")
                    f.write(f"VPP:           {ppv:.4f}\n")
                    f.write(f"VPN:           {npv:.4f}\n\n")
                
                # Validação cruzada se disponível
                if 'cv_results' in self.evaluation_results:
                    f.write("🔄 VALIDAÇÃO CRUZADA:\n")
                    f.write("-" * 20 + "\n")
                    for metric, results in self.evaluation_results['cv_results'].items():
                        if isinstance(results, dict) and 'mean' in results:
                            f.write(f"{metric}: {results['mean']:.4f} (±{results['std']:.4f})\n")
                    f.write("\n")
                
                f.write("ℹ️ INFORMAÇÕES ADICIONAIS:\n")
                f.write("-" * 25 + "\n")
                training_time = self.evaluation_results.get('training_time', 0)
                f.write(f"Tempo de treinamento: {training_time:.2f} segundos\n")
                f.write(f"Threshold otimizado: {self.best_threshold:.4f}\n")
                f.write(f"Número de features: {self.X_train.shape[1] if self.X_train is not None else 'N/A'}\n")
                f.write(f"Amostras de treino: {len(self.X_train) if self.X_train is not None else 'N/A'}\n")
                f.write(f"Amostras de teste: {len(self.X_test) if self.X_test is not None else 'N/A'}\n")
                
                if self.X_train is not None and self.y_train is not None:
                    class_dist_train = {
                        '0': int((self.y_train == 0).sum()),
                        '1': int((self.y_train == 1).sum())
                    }
                    f.write(f"Distribuição de classes (treino): {class_dist_train}\n")
                
                if self.X_test is not None and self.y_test is not None:
                    class_dist_test = {
                        '0': int((self.y_test == 0).sum()),
                        '1': int((self.y_test == 1).sum())
                    }
                    f.write(f"Distribuição de classes (teste): {class_dist_test}\n")
            
            # Relatório em JSON
            json_data = {
                'model_type': 'LogisticRegression',
                'timestamp': timestamp,
                'datetime': datetime.now().isoformat(),
                'config': self.config,
                'metrics': self.evaluation_results,
                'optimal_threshold': self.best_threshold,
                'training_time': self.evaluation_results.get('training_time', 0),
                'data_info': {
                    'train_samples': len(self.X_train) if self.X_train is not None else 0,
                    'test_samples': len(self.X_test) if self.X_test is not None else 0,
                    'features': self.X_train.shape[1] if self.X_train is not None else 0,
                    'train_class_distribution': {
                        '0': int((self.y_train == 0).sum()) if self.y_train is not None else 0,
                        '1': int((self.y_train == 1).sum()) if self.y_train is not None else 0
                    } if self.y_train is not None else {},
                    'test_class_distribution': {
                        '0': int((self.y_test == 0).sum()) if self.y_test is not None else 0,
                        '1': int((self.y_test == 1).sum()) if self.y_test is not None else 0
                    } if self.y_test is not None else {}
                },
                'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else None,
                'confusion_matrix': confusion_matrix(self.y_test, self.y_pred).tolist() if (self.y_test is not None and self.y_pred is not None) else None
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
            
            print(f"✅ Relatório técnico salvo: {report_filename}")
            print(f"✅ Dados JSON salvos: {json_filename}")
            
            return json_path, report_path
            
        except Exception as e:
            print(f"❌ Erro ao salvar relatório: {e}")
            return None, None
    
    def run_complete_pipeline(self, tune_hyperparams=True, optimize_threshold=True):
        """Executa o pipeline completo do modelo"""
        try:
            print("Iniciando pipeline da Regressão Logística...")
            
            # Etapas básicas do pipeline
            self.load_data()
            self.preprocess_data()
            
            # Otimização de hiperparâmetros (opcional)
            if tune_hyperparams:
                print("\n🔧 Executando otimização de hiperparâmetros...")
                self.tune_hyperparameters()
            else:
                print("\n📊 Treinando modelo com hiperparâmetros padrão...")
                self.train_model()
            
            # Fazer predições
            self.make_predictions()
            
            # Avaliação básica
            self.evaluate_model()
            
            # Otimização do limiar (opcional)
            if optimize_threshold:
                print("\n⚖️ Otimizando limiar de decisão...")
                self.optimize_threshold(metric='f1')
            
            # Análises adicionais
            print("\n📈 Executando análises adicionais...")
            self.analyze_feature_importance()
            cv_results = self.cross_validate_model()
            
            # Adicionar resultados da validação cruzada
            self.evaluation_results['cv_results'] = {
                metric: {
                    'mean': results['mean'],
                    'std': results['std']
                } for metric, results in cv_results.items()
            }
            
            # Visualizações
            print("\n📊 Gerando visualizações...")
            self.plot_results()
            self.plot_feature_importance()
            
            # Dashboard técnico já criado pela função _plot_model_info_dashboard na plot_results()
            
            # Salvar modelo e resultados
            print("\n💾 Salvando modelo e resultados...")
            model_path, scaler_path = self.save_model()
            results_path, report_path = self.save_results()
            
            print("\n✅ Pipeline concluído com sucesso!")
            print(f"📁 Modelo salvo em: {model_path}")
            print(f"📁 Resultados salvos em: {results_path}")
            
            # Resumo dos resultados
            self._print_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ Erro no pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _print_summary(self):
        """Imprime resumo dos resultados"""
        print("\n" + "="*60)
        print("📊 RESUMO DOS RESULTADOS")
        print("="*60)
        
        # Métricas principais
        print("\n🎯 Métricas Principais (Limiar = 0.5):")
        main_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in main_metrics:
            if metric in self.evaluation_results:
                print(f"  {metric.upper():>10}: {self.evaluation_results[metric]:.4f}")
        
        # Métricas otimizadas se disponíveis
        if hasattr(self, 'best_threshold'):
            print(f"\n⚖️ Métricas Otimizadas (Limiar = {self.best_threshold:.4f}):")
            opt_metrics = ['accuracy_optimized', 'precision_optimized', 'recall_optimized', 'f1_optimized']
            for metric in opt_metrics:
                if metric in self.evaluation_results:
                    clean_name = metric.replace('_optimized', '').upper()
                    print(f"  {clean_name:>10}: {self.evaluation_results[metric]:.4f}")
        
        # Resultados da validação cruzada
        if 'cv_results' in self.evaluation_results:
            print(f"\n🔄 Validação Cruzada (5-fold):")
            for metric, results in self.evaluation_results['cv_results'].items():
                print(f"  {metric.upper():>10}: {results['mean']:.4f} (±{results['std']:.4f})")
        
        # Hiperparâmetros otimizados se disponíveis
        if 'best_params' in self.evaluation_results:
            print(f"\n🔧 Melhores Hiperparâmetros:")
            for param, value in self.evaluation_results['best_params'].items():
                print(f"  {param:>12}: {value}")
        
        print("="*60)


def main():
    """Execução principal do modelo de Regressão Logística"""
    print("🏥" + "="*58 + "🏥")
    print("    REGRESSÃO LOGÍSTICA - PREDIÇÃO DE READMISSÃO HOSPITALAR")
    print("    Modelo Avançado com Otimizações")
    print("🏥" + "="*58 + "🏥")
    
    try:
        model = LogisticRegressionModel()
        
        # Configurações de otimização
        tune_hyperparams = True    # Ativar otimização de hiperparâmetros
        optimize_threshold = True  # Ativar otimização do limiar
        
        print(f"\n⚙️ Configurações:")
        print(f"  🔧 Otimização de hiperparâmetros: {'✓' if tune_hyperparams else '✗'}")
        print(f"  ⚖️ Otimização de limiar: {'✓' if optimize_threshold else '✗'}")
        print(f"  📊 Validação cruzada: ✓")
        print(f"  📈 Análise de features: ✓")
        
        success = model.run_complete_pipeline(
            tune_hyperparams=tune_hyperparams,
            optimize_threshold=optimize_threshold
        )
        
        if success:
            print(f"\n✅ Modelo treinado com sucesso!")
            return model
        else:
            print("❌ Erro no treinamento do modelo")
            return None
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return None

    def generate_technical_dashboard(self):
        """Gera dashboard técnico completo (já implementado em _plot_model_info_dashboard)"""
        print("\n📊 Dashboard técnico já foi criado através da função _plot_model_info_dashboard!")
        print("   ✅ Relatório técnico completo disponível nos resultados.")
        return True


if __name__ == "__main__":
    main()
