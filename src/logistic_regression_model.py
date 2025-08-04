"""
Módulo para implementação e avaliação do modelo de Regressão Logística
para predição de readmissão hospitalar diabética
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
import json
from datetime import datetime

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
        self.evaluation_results = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred),
            'recall': recall_score(self.y_test, self.y_pred),
            'f1': f1_score(self.y_test, self.y_pred),
            'roc_auc': roc_auc_score(self.y_test, self.y_pred_proba)
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
        """Cria visualizações dos resultados"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Resultados da Regressão Logística', fontsize=16, fontweight='bold')
        
        # 1. Matriz de Confusão (padrão)
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Matriz de Confusão (Limiar = 0.5)')
        axes[0,0].set_xlabel('Predito')
        axes[0,0].set_ylabel('Real')
        
        # 2. Matriz de Confusão (otimizada) se disponível
        if hasattr(self, 'y_pred_optimized'):
            cm_opt = confusion_matrix(self.y_test, self.y_pred_optimized)
            sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens', ax=axes[0,1])
            axes[0,1].set_title(f'Matriz de Confusão (Limiar = {self.best_threshold:.3f})')
            axes[0,1].set_xlabel('Predito')
            axes[0,1].set_ylabel('Real')
        else:
            # Curva ROC se não há limiar otimizado
            fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
            axes[0,1].plot(fpr, tpr, label=f'ROC-AUC = {self.evaluation_results["roc_auc"]:.3f}')
            axes[0,1].plot([0, 1], [0, 1], 'k--')
            axes[0,1].set_xlabel('Taxa de Falsos Positivos')
            axes[0,1].set_ylabel('Taxa de Verdadeiros Positivos')
            axes[0,1].set_title('Curva ROC')
            axes[0,1].legend()
        
        # 3. Curva Precision-Recall
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_pred_proba)
        axes[0,2].plot(recall, precision, 'b-', linewidth=2)
        axes[0,2].set_xlabel('Recall')
        axes[0,2].set_ylabel('Precision')
        axes[0,2].set_title('Curva Precision-Recall')
        axes[0,2].grid(True, alpha=0.3)
        
        # Marcar limiar ótimo se disponível
        if hasattr(self, 'best_threshold'):
            # Encontrar ponto mais próximo do limiar ótimo
            threshold_idx = np.abs(thresholds - self.best_threshold).argmin()
            if threshold_idx < len(precision) and threshold_idx < len(recall):
                axes[0,2].plot(recall[threshold_idx], precision[threshold_idx], 
                              'ro', markersize=8, label=f'Limiar Ótimo = {self.best_threshold:.3f}')
                axes[0,2].legend()
        
        # 4. Métricas de Performance (padrão)
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        values = [self.evaluation_results[m] for m in metrics if m in self.evaluation_results]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        bars = axes[1,0].bar(metrics[:len(values)], values, color=colors[:len(values)])
        axes[1,0].set_title('Métricas de Performance (Limiar = 0.5)')
        axes[1,0].set_ylabel('Score')
        axes[1,0].set_ylim(0, 1)
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, values):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{value:.3f}', ha='center', va='bottom')
        
        # 5. Métricas Otimizadas (se disponível)
        if hasattr(self, 'y_pred_optimized'):
            opt_metrics = ['accuracy_optimized', 'precision_optimized', 'recall_optimized', 'f1_optimized']
            opt_values = [self.evaluation_results[m] for m in opt_metrics if m in self.evaluation_results]
            opt_labels = [m.replace('_optimized', '') for m in opt_metrics[:len(opt_values)]]
            
            bars_opt = axes[1,1].bar(opt_labels, opt_values, color=colors[:len(opt_values)])
            axes[1,1].set_title(f'Métricas Otimizadas (Limiar = {self.best_threshold:.3f})')
            axes[1,1].set_ylabel('Score')
            axes[1,1].set_ylim(0, 1)
            axes[1,1].tick_params(axis='x', rotation=45)
            
            # Adicionar valores nas barras
            for bar, value in zip(bars_opt, opt_values):
                axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                              f'{value:.3f}', ha='center', va='bottom')
        else:
            # Distribuição das Probabilidades
            axes[1,1].hist(self.y_pred_proba[self.y_test == 0], bins=30, alpha=0.7, 
                          label='Classe 0', color='skyblue')
            axes[1,1].hist(self.y_pred_proba[self.y_test == 1], bins=30, alpha=0.7, 
                          label='Classe 1', color='lightcoral')
            axes[1,1].set_xlabel('Probabilidade Predita')
            axes[1,1].set_ylabel('Frequência')
            axes[1,1].set_title('Distribuição das Probabilidades')
            axes[1,1].legend()
        
        # 6. Distribuição das Probabilidades
        axes[1,2].hist(self.y_pred_proba[self.y_test == 0], bins=30, alpha=0.7, 
                      label='Classe 0', color='skyblue')
        axes[1,2].hist(self.y_pred_proba[self.y_test == 1], bins=30, alpha=0.7, 
                      label='Classe 1', color='lightcoral')
        axes[1,2].set_xlabel('Probabilidade Predita')
        axes[1,2].set_ylabel('Frequência')
        axes[1,2].set_title('Distribuição das Probabilidades')
        axes[1,2].legend()
        
        # Adicionar linha do limiar otimizado se disponível
        if hasattr(self, 'best_threshold'):
            axes[1,2].axvline(x=self.best_threshold, color='red', linestyle='--', 
                             label=f'Limiar Ótimo = {self.best_threshold:.3f}')
            axes[1,2].legend()
        
        plt.tight_layout()
        
        # Salvar gráfico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(RESULTS_DIR, 'logistic_regression_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return plot_path
    
    def plot_feature_importance(self, top_n=15):
        """Plota importância das features"""
        if self.feature_importance is None:
            self.analyze_feature_importance(top_n)
        
        plt.figure(figsize=(12, 8))
        
        # Top features
        top_features = self.feature_importance.head(top_n)
        colors = ['red' if coef < 0 else 'blue' for coef in top_features['coefficient']]
        
        plt.barh(range(len(top_features)), top_features['coefficient'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Coeficiente')
        plt.title(f'Top {top_n} Features Mais Importantes - Regressão Logística')
        plt.grid(axis='x', alpha=0.3)
        
        # Adicionar valores
        for i, (coef, feature) in enumerate(zip(top_features['coefficient'], top_features['feature'])):
            plt.text(coef + (0.01 if coef > 0 else -0.01), i, f'{coef:.3f}', 
                    va='center', ha='left' if coef > 0 else 'right')
        
        plt.tight_layout()
        
        # Salvar gráfico
        plot_path = os.path.join(RESULTS_DIR, 'logistic_regression_feature_importance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return plot_path
    
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
        """Salva os resultados em formato JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Criar diretório se não existir
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Preparar dados para salvar
        results_data = {
            'timestamp': timestamp,
            'model_type': 'Logistic Regression',
            'model_config': self.config,
            'metrics': self.evaluation_results,
            'confusion_matrix': confusion_matrix(self.y_test, self.y_pred).tolist(),
            'data_info': {
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'features': self.X_train.shape[1],
                'train_class_distribution': {
                    '0': int((self.y_train == 0).sum()),
                    '1': int((self.y_train == 1).sum())
                },
                'test_class_distribution': {
                    '0': int((self.y_test == 0).sum()),
                    '1': int((self.y_test == 1).sum())
                }
            }
        }
        
        # Adicionar matriz de confusão otimizada se disponível
        if hasattr(self, 'y_pred_optimized'):
            results_data['confusion_matrix_optimized'] = confusion_matrix(
                self.y_test, self.y_pred_optimized
            ).tolist()
        
        # Salvar JSON
        results_path = os.path.join(RESULTS_DIR, f'logistic_regression_results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=4)
        
        # Salvar relatório em texto
        report_path = os.path.join(RESULTS_DIR, f'logistic_regression_report_{timestamp}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO - REGRESSÃO LOGÍSTICA\n")
            f.write("="*60 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Configurações: {self.config}\n\n")
            
            # Métricas principais
            f.write("MÉTRICAS DE PERFORMANCE (Limiar = 0.5):\n")
            main_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            for metric in main_metrics:
                if metric in self.evaluation_results:
                    f.write(f"  {metric.upper():>10}: {self.evaluation_results[metric]:.4f}\n")
            
            # Métricas otimizadas se disponíveis
            if hasattr(self, 'best_threshold'):
                f.write(f"\nMÉTRICAS OTIMIZADAS (Limiar = {self.best_threshold:.4f}):\n")
                opt_metrics = ['accuracy_optimized', 'precision_optimized', 'recall_optimized', 'f1_optimized']
                for metric in opt_metrics:
                    if metric in self.evaluation_results:
                        clean_name = metric.replace('_optimized', '').upper()
                        f.write(f"  {clean_name:>10}: {self.evaluation_results[metric]:.4f}\n")
            
            # Validação cruzada
            if 'cv_results' in self.evaluation_results:
                f.write(f"\nVALIDAÇÃO CRUZADA (5-fold):\n")
                for metric, results in self.evaluation_results['cv_results'].items():
                    f.write(f"  {metric.upper():>10}: {results['mean']:.4f} (±{results['std']:.4f})\n")
            
            # Hiperparâmetros otimizados
            if 'best_params' in self.evaluation_results:
                f.write(f"\nMELHORES HIPERPARÂMETROS:\n")
                for param, value in self.evaluation_results['best_params'].items():
                    f.write(f"  {param:>15}: {value}\n")
            
            f.write(f"\nINFORMAÇÕES DO DATASET:\n")
            f.write(f"  Amostras treino: {len(self.X_train)}\n")
            f.write(f"  Amostras teste: {len(self.X_test)}\n")
            f.write(f"  Features: {self.X_train.shape[1]}\n")
            
            if self.feature_importance is not None:
                f.write(f"\nTOP 10 FEATURES MAIS IMPORTANTES:\n")
                for _, row in self.feature_importance.head(10).iterrows():
                    f.write(f"  {row['feature']:<30}: {row['coefficient']:>8.4f}\n")
        
        return results_path, report_path
    
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


if __name__ == "__main__":
    main()
