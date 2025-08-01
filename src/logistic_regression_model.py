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
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.config import (
    X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE,
    LOGISTIC_REGRESSION_CONFIG, CLASSIFICATION_METRICS,
    MODELS_DIR, RESULTS_DIR, RANDOM_STATE
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
        
    def load_data(self):
        """Carrega os dados de treino e teste"""
        print("Carregando dados...")
        self.X_train = pd.read_csv(X_TRAIN_FILE)
        self.X_test = pd.read_csv(X_TEST_FILE)
        self.y_train = pd.read_csv(Y_TRAIN_FILE).iloc[:, 0]
        self.y_test = pd.read_csv(Y_TEST_FILE).iloc[:, 0]
        
        print(f"Dados carregados: {self.X_train.shape[0]} treino, {self.X_test.shape[0]} teste")
        return True
    
    def preprocess_data(self):
        """Pré-processa os dados (normalização)"""
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        return True
    
    def train_model(self):
        """Treina o modelo de Regressão Logística"""
        self.model = LogisticRegression(**self.config)
        self.model.fit(self.X_train_scaled, self.y_train)
        return True
    
    def make_predictions(self):
        """Faz predições no conjunto de teste"""
        self.y_pred = self.model.predict(self.X_test_scaled)
        self.y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        return True
    
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
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Resultados da Regressão Logística', fontsize=16, fontweight='bold')
        
        # 1. Matriz de Confusão
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Matriz de Confusão')
        axes[0,0].set_xlabel('Predito')
        axes[0,0].set_ylabel('Real')
        
        # 2. Curva ROC
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        axes[0,1].plot(fpr, tpr, label=f'ROC-AUC = {self.evaluation_results["roc_auc"]:.3f}')
        axes[0,1].plot([0, 1], [0, 1], 'k--')
        axes[0,1].set_xlabel('Taxa de Falsos Positivos')
        axes[0,1].set_ylabel('Taxa de Verdadeiros Positivos')
        axes[0,1].set_title('Curva ROC')
        axes[0,1].legend()
        
        # 3. Métricas de Performance
        metrics = list(self.evaluation_results.keys())
        values = list(self.evaluation_results.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        bars = axes[1,0].bar(metrics, values, color=colors[:len(metrics)])
        axes[1,0].set_title('Métricas de Performance')
        axes[1,0].set_ylabel('Score')
        axes[1,0].set_ylim(0, 1)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, values):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Distribuição das Probabilidades
        axes[1,1].hist(self.y_pred_proba[self.y_test == 0], bins=30, alpha=0.7, 
                      label='Classe 0', color='skyblue')
        axes[1,1].hist(self.y_pred_proba[self.y_test == 1], bins=30, alpha=0.7, 
                      label='Classe 1', color='lightcoral')
        axes[1,1].set_xlabel('Probabilidade Predita')
        axes[1,1].set_ylabel('Frequência')
        axes[1,1].set_title('Distribuição das Probabilidades')
        axes[1,1].legend()
        
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
        
        # Salvar JSON
        results_path = os.path.join(RESULTS_DIR, f'logistic_regression_results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=4)
        
        # Salvar relatório em texto
        report_path = os.path.join(RESULTS_DIR, f'logistic_regression_report_{timestamp}.txt')
        with open(report_path, 'w') as f:
            f.write("RELATÓRIO - REGRESSÃO LOGÍSTICA\n")
            f.write("="*50 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Configurações: {self.config}\n\n")
            
            f.write("MÉTRICAS DE PERFORMANCE:\n")
            for metric, value in self.evaluation_results.items():
                f.write(f"  {metric.upper()}: {value:.4f}\n")
            
            f.write(f"\nDATABASE INFO:\n")
            f.write(f"  Amostras treino: {len(self.X_train)}\n")
            f.write(f"  Amostras teste: {len(self.X_test)}\n")
            f.write(f"  Features: {self.X_train.shape[1]}\n")
            
            if self.feature_importance is not None:
                f.write(f"\nTOP 10 FEATURES MAIS IMPORTANTES:\n")
                for _, row in self.feature_importance.head(10).iterrows():
                    f.write(f"  {row['feature']}: {row['coefficient']:.4f}\n")
        
        return results_path, report_path
    
    def run_complete_pipeline(self):
        """Executa o pipeline completo do modelo"""
        try:
            print("Iniciando pipeline da Regressão Logística...")
            
            # Etapas do pipeline
            self.load_data()
            self.preprocess_data()
            self.train_model()
            self.make_predictions()
            self.evaluate_model()
            
            # Análises adicionais
            self.analyze_feature_importance()
            cv_results = self.cross_validate_model()
            
            # Visualizações
            self.plot_results()
            self.plot_feature_importance()
            
            # Salvar modelo e resultados
            model_path, scaler_path = self.save_model()
            results_path, report_path = self.save_results()
            
            print("Pipeline concluído com sucesso!")
            print(f"Modelo salvo em: {model_path}")
            print(f"Resultados salvos em: {results_path}")
            
            return True
            
        except Exception as e:
            print(f"Erro no pipeline: {e}")
            return False


def main():
    """Execução principal do modelo de Regressão Logística"""
    print("Regressão Logística - Predição de Readmissão Hospitalar")
    print("="*60)
    
    try:
        model = LogisticRegressionModel()
        success = model.run_complete_pipeline()
        
        if success:
            print("\n✅ Modelo treinado com sucesso!")
            
            # Exibir métricas finais
            metrics = model.evaluation_results
            print(f"\nMétricas de Performance:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        else:
            print("❌ Erro no treinamento do modelo")
            
    except Exception as e:
        print(f"Erro: {e}")
        return None
    
    return model


if __name__ == "__main__":
    main()
