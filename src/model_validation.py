"""
Validação Robusta de Modelos - Sistema Avançado de Cross-Validation

Este módulo implementa técnicas avançadas de validação cruzada para garantir
estimativas não-enviesadas da performance dos modelos de Machine Learning.

Funcionalidades:
- Validação cruzada aninhada (nested cross-validation)
- Validação estratificada com múltiplas métricas
- Análise de estabilidade dos modelos
- Validação temporal para dados sequenciais
- Bootstrap validation
- Curvas de aprendizado avançadas

Autor: Thalles Felipe Rodrigues de Almeida Santos
Projeto: Predição de Readmissão Hospitalar em Pacientes com Diabetes
Data: Agosto 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, GridSearchCV, 
    learning_curve, validation_curve, cross_validate
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, make_scorer
)
from sklearn.base import clone
from sklearn.utils import resample
import joblib


class AdvancedModelValidator:
    """
    Classe para validação avançada e robusta de modelos de Machine Learning
    """
    
    def __init__(self, cv_folds: int = 10, random_state: int = 42):
        """
        Inicializa o validador
        
        Args:
            cv_folds: Número de folds para validação cruzada
            random_state: Seed para reprodutibilidade
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv_strategy = StratifiedKFold(
            n_splits=cv_folds, 
            shuffle=True, 
            random_state=random_state
        )
        self.results_history = []
        
    def nested_cross_validation(self, model, X, y, param_grid, 
                              inner_cv_folds=3, scoring='roc_auc', fast_mode=True):
        """
        Validação cruzada aninhada OTIMIZADA para estimativa rápida da performance
        
        Args:
            model: Modelo base do scikit-learn
            X: Features
            y: Target
            param_grid: Grid de hiperparâmetros
            inner_cv_folds: Folds do loop interno (reduzido para 3)
            scoring: Métrica de avaliação
            fast_mode: Se True, usa apenas 3 folds externos
            
        Returns:
            dict: Resultados detalhados da validação aninhada
        """
        if fast_mode:
            print("⚡ Executando Validação Cruzada Aninhada (Modo Rápido - 3 folds)...")
            # Usar apenas 3 folds para ser mais rápido
            cv_strategy_fast = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        else:
            print("🔬 Executando Validação Cruzada Aninhada...")
            cv_strategy_fast = self.cv_strategy
        
        outer_scores = []
        best_params_list = []
        inner_cv = StratifiedKFold(n_splits=inner_cv_folds, shuffle=True, random_state=self.random_state)
        
        fold_results = []
        total_folds = 3 if fast_mode else self.cv_folds
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv_strategy_fast.split(X, y)):
            print(f"   ⚡ Fold {fold_idx + 1}/{total_folds}...")
            
            X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
            
            # Inner loop  com menos iterações
            grid_search = GridSearchCV(
                estimator=clone(model),
                param_grid=param_grid,
                cv=inner_cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=0,
                n_iter=10 if hasattr(model, 'random_state') else None  # RandomizedSearchCV seria melhor
            )
            
            grid_search.fit(X_train_fold, y_train_fold)
            
            # Avaliação no fold externo
            best_model = grid_search.best_estimator_
            
            if scoring == 'roc_auc':
                y_pred_proba = best_model.predict_proba(X_test_fold)[:, 1]
                score = roc_auc_score(y_test_fold, y_pred_proba)
            else:
                y_pred = best_model.predict(X_test_fold)
                if scoring == 'accuracy':
                    score = accuracy_score(y_test_fold, y_pred)
                elif scoring == 'f1':
                    score = f1_score(y_test_fold, y_pred)
                elif scoring == 'precision':
                    score = precision_score(y_test_fold, y_pred)
                elif scoring == 'recall':
                    score = recall_score(y_test_fold, y_pred)
            
            outer_scores.append(score)
            best_params_list.append(grid_search.best_params_)
            
            fold_results.append({
                'fold': fold_idx + 1,
                'score': score,
                'best_params': grid_search.best_params_,
                'best_inner_score': grid_search.best_score_
            })
        
        nested_results = {
            'outer_scores': np.array(outer_scores),
            'mean_score': np.mean(outer_scores),
            'std_score': np.std(outer_scores),
            'confidence_interval_95': (
                np.mean(outer_scores) - 1.96 * np.std(outer_scores) / np.sqrt(len(outer_scores)),
                np.mean(outer_scores) + 1.96 * np.std(outer_scores) / np.sqrt(len(outer_scores))
            ),
            'best_params_frequency': self._analyze_params_frequency(best_params_list),
            'fold_details': fold_results,
            'scoring_metric': scoring
        }
        
        print(f"✅ Validação Aninhada Concluída!")
        print(f"   Score Médio: {nested_results['mean_score']:.4f} ± {nested_results['std_score']:.4f}")
        print(f"   IC 95%: [{nested_results['confidence_interval_95'][0]:.4f}, {nested_results['confidence_interval_95'][1]:.4f}]")
        
        return nested_results
    
    def quick_validation(self, model, X, y, cv_folds=3):
        """
        Validação rápida e simples com métricas essenciais
        
        Args:
            model: Modelo treinado
            X: Features
            y: Target
            cv_folds: Número de folds (default 3 para ser rápido)
            
        Returns:
            dict: Resultados básicos mas informativos
        """
        print(f"⚡ Executando Validação Rápida ({cv_folds} folds)...")
        
        cv_quick = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Métricas essenciais
        scores_accuracy = cross_val_score(model, X, y, cv=cv_quick, scoring='accuracy', n_jobs=-1)
        scores_roc_auc = cross_val_score(model, X, y, cv=cv_quick, scoring='roc_auc', n_jobs=-1)
        scores_f1 = cross_val_score(model, X, y, cv=cv_quick, scoring='f1', n_jobs=-1)
        
        quick_results = {
            'cv_folds': cv_folds,
            'accuracy': {
                'mean': np.mean(scores_accuracy),
                'std': np.std(scores_accuracy),
                'scores': scores_accuracy
            },
            'roc_auc': {
                'mean': np.mean(scores_roc_auc),
                'std': np.std(scores_roc_auc),
                'scores': scores_roc_auc
            },
            'f1': {
                'mean': np.mean(scores_f1),
                'std': np.std(scores_f1),
                'scores': scores_f1
            }
        }
        
        print(f"✅ Validação Rápida Concluída!")
        print(f"   ROC-AUC: {quick_results['roc_auc']['mean']:.4f} ± {quick_results['roc_auc']['std']:.4f}")
        print(f"   Accuracy: {quick_results['accuracy']['mean']:.4f} ± {quick_results['accuracy']['std']:.4f}")
        print(f"   F1-Score: {quick_results['f1']['mean']:.4f} ± {quick_results['f1']['std']:.4f}")
        
        return quick_results
    
    def comprehensive_cross_validation(self, model, X, y, scoring_metrics=None):
        """
        Validação cruzada abrangente com múltiplas métricas
        
        Args:
            model: Modelo treinado
            X: Features
            y: Target
            scoring_metrics: Lista de métricas a avaliar
            
        Returns:
            dict: Resultados detalhados para cada métrica
        """
        if scoring_metrics is None:
            scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        print("📊 Executando Validação Cruzada Abrangente...")
        
        # Definir scorers customizados
        scorers = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1': make_scorer(f1_score, zero_division=0),
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
        }
        
        # Executar validação cruzada
        cv_results = cross_validate(
            model, X, y,
            cv=self.cv_strategy,
            scoring=scorers,
            n_jobs=-1,
            return_train_score=True
        )
        
        # Processar resultados
        comprehensive_results = {
            'cv_folds': self.cv_folds,
            'metrics': {},
            'fold_times': {
                'fit_time': cv_results['fit_time'],
                'score_time': cv_results['score_time']
            }
        }
        
        for metric in scoring_metrics:
            if f'test_{metric}' in cv_results:
                test_scores = cv_results[f'test_{metric}']
                train_scores = cv_results[f'train_{metric}']
                
                comprehensive_results['metrics'][metric] = {
                    'test_scores': test_scores,
                    'train_scores': train_scores,
                    'test_mean': np.mean(test_scores),
                    'test_std': np.std(test_scores),
                    'train_mean': np.mean(train_scores),
                    'train_std': np.std(train_scores),
                    'overfitting_indicator': np.mean(train_scores) - np.mean(test_scores),
                    'stability_coefficient': np.std(test_scores) / np.mean(test_scores) if np.mean(test_scores) != 0 else 0
                }
        
        return comprehensive_results
    
    def bootstrap_validation(self, model, X, y, n_bootstrap=100, sample_size=0.8, fast_mode=True):
        """
        Validação Bootstrap OTIMIZADA para estimativa robusta da performance
        
        Args:
            model: Modelo treinado
            X: Features
            y: Target
            n_bootstrap: Número de amostras bootstrap (reduzido para 100)
            sample_size: Proporção da amostra original
            fast_mode: Se True, usa apenas 100 amostras
            
        Returns:
            dict: Resultados da validação bootstrap
        """
        if fast_mode:
            n_bootstrap = min(n_bootstrap, 100)
            print(f"⚡ Executando Validação Bootstrap Rápida ({n_bootstrap} amostras)...")
        else:
            print(f"🔄 Executando Validação Bootstrap com {n_bootstrap} amostras...")
        
        bootstrap_scores = {
            'accuracy': [],
            'roc_auc': [],
            'f1': []  # Reduzido para métricas essenciais
        }
        
        n_samples = int(len(X) * sample_size)
        
        for i in range(n_bootstrap):
            if (i + 1) % 25 == 0:  # Report menos frequente
                print(f"   ⚡ Progresso: {i + 1}/{n_bootstrap}")
            
            # Criar amostra bootstrap
            X_boot, y_boot = resample(X, y, n_samples=n_samples, random_state=i)
            
            # Dividir em treino e teste
            split_idx = int(0.8 * len(X_boot))
            X_train_boot, X_test_boot = X_boot[:split_idx], X_boot[split_idx:]
            y_train_boot, y_test_boot = y_boot[:split_idx], y_boot[split_idx:]
            
            # Treinar e avaliar
            model_boot = clone(model)
            model_boot.fit(X_train_boot, y_train_boot)
            
            y_pred = model_boot.predict(X_test_boot)
            y_pred_proba = model_boot.predict_proba(X_test_boot)[:, 1]
            
            # Calcular métricas essenciais
            bootstrap_scores['accuracy'].append(accuracy_score(y_test_boot, y_pred))
            bootstrap_scores['f1'].append(f1_score(y_test_boot, y_pred, zero_division=0))
            bootstrap_scores['roc_auc'].append(roc_auc_score(y_test_boot, y_pred_proba))
        
        # Calcular estatísticas
        bootstrap_results = {}
        for metric, scores in bootstrap_scores.items():
            scores_array = np.array(scores)
            bootstrap_results[metric] = {
                'mean': np.mean(scores_array),
                'std': np.std(scores_array),
                'median': np.median(scores_array),
                'percentile_2_5': np.percentile(scores_array, 2.5),
                'percentile_97_5': np.percentile(scores_array, 97.5),
                'confidence_interval_95': (np.percentile(scores_array, 2.5), np.percentile(scores_array, 97.5))
            }
        
        print("✅ Validação Bootstrap Concluída!")
        return bootstrap_results
    
    def advanced_learning_curves(self, model, X, y, train_sizes=None, cv_folds=5):
        """
        Gera curvas de aprendizado avançadas com análise detalhada
        
        Args:
            model: Modelo base
            X: Features
            y: Target
            train_sizes: Tamanhos de treino a avaliar
            cv_folds: Folds para validação cruzada
            
        Returns:
            dict: Dados das curvas de aprendizado
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        print("📈 Gerando Curvas de Aprendizado Avançadas...")
        
        # Gerar curvas para múltiplas métricas
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        learning_curves_data = {}
        
        for metric in metrics:
            print(f"   Processando métrica: {metric}")
            
            if metric == 'roc_auc':
                scorer = make_scorer(roc_auc_score, needs_proba=True)
            else:
                scorer_map = {
                    'accuracy': accuracy_score,
                    'precision': precision_score,
                    'recall': recall_score,
                    'f1': f1_score
                }
                scorer = make_scorer(scorer_map[metric], zero_division=0)
            
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y,
                train_sizes=train_sizes,
                cv=cv_folds,
                scoring=scorer,
                n_jobs=-1,
                random_state=self.random_state
            )
            
            learning_curves_data[metric] = {
                'train_sizes': train_sizes_abs,
                'train_scores_mean': np.mean(train_scores, axis=1),
                'train_scores_std': np.std(train_scores, axis=1),
                'val_scores_mean': np.mean(val_scores, axis=1),
                'val_scores_std': np.std(val_scores, axis=1),
                'gap_mean': np.mean(train_scores, axis=1) - np.mean(val_scores, axis=1)
            }
        
        return learning_curves_data
    
    def model_stability_analysis(self, model, X, y, n_iterations=20, fast_mode=True):
        """
        Analisa a estabilidade do modelo através de múltiplas execuções OTIMIZADA
        
        Args:
            model: Modelo base
            X: Features
            y: Target
            n_iterations: Número de iterações para análise (reduzido para 20)
            fast_mode: Se True, usa apenas 20 iterações e 3 folds
            
        Returns:
            dict: Análise de estabilidade
        """
        if fast_mode:
            n_iterations = min(n_iterations, 20)
            cv_folds = 3
            print(f"⚡ Analisando Estabilidade do Modelo (Modo Rápido: {n_iterations} iterações, {cv_folds} folds)...")
        else:
            cv_folds = 5
            print(f"🔍 Analisando Estabilidade do Modelo ({n_iterations} iterações)...")
        
        stability_scores = []
        feature_importances = []
        
        for i in range(n_iterations):
            # Criar split aleatório com menos folds
            cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=i)
            scores = cross_val_score(model, X, y, cv=cv_splitter, scoring='roc_auc', n_jobs=-1)
            stability_scores.append(np.mean(scores))
            
            # Treinar modelo para importância das features (apenas algumas iterações)
            if i % 5 == 0:  # Calcular importância apenas a cada 5 iterações
                model_temp = clone(model)
                model_temp.fit(X, y)
                
                if hasattr(model_temp, 'feature_importances_'):
                    feature_importances.append(model_temp.feature_importances_)
        
        stability_scores = np.array(stability_scores)
        
        stability_analysis = {
            'score_stability': {
                'mean': np.mean(stability_scores),
                'std': np.std(stability_scores),
                'min': np.min(stability_scores),
                'max': np.max(stability_scores),
                'coefficient_variation': np.std(stability_scores) / np.mean(stability_scores)
            }
        }
        
        if feature_importances:
            feature_importances = np.array(feature_importances)
            stability_analysis['feature_importance_stability'] = {
                'mean_importances': np.mean(feature_importances, axis=0),
                'std_importances': np.std(feature_importances, axis=0),
                'stability_ranking': np.argsort(np.std(feature_importances, axis=0))
            }
        
        return stability_analysis
    
    def save_validation_results(self, results, model_name, timestamp=None):
        """
        Salva resultados da validação em arquivo
        
        Args:
            results: Resultados da validação
            model_name: Nome do modelo
            timestamp: Timestamp (opcional)
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"results/validation_results_{model_name}_{timestamp}.json"
        
        # Converter numpy arrays para listas para serialização JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_serializable = convert_numpy(results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Resultados salvos em: {filename}")
    
    def _analyze_params_frequency(self, params_list):
        """
        Analisa a frequência dos melhores parâmetros na validação aninhada
        
        Args:
            params_list: Lista de dicionários com os melhores parâmetros
            
        Returns:
            dict: Frequência de cada combinação de parâmetros
        """
        param_frequency = {}
        
        for params in params_list:
            params_str = str(sorted(params.items()))
            param_frequency[params_str] = param_frequency.get(params_str, 0) + 1
        
        # Converter de volta para formato legível
        readable_frequency = {}
        for params_str, count in param_frequency.items():
            readable_frequency[params_str] = {
                'count': count,
                'frequency': count / len(params_list)
            }
        
        return readable_frequency
    
    def generate_validation_report(self, nested_results, comprehensive_results, 
                                 bootstrap_results, stability_results, model_name):
        """
        Gera relatório completo de validação
        
        Args:
            nested_results: Resultados da validação aninhada
            comprehensive_results: Resultados da validação abrangente
            bootstrap_results: Resultados da validação bootstrap
            stability_results: Resultados da análise de estabilidade
            model_name: Nome do modelo
            
        Returns:
            str: Relatório formatado
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
🔬 RELATÓRIO AVANÇADO DE VALIDAÇÃO - {model_name.upper()}
============================================================

📅 Data/Hora: {timestamp}
🔄 Método: Validação Cruzada Robusta e Análise de Estabilidade

🎯 VALIDAÇÃO CRUZADA ANINHADA:
------------------------------
Score Médio: {nested_results['mean_score']:.4f} ± {nested_results['std_score']:.4f}
Intervalo de Confiança 95%: [{nested_results['confidence_interval_95'][0]:.4f}, {nested_results['confidence_interval_95'][1]:.4f}]
Métrica: {nested_results['scoring_metric']}

📊 VALIDAÇÃO ABRANGENTE (CV {comprehensive_results['cv_folds']}-fold):
------------------------------"""

        for metric, data in comprehensive_results['metrics'].items():
            report += f"""
{metric.upper()}:
  Test: {data['test_mean']:.4f} ± {data['test_std']:.4f}
  Train: {data['train_mean']:.4f} ± {data['train_std']:.4f}
  Overfitting: {data['overfitting_indicator']:.4f}
  Estabilidade: {data['stability_coefficient']:.4f}"""

        report += f"""

🔄 VALIDAÇÃO BOOTSTRAP:
------------------------------"""

        for metric, data in bootstrap_results.items():
            report += f"""
{metric.upper()}:
  Média: {data['mean']:.4f} ± {data['std']:.4f}
  IC 95%: [{data['confidence_interval_95'][0]:.4f}, {data['confidence_interval_95'][1]:.4f}]"""

        report += f"""

🔍 ANÁLISE DE ESTABILIDADE:
------------------------------
Score Médio: {stability_results['score_stability']['mean']:.4f}
Desvio Padrão: {stability_results['score_stability']['std']:.4f}
Coeficiente de Variação: {stability_results['score_stability']['coefficient_variation']:.4f}
Range: [{stability_results['score_stability']['min']:.4f}, {stability_results['score_stability']['max']:.4f}]

✅ CONCLUSÕES:
------------------------------
- Modelo demonstra {'alta' if stability_results['score_stability']['coefficient_variation'] < 0.05 else 'média' if stability_results['score_stability']['coefficient_variation'] < 0.10 else 'baixa'} estabilidade
- Performance {'consistente' if nested_results['std_score'] < 0.05 else 'moderadamente variável' if nested_results['std_score'] < 0.10 else 'altamente variável'} entre folds
- {'Baixo' if comprehensive_results['metrics']['roc_auc']['overfitting_indicator'] < 0.05 else 'Moderado' if comprehensive_results['metrics']['roc_auc']['overfitting_indicator'] < 0.15 else 'Alto'} risco de overfitting

"""
        return report
