"""
Análise Avançada de Features - Sistema Completo de Feature Engineering

Este módulo implementa técnicas avançadas para análise, seleção e interpretação
de features em modelos de Machine Learning para predição médica.

Funcionalidades:
- Análise SHAP para interpretabilidade explícita
- Seleção recursiva de features com validação cruzada
- Análise de importância por permutação
- Detecção de correlações e multicolinearidade
- Análise de interações entre features
- Feature importance stability analysis
- Análise de contribuição individual por paciente

Autor: Thalles Felipe Rodrigues de Almeida Santos
Projeto: Predição de Readmissão Hospitalar em Pacientes com Diabetes
Data: Agosto 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import joblib
import json

warnings.filterwarnings('ignore')

# Configurar matplotlib
import matplotlib
matplotlib.use('Agg')
plt.ioff()

# Scikit-learn imports
from sklearn.feature_selection import (
    RFECV, SelectKBest, f_classif, chi2, mutual_info_classif,
    VarianceThreshold, SelectFromModel
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

# SHAP para interpretabilidade
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP não disponível. Instale com: pip install shap")

# Estatísticas avançadas
from scipy import stats
from scipy.stats import chi2_contingency, spearmanr, pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor


class AdvancedFeatureAnalyzer:
    """
    Classe para análise avançada e completa de features
    """
    
    def __init__(self, random_state: int = 42):
        """
        Inicializa o analisador de features
        
        Args:
            random_state: Seed para reprodutibilidade
        """
        self.random_state = random_state
        self.feature_rankings = {}
        self.importance_results = {}
        self.shap_values = None
        self.selected_features = None
        
    def comprehensive_feature_importance(self, models_dict, X_train, y_train, X_test, y_test):
        """
        Análise abrangente de importância usando múltiplos métodos
        
        Args:
            models_dict: Dicionário com modelos treinados {'nome': modelo}
            X_train: Features de treino
            y_train: Target de treino
            X_test: Features de teste
            y_test: Target de teste
            
        Returns:
            dict: Resultados completos de importância
        """
        print("🔍 Executando Análise Abrangente de Importância de Features...")
        
        importance_results = {}
        feature_names = X_train.columns.tolist()
        
        for model_name, model in models_dict.items():
            print(f"   Analisando modelo: {model_name}")
            
            model_results = {
                'feature_names': feature_names,
                'methods': {}
            }
            
            # 1. Importância nativa do modelo (se disponível)
            if hasattr(model, 'feature_importances_'):
                model_results['methods']['native_importance'] = {
                    'values': model.feature_importances_.tolist(),
                    'ranking': np.argsort(model.feature_importances_)[::-1].tolist()
                }
            
            # 2. Importância por coeficientes (para modelos lineares)
            if hasattr(model, 'coef_'):
                coef_abs = np.abs(model.coef_[0])
                model_results['methods']['coefficient_importance'] = {
                    'values': coef_abs.tolist(),
                    'ranking': np.argsort(coef_abs)[::-1].tolist()
                }
            
            # 3. Importância por permutação
            perm_importance = permutation_importance(
                model, X_test, y_test,
                n_repeats=10,
                random_state=self.random_state,
                scoring='roc_auc'
            )
            
            model_results['methods']['permutation_importance'] = {
                'values': perm_importance.importances_mean.tolist(),
                'std': perm_importance.importances_std.tolist(),
                'ranking': np.argsort(perm_importance.importances_mean)[::-1].tolist()
            }
            
            # 4. Análise SHAP (se disponível)
            if SHAP_AVAILABLE:
                try:
                    shap_results = self._calculate_shap_importance(model, X_test, model_name)
                    if shap_results:
                        model_results['methods']['shap_importance'] = shap_results
                except Exception as e:
                    print(f"   ⚠️  Erro no SHAP para {model_name}: {e}")
            
            importance_results[model_name] = model_results
        
        # 5. Métodos agnósticos de modelo
        print("   Executando métodos agnósticos...")
        
        # Correlação com target
        correlations = []
        for col in feature_names:
            if X_train[col].dtype in ['int64', 'float64']:
                corr, _ = pearsonr(X_train[col], y_train)
                correlations.append(abs(corr))
            else:
                correlations.append(0.0)
        
        importance_results['model_agnostic'] = {
            'feature_names': feature_names,
            'methods': {
                'correlation_with_target': {
                    'values': correlations,
                    'ranking': np.argsort(correlations)[::-1].tolist()
                }
            }
        }
        
        # Mutual Information
        mi_scores = mutual_info_classif(X_train, y_train, random_state=self.random_state)
        importance_results['model_agnostic']['methods']['mutual_information'] = {
            'values': mi_scores.tolist(),
            'ranking': np.argsort(mi_scores)[::-1].tolist()
        }
        
        # Univariate Statistical Tests
        f_scores, _ = f_classif(X_train, y_train)
        importance_results['model_agnostic']['methods']['f_statistic'] = {
            'values': f_scores.tolist(),
            'ranking': np.argsort(f_scores)[::-1].tolist()
        }
        
        self.importance_results = importance_results
        return importance_results
    
    def recursive_feature_elimination_cv(self, model, X_train, y_train, 
                                       min_features=10, step=1, cv_folds=5):
        """
        Seleção recursiva de features com validação cruzada
        
        Args:
            model: Modelo base para seleção
            X_train: Features de treino
            y_train: Target de treino
            min_features: Número mínimo de features
            step: Passo de eliminação
            cv_folds: Folds para validação cruzada
            
        Returns:
            dict: Resultados da seleção recursiva
        """
        print("🔄 Executando Seleção Recursiva de Features com CV...")
        
        # RFECV
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        rfecv = RFECV(
            estimator=clone(model),
            step=step,
            cv=cv_strategy,
            scoring='roc_auc',
            min_features_to_select=min_features,
            n_jobs=-1
        )
        
        rfecv.fit(X_train, y_train)
        
        # Análise detalhada dos resultados
        rfe_results = {
            'optimal_features': rfecv.n_features_,
            'feature_ranking': rfecv.ranking_.tolist(),
            'cv_scores': rfecv.cv_results_['mean_test_score'].tolist(),
            'cv_scores_std': rfecv.cv_results_['std_test_score'].tolist(),
            'selected_features': X_train.columns[rfecv.support_].tolist(),
            'eliminated_features': X_train.columns[~rfecv.support_].tolist(),
            'feature_elimination_order': [],
            'score_improvement': []
        }
        
        # Calcular ordem de eliminação e melhoria de score
        for n_features in range(len(X_train.columns), min_features - 1, -step):
            idx = len(X_train.columns) - n_features
            if idx < len(rfecv.cv_results_['mean_test_score']):
                score = rfecv.cv_results_['mean_test_score'][idx]
                rfe_results['score_improvement'].append(score)
        
        print(f"   ✅ Features selecionadas: {rfecv.n_features_}/{len(X_train.columns)}")
        print(f"   📈 Score CV: {rfecv.cv_results_['mean_test_score'][rfecv.n_features_ - min_features]:.4f}")
        
        self.selected_features = rfecv.support_
        return rfe_results
    
    def multicollinearity_analysis(self, X_train):
        """
        Análise detalhada de multicolinearidade
        
        Args:
            X_train: Features de treino
            
        Returns:
            dict: Análise de multicolinearidade
        """
        print("📊 Executando Análise de Multicolinearidade...")
        
        # Selecionar apenas features numéricas
        numeric_features = X_train.select_dtypes(include=[np.number])
        
        if len(numeric_features.columns) == 0:
            return {'error': 'Nenhuma feature numérica encontrada'}
        
        # Matriz de correlação
        correlation_matrix = numeric_features.corr()
        
        # Encontrar pares altamente correlacionados
        high_corr_pairs = []
        correlation_threshold = 0.8
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = abs(correlation_matrix.iloc[i, j])
                if corr_value > correlation_threshold:
                    high_corr_pairs.append({
                        'feature_1': correlation_matrix.columns[i],
                        'feature_2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        # Variance Inflation Factor (VIF)
        vif_data = []
        try:
            for i, feature in enumerate(numeric_features.columns):
                vif_value = variance_inflation_factor(numeric_features.values, i)
                vif_data.append({
                    'feature': feature,
                    'vif': vif_value
                })
        except Exception as e:
            print(f"   ⚠️  Erro no cálculo VIF: {e}")
            vif_data = []
        
        multicollinearity_results = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlation_pairs': high_corr_pairs,
            'vif_analysis': vif_data,
            'recommendations': []
        }
        
        # Gerar recomendações
        if high_corr_pairs:
            multicollinearity_results['recommendations'].append(
                f"Encontrados {len(high_corr_pairs)} pares com correlação > {correlation_threshold}"
            )
        
        high_vif_features = [item for item in vif_data if item['vif'] > 10]
        if high_vif_features:
            multicollinearity_results['recommendations'].append(
                f"{len(high_vif_features)} features com VIF > 10 (possível multicolinearidade)"
            )
        
        return multicollinearity_results
    
    def feature_interaction_analysis(self, model, X_train, y_train, top_features=10):
        """
        Análise de interações entre features
        
        Args:
            model: Modelo treinado
            X_train: Features de treino
            y_train: Target de treino
            top_features: Número de top features para analisar interações
            
        Returns:
            dict: Análise de interações
        """
        print("🔗 Executando Análise de Interações entre Features...")
        
        # Selecionar top features baseado na importância
        if hasattr(model, 'feature_importances_'):
            top_indices = np.argsort(model.feature_importances_)[-top_features:]
            top_feature_names = X_train.columns[top_indices].tolist()
        else:
            # Use correlação como fallback
            correlations = []
            for col in X_train.columns:
                if X_train[col].dtype in ['int64', 'float64']:
                    corr, _ = pearsonr(X_train[col], y_train)
                    correlations.append(abs(corr))
                else:
                    correlations.append(0.0)
            
            top_indices = np.argsort(correlations)[-top_features:]
            top_feature_names = X_train.columns[top_indices].tolist()
        
        # Análise de interações par a par
        interaction_results = {
            'analyzed_features': top_feature_names,
            'pairwise_interactions': [],
            'interaction_importance': {}
        }
        
        base_score = cross_val_score(model, X_train[top_feature_names], y_train, 
                                   cv=3, scoring='roc_auc').mean()
        
        for i in range(len(top_feature_names)):
            for j in range(i + 1, len(top_feature_names)):
                feature_1 = top_feature_names[i]
                feature_2 = top_feature_names[j]
                
                # Criar feature de interação
                X_interaction = X_train[top_feature_names].copy()
                
                # Multiplicação para features numéricas
                if (X_train[feature_1].dtype in ['int64', 'float64'] and 
                    X_train[feature_2].dtype in ['int64', 'float64']):
                    
                    interaction_name = f"{feature_1}_x_{feature_2}"
                    X_interaction[interaction_name] = X_train[feature_1] * X_train[feature_2]
                    
                    # Avaliar melhoria com interação
                    interaction_score = cross_val_score(
                        model, X_interaction, y_train, cv=3, scoring='roc_auc'
                    ).mean()
                    
                    improvement = interaction_score - base_score
                    
                    interaction_results['pairwise_interactions'].append({
                        'feature_1': feature_1,
                        'feature_2': feature_2,
                        'interaction_name': interaction_name,
                        'base_score': base_score,
                        'interaction_score': interaction_score,
                        'improvement': improvement
                    })
        
        # Ordenar por melhoria
        interaction_results['pairwise_interactions'].sort(
            key=lambda x: x['improvement'], reverse=True
        )
        
        return interaction_results
    
    def feature_stability_analysis(self, model, X_train, y_train, n_iterations=20):
        """
        Análise de estabilidade da importância das features
        
        Args:
            model: Modelo base
            X_train: Features de treino
            y_train: Target de treino
            n_iterations: Número de iterações para análise
            
        Returns:
            dict: Análise de estabilidade
        """
        print(f"📈 Executando Análise de Estabilidade das Features ({n_iterations} iterações)...")
        
        feature_names = X_train.columns.tolist()
        importance_matrix = []
        
        for i in range(n_iterations):
            # Bootstrap sampling
            n_samples = len(X_train)
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X_train.iloc[bootstrap_indices]
            y_bootstrap = y_train.iloc[bootstrap_indices]
            
            # Treinar modelo
            model_temp = clone(model)
            model_temp.fit(X_bootstrap, y_bootstrap)
            
            # Extrair importância
            if hasattr(model_temp, 'feature_importances_'):
                importance_matrix.append(model_temp.feature_importances_)
            else:
                # Usar importância por permutação como fallback
                perm_imp = permutation_importance(
                    model_temp, X_bootstrap, y_bootstrap,
                    n_repeats=3, random_state=i
                )
                importance_matrix.append(perm_imp.importances_mean)
        
        importance_matrix = np.array(importance_matrix)
        
        # Calcular estatísticas de estabilidade
        stability_results = {
            'feature_names': feature_names,
            'importance_stats': {
                'mean': importance_matrix.mean(axis=0).tolist(),
                'std': importance_matrix.std(axis=0).tolist(),
                'min': importance_matrix.min(axis=0).tolist(),
                'max': importance_matrix.max(axis=0).tolist(),
                'coefficient_variation': (importance_matrix.std(axis=0) / 
                                        (importance_matrix.mean(axis=0) + 1e-8)).tolist()
            },
            'ranking_stability': {},
            'top_stable_features': [],
            'unstable_features': []
        }
        
        # Análise de estabilidade do ranking
        rankings = []
        for iteration_importance in importance_matrix:
            ranking = np.argsort(iteration_importance)[::-1]
            rankings.append(ranking)
        
        rankings = np.array(rankings)
        
        # Calcular estabilidade do ranking para cada feature
        for i, feature in enumerate(feature_names):
            positions = []
            for ranking in rankings:
                position = np.where(ranking == i)[0][0]
                positions.append(position)
            
            stability_results['ranking_stability'][feature] = {
                'mean_position': np.mean(positions),
                'std_position': np.std(positions),
                'position_range': (int(np.min(positions)), int(np.max(positions)))
            }
        
        # Identificar features estáveis e instáveis
        cv_threshold = 0.5  # Coeficiente de variação
        for i, feature in enumerate(feature_names):
            cv = stability_results['importance_stats']['coefficient_variation'][i]
            std_pos = stability_results['ranking_stability'][feature]['std_position']
            
            if cv < cv_threshold and std_pos < 5:
                stability_results['top_stable_features'].append(feature)
            elif cv > 1.0 or std_pos > 10:
                stability_results['unstable_features'].append(feature)
        
        return stability_results
    
    def _calculate_shap_importance(self, model, X_test, model_name):
        """
        Calcula importância SHAP para interpretabilidade
        
        Args:
            model: Modelo treinado
            X_test: Features de teste
            model_name: Nome do modelo
            
        Returns:
            dict: Resultados SHAP
        """
        try:
            # Selecionar explainer apropriado
            if hasattr(model, 'predict_proba'):
                if 'forest' in model_name.lower() or 'tree' in model_name.lower():
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.KernelExplainer(model.predict_proba, X_test.sample(100))
            else:
                explainer = shap.KernelExplainer(model.predict, X_test.sample(100))
            
            # Calcular SHAP values (limitando a 500 amostras para performance)
            sample_size = min(500, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=self.random_state)
            
            shap_values = explainer.shap_values(X_sample)
            
            # Para classificação binária, pegar valores da classe positiva
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            
            # Calcular importância média absoluta
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            shap_results = {
                'values': mean_abs_shap.tolist(),
                'ranking': np.argsort(mean_abs_shap)[::-1].tolist(),
                'sample_size': sample_size
            }
            
            # Salvar gráficos SHAP
            self._save_shap_plots(shap_values, X_sample, model_name)
            
            return shap_results
            
        except Exception as e:
            print(f"   ⚠️  Erro no cálculo SHAP: {e}")
            return None
    
    def _save_shap_plots(self, shap_values, X_sample, model_name):
        """
        Salva gráficos SHAP
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(f'results/shap_summary_{model_name}_{timestamp}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Bar plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
            plt.tight_layout()
            plt.savefig(f'results/shap_bar_{model_name}_{timestamp}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"   ⚠️  Erro ao salvar gráficos SHAP: {e}")
    
    def generate_feature_report(self, importance_results, rfe_results, 
                              multicollinearity_results, interaction_results, 
                              stability_results, model_name):
        """
        Gera relatório completo de análise de features
        
        Args:
            importance_results: Resultados de importância
            rfe_results: Resultados RFE
            multicollinearity_results: Análise multicolinearidade
            interaction_results: Análise de interações
            stability_results: Análise de estabilidade
            model_name: Nome do modelo
            
        Returns:
            str: Relatório formatado
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
🔬 RELATÓRIO AVANÇADO DE ANÁLISE DE FEATURES - {model_name.upper()}
============================================================

📅 Data/Hora: {timestamp}
🔍 Análise: Importância, Seleção, Interações e Estabilidade

📊 SELEÇÃO RECURSIVA DE FEATURES (RFECV):
------------------------------
Features Originais: {len(rfe_results['feature_ranking'])}
Features Selecionadas: {rfe_results['optimal_features']}
Score CV Ótimo: {max(rfe_results['cv_scores']):.4f}

Top 10 Features Selecionadas:
"""
        
        for i, feature in enumerate(rfe_results['selected_features'][:10]):
            report += f"  {i+1:2d}. {feature}\n"
        
        report += f"""
🔗 ANÁLISE DE MULTICOLINEARIDADE:
------------------------------
Pares Altamente Correlacionados: {len(multicollinearity_results.get('high_correlation_pairs', []))}"""
        
        if 'vif_analysis' in multicollinearity_results:
            high_vif = [item for item in multicollinearity_results['vif_analysis'] if item['vif'] > 10]
            report += f"\nFeatures com VIF > 10: {len(high_vif)}"
        
        report += f"""

🔗 ANÁLISE DE INTERAÇÕES:
------------------------------
Features Analisadas: {len(interaction_results['analyzed_features'])}
Interações Testadas: {len(interaction_results['pairwise_interactions'])}

Top 5 Interações Mais Importantes:"""
        
        for i, interaction in enumerate(interaction_results['pairwise_interactions'][:5]):
            report += f"""
  {i+1}. {interaction['feature_1']} × {interaction['feature_2']}
     Melhoria: {interaction['improvement']:+.4f}"""
        
        report += f"""

📈 ANÁLISE DE ESTABILIDADE:
------------------------------
Features Estáveis: {len(stability_results['top_stable_features'])}
Features Instáveis: {len(stability_results['unstable_features'])}

Top 10 Features Mais Estáveis:"""
        
        stable_features = stability_results['top_stable_features'][:10]
        for i, feature in enumerate(stable_features):
            cv = stability_results['importance_stats']['coefficient_variation'][
                stability_results['feature_names'].index(feature)
            ]
            report += f"\n  {i+1:2d}. {feature} (CV: {cv:.3f})"
        
        if stability_results['unstable_features']:
            report += f"""

⚠️  Features Instáveis (requem atenção):"""
            for feature in stability_results['unstable_features'][:5]:
                cv = stability_results['importance_stats']['coefficient_variation'][
                    stability_results['feature_names'].index(feature)
                ]
                report += f"\n  • {feature} (CV: {cv:.3f})"
        
        report += f"""

✅ RECOMENDAÇÕES:
------------------------------
• Usar as {rfe_results['optimal_features']} features selecionadas pelo RFECV
• Considerar remoção de features com alta multicolinearidade
• Avaliar inclusão das top 3 interações de features
• Monitorar estabilidade das features instáveis em produção
• Aplicar regularização para features com alta variabilidade

"""
        return report
    
    def save_analysis_results(self, all_results, model_name, timestamp=None):
        """
        Salva todos os resultados da análise
        
        Args:
            all_results: Dicionário com todos os resultados
            model_name: Nome do modelo
            timestamp: Timestamp opcional
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"results/feature_analysis_{model_name}_{timestamp}.json"
        
        # Converter numpy arrays para listas
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
        
        results_serializable = convert_numpy(all_results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Análise salva em: {filename}")
