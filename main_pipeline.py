"""
Pipeline Principal de Análise de Readmissão Hospitalar Diabética

Este script orquestra a execução completa do pipeline de Machine Learning, incluindo:
1. Análise Exploratória de Dados (EDA) - Compreensão inicial dos padrões
2. Limpeza e Pré-processamento - Tratamento de dados faltantes e inconsistências  
3. Engenharia de Features - Transformação e preparação para modelagem
4. Treinamento de Modelos ML - Regressão Logística e Random Forest
5. Avaliação e Validação - Métricas de performance e visualizações
6. Geração de Relatórios - Documentação completa dos resultados

Modos de Execução:
    python main_pipeline.py                # Pipeline completo com otimizações
    python main_pipeline.py --demo        # Demonstração rápida (apenas modelagem)
    python main_pipeline.py --fast        # Pipeline completo sem otimização de hiperparâmetros

Autor: Thalles Felipe Rodrigues de Almeida Santos
Projeto: Predição de Readmissão Hospitalar em Pacientes com Diabetes Usando Aprendizado de Máquina
Instituição: Universidade Federal de Ouro Preto (UFOP)
Disciplina: Inteligência Artificial
Professor: Jadson Castro Gertrudes
Data: Agosto 2025
"""

import os
import sys
import time
import traceback
from datetime import datetime

# Adicionar diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importações dos módulos do projeto
from src.exploratory_analysis import ExploratoryDataAnalysis
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.logistic_regression_model import LogisticRegressionModel
from src.config import (
    RAW_DATA_FILE, CLEAN_DATA_FILE, PROCESSED_DATA_FILE,
    MODELS_DIR, RESULTS_DIR, DATA_DIR
)


def ensure_directories():
    """Garante que todos os diretórios necessários existam"""
    directories = [DATA_DIR, RESULTS_DIR, MODELS_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Diretório verificado: {directory}")


def print_pipeline_header():
    """Imprime cabeçalho do pipeline"""
    print("🏥" + "="*70 + "🏥")
    print("    PIPELINE COMPLETO DE ANÁLISE DE READMISSÃO HOSPITALAR DIABÉTICA")
    print("    Projeto: Predição de Readmissão Hospitalar em Pacientes com Diabetes Usando Aprendizado de Máquina")
    print("    Dataset: UCI Diabetes 130-US hospitals (1999-2008)")
    print("    Objetivo: Predizer readmissão em < 30 dias")
    print("🏥" + "="*70 + "🏥")


def print_stage_header(stage_num, stage_name, description=""):
    """Imprime cabeçalho padronizado para cada etapa"""
    print(f"\n{'='*20} ETAPA {stage_num}: {stage_name.upper()} {'='*20}")
    if description:
        print(f"Objetivo: {description}")
    print("=" * (42 + len(stage_name)))


def run_exploratory_analysis():
    """Executa análise exploratória dos dados"""
    print_stage_header(1, "Análise Exploratória", 
                      "Compreensão inicial dos dados, identificação de padrões e problemas")
    
    try:
        eda = ExploratoryDataAnalysis(RAW_DATA_FILE)
        df_analyzed = eda.run_complete_analysis()
        
        print(f"✅ Análise exploratória concluída!")
        print(f"   📊 Dataset: {df_analyzed.shape[0]:,} registros, {df_analyzed.shape[1]} colunas")
        print(f"   📁 Relatórios gerados em: {RESULTS_DIR}")
        
        return df_analyzed
        
    except Exception as e:
        print(f"❌ Erro na análise exploratória: {e}")
        raise


def run_data_cleaning():
    """Executa limpeza e pré-processamento dos dados"""
    print_stage_header(2, "Limpeza dos Dados", 
                      "Tratamento de dados faltantes, remoção de inconsistências")
    
    try:
        cleaner = DataCleaner(RAW_DATA_FILE)
        df_clean = cleaner.clean_data(CLEAN_DATA_FILE)
        
        # Mostrar resumo da limpeza
        cleaner.get_cleaning_summary()
        
        # Obter estatísticas diretamente dos dados
        original_records = cleaner.cleaning_stats.get('original_records', 'N/A')
        final_records = len(df_clean) if df_clean is not None else 0
        removed_records = (original_records - final_records) if isinstance(original_records, int) else 'N/A'
        
        print(f"✅ Limpeza dos dados concluída!")
        print(f"   📊 Registros originais: {original_records:,}" if isinstance(original_records, int) else f"   📊 Registros originais: {original_records}")
        print(f"   📊 Registros finais: {final_records:,}")
        print(f"   🗑️ Registros removidos: {removed_records:,}" if isinstance(removed_records, int) else f"   🗑️ Registros removidos: {removed_records}")
        print(f"   💾 Dados limpos salvos em: {CLEAN_DATA_FILE}")
        
        return df_clean
        
    except Exception as e:
        print(f"❌ Erro na limpeza dos dados: {e}")
        raise


def run_feature_engineering():
    """Executa engenharia de features e preparação para modelagem"""
    print_stage_header(3, "Engenharia de Features", 
                      "Transformação e preparação final dos dados para modelagem")
    
    try:
        engineer = FeatureEngineer(CLEAN_DATA_FILE)
        
        # Processar features e obter conjuntos de treino/teste
        X_train, X_test, y_train, y_test = engineer.process_features(PROCESSED_DATA_FILE)
        
        # Salvar datasets processados
        engineer.save_processed_datasets()
        
        # Obter relatório de processamento
        report = engineer.get_processing_report()
        
        print(f"✅ Engenharia de features concluída!")
        print(f"   🎯 Conjunto de treino: {X_train.shape[0]:,} amostras, {X_train.shape[1]} features")
        print(f"   🎯 Conjunto de teste: {X_test.shape[0]:,} amostras, {X_test.shape[1]} features")
        print(f"   📊 Taxa de readmissão (treino): {y_train.mean():.1%}")
        print(f"   📊 Taxa de readmissão (teste): {y_test.mean():.1%}")
        print(f"   💾 Datasets salvos em: {DATA_DIR}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"❌ Erro na engenharia de features: {e}")
        raise


def run_machine_learning_models(fast_mode=False, advanced_analysis=True):
    """Executa treinamento e avaliação de modelos de Machine Learning com análises avançadas"""
    print_stage_header(4, "Modelagem de Machine Learning Avançada", 
                      "Treinamento, validação robusta e análise avançada de features")
    
    if advanced_analysis:
        print("✨ MODO AVANÇADO ATIVADO:")
        print("   🔬 Validação Cruzada Aninhada")
        print("   🧬 Análise Avançada de Features")
        print("   🎨 Dashboard Interativo")
        print("   📋 Relatórios Detalhados")
        print("=" * 70)
    
    trained_models = {}
    
    try:
        # 1. Regressão Logística
        print("🤖 Inicializando Regressão Logística...")
        lr_model = LogisticRegressionModel()
        
        # Executar pipeline completo do modelo (com ou sem otimização)
        lr_success = lr_model.run_complete_pipeline(
            tune_hyperparams=not fast_mode,    # Otimizar hiperparâmetros apenas se não estiver em modo rápido
            optimize_threshold=True            # Sempre otimizar limiar de decisão
        )
        
        if lr_success:
            trained_models['logistic_regression'] = lr_model
            print(f"✅ Regressão Logística executada com sucesso!")
            
            # Mostrar métricas principais
            if hasattr(lr_model, 'evaluation_results') and lr_model.evaluation_results:
                metrics = lr_model.evaluation_results
                print(f"   📈 Accuracy: {metrics.get('accuracy', 0):.1%}")
                print(f"   📈 Precision: {metrics.get('precision', 0):.1%}")
                print(f"   📈 Recall: {metrics.get('recall', 0):.1%}")
                print(f"   📈 F1-Score: {metrics.get('f1', 0):.1%}")
                print(f"   📈 ROC-AUC: {metrics.get('roc_auc', 0):.3f}")
            
        else:
            print(f"❌ Erro na execução da Regressão Logística")
        
        # 2. Random Forest com Análises Avançadas
        print("\n🌲 Inicializando Random Forest Avançado...")
        
        try:
            from src.random_forest_model import RandomForestModel
            
            rf_model = RandomForestModel()
            
            if fast_mode:
                print("⚡ Modo rápido: configurações otimizadas sem busca avançada")
                # Configurar para demo rápida
                rf_model.model_config.update({
                    'n_estimators': 200,
                    'max_depth': 12,
                    'min_samples_split': 8,
                    'min_samples_leaf': 3
                })
                rf_success = rf_model.run_complete_pipeline(
                    tune_hyperparams=False,
                    optimize_threshold=True,
                    cv_folds=3
                )
            else:
                print("🚀 Executando pipeline completo com análises avançadas...")
                rf_success = rf_model.run_complete_pipeline(
                    tune_hyperparams=True,
                    method='random_search',
                    optimize_threshold=True
                )
            
            if rf_success:
                trained_models['random_forest'] = rf_model
                print(f"✅ Random Forest Avançado executado com sucesso!")
                print(f"   📈 Accuracy: {rf_model.metrics['accuracy']:.1%}")
                print(f"   📈 Precision: {rf_model.metrics['precision']:.1%}")
                print(f"   📈 Recall: {rf_model.metrics['recall']:.1%}")
                print(f"   📈 F1-Score: {rf_model.metrics['f1']:.1%}")
                print(f"   📈 ROC-AUC: {rf_model.metrics['roc_auc']:.3f}")
                if rf_model.oob_score:
                    print(f"   📈 OOB Score: {rf_model.oob_score:.3f}")
                
                # ✨ DESTACAR RESULTADOS DAS ANÁLISES AVANÇADAS ✨
                if advanced_analysis and not fast_mode:
                    print(f"\n🔬 RESULTADOS DAS ANÁLISES AVANÇADAS:")
                    
                    # Validação aninhada
                    if 'nested_cv' in rf_model.validation_results:
                        nested_score = rf_model.validation_results['nested_cv']['mean_score']
                        nested_std = rf_model.validation_results['nested_cv']['std_score']
                        print(f"   🎯 Validação Aninhada: {nested_score:.4f} ± {nested_std:.4f}")
                    
                    # Features selecionadas
                    if 'rfe' in rf_model.feature_analysis_results:
                        optimal_features = rf_model.feature_analysis_results['rfe']['optimal_features']
                        total_features = len(rf_model.feature_analysis_results['rfe']['feature_ranking'])
                        print(f"   🧬 Features Otimizadas: {optimal_features}/{total_features}")
                    
                    # Estabilidade
                    if 'stability' in rf_model.validation_results:
                        stability_cv = rf_model.validation_results['stability']['score_stability']['coefficient_variation']
                        stability_level = "Alta" if stability_cv < 0.05 else "Média" if stability_cv < 0.10 else "Baixa"
                        print(f"   📊 Estabilidade do Modelo: {stability_level} (CV: {stability_cv:.4f})")
                    
                    # Multicolinearidade
                    if 'multicollinearity' in rf_model.feature_analysis_results:
                        high_corr = len(rf_model.feature_analysis_results['multicollinearity'].get('high_correlation_pairs', []))
                        print(f"   🔗 Pares Correlacionados: {high_corr}")
                    
                    # Interações
                    if 'interactions' in rf_model.feature_analysis_results:
                        interactions = len(rf_model.feature_analysis_results['interactions']['pairwise_interactions'])
                        if interactions > 0:
                            best_improvement = rf_model.feature_analysis_results['interactions']['pairwise_interactions'][0]['improvement']
                            print(f"   ⚡ Melhor Interação: {best_improvement:+.4f} melhoria")
                    
                    # Relatórios
                    if rf_model.advanced_reports:
                        print(f"   📋 Relatórios Avançados Gerados:")
                        if 'validation' in rf_model.advanced_reports:
                            print(f"      • Validação Robusta")
                        if 'features' in rf_model.advanced_reports:
                            print(f"      • Análise de Features")
            else:
                print(f"❌ Erro na execução do Random Forest")
                
        except ImportError as e:
            print(f"⚠️ Random Forest não disponível: {e}")
        except Exception as e:
            print(f"❌ Erro no Random Forest: {e}")
            import traceback
            traceback.print_exc()
        
        # Comparação de modelos (se múltiplos foram treinados)
        if len(trained_models) > 1:
            print(f"\n📊 COMPARAÇÃO DE MODELOS:")
            print(f"{'='*70}")
            print(f"{'Modelo':<20}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'ROC-AUC':<12}")
            print(f"{'='*70}")
            
            for model_name, model in trained_models.items():
                name_display = model_name.replace('_', ' ').title()
                if hasattr(model, 'metrics'):
                    acc = model.metrics.get('accuracy', 0)
                    prec = model.metrics.get('precision', 0)
                    rec = model.metrics.get('recall', 0)
                    f1 = model.metrics.get('f1', 0)
                    auc = model.metrics.get('roc_auc', 0)
                elif hasattr(model, 'evaluation_results'):
                    metrics = model.evaluation_results
                    acc = metrics.get('accuracy', 0)
                    prec = metrics.get('precision', 0)
                    rec = metrics.get('recall', 0)
                    f1 = metrics.get('f1', 0)
                    auc = metrics.get('roc_auc', 0)
                else:
                    acc = prec = rec = f1 = auc = 0
                    
                print(f"{name_display:<20}{acc:<12.4f}{prec:<12.4f}{rec:<12.4f}{f1:<12.4f}{auc:<12.4f}")
        
        # ✨ SEÇÃO ESPECIAL PARA ANÁLISES AVANÇADAS ✨
        if advanced_analysis and not fast_mode and 'random_forest' in trained_models:
            print(f"\n🔬 RESUMO DAS ANÁLISES AVANÇADAS:")
            print(f"{'='*70}")
            
            rf_model = trained_models['random_forest']
            
            # Mostrar benefícios das análises
            print(f"🎯 BENEFÍCIOS OBTIDOS:")
            
            # Validação robusta
            if 'nested_cv' in rf_model.validation_results:
                nested_score = rf_model.validation_results['nested_cv']['mean_score']
                basic_score = rf_model.metrics.get('roc_auc', 0)
                reliability = "Alta" if abs(nested_score - basic_score) < 0.02 else "Média"
                print(f"   ✅ Confiabilidade da Validação: {reliability}")
                print(f"      Score Aninhado: {nested_score:.4f}")
                print(f"      Score Básico: {basic_score:.4f}")
            
            # Otimização de features
            if 'rfe' in rf_model.feature_analysis_results:
                original_features = len(rf_model.feature_analysis_results['rfe']['feature_ranking'])
                optimal_features = rf_model.feature_analysis_results['rfe']['optimal_features']
                reduction = (1 - optimal_features/original_features) * 100
                print(f"   ✅ Redução de Features: {reduction:.1f}%")
                print(f"      Original: {original_features} → Otimizado: {optimal_features}")
            
            # Estabilidade
            if 'stability' in rf_model.validation_results:
                stability = rf_model.validation_results['stability']['score_stability']
                cv = stability['coefficient_variation']
                stability_grade = "A" if cv < 0.05 else "B" if cv < 0.10 else "C"
                print(f"   ✅ Nota de Estabilidade: {stability_grade}")
                print(f"      Coeficiente de Variação: {cv:.4f}")
            
            # Insights de features
            if 'feature_stability' in rf_model.feature_analysis_results:
                stable_features = len(rf_model.feature_analysis_results['feature_stability']['top_stable_features'])
                unstable_features = len(rf_model.feature_analysis_results['feature_stability']['unstable_features'])
                print(f"   ✅ Features Estáveis: {stable_features}")
                print(f"   ⚠️  Features Instáveis: {unstable_features}")
            
            print(f"\n📋 RECURSOS ADICIONAIS DISPONÍVEIS:")
            print(f"   � Dashboard Interativo (requer: pip install dash plotly)")
            print(f"   📈 Análise SHAP (requer: pip install shap)")
            print(f"   📑 Relatórios Detalhados em: {RESULTS_DIR}")
        
        print(f"\n�💾 Modelos salvos em: {MODELS_DIR}")
        print(f"📊 Resultados salvos em: {RESULTS_DIR}")
        
        if fast_mode:
            print(f"⚡ Modo rápido: algumas análises avançadas foram puladas")
        elif not advanced_analysis:
            print(f"📝 Modo básico: análises avançadas não executadas")
        
        # Retornar o melhor modelo
        best_model = None
        best_score = 0
        
        for model_name, model in trained_models.items():
            if hasattr(model, 'metrics') and 'roc_auc' in model.metrics:
                score = model.metrics['roc_auc']
            elif hasattr(model, 'evaluation_results') and 'roc_auc' in model.evaluation_results:
                score = model.evaluation_results['roc_auc']
            else:
                score = 0
            
            if score > best_score:
                best_score = score
                best_model = model
        
        return best_model or trained_models.get('logistic_regression') or trained_models.get('random_forest')
            
    except Exception as e:
        print(f"❌ Erro no treinamento dos modelos: {e}")
        import traceback
        traceback.print_exc()
        return None
        raise


def print_pipeline_summary(X_train, X_test, y_train, y_test, lr_model, execution_time):
    """Imprime resumo final do pipeline"""
    print("\n" + "🎯" + "="*68 + "🎯")
    print("    PIPELINE EXECUTADO COM SUCESSO!")
    print("🎯" + "="*68 + "🎯")
    
    print(f"\n📊 RESUMO DOS DADOS:")
    print(f"   • Conjunto de treino: {X_train.shape[0]:,} amostras")
    print(f"   • Conjunto de teste: {X_test.shape[0]:,} amostras")  
    print(f"   • Total de features: {X_train.shape[1]:,}")
    print(f"   • Taxa de readmissão (treino): {y_train.mean():.1%}")
    print(f"   • Taxa de readmissão (teste): {y_test.mean():.1%}")
    
    print(f"\n🤖 MODELOS TREINADOS:")
    if lr_model:
        print(f"   ✅ Regressão Logística")
        if hasattr(lr_model, 'evaluation_results') and lr_model.evaluation_results:
            metrics = lr_model.evaluation_results
            print(f"      └─ Accuracy: {metrics.get('accuracy', 0):.1%}")
            print(f"      └─ F1-Score: {metrics.get('f1', 0):.1%}")
            print(f"      └─ ROC-AUC: {metrics.get('roc_auc', 0):.3f}")
    else:
        print(f"   ❌ Nenhum modelo treinado com sucesso")
    
    print(f"\n📁 ARQUIVOS GERADOS:")
    print(f"   • Dados limpos: {os.path.basename(CLEAN_DATA_FILE)}")
    print(f"   • Dados processados: {os.path.basename(PROCESSED_DATA_FILE)}")
    print(f"   • Conjuntos de treino/teste: data/X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    print(f"   • Modelos treinados: models/")
    print(f"   • Relatórios e gráficos: results/")
    
    print(f"\n⏱️ TEMPO DE EXECUÇÃO: {execution_time:.1f} segundos")


def handle_pipeline_error(error, stage=""):
    """Trata erros do pipeline de forma padronizada"""
    print(f"\n❌ ERRO NO PIPELINE{' - ' + stage if stage else ''}")
    print(f"   Tipo: {type(error).__name__}")
    print(f"   Mensagem: {str(error)}")
    print(f"\n📋 Detalhes técnicos:")
    traceback.print_exc()
    
    print(f"\n💡 Sugestões para resolução:")
    print(f"   • Verifique se os arquivos de dados estão presentes")
    print(f"   • Confirme se todas as dependências estão instaladas")
    print(f"   • Execute: python scripts/validate_setup.py")
    print(f"   • Consulte os logs de erro para mais detalhes")


def main(fast_mode=False):
    """Função principal que executa todo o pipeline de forma robusta"""
    # Inicializar timing
    start_time = time.time()
    
    # Imprimir cabeçalho
    print_pipeline_header()
    
    if fast_mode:
        print("⚡ MODO RÁPIDO ATIVADO - Sem otimização de hiperparâmetros")
        print("=" * 70)
    
    # Variáveis para tracking de resultados
    X_train = X_test = y_train = y_test = lr_model = None
    
    try:
        # Verificar e criar diretórios necessários
        ensure_directories()
        
        # Etapa 1: Análise Exploratória
        df_analyzed = run_exploratory_analysis()
        
        # Etapa 2: Limpeza dos Dados
        df_clean = run_data_cleaning()
        
        # Etapa 3: Engenharia de Features
        X_train, X_test, y_train, y_test = run_feature_engineering()
        
        # Etapa 4: Modelos de Machine Learning
        lr_model = run_machine_learning_models(fast_mode=fast_mode)
        
        # Calcular tempo de execução
        execution_time = time.time() - start_time
        
        # Imprimir resumo final
        print_pipeline_summary(X_train, X_test, y_train, y_test, lr_model, execution_time)
        
        # Retornar resultados principais
        return {
            'X_train': X_train,
            'X_test': X_test, 
            'y_train': y_train,
            'y_test': y_test,
            'model': lr_model,
            'execution_time': execution_time,
            'fast_mode': fast_mode
        }
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Pipeline interrompido pelo usuário")
        print(f"   Tempo decorrido: {time.time() - start_time:.1f} segundos")
        return None
        
    except FileNotFoundError as e:
        handle_pipeline_error(e, "Arquivo não encontrado")
        print(f"\n💡 Verifique se o arquivo de dados existe em: {RAW_DATA_FILE}")
        return None
        
    except ImportError as e:
        handle_pipeline_error(e, "Dependência não encontrada")
        print(f"\n💡 Execute: pip install -r requirements.txt")
        return None
        
    except Exception as e:
        handle_pipeline_error(e, "Erro geral")
        return None


def quick_demo():
    """Execução rápida para demonstração (apenas modelo, sem análise exploratória completa)"""
    print("🚀 DEMONSTRAÇÃO RÁPIDA - APENAS MODELAGEM")
    print("="*50)
    
    try:
        lr_model = LogisticRegressionModel()
        success = lr_model.run_complete_pipeline(tune_hyperparams=False, optimize_threshold=True)
        
        if success:
            print("✅ Demonstração concluída com sucesso!")
            if hasattr(lr_model, 'evaluation_results') and lr_model.evaluation_results:
                metrics = lr_model.evaluation_results
                print(f"📈 Métricas principais:")
                print(f"   • Accuracy: {metrics.get('accuracy', 0):.1%}")
                print(f"   • F1-Score: {metrics.get('f1', 0):.1%}")
                print(f"   • ROC-AUC: {metrics.get('roc_auc', 0):.3f}")
        else:
            print("❌ Erro na demonstração")
            
    except Exception as e:
        print(f"❌ Erro: {e}")


if __name__ == "__main__":
    # Verificar argumentos da linha de comando
    if len(sys.argv) > 1:
        if sys.argv[1] == '--demo':
            quick_demo()
        elif sys.argv[1] == '--fast':
            result = main(fast_mode=True)
        else:
            print("Uso: python main_pipeline.py [--demo|--fast]")
            print("  --demo: Apenas demonstração do modelo")
            print("  --fast: Pipeline completo sem otimização de hiperparâmetros") 
            sys.exit(1)
    else:
        result = main(fast_mode=False)
    
    # Exit code baseado no sucesso
    if 'result' in locals() and result is None:
        sys.exit(1)
    else:
        sys.exit(0)
