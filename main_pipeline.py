"""
Script principal para execução do pipeline completo de análise de readmissão hospitalar diabética

Este script executa todas as etapas do projeto:
1. Análise exploratória dos dados
2. Limpeza e pré-processamento dos dados  
3. Engenharia de features e preparação para modelagem
4. Treinamento e avaliação do modelo de Regressão Logística
5. Geração de relatórios e visualizações

Uso:
    python main_pipeline.py                # Pipeline completo
    python main_pipeline.py --demo        # Demonstração rápida (apenas modelagem)
    python main_pipeline.py --fast        # Pipeline completo sem otimização de hiperparâmetros

Autor: Projeto BCC325 - Inteligência Artificial UFOP
Data: 2025
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
    print("    Projeto: BCC325 - Inteligência Artificial - UFOP")
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


def run_machine_learning_models(fast_mode=False):
    """Executa treinamento e avaliação do modelo de Regressão Logística"""
    print_stage_header(4, "Modelagem de Machine Learning", 
                      "Treinamento e avaliação do modelo de Regressão Logística")
    
    try:
        # Inicializar modelo de Regressão Logística
        print("🤖 Inicializando Regressão Logística...")
        lr_model = LogisticRegressionModel()
        
        # Executar pipeline completo do modelo (com ou sem otimização)
        success = lr_model.run_complete_pipeline(
            tune_hyperparams=not fast_mode,    # Otimizar hiperparâmetros apenas se não estiver em modo rápido
            optimize_threshold=True            # Sempre otimizar limiar de decisão
        )
        
        if success:
            print(f"✅ Regressão Logística executada com sucesso!")
            
            # Mostrar métricas principais
            if hasattr(lr_model, 'evaluation_results') and lr_model.evaluation_results:
                metrics = lr_model.evaluation_results
                print(f"   📈 Accuracy: {metrics.get('accuracy', 0):.1%}")
                print(f"   📈 Precision: {metrics.get('precision', 0):.1%}")
                print(f"   📈 Recall: {metrics.get('recall', 0):.1%}")
                print(f"   📈 F1-Score: {metrics.get('f1', 0):.1%}")
                print(f"   📈 ROC-AUC: {metrics.get('roc_auc', 0):.3f}")
                
            print(f"   💾 Modelo salvo em: {MODELS_DIR}")
            print(f"   📊 Resultados salvos em: {RESULTS_DIR}")
            
            if fast_mode:
                print(f"   ⚡ Modo rápido: hiperparâmetros padrão utilizados")
            
            return lr_model
        else:
            print(f"❌ Erro na execução da Regressão Logística")
            return None
            
    except Exception as e:
        print(f"❌ Erro no treinamento do modelo: {e}")
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
    
    print(f"\n🔬 INTERPRETAÇÃO CLÍNICA:")
    print(f"   • O modelo pode auxiliar na identificação de pacientes com maior risco")
    print(f"   • Útil para otimização de recursos hospitalares e cuidados preventivos")
    print(f"   • Recomenda-se validação adicional com dados mais recentes")
    
    print(f"\n🚀 PRÓXIMOS PASSOS SUGERIDOS:")
    print(f"   • Implementar Random Forest e XGBoost para comparação")
    print(f"   • Validação cruzada mais robusta")
    print(f"   • Análise de feature importance mais detalhada")
    print(f"   • Implementação em ambiente de produção")


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
