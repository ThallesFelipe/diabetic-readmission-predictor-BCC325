"""
Demonstração da Regressão Logística para Predição de Readmissão Hospitalar Diabética

Este script executa apenas o modelo de Regressão Logística com dados pré-processados,
ideal para demonstrações rápidas e testes do modelo.

Uso: python scripts/demo_logistic_regression.py
"""

import os
import sys
import time
import traceback

# Adicionar diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logistic_regression_model import LogisticRegressionModel


def print_demo_header():
    """Imprime cabeçalho da demonstração"""
    print("🤖" + "="*70 + "🤖")
    print("    DEMONSTRAÇÃO: REGRESSÃO LOGÍSTICA")
    print("    Predição de Readmissão Hospitalar Diabética")
    print("    Modelo: Logistic Regression com class_weight='balanced'")
    print("🤖" + "="*70 + "🤖")


def print_demo_info():
    """Imprime informações sobre a demonstração"""
    print(f"\n📋 INFORMAÇÕES DA DEMONSTRAÇÃO:")
    print(f"  🎯 Objetivo: Predizer readmissão hospitalar em <30 dias")
    print(f"  👥 Pacientes: Diabéticos hospitalizados (1999-2008)")
    print(f"  📊 Dataset: UCI - Diabetes 130-US hospitals")
    print(f"  🔧 Modelo: Regressão Logística balanceada")
    print(f"  📈 Métricas: Accuracy, Precision, Recall, F1, ROC-AUC")


def check_prerequisites():
    """Verifica se os pré-requisitos estão atendidos"""
    print(f"\n🔍 Verificando pré-requisitos...")
    
    required_files = [
        'data/X_train.csv',
        'data/X_test.csv', 
        'data/y_train.csv',
        'data/y_test.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"  ✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ Arquivos faltando: {missing_files}")
        print(f"💡 Execute primeiro: python main_pipeline.py")
        return False
    
    print(f"  ✅ Todos os arquivos necessários estão presentes!")
    return True


def print_results_summary(success, execution_time):
    """Imprime resumo dos resultados"""
    if success:
        print(f"\n🎉" + "="*70 + "🎉")
        print(f"    DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO!")
        print(f"    Tempo de execução: {execution_time:.2f} segundos")
        print(f"🎉" + "="*70 + "🎉")
        
        print(f"\n📁 ARQUIVOS GERADOS:")
        print(f"  📊 results/logistic_regression_results.png")
        print(f"  🔍 results/logistic_regression_feature_importance.png")
        print(f"  📋 results/logistic_regression_report_*.txt")
        print(f"  💾 models/logistic_regression_model_*.joblib")
        
        print(f"\n🏥 INTERPRETAÇÃO CLÍNICA:")
        print(f"  • Accuracy ~65%: O modelo acerta 2 em cada 3 casos")
        print(f"  • Precision baixa: Muitos falsos positivos (alertas desnecessários)")
        print(f"  • Recall moderado: Alguns casos de risco passam despercebidos")
        print(f"  • Uso recomendado: Ferramenta de triagem, não diagnóstico final")
        
    else:
        print(f"\n❌ DEMONSTRAÇÃO FALHOU")
        print(f"   Verifique os logs de erro acima")


def main():
    """Demonstração da Regressão Logística"""
    
    print_demo_header()
    print_demo_info()
    
    # Verificar pré-requisitos
    if not check_prerequisites():
        sys.exit(1)
    
    print(f"\n🚀 Iniciando execução do modelo...")
    start_time = time.time()
    
    try:
        # Criar e executar modelo
        lr_model = LogisticRegressionModel()
        success = lr_model.run_complete_pipeline()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Mostrar resultados
        print_results_summary(success, execution_time)
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Demonstração interrompida pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
