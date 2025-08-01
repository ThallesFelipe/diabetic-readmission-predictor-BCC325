"""
Demonstração da Regressão Logística para Predição de Readmissão Hospitalar Diabética
"""

import os
import sys
import traceback

# Adicionar diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.logistic_regression_model import LogisticRegressionModel


def main():
    """Demonstração da Regressão Logística"""
    
    print("Demonstração: Regressão Logística para Predição de Readmissão Hospitalar")
    print("="*70)
    
    print(f"\nObjetivo: Predizer readmissão hospitalar em <30 dias")
    print(f"Pacientes: Diabéticos hospitalizados")
    print(f"Dataset: Dados processados com engenharia de features")
    print(f"Modelo: Regressão Logística com class_weight='balanced'")
    
    print(f"\nIniciando execução...")
    
    try:
        # Criar e executar modelo
        lr_model = LogisticRegressionModel()
        success = lr_model.run_complete_pipeline()
        
        if success:
            print(f"\n✅ Demonstração concluída com sucesso!")
            
            print(f"\nArquivos gerados:")
            print(f"  📊 results/logistic_regression_results.png")
            print(f"  🔍 results/logistic_regression_feature_importance.png")
            print(f"  📋 results/logistic_regression_report_*.txt")
            print(f"  💾 models/logistic_regression_model_*.joblib")
            
            print(f"\nComo interpretar os resultados:")
            print(f"  1. Verifique as métricas no terminal")
            print(f"  2. Analise os gráficos gerados")
            print(f"  3. Examine as features mais importantes")
            
        else:
            print(f"❌ Erro durante execução")
            
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
