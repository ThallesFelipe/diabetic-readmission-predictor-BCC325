"""
Script para executar e testar o modelo de Regressão Logística
"""

import sys
import os

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logistic_regression_model import LogisticRegressionModel


def main():
    """Função principal para testar a Regressão Logística"""
    print("TESTE DA REGRESSÃO LOGÍSTICA")
    print("="*40)
    
    try:
        # Criar instância do modelo
        lr_model = LogisticRegressionModel()
        
        # Executar pipeline completo
        success = lr_model.run_complete_pipeline()
        
        if success:
            print("\n✅ Teste concluído com sucesso!")
            print("  - Modelo treinado e avaliado")
            print("  - Resultados salvos")
            print("  - Visualizações criadas")
        else:
            print("\n❌ Erro no teste")
            
    except Exception as e:
        print(f"\nErro durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
