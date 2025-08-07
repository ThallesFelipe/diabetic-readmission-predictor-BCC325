"""
Exemplo de uso do modelo Random Forest

Este script demonstra como usar o modelo Random Forest para predição
de readmissão hospitalar diabética.
"""

import sys
import os

# Adicionar diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.random_forest_model import RandomForestModel

def main():
    """Exemplo básico de uso do modelo Random Forest"""
    print("🌲 Exemplo: Modelo Random Forest")
    print("="*50)
    
    # Inicializar o modelo
    model = RandomForestModel()
    
    # Executar pipeline completo
    print("🚀 Executando pipeline completo...")
    success = model.run_complete_pipeline(
        tune_hyperparams=True,
        optimize_threshold=True
    )
    
    if success:
        print("✅ Modelo treinado com sucesso!")
        
        # Mostrar resultados
        if hasattr(model, 'evaluation_results'):
            metrics = model.evaluation_results
            print(f"\n📊 Resultados:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"   • {metric}: {value:.4f}")
                else:
                    print(f"   • {metric}: {value}")
        
        # Salvar modelo
        model.save_model()
        print("💾 Modelo salvo com sucesso!")
        
    else:
        print("❌ Erro no treinamento do modelo")

if __name__ == "__main__":
    main()
