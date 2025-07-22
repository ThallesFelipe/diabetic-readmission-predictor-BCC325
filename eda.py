"""
Script legado - Redirecionado para o novo pipeline modularizado

Este arquivo foi refatorado e modularizado. Use os novos módulos:
- src/exploratory_analysis.py para análise exploratória
- src/data_cleaning.py para limpeza dos dados  
- src/feature_engineering.py para engenharia de features
- main_pipeline.py para executar o pipeline completo
"""

import sys
import os

# Adicionar o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("="*80)
print("🔄 REDIRECIONAMENTO PARA NOVO PIPELINE MODULARIZADO")
print("="*80)

print("\n📋 Este arquivo foi refatorado em módulos organizados:")
print("✅ src/exploratory_analysis.py - Análise exploratória completa")
print("✅ src/data_cleaning.py - Pipeline de limpeza de dados")
print("✅ src/feature_engineering.py - Engenharia de features")
print("✅ main_pipeline.py - Pipeline completo automatizado")

print("\n🚀 Para executar a análise, use:")
print("python main_pipeline.py")

print("\n📊 Para análise interativa, use:")
print("jupyter notebook notebooks/diabetic_readmission_analysis.ipynb")

print("\n🔧 Para usar módulos específicos:")
print("from src.exploratory_analysis import ExploratoryDataAnalysis")
print("from src.data_cleaning import DataCleaner")
print("from src.feature_engineering import FeatureEngineer")

# Executar o pipeline automaticamente
try:
    print("\n⚡ Executando pipeline automaticamente...")
    from main_pipeline import main
    main()
except ImportError as e:
    print(f"❌ Erro ao importar módulos: {e}")
    print("Certifique-se de que todos os arquivos estão no lugar correto")
except Exception as e:
    print(f"❌ Erro durante execução: {e}")
    print("Execute 'python main_pipeline.py' manualmente")