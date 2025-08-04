#!/usr/bin/env python3
"""
Script para executar apenas a análise exploratória dos dados
Uso: python scripts/run_eda.py
"""

import os
import sys
import time

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exploratory_analysis import ExploratoryDataAnalysis
from src.config import RAW_DATA_FILE


def main():
    """Executa apenas a análise exploratória"""
    print("📊" + "="*60 + "📊")
    print("  ANÁLISE EXPLORATÓRIA - DADOS DIABÉTICOS")
    print("📊" + "="*60 + "📊")
    
    start_time = time.time()
    
    try:
        # Executar análise exploratória
        eda = ExploratoryDataAnalysis(RAW_DATA_FILE)
        df_analyzed = eda.run_complete_analysis()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\n✅ Análise exploratória concluída!")
        print(f"⏱️ Tempo de execução: {execution_time:.2f} segundos")
        print(f"📄 Dataset analisado: {df_analyzed.shape[0]:,} registros, {df_analyzed.shape[1]} colunas")
        
        return df_analyzed
        
    except Exception as e:
        print(f"\n❌ Erro durante análise: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
