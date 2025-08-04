#!/usr/bin/env python3
"""
Script para executar todo o pipeline de análise de readmissão hospitalar diabética
Uso: python scripts/run_full_pipeline.py
"""

import os
import sys
import time

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_pipeline import main


def print_header():
    """Imprime cabeçalho do pipeline"""
    print("🏥" + "="*68 + "🏥")
    print("    PIPELINE COMPLETO - PREDIÇÃO DE READMISSÃO HOSPITALAR")
    print("    Projeto: Inteligência Artificial - BCC325 UFOP")
    print("    Dataset: Diabetes 130-US hospitals (1999-2008)")
    print("🏥" + "="*68 + "🏥")


def print_footer():
    """Imprime rodapé com informações finais"""
    print("\n🎯" + "="*68 + "🎯")
    print("    PIPELINE EXECUTADO COM SUCESSO!")
    print("    Verifique os arquivos gerados em:")
    print("      📊 results/ - Gráficos e relatórios")
    print("      💾 models/ - Modelos treinados")
    print("      📁 data/ - Datasets processados")
    print("🎯" + "="*68 + "🎯")


if __name__ == "__main__":
    print_header()
    
    start_time = time.time()
    
    try:
        # Executar pipeline principal
        result = main()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if result is not None:
            print_footer()
            print(f"\n⏱️ Tempo total de execução: {execution_time:.2f} segundos")
        else:
            print(f"\n❌ Pipeline falhou durante a execução")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Pipeline interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
