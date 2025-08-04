#!/usr/bin/env python3
"""
Script para validar a instalação e configuração do projeto
Uso: python scripts/validate_setup.py
"""

import os
import sys
import importlib
import pandas as pd

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_dependencies():
    """Verifica se todas as dependências estão instaladas"""
    print("🔍 Verificando dependências...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'jupyter'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NÃO INSTALADO")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Pacotes faltando: {', '.join(missing_packages)}")
        print(f"   Execute: pip install {' '.join(missing_packages)}")
        return False
    else:
        print(f"✅ Todas as dependências estão instaladas!")
        return True


def check_data_files():
    """Verifica se os arquivos de dados existem"""
    print("\n📁 Verificando arquivos de dados...")
    
    required_files = [
        'data/diabetic_data.csv',
        'data/IDS_mapping.csv'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  ✅ {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {file_path} - NÃO ENCONTRADO")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ Arquivos faltando: {', '.join(missing_files)}")
        print(f"   Certifique-se de que os dados estão na pasta 'data/'")
        return False
    else:
        print(f"✅ Todos os arquivos de dados estão presentes!")
        return True


def check_project_structure():
    """Verifica se a estrutura do projeto está correta"""
    print("\n🏗️ Verificando estrutura do projeto...")
    
    required_dirs = [
        'src', 'data', 'notebooks', 'models', 'results', 'scripts', 'docs'
    ]
    
    required_files = [
        'src/config.py',
        'src/data_cleaning.py',
        'src/exploratory_analysis.py',
        'src/feature_engineering.py',
        'src/logistic_regression_model.py',
        'src/id_mapping_utils.py',
        'main_pipeline.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_items = []
    
    # Verificar diretórios
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ - NÃO ENCONTRADO")
            missing_items.append(dir_path)
    
    # Verificar arquivos
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - NÃO ENCONTRADO")
            missing_items.append(file_path)
    
    if missing_items:
        print(f"\n⚠️ Itens faltando: {', '.join(missing_items)}")
        return False
    else:
        print(f"✅ Estrutura do projeto está correta!")
        return True


def test_data_loading():
    """Testa o carregamento dos dados principais"""
    print("\n📊 Testando carregamento de dados...")
    
    try:
        # Testar carregamento do dataset principal
        df = pd.read_csv('data/diabetic_data.csv')
        print(f"  ✅ diabetic_data.csv carregado: {df.shape[0]:,} registros, {df.shape[1]} colunas")
        
        # Testar carregamento dos mapeamentos
        mappings = pd.read_csv('data/IDS_mapping.csv')
        print(f"  ✅ IDS_mapping.csv carregado: {len(mappings)} linhas")
        
        # Verificar colunas essenciais
        essential_cols = ['patient_nbr', 'readmitted', 'age', 'gender']
        missing_cols = [col for col in essential_cols if col not in df.columns]
        
        if missing_cols:
            print(f"  ⚠️ Colunas essenciais faltando: {missing_cols}")
            return False
        else:
            print(f"  ✅ Colunas essenciais presentes")
            return True
            
    except Exception as e:
        print(f"  ❌ Erro ao carregar dados: {e}")
        return False


def test_imports():
    """Testa se os módulos do projeto podem ser importados"""
    print("\n🐍 Testando imports dos módulos...")
    
    modules_to_test = [
        'src.config',
        'src.data_cleaning',
        'src.exploratory_analysis',
        'src.feature_engineering',
        'src.logistic_regression_model',
        'src.id_mapping_utils'
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except Exception as e:
            print(f"  ❌ {module} - ERRO: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️ Módulos com problemas: {', '.join(failed_imports)}")
        return False
    else:
        print(f"✅ Todos os módulos podem ser importados!")
        return True


def main():
    """Executa todas as validações"""
    print("🔧" + "="*60 + "🔧")
    print("  VALIDAÇÃO DE CONFIGURAÇÃO DO PROJETO")
    print("  Predição de Readmissão Hospitalar Diabética")
    print("🔧" + "="*60 + "🔧")
    
    all_checks = [
        check_dependencies(),
        check_data_files(),
        check_project_structure(),
        test_data_loading(),
        test_imports()
    ]
    
    print("\n" + "="*60)
    
    if all(all_checks):
        print("🎉 VALIDAÇÃO CONCLUÍDA COM SUCESSO!")
        print("   O projeto está configurado corretamente e pronto para uso.")
        print("\n💡 Próximos passos:")
        print("   1. Execute: python scripts/run_eda.py (análise exploratória)")
        print("   2. Execute: python scripts/run_full_pipeline.py (pipeline completo)")
        print("   3. Execute: python scripts/demo_logistic_regression.py (demonstração)")
        return True
    else:
        print("❌ VALIDAÇÃO FALHOU!")
        print("   Corrija os problemas identificados antes de prosseguir.")
        return False


if __name__ == "__main__":
    main()
