"""
Utilitário de Mapeamento de IDs e Enriquecimento de Dados Médicos

Este módulo fornece funcionalidades robustas e profissionais para carregar e aplicar
mapeamentos de IDs do sistema hospitalar, enriquecendo os dados com descrições legíveis
e validações de consistência.

Funcionalidades Principais:
- Carregamento inteligente de mapeamentos do arquivo IDS_mapping.csv
- Aplicação automática de mapeamentos com validação de integridade
- Enriquecimento de dados com descrições humanamente legíveis
- Validação de consistência e identificação de códigos órfãos
- Relatórios detalhados de qualidade dos mapeamentos
- Tratamento robusto de erros e dados inconsistentes
- Suporte para múltiplos tipos de mapeamento simultâneos
- Logging detalhado de operações de transformação
- Validação de completude e cobertura dos mapeamentos

Tipos de Mapeamento Suportados:
- admission_type_id: Tipos de admissão hospitalar
- discharge_disposition_id: Tipos de alta hospitalar  
- admission_source_id: Origens de admissão

Autor: Thalles Felipe Rodrigues de Almeida Santos
Projeto: Predição de Readmissão Hospitalar em Pacientes com Diabetes Usando Aprendizado de Máquina
Instituição: Universidade Federal de Ouro Preto (UFOP)
Disciplina: Inteligência Artificial
Professor: Jadson Castro Gertrudes
Data: Agosto 2025

"""

import pandas as pd
import os
from io import StringIO

# Import condicional para config
try:
    from src.config import MAPPING_FILE
except ImportError:
    # Fallback para quando executado diretamente
    MAPPING_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'IDS_mapping.csv')


class IDMappingUtils:
    """Classe para gerenciar mapeamentos de IDs e enriquecer dados com descrições"""
    
    def __init__(self, mapping_file_path=None):
        """
        Inicializa o utilitário de mapeamento de IDs
        
        Args:
            mapping_file_path (str): Caminho para o arquivo IDS_mapping.csv
        """
        self.mapping_file_path = mapping_file_path or MAPPING_FILE
        self.mappings = {}
        
    def load_mappings(self):
        """
        Carrega todos os mapeamentos do arquivo IDS_mapping.csv de forma robusta usando pandas.
        
        Returns:
            dict: Dicionário com os mapeamentos para cada tipo de ID
        """
        print(f"Carregando mapeamentos de IDs de: {self.mapping_file_path}")
        
        if not os.path.exists(self.mapping_file_path):
            raise FileNotFoundError(f"Arquivo de mapeamento não encontrado: {self.mapping_file_path}")
        
        with open(self.mapping_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Encontrar as linhas que definem o início de cada tabela de mapeamento
        header_indices = [i for i, line in enumerate(lines) if '_id,description' in line]
        
        for i, start_index in enumerate(header_indices):
            # O nome do mapeamento vem da linha de cabeçalho
            header_line = lines[start_index].strip()
            mapping_name = header_line.split(',')[0].replace('_id', '')
            
            # O fim do bloco é o início do próximo bloco ou o fim do arquivo
            end_index = header_indices[i+1] if i + 1 < len(header_indices) else len(lines)
            
            # Extrair as linhas de dados para este bloco
            data_lines = lines[start_index + 1 : end_index]
            
            # Filtrar linhas vazias
            csv_block = [line for line in data_lines if line.strip() and line.strip() != ',']
            
            if not csv_block:
                print(f"Nenhum dado encontrado para o mapeamento '{mapping_name}'")
                continue
            
            # Usar pandas para ler o bloco de CSV
            # O StringIO trata uma lista de strings como se fosse um arquivo
            df_map = pd.read_csv(StringIO("".join(csv_block)), header=None, names=['id', 'description'])
            
            # Criar o dicionário de mapeamento
            # pd.to_numeric com errors='coerce' transforma o que não for número em NaN, que é então filtrado
            df_map['id'] = pd.to_numeric(df_map['id'], errors='coerce')
            df_map.dropna(subset=['id'], inplace=True)
            df_map['id'] = df_map['id'].astype(int)
            
            # Remover aspas das descrições se houver
            df_map['description'] = df_map['description'].str.strip(' "')
            
            mapping_dict = pd.Series(df_map.description.values, index=df_map.id).to_dict()
            
            self.mappings[mapping_name] = mapping_dict
            print(f"Carregado mapeamento para '{mapping_name}': {len(mapping_dict)} itens")
        
        print(f"Total de mapeamentos carregados: {len(self.mappings)}")
        return self.mappings
    
    def get_mapping(self, mapping_type):
        """
        Obtém um mapeamento específico
        
        Args:
            mapping_type (str): Tipo de mapeamento (admission_type, discharge_disposition, admission_source)
            
        Returns:
            dict: Dicionário com o mapeamento ID -> descrição
        """
        return self.mappings.get(mapping_type, {})
    
    def apply_mappings_to_dataframe(self, df, mappings_to_apply=None):
        """
        Aplica mapeamentos de IDs a um DataFrame, adicionando colunas descritivas
        
        Args:
            df (pd.DataFrame): DataFrame com os dados
            mappings_to_apply (list): Lista de mapeamentos a aplicar. Se None, aplica todos disponíveis
            
        Returns:
            pd.DataFrame: DataFrame com colunas descritivas adicionadas
        """
        df_enriched = df.copy()
        
        if mappings_to_apply is None:
            mappings_to_apply = list(self.mappings.keys())
        
        applied_mappings = []
        
        for mapping_type in mappings_to_apply:
            id_column = f"{mapping_type}_id"
            desc_column = f"{mapping_type}_desc"
            
            if id_column in df_enriched.columns and mapping_type in self.mappings:
                mapping_dict = self.mappings[mapping_type]
                
                # Aplicar mapeamento
                df_enriched[desc_column] = df_enriched[id_column].map(mapping_dict)
                
                # Preencher valores não mapeados com um valor padrão
                df_enriched[desc_column] = df_enriched[desc_column].fillna('Unknown')
                
                # Contar valores mapeados (antes de preencher com 'Unknown')
                mapped_count = df_enriched[id_column].isin(mapping_dict.keys()).sum()
                total_count = len(df_enriched)
                
                print(f"Aplicado mapeamento '{mapping_type}': {mapped_count}/{total_count} valores mapeados")
                applied_mappings.append(mapping_type)
                
                # Verificar valores não mapeados
                unmapped_ids = df_enriched[~df_enriched[id_column].isin(mapping_dict.keys())][id_column].unique()
                if len(unmapped_ids) > 0:
                    print(f"  Valores não mapeados em {id_column}: {sorted(unmapped_ids)} (preenchidos com 'Unknown')")
        
        print(f"Mapeamentos aplicados: {applied_mappings}")
        return df_enriched
    
    def get_mapping_summary(self):
        """
        Retorna um resumo dos mapeamentos disponíveis
        
        Returns:
            dict: Resumo dos mapeamentos
        """
        summary = {}
        for mapping_type, mapping_dict in self.mappings.items():
            summary[mapping_type] = {
                'count': len(mapping_dict),
                'id_range': f"{min(mapping_dict.keys())}-{max(mapping_dict.keys())}" if mapping_dict else "N/A",
                'sample_values': list(mapping_dict.items())[:3] if mapping_dict else []
            }
        return summary
    
    def validate_mappings(self, df):
        """
        Valida se os IDs no DataFrame têm mapeamentos disponíveis
        
        Args:
            df (pd.DataFrame): DataFrame para validar
            
        Returns:
            dict: Relatório de validação
        """
        validation_report = {}
        
        for mapping_type in self.mappings.keys():
            id_column = f"{mapping_type}_id"
            
            if id_column in df.columns:
                unique_ids = set(df[id_column].dropna().unique())
                mapped_ids = set(self.mappings[mapping_type].keys())
                
                validation_report[mapping_type] = {
                    'total_unique_ids': len(unique_ids),
                    'mapped_ids': len(unique_ids.intersection(mapped_ids)),
                    'unmapped_ids': list(unique_ids - mapped_ids),
                    'coverage_rate': len(unique_ids.intersection(mapped_ids)) / len(unique_ids) if unique_ids else 0
                }
        
        return validation_report


def load_and_apply_id_mappings(df, mapping_file_path=None, mappings_to_apply=None):
    """
    Função utilitária para carregar e aplicar mapeamentos de ID em um só passo
    
    Args:
        df (pd.DataFrame): DataFrame para enriquecer
        mapping_file_path (str): Caminho para o arquivo de mapeamento
        mappings_to_apply (list): Lista de mapeamentos a aplicar
        
    Returns:
        pd.DataFrame: DataFrame enriquecido com descrições
    """
    mapper = IDMappingUtils(mapping_file_path)
    mapper.load_mappings()
    return mapper.apply_mappings_to_dataframe(df, mappings_to_apply)


def main():
    """Função principal para testar o utilitário de mapeamentos"""
    mapper = IDMappingUtils()
    
    try:
        mappings = mapper.load_mappings()
        print("\nMapeamentos carregados:")
        for mapping_type, mapping_dict in mappings.items():
            print(f"  {mapping_type}: {len(mapping_dict)} itens")
        
        print("\nResumo dos mapeamentos:")
        summary = mapper.get_mapping_summary()
        for mapping_type, info in summary.items():
            print(f"  {mapping_type}:")
            print(f"    Count: {info['count']}")
            print(f"    Range: {info['id_range']}")
            print(f"    Sample: {info['sample_values']}")
        
    except Exception as e:
        print(f"Erro ao carregar mapeamentos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
