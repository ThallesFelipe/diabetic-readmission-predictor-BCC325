"""
Utilitário para carregar e aplicar mapeamentos de IDs do arquivo IDS_mapping.csv
Este módulo enriquece os dados com descrições legíveis dos códigos IDs
"""

import pandas as pd
import os

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
        Carrega todos os mapeamentos do arquivo IDS_mapping.csv
        
        Returns:
            dict: Dicionário com os mapeamentos para cada tipo de ID
        """
        print(f"Carregando mapeamentos de IDs de: {self.mapping_file_path}")
        
        if not os.path.exists(self.mapping_file_path):
            raise FileNotFoundError(f"Arquivo de mapeamento não encontrado: {self.mapping_file_path}")
        
        # Ler o arquivo linha por linha
        with open(self.mapping_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        current_mapping = None
        current_dict = {}
        
        for line in lines:
            line = line.strip()
            
            # Linha vazia indica fim de uma seção
            if not line or line == ',':
                if current_mapping and current_dict:
                    self.mappings[current_mapping] = current_dict
                    print(f"Carregado mapeamento para '{current_mapping}': {len(current_dict)} itens")
                current_mapping = None
                current_dict = {}
                continue
            
            # Verificar se é uma linha de cabeçalho (contém "_id,description")
            if '_id,description' in line:
                # Novo tipo de mapeamento
                id_column = line.split(',')[0].strip()
                current_mapping = id_column.replace('_id', '') if id_column.endswith('_id') else id_column
                current_dict = {}
                continue
            
            # Linha de dados
            if current_mapping is not None:
                # Encontrar a primeira vírgula para separar ID da descrição
                comma_index = line.find(',')
                if comma_index == -1:
                    continue
                    
                id_part = line[:comma_index].strip()
                desc_part = line[comma_index + 1:].strip()
                
                # Verificar se temos ID válido
                if not id_part or not desc_part:
                    continue
                    
                try:
                    id_value = int(id_part)
                    # Remover aspas se houver
                    description = desc_part.strip(' "')
                    current_dict[id_value] = description
                except ValueError:
                    continue
        
        # Adicionar o último mapeamento se existir
        if current_mapping and current_dict:
            self.mappings[current_mapping] = current_dict
            print(f"Carregado mapeamento para '{current_mapping}': {len(current_dict)} itens")
        
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
                
                # Contar valores mapeados
                mapped_count = df_enriched[desc_column].notna().sum()
                total_count = len(df_enriched)
                
                print(f"Aplicado mapeamento '{mapping_type}': {mapped_count}/{total_count} valores mapeados")
                applied_mappings.append(mapping_type)
                
                # Verificar valores não mapeados
                unmapped = df_enriched[df_enriched[desc_column].isna()][id_column].unique()
                if len(unmapped) > 0:
                    print(f"  Valores não mapeados em {id_column}: {sorted(unmapped)}")
        
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
