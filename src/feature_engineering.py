"""
Módulo de Engenharia de Features e Preparação de Dados para Machine Learning

Este módulo implementa o pipeline completo e profissional de Feature Engineering
para o sistema de predição de readmissão hospitalar diabética, incluindo:

Funcionalidades de Transformação:
1. Carregamento e validação rigorosa de dados limpos
2. Remoção inteligente de colunas desnecessárias para modelagem
3. Identificação automática e tratamento de colunas categóricas e numéricas
4. One-Hot Encoding otimizado com controle de alta cardinalidade
5. Divisão estratificada e balanceada dos dados
6. Escalonamento e normalização de features numéricas
7. Validações estatísticas e verificações de qualidade
8. Geração de relatórios detalhados de transformações
9. Salvamento organizado dos conjuntos finais
10. Pipeline sklearn compatível para produção

Pré-requisitos Técnicos:
- Dados limpos em formato CSV com coluna 'target' binária (0/1)
- Arquivo de configuração (config.py) com parâmetros adequados
- Estrutura de diretórios definida (data/, results/, models/)

Autor: Thalles Felipe Rodrigues de Almeida Santos
Projeto: Predição de Readmissão Hospitalar em Pacientes com Diabetes Usando Aprendizado de Máquina
Instituição: Universidade Federal de Ouro Preto (UFOP)
Disciplina: Inteligência Artificial
Professor: Jadson Castro Gertrudes
Data: Agosto 2025
"""

import pandas as pd
import numpy as np
import os
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import condicional para config
try:
    from src.config import (
        CLEAN_DATA_FILE, PROCESSED_DATA_FILE, COLUMNS_TO_DROP_MODELING,
        TEST_SIZE, RANDOM_STATE, DATA_DIR
    )
except ImportError:
    # Fallback para quando executado diretamente
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import (
        CLEAN_DATA_FILE, PROCESSED_DATA_FILE, COLUMNS_TO_DROP_MODELING,
        TEST_SIZE, RANDOM_STATE, DATA_DIR
    )

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FeatureEngineer:
    """
    Classe responsável pela engenharia de features e preparação dos dados
    
    Esta classe implementa um pipeline completo de transformação de dados,
    desde o carregamento dos dados limpos até a preparação final para modelagem.
    
    Atributos:
        clean_data_path (str): Caminho para o arquivo de dados limpos
        df_clean (pd.DataFrame): DataFrame com dados limpos carregados
        df_processed (pd.DataFrame): DataFrame após processamento
        X_train, X_test (pd.DataFrame): Features de treino e teste
        y_train, y_test (pd.Series): Targets de treino e teste
        high_cardinality_threshold (int): Limite para considerar alta cardinalidade
        stratify_enabled (bool): Se deve usar estratificação na divisão
    """
    
    def __init__(self, clean_data_path=None, high_cardinality_threshold=50, stratify_enabled=True, apply_scaling=True):
        """
        Inicializa o engenheiro de features
        
        Args:
            clean_data_path (str): Caminho para dados limpos (padrão: config.CLEAN_DATA_FILE)
            high_cardinality_threshold (int): Limite para alta cardinalidade (padrão: 50)
            stratify_enabled (bool): Se deve usar estratificação (padrão: True)
            apply_scaling (bool): Se deve aplicar escalonamento nas features numéricas (padrão: True)
        """
        self.clean_data_path = clean_data_path or CLEAN_DATA_FILE
        self.high_cardinality_threshold = high_cardinality_threshold
        self.stratify_enabled = stratify_enabled
        self.apply_scaling = apply_scaling
        
        # Atributos de dados
        self.df_clean = None
        self.df_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Objetos de transformação para produção
        self.scaler = None
        self.numeric_features = None
        self.categorical_features = None
        
        # Metadados para rastreamento
        self.processing_metadata = {
            'columns_removed': [],
            'high_cardinality_columns': [],
            'encoding_stats': {},
            'feature_stats': {},
            'scaling_info': {}
        }
        
    def load_clean_data(self):
        """
        Carrega os dados limpos do arquivo CSV com validações
        
        Returns:
            pd.DataFrame: DataFrame com dados carregados e validados
            
        Raises:
            FileNotFoundError: Se o arquivo não for encontrado
            ValueError: Se o DataFrame estiver vazio ou faltar coluna 'target'
        """
        logging.info(f"Carregando dados limpos de: {self.clean_data_path}")
        
        try:
            self.df_clean = pd.read_csv(self.clean_data_path)
            logging.info(f"Dados carregados com sucesso. Shape: {self.df_clean.shape}")
            
            # Validações básicas
            if self.df_clean.empty:
                raise ValueError("DataFrame carregado está vazio")
                
            if 'target' not in self.df_clean.columns:
                raise ValueError("Coluna 'target' não encontrada no DataFrame")
                
            # Verificar se target é binário
            unique_targets = self.df_clean['target'].unique()
            if not set(unique_targets).issubset({0, 1}):
                raise ValueError(f"Target deve ser binário (0 e 1), encontrado: {unique_targets}")
                
            logging.info(f"Validações básicas aprovadas. Target possui {len(unique_targets)} valores únicos")
            
            return self.df_clean
            
        except FileNotFoundError:
            logging.error(f"Arquivo não encontrado: {self.clean_data_path}")
            raise
        except Exception as e:
            logging.error(f"Erro ao carregar dados: {str(e)}")
            raise
    
    def remove_unnecessary_columns(self):
        """
        Remove colunas desnecessárias para modelagem com validações aprimoradas
        
        Remove:
        - Colunas definidas em COLUMNS_TO_DROP_MODELING
        - Colunas ID que possuem colunas descritivas correspondentes
        - Identificadores únicos restantes (ex: encounter_id)
        """
        logging.info("Removendo colunas desnecessárias...")
        
        # Identificar colunas descritivas (criadas pelos mapeamentos)
        desc_cols = [col for col in self.df_clean.columns if col.endswith('_desc')]
        
        # Identificar colunas ID que têm descrições correspondentes
        id_cols = []
        for col in self.df_clean.columns:
            if col.endswith('_id'):
                desc_equivalent = col.replace('_id', '_desc')
                if desc_equivalent in desc_cols:
                    id_cols.append(col)
        
        # Identificar possíveis identificadores únicos restantes
        unique_id_cols = []
        for col in self.df_clean.columns:
            if col not in COLUMNS_TO_DROP_MODELING and col not in id_cols:
                # Verificar se a coluna tem alta cardinalidade (possível ID)
                if self.df_clean[col].dtype in ['object', 'int64'] and \
                   self.df_clean[col].nunique() / len(self.df_clean) > 0.9:
                    unique_id_cols.append(col)
                    logging.warning(f"Coluna {col} detectada como possível ID único (cardinalidade: {self.df_clean[col].nunique()})")
        
        # Combinar todas as colunas a serem removidas
        cols_to_remove = list(COLUMNS_TO_DROP_MODELING) + id_cols + unique_id_cols
        
        # Log detalhado das remoções
        logging.info(f"Colunas configuradas para remoção: {COLUMNS_TO_DROP_MODELING}")
        if id_cols:
            logging.info(f"Colunas ID com mapeamentos removidas: {id_cols}")
            logging.info(f"Colunas descritivas mantidas: {desc_cols}")
        if unique_id_cols:
            logging.info(f"IDs únicos detectados e removidos: {unique_id_cols}")
        
        # Remover apenas as colunas que existem no DataFrame
        existing_cols_to_remove = [col for col in cols_to_remove if col in self.df_clean.columns]
        missing_cols = [col for col in cols_to_remove if col not in self.df_clean.columns]
        
        if missing_cols:
            logging.warning(f"Colunas não encontradas (ignoradas): {missing_cols}")
        
        self.df_processed = self.df_clean.drop(existing_cols_to_remove, axis=1)
        
        # Salvar metadados
        self.processing_metadata['columns_removed'] = existing_cols_to_remove
        
        logging.info(f"Total de colunas removidas: {len(existing_cols_to_remove)}")
        logging.info(f"Dimensões após remoção: {self.df_processed.shape}")
        
        return self.df_processed
    
    def identify_column_types(self):
        """
        Identifica tipos de colunas para processamento com validações
        
        Returns:
            tuple: (numeric_cols, categorical_cols) - listas de nomes de colunas
        """
        logging.info("Identificando tipos de colunas...")
        
        numeric_cols = self.df_processed.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = self.df_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Remover target e possíveis identificadores da lista de features
        excluded_cols = ['target', 'encounter_id', 'patient_nbr']
        numeric_cols = [col for col in numeric_cols if col not in excluded_cols]
        categorical_cols = [col for col in categorical_cols if col not in excluded_cols]
        
        # Identificar colunas com alta cardinalidade
        high_cardinality_cols = []
        for col in categorical_cols:
            unique_count = self.df_processed[col].nunique()
            if unique_count > self.high_cardinality_threshold:
                high_cardinality_cols.append((col, unique_count))
                
        self.processing_metadata['high_cardinality_columns'] = high_cardinality_cols
        
        # Logs detalhados
        logging.info(f"Colunas numéricas ({len(numeric_cols)}): {numeric_cols[:5]}{'...' if len(numeric_cols) > 5 else ''}")
        logging.info(f"Colunas categóricas ({len(categorical_cols)}): {categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''}")
        
        if high_cardinality_cols:
            logging.warning("Colunas com alta cardinalidade detectadas:")
            for col, count in high_cardinality_cols:
                logging.warning(f"  {col}: {count} categorias únicas")
        
        # Estatísticas para features numéricas
        if numeric_cols:
            logging.info("Estatísticas das features numéricas:")
            for col in numeric_cols[:3]:  # Mostrar apenas as primeiras 3
                stats = self.df_processed[col].describe()
                logging.info(f"  {col}: média={stats['mean']:.2f}, std={stats['std']:.2f}")
        
        return numeric_cols, categorical_cols
    
    def apply_one_hot_encoding(self, categorical_cols, max_categories_per_col=None):
        """
        Aplica One-Hot Encoding nas colunas categóricas com controle de cardinalidade
        
        Args:
            categorical_cols (list): Lista de colunas categóricas
            max_categories_per_col (int): Máximo de categorias por coluna (padrão: None)
            
        Returns:
            pd.DataFrame: DataFrame com encoding aplicado
        """
        logging.info("Aplicando One-Hot Encoding...")
        logging.info(f"Processando {len(categorical_cols)} colunas categóricas")
        logging.info(f"Dimensões antes do encoding: {self.df_processed.shape}")
        
        # Processar colunas com alta cardinalidade
        processed_categorical_cols = []
        encoding_stats = {}
        
        for col in categorical_cols:
            unique_count = self.df_processed[col].nunique()
            
            if max_categories_per_col and unique_count > max_categories_per_col:
                # Manter apenas as top N categorias mais frequentes
                top_categories = self.df_processed[col].value_counts().head(max_categories_per_col).index
                
                # Criar nova coluna com categorias agrupadas
                col_processed = f"{col}_processed"
                self.df_processed[col_processed] = self.df_processed[col].apply(
                    lambda x: x if x in top_categories else 'Others'
                )
                processed_categorical_cols.append(col_processed)
                
                # Remover coluna original
                self.df_processed.drop(col, axis=1, inplace=True)
                
                logging.info(f"  {col}: {unique_count} → {len(top_categories)+1} categorias (agrupamento aplicado)")
                encoding_stats[col] = {
                    'original_categories': unique_count,
                    'final_categories': len(top_categories) + 1,
                    'grouped': True
                }
            else:
                processed_categorical_cols.append(col)
                encoding_stats[col] = {
                    'original_categories': unique_count,
                    'final_categories': unique_count,
                    'grouped': False
                }
        
        # Aplicar One-Hot Encoding
        df_encoded = pd.get_dummies(
            self.df_processed, 
            columns=processed_categorical_cols, 
            drop_first=True,
            dummy_na=False  # Não criar dummy para NaN
        )
        
        # Salvar estatísticas
        self.processing_metadata['encoding_stats'] = encoding_stats
        
        new_features_count = df_encoded.shape[1] - self.df_processed.shape[1] + len(processed_categorical_cols)
        logging.info(f"Dimensões após encoding: {df_encoded.shape}")
        logging.info(f"Novas features criadas: {new_features_count}")
        
        return df_encoded
    
    def prepare_features_and_target(self, df_encoded):
        """
        Separa features (X) e variável alvo (y) com validações
        
        Args:
            df_encoded (pd.DataFrame): DataFrame com encoding aplicado
            
        Returns:
            tuple: (X, y) - features e target
            
        Raises:
            ValueError: Se coluna 'target' não existe ou target não é binário
        """
        logging.info("Separando features (X) e variável alvo (y)...")
        
        # Validação da existência da coluna target
        if 'target' not in df_encoded.columns:
            raise ValueError("Coluna 'target' não encontrada no DataFrame codificado")
        
        X = df_encoded.drop('target', axis=1)
        y = df_encoded['target']
        
        # Validações do target
        unique_targets = y.unique()
        if not set(unique_targets).issubset({0, 1}):
            raise ValueError(f"Target deve ser binário (0 e 1), encontrado: {unique_targets}")
        
        # Verificar balanceamento
        target_counts = y.value_counts()
        target_proportions = y.value_counts(normalize=True)
        
        logging.info(f"Features (X): {X.shape}")
        logging.info(f"Target (y): {y.shape}")
        logging.info(f"Distribuição da variável alvo:")
        for value, count in target_counts.items():
            prop = target_proportions[value]
            logging.info(f"  Classe {value}: {count:,} ({prop:.1%})")
        
        # Calcular taxa de desbalanceamento
        if len(target_counts) == 2:
            imbalance_ratio = target_counts.max() / target_counts.min()
            logging.info(f"Taxa de desbalanceamento: {imbalance_ratio:.1f}:1")
            
            if imbalance_ratio > 10:
                logging.warning("Dataset altamente desbalanceado detectado! Considere usar técnicas de balanceamento.")
        
        # Verificar dados faltantes
        missing_features = X.isnull().sum().sum()
        missing_target = y.isnull().sum()
        
        if missing_features > 0:
            logging.warning(f"Features com dados faltantes: {missing_features}")
        if missing_target > 0:
            logging.warning(f"Target com dados faltantes: {missing_target}")
        
        return X, y
    
    def split_train_test(self, X, y):
        """
        Divide os dados em conjuntos de treino e teste com controle de estratificação
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logging.info("Dividindo dados em treino e teste...")
        logging.info(f"Proporção de teste: {TEST_SIZE} ({TEST_SIZE*100}%)")
        logging.info(f"Random state: {RANDOM_STATE}")
        logging.info(f"Estratificação: {'Sim' if self.stratify_enabled else 'Não'}")
        
        # Configurar estratificação
        stratify = y if self.stratify_enabled else None
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=TEST_SIZE, 
                random_state=RANDOM_STATE, 
                stratify=stratify
            )
        except ValueError as e:
            if "stratify" in str(e).lower():
                logging.warning(f"Erro na estratificação: {e}")
                logging.warning("Tentando divisão sem estratificação...")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=TEST_SIZE, 
                    random_state=RANDOM_STATE, 
                    stratify=None
                )
            else:
                raise
        
        # Verificar distribuição após divisão
        train_counts = y_train.value_counts()
        test_counts = y_test.value_counts()
        train_props = y_train.value_counts(normalize=True)
        test_props = y_test.value_counts(normalize=True)
        
        logging.info("Distribuição após divisão:")
        logging.info(f"  Treino: Total={len(y_train):,}")
        for class_val in sorted(train_counts.index):
            count = train_counts[class_val]
            prop = train_props[class_val]
            logging.info(f"    Classe {class_val}: {count:,} ({prop:.1%})")
            
        logging.info(f"  Teste:  Total={len(y_test):,}")
        for class_val in sorted(test_counts.index):
            count = test_counts[class_val]
            prop = test_props[class_val]
            logging.info(f"    Classe {class_val}: {count:,} ({prop:.1%})")
        
        # Verificar se a estratificação funcionou
        if self.stratify_enabled and len(train_props) == len(test_props):
            max_diff = max(abs(train_props[i] - test_props[i]) for i in train_props.index)
            if max_diff < 0.01:  # Diferença menor que 1%
                logging.info("✅ Estratificação bem-sucedida (diferença < 1%)")
            else:
                logging.warning(f"⚠️ Estratificação com diferença: {max_diff:.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def apply_feature_scaling(self, X_train, X_test):
        """
        Aplica escalonamento nas features numéricas com prevenção de data leakage
        
        IMPORTANTE: O scaler é ajustado (fit) APENAS nos dados de treino para evitar
        data leakage. Em seguida, transforma tanto treino quanto teste.
        
        Args:
            X_train (pd.DataFrame): Features de treino
            X_test (pd.DataFrame): Features de teste
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled) - DataFrames com features escaladas
        """
        if not self.apply_scaling:
            logging.info("Escalonamento desabilitado - pulando etapa")
            return X_train.copy(), X_test.copy()
            
        logging.info("Aplicando escalonamento nas features numéricas...")
        
        # Identificar features numéricas (excluir binárias do one-hot encoding)
        numeric_features = []
        for col in X_train.columns:
            # Uma feature é considerada numérica se:
            # 1. Tem dtype numérico E
            # 2. Não é binária (tem mais de 2 valores únicos) E
            # 3. Não parece ser resultado de one-hot encoding
            if (X_train[col].dtype in ['int64', 'float64'] and 
                X_train[col].nunique() > 2 and
                not set(X_train[col].unique()).issubset({0, 1})):
                numeric_features.append(col)
        
        self.numeric_features = numeric_features
        
        if not numeric_features:
            logging.info("Nenhuma feature numérica encontrada para escalonamento")
            self.processing_metadata['scaling_info'] = {'applied': False, 'reason': 'no_numeric_features'}
            return X_train.copy(), X_test.copy()
        
        logging.info(f"Features numéricas identificadas para escalonamento: {len(numeric_features)}")
        for feature in numeric_features[:5]:  # Mostrar primeiras 5
            mean_val = X_train[feature].mean()
            std_val = X_train[feature].std()
            min_val = X_train[feature].min()
            max_val = X_train[feature].max()
            logging.info(f"  {feature}: μ={mean_val:.2f}, σ={std_val:.2f}, range=[{min_val:.1f}, {max_val:.1f}]")
        
        if len(numeric_features) > 5:
            logging.info(f"  ... e mais {len(numeric_features) - 5} features numéricas")
        
        # Criar cópias para não modificar os originais
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # PASSO CRÍTICO: Ajustar scaler APENAS nos dados de treino
        self.scaler = StandardScaler()
        
        try:
            # 1. Fit no treino (aprende média e desvio padrão)
            X_train_scaled[numeric_features] = self.scaler.fit_transform(X_train[numeric_features])
            
            # 2. Transform no teste (aplica mesma transformação aprendida)
            X_test_scaled[numeric_features] = self.scaler.transform(X_test[numeric_features])
            
            # Verificar se a transformação funcionou
            train_means = X_train_scaled[numeric_features].mean()
            train_stds = X_train_scaled[numeric_features].std()
            
            # Verificar se treino está padronizado (média ≈ 0, std ≈ 1)
            if (abs(train_means).max() < 0.01) and (abs(train_stds - 1).max() < 0.01):
                logging.info("✅ Escalonamento aplicado com sucesso")
                logging.info(f"   Treino padronizado: média ≈ 0, std ≈ 1")
            else:
                logging.warning("⚠️ Escalonamento pode não ter funcionado perfeitamente")
                
            # Verificar diferenças entre treino e teste (devem ser pequenas se bem estratificado)
            test_means = X_test_scaled[numeric_features].mean()
            test_stds = X_test_scaled[numeric_features].std()
            mean_diff = abs(train_means - test_means).max()
            std_diff = abs(train_stds - test_stds).max()
            
            logging.info(f"   Diferença treino-teste: média_max={mean_diff:.3f}, std_max={std_diff:.3f}")
            
            if mean_diff > 0.5 or std_diff > 0.5:
                logging.warning("⚠️ Grande diferença entre treino e teste após escalonamento")
                logging.warning("   Isso pode indicar datasets muito diferentes")
            
            # Salvar informações do escalonamento
            scaling_info = {
                'applied': True,
                'n_features_scaled': len(numeric_features),
                'features_scaled': numeric_features,
                'scaler_params': {
                    'mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
                    'scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None
                },
                'train_stats_after': {
                    'mean_max': abs(train_means).max(),
                    'std_mean': train_stds.mean()
                }
            }
            
            self.processing_metadata['scaling_info'] = scaling_info
            
            logging.info(f"Estatísticas pós-escalonamento salvas nos metadados")
            
            return X_train_scaled, X_test_scaled
            
        except Exception as e:
            logging.error(f"Erro durante escalonamento: {e}")
            logging.warning("Continuando sem escalonamento...")
            self.processing_metadata['scaling_info'] = {'applied': False, 'error': str(e)}
            return X_train.copy(), X_test.copy()
    
    def get_feature_summary(self, X):
        """
        Exibe resumo detalhado das features finais com estatísticas
        
        Args:
            X (pd.DataFrame): DataFrame de features
        """
        logging.info("Gerando resumo das features finais...")
        
        # Estatísticas básicas
        total_features = X.shape[1]
        feature_types = X.dtypes.value_counts()
        
        logging.info(f"Total de features: {total_features}")
        logging.info(f"Tipos de dados:")
        for dtype, count in feature_types.items():
            logging.info(f"  {dtype}: {count} features")
        
        # Identificar tipos de features
        numeric_features = X.select_dtypes(include=np.number).columns
        binary_features = []
        categorical_features = []
        
        for col in X.columns:
            unique_vals = X[col].nunique()
            if unique_vals == 2 and set(X[col].unique()).issubset({0, 1}):
                binary_features.append(col)
            elif X[col].dtype == 'object' or (X[col].dtype in ['int64', 'float64'] and unique_vals < 10):
                if col not in binary_features:
                    categorical_features.append(col)
        
        logging.info(f"Distribuição por tipo de feature:")
        logging.info(f"  Features numéricas: {len(numeric_features)}")
        logging.info(f"  Features binárias: {len(binary_features)}")
        logging.info(f"  Features categóricas: {len(categorical_features)}")
        
        # Estatísticas para features numéricas
        if len(numeric_features) > 0:
            logging.info("Estatísticas das features numéricas:")
            numeric_stats = X[numeric_features].describe()
            for col in numeric_features[:5]:  # Mostrar primeiras 5
                mean_val = numeric_stats.loc['mean', col]
                std_val = numeric_stats.loc['std', col]
                min_val = numeric_stats.loc['min', col]
                max_val = numeric_stats.loc['max', col]
                logging.info(f"  {col}: μ={mean_val:.2f}, σ={std_val:.2f}, min={min_val:.2f}, max={max_val:.2f}")
            
            if len(numeric_features) > 5:
                logging.info(f"  ... e mais {len(numeric_features) - 5} features numéricas")
        
        # Verificar dados faltantes
        missing_data = X.isnull().sum()
        if missing_data.sum() > 0:
            missing_features = missing_data[missing_data > 0]
            logging.warning(f"Features com dados faltantes: {len(missing_features)}")
            for feature, count in missing_features.head(5).items():
                pct = (count / len(X)) * 100
                logging.warning(f"  {feature}: {count} ({pct:.1f}%)")
        else:
            logging.info("✅ Nenhum dado faltante encontrado")
        
        # Salvar estatísticas nos metadados
        self.processing_metadata['feature_stats'] = {
            'total_features': total_features,
            'numeric_features': len(numeric_features),
            'binary_features': len(binary_features),
            'categorical_features': len(categorical_features),
            'missing_data_features': len(missing_data[missing_data > 0])
        }
        
        # Mostrar exemplos de features
        logging.info(f"Exemplos de features: {list(X.columns[:10])}")
        if total_features > 10:
            logging.info(f"... e mais {total_features - 10} features")
        
        return self.processing_metadata['feature_stats']
    
    def process_features(self, save_path=None, max_categories_per_col=None):
        """
        Executa todo o pipeline de engenharia de features com tratamento de erros robusto
        
        Args:
            save_path (str): Caminho para salvar dados processados (opcional)
            max_categories_per_col (int): Máximo de categorias por coluna categórica (opcional)
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
            
        Raises:
            Exception: Se alguma etapa do pipeline falhar
        """
        logging.info("=" * 60)
        logging.info("INICIANDO ENGENHARIA DE FEATURES")
        logging.info("=" * 60)
        
        pipeline_start_time = pd.Timestamp.now()
        
        try:
            # Etapa 1: Carregar dados
            logging.info("ETAPA 1/6: Carregamento de dados")
            if self.df_clean is None:
                self.load_clean_data()
            
            # Etapa 2: Remover colunas desnecessárias
            logging.info("ETAPA 2/6: Remoção de colunas desnecessárias")
            self.remove_unnecessary_columns()
            
            # Etapa 3: Identificar tipos de colunas
            logging.info("ETAPA 3/6: Identificação de tipos de colunas")
            numeric_cols, categorical_cols = self.identify_column_types()
            
            # Etapa 4: Aplicar One-Hot Encoding
            logging.info("ETAPA 4/6: One-Hot Encoding")
            df_encoded = self.apply_one_hot_encoding(categorical_cols, max_categories_per_col)
            
            # Etapa 5: Preparar features e target
            logging.info("ETAPA 5/6: Preparação de features e target")
            X, y = self.prepare_features_and_target(df_encoded)
            
            # Etapa 6: Divisão treino/teste
            logging.info("ETAPA 6/7: Divisão treino/teste")
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test(X, y)
            
            # Etapa 7: Escalonamento de features (NOVA)
            logging.info("ETAPA 7/7: Escalonamento de features numéricas")
            self.X_train, self.X_test = self.apply_feature_scaling(self.X_train, self.X_test)
            
            # Gerar resumo final
            feature_stats = self.get_feature_summary(X)
            
            # Salvar dados processados se solicitado
            if save_path:
                try:
                    # Garantir que o diretório existe
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    df_encoded.to_csv(save_path, index=False)
                    logging.info(f"Dados processados salvos em: {save_path}")
                except Exception as e:
                    logging.error(f"Erro ao salvar dados processados: {e}")
            
            # Salvar metadados do processamento
            self._save_processing_metadata()
            
            # Tempo total de processamento
            pipeline_end_time = pd.Timestamp.now()
            processing_time = (pipeline_end_time - pipeline_start_time).total_seconds()
            
            logging.info("=" * 60)
            logging.info("ENGENHARIA DE FEATURES CONCLUÍDA COM SUCESSO")
            logging.info(f"Tempo total de processamento: {processing_time:.2f} segundos")
            logging.info(f"Features finais: {X.shape[1]}")
            logging.info(f"Features escaladas: {len(self.numeric_features) if self.numeric_features else 0}")
            logging.info(f"Amostras de treino: {self.X_train.shape[0]:,}")
            logging.info(f"Amostras de teste: {self.X_test.shape[0]:,}")
            if self.apply_scaling and self.processing_metadata['scaling_info'].get('applied'):
                logging.info("✅ Escalonamento aplicado com sucesso")
            logging.info("=" * 60)
            
            return self.X_train, self.X_test, self.y_train, self.y_test
            
        except Exception as e:
            logging.error(f"ERRO NO PIPELINE DE FEATURE ENGINEERING: {str(e)}")
            logging.error("Verificando etapas concluídas...")
            
            # Diagnóstico do estado atual
            if self.df_clean is not None:
                logging.info(f"✅ Dados carregados: {self.df_clean.shape}")
            else:
                logging.error("❌ Falha no carregamento de dados")
                
            if self.df_processed is not None:
                logging.info(f"✅ Dados processados: {self.df_processed.shape}")
            else:
                logging.error("❌ Falha no processamento de dados")
            
            raise
    
    def save_processed_datasets(self, output_dir=None):
        """
        Salva os conjuntos de treino e teste separadamente com validações
        
        Args:
            output_dir (str): Diretório de saída (padrão: DATA_DIR)
            
        Raises:
            ValueError: Se os dados não foram processados
            OSError: Se houver problemas na criação de diretórios ou escrita
        """
        if any(data is None for data in [self.X_train, self.X_test, self.y_train, self.y_test]):
            raise ValueError("Dados não foram processados. Execute process_features() primeiro.")
        
        output_dir = output_dir or DATA_DIR
        
        # Garantir que o diretório existe
        try:
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Diretório de saída: {output_dir}")
        except OSError as e:
            logging.error(f"Erro ao criar diretório {output_dir}: {e}")
            raise
        
        # Definir caminhos dos arquivos
        files_to_save = {
            'X_train.csv': (self.X_train, "Features de treino"),
            'X_test.csv': (self.X_test, "Features de teste"),
            'y_train.csv': (self.y_train, "Target de treino"),
            'y_test.csv': (self.y_test, "Target de teste")
        }
        
        # Salvar arquivo combinado para análise
        combined_train = self.X_train.copy()
        combined_train['target'] = self.y_train
        combined_test = self.X_test.copy()
        combined_test['target'] = self.y_test
        
        files_to_save.update({
            'train_combined.csv': (combined_train, "Dados de treino combinados (X + y)"),
            'test_combined.csv': (combined_test, "Dados de teste combinados (X + y)")
        })
        
        logging.info("Salvando conjuntos de dados...")
        saved_files = {}
        
        for filename, (data, description) in files_to_save.items():
            filepath = os.path.join(output_dir, filename)
            
            try:
                if 'y_' in filename:
                    # Salvar targets com header
                    data.to_csv(filepath, index=False, header=['target'])
                else:
                    # Salvar features normalmente
                    data.to_csv(filepath, index=False)
                
                file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                saved_files[filename] = {
                    'path': filepath,
                    'shape': data.shape,
                    'size_mb': file_size,
                    'description': description
                }
                
                logging.info(f"✅ {description}: {filepath}")
                logging.info(f"   Shape: {data.shape}, Tamanho: {file_size:.2f} MB")
                
            except Exception as e:
                logging.error(f"❌ Erro ao salvar {filename}: {e}")
                raise
        
        # Validar arquivos salvos
        logging.info("Validando arquivos salvos...")
        for filename, info in saved_files.items():
            try:
                # Tentar recarregar para validar
                test_df = pd.read_csv(info['path'])
                
                # Para arquivos y_, ajustar validação (Series foram salvos como DataFrame com header)
                if 'y_' in filename:
                    # Comparar apenas o número de linhas (shape original é Series)
                    expected_rows = info['shape'][0]
                    actual_rows = test_df.shape[0]
                    if actual_rows != expected_rows:
                        logging.warning(f"⚠️ Tamanho inconsistente para {filename}: esperado {expected_rows} linhas, encontrado {actual_rows}")
                    else:
                        logging.info(f"✅ {filename} validado ({actual_rows} linhas)")
                else:
                    # Para outros arquivos, comparar shape completo
                    if test_df.shape != info['shape']:
                        logging.warning(f"⚠️ Tamanho inconsistente para {filename}: esperado {info['shape']}, encontrado {test_df.shape}")
                    else:
                        logging.info(f"✅ {filename} validado")
            except Exception as e:
                logging.error(f"❌ Erro na validação de {filename}: {e}")
        
        # Resumo final
        total_size = sum(info['size_mb'] for info in saved_files.values())
        logging.info(f"Resumo do salvamento:")
        logging.info(f"  Total de arquivos: {len(saved_files)}")
        logging.info(f"  Tamanho total: {total_size:.2f} MB")
        logging.info(f"  Diretório: {output_dir}")
        
        # Salvar scaler se aplicável
        scaler_path = self.save_scaler(output_dir)
        if scaler_path:
            logging.info(f"  Scaler salvo: {os.path.basename(scaler_path)}")
        
        return saved_files
    
    def save_scaler(self, output_dir=None):
        """
        Salva o scaler treinado para uso em produção
        
        Args:
            output_dir (str): Diretório de saída (padrão: DATA_DIR)
            
        Returns:
            str: Caminho do arquivo salvo ou None se não aplicável
        """
        if self.scaler is None or not self.apply_scaling:
            logging.info("Nenhum scaler para salvar (escalonamento não aplicado)")
            return None
            
        output_dir = output_dir or DATA_DIR
        scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
        
        try:
            joblib.dump(self.scaler, scaler_path)
            logging.info(f"✅ Scaler salvo em: {scaler_path}")
            
            # Salvar também informações sobre as features escaladas
            scaler_info = {
                'numeric_features': self.numeric_features,
                'scaler_type': 'StandardScaler',
                'n_features_scaled': len(self.numeric_features) if self.numeric_features else 0,
                'created_at': pd.Timestamp.now().isoformat()
            }
            
            info_path = os.path.join(output_dir, 'scaler_info.json')
            import json
            with open(info_path, 'w') as f:
                json.dump(scaler_info, f, indent=2)
            
            logging.info(f"✅ Informações do scaler salvas em: {info_path}")
            return scaler_path
            
        except Exception as e:
            logging.error(f"Erro ao salvar scaler: {e}")
            return None
    
    def load_scaler(self, scaler_path=None):
        """
        Carrega um scaler previamente salvo
        
        Args:
            scaler_path (str): Caminho para o arquivo do scaler
            
        Returns:
            StandardScaler: Scaler carregado
        """
        scaler_path = scaler_path or os.path.join(DATA_DIR, 'feature_scaler.pkl')
        
        try:
            self.scaler = joblib.load(scaler_path)
            logging.info(f"✅ Scaler carregado de: {scaler_path}")
            return self.scaler
        except Exception as e:
            logging.error(f"Erro ao carregar scaler: {e}")
            return None
    
    def create_sklearn_pipeline(self, max_categories_per_col=None):
        """
        Cria um pipeline Scikit-learn com ColumnTransformer (abordagem avançada)
        
        Esta é uma implementação alternativa mais elegante que encapsula todo o
        pré-processamento em um único objeto Pipeline da Scikit-learn.
        
        Args:
            max_categories_per_col (int): Máximo de categorias por coluna categórica
            
        Returns:
            Pipeline: Pipeline completo de pré-processamento
            
        Nota:
            Este método é uma alternativa ao processo manual. Use apenas se quiser
            uma abordagem mais integrada com o ecossistema Scikit-learn.
        """
        logging.info("Criando Pipeline Scikit-learn com ColumnTransformer...")
        
        if self.df_processed is None:
            raise ValueError("Execute primeiro o processamento básico (até remove_unnecessary_columns)")
        
        # Identificar tipos de colunas (excluindo target)
        feature_df = self.df_processed.drop('target', axis=1, errors='ignore')
        
        numeric_features = []
        categorical_features = []
        
        for col in feature_df.columns:
            if feature_df[col].dtype in ['int64', 'float64']:
                # Features numéricas (não binárias)
                if feature_df[col].nunique() > 2:
                    numeric_features.append(col)
            else:
                # Features categóricas
                unique_count = feature_df[col].nunique()
                if max_categories_per_col and unique_count <= max_categories_per_col:
                    categorical_features.append(col)
                elif not max_categories_per_col:
                    categorical_features.append(col)
                else:
                    logging.warning(f"Coluna {col} tem {unique_count} categorias (> {max_categories_per_col}) - será ignorada")
        
        logging.info(f"Pipeline configurado para:")
        logging.info(f"  Features numéricas: {len(numeric_features)} ({numeric_features[:3]}...)")
        logging.info(f"  Features categóricas: {len(categorical_features)} ({categorical_features[:3]}...)")
        
        # Configurar transformadores
        transformers = []
        
        # Transformador para features numéricas
        if numeric_features and self.apply_scaling:
            numeric_transformer = StandardScaler()
            transformers.append(('num', numeric_transformer, numeric_features))
            logging.info("  → StandardScaler para features numéricas")
        elif numeric_features:
            transformers.append(('num', 'passthrough', numeric_features))
            logging.info("  → Passthrough para features numéricas (sem escalonamento)")
        
        # Transformador para features categóricas
        if categorical_features:
            categorical_transformer = OneHotEncoder(
                drop='first',           # Remove primeira categoria para evitar multicolinearidade
                sparse_output=False,    # Retorna array denso
                handle_unknown='ignore' # Ignora categorias não vistas no treino
            )
            transformers.append(('cat', categorical_transformer, categorical_features))
            logging.info("  → OneHotEncoder para features categóricas")
        
        if not transformers:
            raise ValueError("Nenhuma feature válida encontrada para o pipeline")
        
        # Criar ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',  # Descartar colunas não especificadas
            verbose_feature_names_out=False
        )
        
        # Criar Pipeline completo
        pipeline = Pipeline([
            ('preprocessor', preprocessor)
        ])
        
        logging.info("✅ Pipeline Scikit-learn criado com sucesso")
        logging.info("   Para usar: pipeline.fit(X_train, y_train)")
        logging.info("   Para transformar: X_transformed = pipeline.transform(X_new)")
        
        # Salvar referências para uso posterior
        self.sklearn_pipeline = pipeline
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        return pipeline
    
    def demonstrate_sklearn_pipeline(self):
        """
        Demonstra o uso do Pipeline Scikit-learn como alternativa
        
        Esta função mostra como usar a abordagem ColumnTransformer
        """
        logging.info("🔬 DEMONSTRAÇÃO: Pipeline Scikit-learn")
        logging.info("=" * 50)
        
        try:
            # Carregar e processar dados básicos se necessário
            if self.df_clean is None:
                self.load_clean_data()
            if self.df_processed is None:
                self.remove_unnecessary_columns()
            
            # Separar X e y
            X = self.df_processed.drop('target', axis=1)
            y = self.df_processed['target']
            
            # Divisão treino/teste
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )
            
            # Criar pipeline
            pipeline = self.create_sklearn_pipeline(max_categories_per_col=20)
            
            # Ajustar pipeline nos dados de treino
            logging.info("Ajustando pipeline nos dados de treino...")
            pipeline.fit(X_train)
            
            # Transformar dados
            X_train_transformed = pipeline.transform(X_train)
            X_test_transformed = pipeline.transform(X_test)
            
            # Converter de volta para DataFrame para facilitar uso
            # (Opcional - pode trabalhar com arrays numpy também)
            feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
            
            X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
            X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)
            
            logging.info("✅ Pipeline aplicado com sucesso!")
            logging.info(f"   Shape original: {X_train.shape} → Transformado: {X_train_df.shape}")
            logging.info(f"   Features finais: {len(feature_names)}")
            
            # Verificar se escalonamento foi aplicado (se habilitado)
            if self.apply_scaling and self.numeric_features:
                # Verificar se features numéricas estão padronizadas
                numeric_cols_transformed = [col for col in feature_names if any(nf in col for nf in self.numeric_features[:3])]
                if numeric_cols_transformed:
                    sample_col = numeric_cols_transformed[0]
                    mean_val = X_train_df[sample_col].mean()
                    std_val = X_train_df[sample_col].std()
                    logging.info(f"   Verificação escalonamento - {sample_col}: μ={mean_val:.3f}, σ={std_val:.3f}")
            
            # Salvar pipeline para uso futuro
            pipeline_path = os.path.join(DATA_DIR, 'sklearn_preprocessing_pipeline.pkl')
            try:
                joblib.dump(pipeline, pipeline_path)
                logging.info(f"✅ Pipeline salvo em: {pipeline_path}")
            except Exception as e:
                logging.warning(f"Não foi possível salvar pipeline: {e}")
            
            logging.info("📊 Vantagens do Pipeline Scikit-learn:")
            logging.info("   • Encapsulamento completo do pré-processamento")
            logging.info("   • Prevenção automática de data leakage")
            logging.info("   • Facilidade para aplicar em novos dados")
            logging.info("   • Integração perfeita com modelos Scikit-learn")
            
            return pipeline, X_train_df, X_test_df, y_train, y_test
            
        except Exception as e:
            logging.error(f"Erro na demonstração do pipeline: {e}")
            raise
    
    def _save_processing_metadata(self, output_dir=None):
        """
        Salva metadados do processamento para rastreabilidade
        
        Args:
            output_dir (str): Diretório de saída (padrão: DATA_DIR)
        """
        import json
        from datetime import datetime
        
        output_dir = output_dir or DATA_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        # Enriquecer metadados com informações adicionais
        metadata = self.processing_metadata.copy()
        metadata.update({
            'timestamp': datetime.now().isoformat(),
            'pipeline_config': {
                'high_cardinality_threshold': self.high_cardinality_threshold,
                'stratify_enabled': self.stratify_enabled,
                'test_size': TEST_SIZE,
                'random_state': RANDOM_STATE
            },
            'data_info': {
                'original_shape': self.df_clean.shape if self.df_clean is not None else None,
                'processed_shape': self.df_processed.shape if self.df_processed is not None else None,
                'final_train_shape': self.X_train.shape if self.X_train is not None else None,
                'final_test_shape': self.X_test.shape if self.X_test is not None else None
            }
        })
        
        # Salvar metadados
        metadata_path = os.path.join(output_dir, 'feature_engineering_metadata.json')
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logging.info(f"Metadados salvos em: {metadata_path}")
        except Exception as e:
            logging.error(f"Erro ao salvar metadados: {e}")
    
    def load_processed_data(self, data_dir=None):
        """
        Carrega dados já processados dos arquivos CSV
        
        Args:
            data_dir (str): Diretório dos dados (padrão: DATA_DIR)
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
            
        Raises:
            FileNotFoundError: Se algum arquivo não for encontrado
        """
        data_dir = data_dir or DATA_DIR
        
        required_files = {
            'X_train.csv': 'Features de treino',
            'X_test.csv': 'Features de teste',
            'y_train.csv': 'Target de treino',
            'y_test.csv': 'Target de teste'
        }
        
        logging.info("Carregando dados processados...")
        
        # Verificar se todos os arquivos existem
        missing_files = []
        for filename in required_files.keys():
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                missing_files.append(filename)
        
        if missing_files:
            raise FileNotFoundError(f"Arquivos não encontrados: {missing_files}")
        
        # Carregar dados
        try:
            self.X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
            self.X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
            self.y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))['target']
            self.y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))['target']
            
            logging.info("Dados carregados com sucesso:")
            logging.info(f"  X_train: {self.X_train.shape}")
            logging.info(f"  X_test: {self.X_test.shape}")
            logging.info(f"  y_train: {self.y_train.shape}")
            logging.info(f"  y_test: {self.y_test.shape}")
            
            return self.X_train, self.X_test, self.y_train, self.y_test
            
        except Exception as e:
            logging.error(f"Erro ao carregar dados processados: {e}")
            raise
    
    def get_processing_report(self):
        """
        Gera relatório detalhado do processamento realizado
        
        Returns:
            dict: Relatório completo das transformações
        """
        if not self.processing_metadata:
            logging.warning("Nenhum processamento foi realizado ainda")
            return {}
        
        report = {
            'pipeline_summary': {
                'total_columns_removed': len(self.processing_metadata.get('columns_removed', [])),
                'high_cardinality_detected': len(self.processing_metadata.get('high_cardinality_columns', [])),
                'encoding_applied': len(self.processing_metadata.get('encoding_stats', {})),
                'final_features': self.processing_metadata.get('feature_stats', {}).get('total_features', 0)
            },
            'data_transformation': {
                'columns_removed': self.processing_metadata.get('columns_removed', []),
                'high_cardinality_columns': self.processing_metadata.get('high_cardinality_columns', []),
                'encoding_stats': self.processing_metadata.get('encoding_stats', {}),
                'feature_stats': self.processing_metadata.get('feature_stats', {})
            },
            'validation_status': {
                'data_loaded': self.df_clean is not None,
                'data_processed': self.df_processed is not None,
                'features_ready': all(data is not None for data in [self.X_train, self.X_test, self.y_train, self.y_test])
            }
        }
        
        return report


def main():
    """
    Função principal para executar a engenharia de features com configurações flexíveis
    
    Esta função demonstra o uso completo da classe FeatureEngineer com opções
    avançadas de configuração e tratamento de erros.
    """
    logging.info("🔧 INICIANDO PIPELINE DE FEATURE ENGINEERING")
    logging.info("=" * 70)
    
    try:
        # Configurações do pipeline
        config = {
            'high_cardinality_threshold': 50,  # Limite para alta cardinalidade
            'stratify_enabled': True,          # Usar estratificação
            'max_categories_per_col': 20,      # Máximo de categorias por coluna
            'apply_scaling': True,             # Aplicar escalonamento (NOVO)
        }
        
        logging.info("Configurações do pipeline:")
        for key, value in config.items():
            logging.info(f"  {key}: {value}")
        
        # Inicializar engenheiro de features
        engineer = FeatureEngineer(
            high_cardinality_threshold=config['high_cardinality_threshold'],
            stratify_enabled=config['stratify_enabled'],
            apply_scaling=config['apply_scaling']
        )
        
        # Executar pipeline completo
        X_train, X_test, y_train, y_test = engineer.process_features(
            save_path=PROCESSED_DATA_FILE,
            max_categories_per_col=config['max_categories_per_col']
        )
        
        # Salvar conjuntos finais
        saved_files = engineer.save_processed_datasets()
        
        # Gerar relatório
        report = engineer.get_processing_report()
        
        # Exibir resumo final
        logging.info("🎯 RESUMO FINAL")
        logging.info("=" * 40)
        logging.info(f"✅ Features finais: {X_train.shape[1]}")
        logging.info(f"✅ Amostras de treino: {X_train.shape[0]:,}")
        logging.info(f"✅ Amostras de teste: {X_test.shape[0]:,}")
        logging.info(f"✅ Arquivos salvos: {len(saved_files)}")
        logging.info(f"✅ Colunas removidas: {report['pipeline_summary']['total_columns_removed']}")
        logging.info(f"✅ Encodings aplicados: {report['pipeline_summary']['encoding_applied']}")
        
        # Verificar balanceamento
        train_balance = y_train.value_counts()
        test_balance = y_test.value_counts()
        
        logging.info("📊 Distribuição das classes:")
        logging.info(f"  Treino - Classe 0: {train_balance[0]:,}, Classe 1: {train_balance[1]:,}")
        logging.info(f"  Teste  - Classe 0: {test_balance[0]:,}, Classe 1: {test_balance[1]:,}")
        
        logging.info("🎉 PIPELINE CONCLUÍDO COM SUCESSO!")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logging.error(f"❌ ERRO NO PIPELINE: {str(e)}")
        logging.error("Verifique os logs acima para mais detalhes")
        raise
    
    finally:
        logging.info("=" * 70)


def create_feature_engineer_with_best_practices(data_path=None, **kwargs):
    """
    Função de conveniência para criar FeatureEngineer com configurações otimizadas
    
    Args:
        data_path (str): Caminho para dados limpos
        **kwargs: Argumentos adicionais para personalização
        
    Returns:
        FeatureEngineer: Instância configurada com boas práticas
    """
    # Configurações padrão otimizadas
    default_config = {
        'high_cardinality_threshold': 50,
        'stratify_enabled': True,
        'apply_scaling': True  # Escalonamento habilitado por padrão
    }
    
    # Atualizar com configurações personalizadas
    config = {**default_config, **kwargs}
    
    logging.info("Criando FeatureEngineer com configurações otimizadas:")
    for key, value in config.items():
        logging.info(f"  {key}: {value}")
    
    return FeatureEngineer(clean_data_path=data_path, **config)


def validate_feature_engineering_setup():
    """
    Valida se o ambiente está configurado corretamente para Feature Engineering
    
    Returns:
        bool: True se tudo estiver configurado corretamente
        
    Raises:
        EnvironmentError: Se alguma dependência estiver faltando
    """
    logging.info("🔍 Validando configuração do ambiente...")
    
    # Verificar arquivos necessários
    required_files = [CLEAN_DATA_FILE]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        raise EnvironmentError(f"Arquivos necessários não encontrados: {missing_files}")
    
    # Verificar diretórios
    required_dirs = [DATA_DIR, os.path.dirname(PROCESSED_DATA_FILE)]
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Verificar imports críticos
    try:
        import sklearn
        import pandas
        import numpy
        logging.info(f"✅ Dependências OK - sklearn: {sklearn.__version__}, pandas: {pandas.__version__}")
    except ImportError as e:
        raise EnvironmentError(f"Dependência faltando: {e}")
    
    # Verificar dados básicos
    try:
        df_test = pd.read_csv(CLEAN_DATA_FILE, nrows=5)
        if 'target' not in df_test.columns:
            raise EnvironmentError("Coluna 'target' não encontrada nos dados limpos")
        logging.info(f"✅ Dados básicos OK - {df_test.shape[1]} colunas detectadas")
    except Exception as e:
        raise EnvironmentError(f"Erro ao validar dados: {e}")
    
    logging.info("✅ Ambiente validado com sucesso!")
    return True


def quick_feature_engineering_demo():
    """
    Demonstração rápida do pipeline de Feature Engineering
    
    Esta função executa uma versão simplificada do pipeline para testes rápidos
    """
    logging.info("🚀 DEMO RÁPIDA - FEATURE ENGINEERING")
    logging.info("=" * 50)
    
    try:
        # Validar ambiente
        validate_feature_engineering_setup()
        
        # Criar engenheiro com configurações de demo
        engineer = create_feature_engineer_with_best_practices(
            high_cardinality_threshold=30,  # Mais restritivo para demo
            stratify_enabled=True,
            apply_scaling=True              # Testar escalonamento
        )
        
        # Executar pipeline com sample dos dados
        logging.info("Executando pipeline com configurações de demonstração...")
        
        # Carregar apenas uma amostra para demonstração
        df_sample = pd.read_csv(CLEAN_DATA_FILE, nrows=1000)
        engineer.df_clean = df_sample
        
        # Executar pipeline
        X_train, X_test, y_train, y_test = engineer.process_features()
        
        # Mostrar resultados
        logging.info("📊 RESULTADOS DA DEMO:")
        logging.info(f"  Amostra original: {df_sample.shape}")
        logging.info(f"  Features finais: {X_train.shape[1]}")
        logging.info(f"  Treino: {X_train.shape[0]} amostras")
        logging.info(f"  Teste: {X_test.shape[0]} amostras")
        
        # Relatório simplificado
        report = engineer.get_processing_report()
        logging.info(f"  Transformações: {report['pipeline_summary']}")
        
        logging.info("✅ Demo concluída com sucesso!")
        
        return engineer
        
    except Exception as e:
        logging.error(f"❌ Erro na demo: {e}")
        raise

if __name__ == "__main__":
    """
    Executar o módulo diretamente para processamento completo ou demonstração
    
    Uso:
        python feature_engineering.py           # Pipeline completo
        python feature_engineering.py --demo    # Demonstração rápida
        python feature_engineering.py --validate # Apenas validação
    """
    import sys
    
    # Configurar logging para execução direta
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('feature_engineering.log')
        ]
    )
    
    # Processar argumentos de linha de comando
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg == '--demo':
            logging.info("Executando demonstração rápida...")
            quick_feature_engineering_demo()
            
        elif arg == '--validate':
            logging.info("Executando apenas validação...")
            validate_feature_engineering_setup()
            logging.info("Validação concluída!")
            
        elif arg == '--help':
            print("""
Uso do Feature Engineering Module:

python feature_engineering.py           # Pipeline completo
python feature_engineering.py --demo    # Demonstração rápida  
python feature_engineering.py --validate # Apenas validação
python feature_engineering.py --help    # Esta mensagem

O pipeline completo executa todas as etapas de Feature Engineering
e salva os conjuntos finais de treino e teste.
            """)
        else:
            logging.warning(f"Argumento desconhecido: {arg}")
            logging.info("Use --help para ver opções disponíveis")
            main()
    else:
        # Execução padrão - pipeline completo
        main()
