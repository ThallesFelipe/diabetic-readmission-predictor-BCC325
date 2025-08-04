"""
Módulo para análise exploratória dos dados de readmissão hospitalar diabética

Este módulo implementa uma análise exploratória completa com visualizações gráficas,
estatísticas descritivas detalhadas e análises de relações entre variáveis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import sys
from datetime import datetime
from scipy import stats

# Adicionar o diretório pai ao path para importações
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RAW_DATA_FILE, CLEAN_DATA_FILE, RESULTS_DIR
from src.id_mapping_utils import IDMappingUtils

# Configurações de visualização
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Criar diretório de resultados se não existir
os.makedirs(RESULTS_DIR, exist_ok=True)

class ExploratoryDataAnalysis:
    """
    Classe para análise exploratória abrangente dos dados
    
    Esta classe implementa uma análise exploratória completa com:
    - Visualizações gráficas interativas
    - Estatísticas descritivas detalhadas
    - Análise de correlações e relações entre variáveis
    - Geração de relatórios automatizados
    """
    
    def __init__(self, data_path=None):
        """
        Inicializa a análise exploratória
        
        Args:
            data_path (str): Caminho para o arquivo de dados. 
                           Por padrão usa dados limpos (CLEAN_DATA_FILE)
        """
        # Priorizar dados limpos se disponíveis
        if data_path is None:
            if os.path.exists(CLEAN_DATA_FILE):
                self.data_path = CLEAN_DATA_FILE
                print(f"✅ Usando dados limpos: {CLEAN_DATA_FILE}")
            else:
                self.data_path = RAW_DATA_FILE
                print(f"⚠️  Dados limpos não encontrados. Usando dados brutos: {RAW_DATA_FILE}")
        else:
            self.data_path = data_path
            
        self.df = None
        self.id_mapper = IDMappingUtils()
        self.results_dir = RESULTS_DIR
        self.figures = []  # Lista para armazenar figuras geradas
        
        # Carregar mapeamentos de IDs
        try:
            self.id_mapper.load_mappings()
        except Exception as e:
            print(f"Aviso: Não foi possível carregar mapeamentos de IDs: {e}")
    
    def save_figure(self, fig, filename, dpi=300):
        """Salva figura em alta qualidade"""
        filepath = os.path.join(self.results_dir, f"{filename}.png")
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        self.figures.append(filepath)
        plt.close(fig)
        print(f"  📊 Gráfico salvo: {filename}.png")
        
    def load_data(self):
        """Carrega os dados do arquivo CSV"""
        print(f"Carregando dados de: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"Dados carregados. Shape: {self.df.shape}")
        return self.df
    
    def basic_info(self):
        """Exibe informações básicas sobre o dataset"""
        print("\n" + "="*60)
        print("INFORMAÇÕES BÁSICAS DO DATASET")
        print("="*60)
        
        print(f"\nTamanho do dataset: {self.df.shape}")
        print(f"Número de colunas: {self.df.shape[1]}")
        print(f"Número de registros: {self.df.shape[0]}")
        print(f"Número de pacientes únicos: {self.df['patient_nbr'].nunique()}")
        print(f"Número de encontros: {len(self.df)}")
        
        print(f"\nInformações dos tipos de dados:")
        print(self.df.info())
        
        print(f"\nPrimeiras 5 linhas:")
        print(self.df.head())
        
    def analyze_target_variable(self):
        """Analisa a variável alvo 'readmitted' com visualizações"""
        print("\n" + "="*60)
        print("ANÁLISE DA VARIÁVEL ALVO")
        print("="*60)
        
        # Verificar se a coluna 'readmitted' existe
        if 'readmitted' not in self.df.columns:
            print("❌ Coluna 'readmitted' não encontrada no dataset!")
            # Tentar encontrar colunas similares
            similar_cols = [col for col in self.df.columns if 'readmit' in col.lower()]
            if similar_cols:
                print(f"Colunas similares encontradas: {similar_cols}")
            return
        
        # Verificar se a coluna 'target' já existe (deve vir dos dados limpos)
        if 'target' not in self.df.columns:
            print("⚠️  Coluna 'target' não encontrada! Verificando se pode ser criada...")
            if 'readmitted' in self.df.columns:
                print("🎯 Criando variável target a partir de 'readmitted'...")
                self.df['target'] = self.df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
                print("✅ Variável target criada temporariamente para análise!")
            else:
                print("❌ Não foi possível criar a variável target - 'readmitted' não encontrada!")
                return
        else:
            print("✅ Variável target encontrada nos dados limpos!")
        
        print("\nDistribuição da variável 'readmitted':")
        if 'readmitted' in self.df.columns:
            readmitted_counts = self.df['readmitted'].value_counts()
            print(readmitted_counts)
            print(f"\nDistribuição percentual:")
            readmitted_pct = self.df['readmitted'].value_counts(normalize=True) * 100
            print(readmitted_pct.round(2))
        
        print(f"\n📊 Estatísticas da variável target:")
        print(f"Proporção de readmissões em <30 dias: {self.df['target'].mean():.3f}")
        print(f"Total de readmissões em <30 dias: {self.df['target'].sum():,}")
        print(f"Total de casos: {len(self.df):,}")
        
        # Verificar se há dados válidos para visualização
        if readmitted_counts.empty:
            print("⚠️  Sem dados válidos para visualização da variável target")
            return
        
        # Visualizações
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Gráfico de barras - readmitted
        ax1 = axes[0]
        order = ['NO', '>30', '<30']
        # Filtrar apenas valores que existem nos dados
        available_order = [val for val in order if val in readmitted_counts.index]
        
        if available_order:
            sns.countplot(data=self.df, x='readmitted', order=available_order, ax=ax1)
            ax1.set_title('Distribuição da Variável Readmitted', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Status de Readmissão')
            ax1.set_ylabel('Contagem')
            
            # Adicionar valores nas barras
            for i, cat in enumerate(available_order):
                if cat in readmitted_counts.index:
                    v = readmitted_counts[cat]
                    pct = readmitted_pct[cat] if cat in readmitted_pct.index else 0
                    ax1.text(i, v + max(readmitted_counts.values)*0.01, f'{v:,}\n({pct:.1f}%)', 
                            ha='center', va='bottom', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'Dados insuficientes\npara visualização', 
                    ha='center', va='center', transform=ax1.transAxes)
        
        # Gráfico de pizza - target binário
        ax2 = axes[1]
        target_counts = self.df['target'].value_counts()
        if len(target_counts) > 1:
            labels = ['Não Readmitido (<30d)', 'Readmitido (<30d)']
            colors = ['lightblue', 'salmon']
            wedges, texts, autotexts = ax2.pie(target_counts.values, labels=labels, 
                                              autopct='%1.1f%%', colors=colors, startangle=90)
            ax2.set_title('Distribuição da Variável Target\n(Readmissão <30 dias)', 
                         fontsize=14, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Dados de target\ninsuficientes', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # Gráfico de barras horizontais - comparação
        ax3 = axes[2]
        comparison_data = pd.DataFrame({
            'Categoria': ['Total de Casos', 'Readmissões <30d', 'Readmissões >30d', 'Não Readmitidos'],
            'Valor': [len(self.df), 
                     (self.df['readmitted'] == '<30').sum(),
                     (self.df['readmitted'] == '>30').sum(),
                     (self.df['readmitted'] == 'NO').sum()]
        })
        
        bars = ax3.barh(comparison_data['Categoria'], comparison_data['Valor'], 
                       color=['darkblue', 'red', 'orange', 'green'])
        ax3.set_title('Comparação de Categorias', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Número de Casos')
        
        # Adicionar valores nas barras
        for i, (bar, value) in enumerate(zip(bars, comparison_data['Valor'])):
            ax3.text(value + max(comparison_data['Valor'])*0.01, i, f'{value:,}', 
                    va='center', fontweight='bold')
        
        plt.tight_layout()
        self.save_figure(fig, 'target_variable_analysis')
        
        # Estatísticas detalhadas
        print(f"\n📊 ESTATÍSTICAS DETALHADAS:")
        print(f"  • Taxa de readmissão geral: {((self.df['readmitted'] != 'NO').mean() * 100):.2f}%")
        print(f"  • Taxa de readmissão <30 dias: {(self.df['target'].mean() * 100):.2f}%")
        print(f"  • Taxa de readmissão >30 dias: {((self.df['readmitted'] == '>30').mean() * 100):.2f}%")
        
        # Verificar se há dados para calcular a razão
        readmit_30_count = (self.df['readmitted'] == '<30').sum()
        readmit_gt30_count = (self.df['readmitted'] == '>30').sum()
        if readmit_gt30_count > 0:
            ratio = readmit_30_count / readmit_gt30_count
            print(f"  • Razão readmissão <30d / >30d: {ratio:.2f}")
        else:
            print(f"  • Razão readmissão <30d / >30d: N/A (sem readmissões >30d)")
    
    def analyze_missing_data(self):
        """Analisa dados faltantes no dataset com visualizações"""
        print("\n" + "="*60)
        print("ANÁLISE DE DADOS FALTANTES")
        print("="*60)
        
        # Calcular porcentagem de dados faltantes por coluna
        missing_data = pd.DataFrame({
            'Missing_Count': self.df.isnull().sum(),
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df)) * 100
        })
        
        # Filtrar apenas colunas com dados faltantes
        missing_data = missing_data[missing_data['Missing_Count'] > 0]
        missing_data = missing_data.sort_values('Missing_Percentage', ascending=False)
        
        print(f"\nColunas com dados faltantes ({len(missing_data)} de {len(self.df.columns)}):")
        for col, row in missing_data.head(10).iterrows():
            print(f"  {col}: {row['Missing_Count']:,} ({row['Missing_Percentage']:.1f}%)")
        
        # Analisar valores especiais ('?', 'NULL', etc.)
        print(f"\nAnálise de valores especiais:")
        special_values = ['?', 'NULL', 'null', 'None', '']
        special_counts = {}
        for value in special_values:
            count = (self.df == value).sum().sum()
            if count > 0:
                special_counts[value] = count
                print(f"  Valores '{value}': {count:,}")
        
        # Análise especial para valores '?' (comum neste dataset)
        if '?' in self.df.values:
            print(f"\n🔍 Análise detalhada de valores '?':")
            for col in self.df.columns:
                question_count = (self.df[col] == '?').sum()
                if question_count > 0:
                    pct = (question_count / len(self.df)) * 100
                    print(f"  {col}: {question_count:,} ({pct:.1f}%)")
        
        # Visualizações
        if len(missing_data) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Mapa de calor de dados faltantes
            ax1 = axes[0, 0]
            # Selecionar colunas com mais dados faltantes para visualização
            cols_to_show = missing_data.head(15).index.tolist()
            missing_matrix = self.df[cols_to_show].isnull()
            sns.heatmap(missing_matrix, cbar=True, cmap='viridis', ax=ax1)
            ax1.set_title('Mapa de Calor - Dados Faltantes\n(Top 15 colunas)', 
                         fontsize=14, fontweight='bold')
            ax1.set_xlabel('Colunas')
            ax1.set_ylabel('Registros (amostra)')
            
            # Gráfico de barras - porcentagem de dados faltantes
            ax2 = axes[0, 1]
            top_missing = missing_data.head(15)
            bars = ax2.barh(range(len(top_missing)), top_missing['Missing_Percentage'], 
                           color='coral')
            ax2.set_yticks(range(len(top_missing)))
            ax2.set_yticklabels(top_missing.index, fontsize=10)
            ax2.set_xlabel('Porcentagem de Dados Faltantes (%)')
            ax2.set_title('Top 15 Colunas com Dados Faltantes', 
                         fontsize=14, fontweight='bold')
            
            # Adicionar valores nas barras
            for i, (bar, value) in enumerate(zip(bars, top_missing['Missing_Percentage'])):
                ax2.text(value + 0.5, i, f'{value:.1f}%', va='center', fontsize=9)
            
            # Distribuição geral de missingness
            ax3 = axes[1, 0]
            missingness_per_row = self.df.isnull().sum(axis=1)
            ax3.hist(missingness_per_row, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Número de Valores Faltantes por Registro')
            ax3.set_ylabel('Frequência')
            ax3.set_title('Distribuição de Valores Faltantes por Registro', 
                         fontsize=14, fontweight='bold')
            ax3.axvline(missingness_per_row.mean(), color='red', linestyle='--', 
                       label=f'Média: {missingness_per_row.mean():.1f}')
            ax3.legend()
            
            # Padrões de dados faltantes (combinações)
            ax4 = axes[1, 1]
            if len(cols_to_show) >= 3:
                # Analisar padrões de missing data para as top 5 colunas
                top_5_cols = cols_to_show[:5]
                missing_patterns = self.df[top_5_cols].isnull().astype(int)
                pattern_counts = missing_patterns.groupby(top_5_cols).size().sort_values(ascending=False).head(10)
                
                if len(pattern_counts) > 1:
                    ax4.bar(range(len(pattern_counts)), pattern_counts.values, color='lightgreen')
                    ax4.set_xlabel('Padrões de Dados Faltantes (Top 10)')
                    ax4.set_ylabel('Frequência')
                    ax4.set_title('Padrões de Combinação\nDados Faltantes (Top 5 colunas)', 
                                 fontsize=14, fontweight='bold')
                    ax4.set_xticks(range(len(pattern_counts)))
                    ax4.set_xticklabels([f'P{i+1}' for i in range(len(pattern_counts))], rotation=45)
                else:
                    ax4.text(0.5, 0.5, 'Padrões insuficientes\npara análise', 
                            ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                    ax4.set_title('Padrões de Dados Faltantes', fontsize=14, fontweight='bold')
            else:
                ax4.text(0.5, 0.5, 'Dados insuficientes\npara análise de padrões', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Padrões de Dados Faltantes', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            self.save_figure(fig, 'missing_data_analysis')
        
        # Estatísticas detalhadas
        print(f"\n📊 ESTATÍSTICAS DE DADOS FALTANTES:")
        total_cells = len(self.df) * len(self.df.columns)
        total_missing = self.df.isnull().sum().sum()
        print(f"  • Total de células: {total_cells:,}")
        print(f"  • Total de valores faltantes: {total_missing:,}")
        print(f"  • Porcentagem geral de missing: {(total_missing/total_cells)*100:.2f}%")
        print(f"  • Registros completamente preenchidos: {len(self.df.dropna()):,} ({(len(self.df.dropna())/len(self.df))*100:.1f}%)")
        print(f"  • Colunas sem dados faltantes: {len(self.df.columns) - len(missing_data)}")
        
        if special_counts:
            print(f"  • Total de valores especiais: {sum(special_counts.values()):,}")
    
    def analyze_demographic_data(self):
        """Analisa dados demográficos com visualizações"""
        print("\n" + "="*60)
        print("ANÁLISE DE DADOS DEMOGRÁFICOS")
        print("="*60)
        
        # Preparar dados demográficos
        demographic_cols = ['age', 'gender', 'race']
        available_cols = [col for col in demographic_cols if col in self.df.columns]
        
        if not available_cols:
            print("⚠️  Colunas demográficas não encontradas no dataset.")
            print("Colunas disponíveis:", list(self.df.columns)[:10], "...")
            return
        
        # Criar subplots baseado nas colunas disponíveis
        n_cols = min(len(available_cols), 3)  # Máximo de 3 colunas para visualização
        fig, axes = plt.subplots(2, n_cols, figsize=(6*n_cols, 12))
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        elif n_cols == 2:
            axes = axes.reshape(2, 2)
        
        col_idx = 0
        
        # Processar apenas as primeiras 3 colunas demográficas
        cols_to_process = available_cols[:3]
        
        # Análise de idade
        if 'age' in cols_to_process and col_idx < n_cols:
            print(f"\nDistribuição por faixa etária:")
            age_counts = self.df['age'].value_counts().sort_index()
            age_target = self.df.groupby('age')['target'].agg(['count', 'mean']).round(3)
            age_target.columns = ['Total', 'Taxa_Readmissao']
            
            for age_range, count in age_counts.items():
                pct = (count / len(self.df)) * 100
                readmission_rate = age_target.loc[age_range, 'Taxa_Readmissao'] * 100
                print(f"  {age_range}: {count:,} ({pct:.1f}%) - Taxa readmissão: {readmission_rate:.1f}%")
            
            # Gráfico de barras - distribuição por idade
            ax1 = axes[0, col_idx]
            bars1 = ax1.bar(range(len(age_counts)), age_counts.values, color='lightblue', alpha=0.8)
            ax1.set_title('Distribuição por Faixa Etária', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Faixa Etária')
            ax1.set_ylabel('Número de Pacientes')
            ax1.set_xticks(range(len(age_counts)))
            ax1.set_xticklabels(age_counts.index, rotation=45)
            
            # Adicionar valores nas barras
            for i, v in enumerate(age_counts.values):
                ax1.text(i, v + max(age_counts.values)*0.01, f'{v:,}', 
                        ha='center', va='bottom', fontsize=10)
            
            # Gráfico de linha - taxa de readmissão por idade
            ax2 = axes[1, col_idx]
            ax2.plot(range(len(age_target)), age_target['Taxa_Readmissao'] * 100, 
                    marker='o', linewidth=2, markersize=8, color='red')
            ax2.set_title('Taxa de Readmissão por Faixa Etária', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Faixa Etária')
            ax2.set_ylabel('Taxa de Readmissão (%)')
            ax2.set_xticks(range(len(age_target)))
            ax2.set_xticklabels(age_target.index, rotation=45)
            ax2.grid(True, alpha=0.3)
            
            col_idx += 1
        
        # Análise de gênero
        if 'gender' in cols_to_process and col_idx < n_cols:
            print(f"\nDistribuição por gênero:")
            gender_counts = self.df['gender'].value_counts()
            gender_target = self.df.groupby('gender')['target'].agg(['count', 'mean']).round(3)
            gender_target.columns = ['Total', 'Taxa_Readmissao']
            
            for gender, count in gender_counts.items():
                pct = (count / len(self.df)) * 100
                if gender in gender_target.index:
                    readmission_rate = gender_target.loc[gender, 'Taxa_Readmissao'] * 100
                    print(f"  {gender}: {count:,} ({pct:.1f}%) - Taxa readmissão: {readmission_rate:.1f}%")
                else:
                    print(f"  {gender}: {count:,} ({pct:.1f}%)")
            
            # Gráfico de pizza - distribuição por gênero
            ax1 = axes[0, col_idx]
            colors = ['lightcoral', 'lightskyblue', 'lightgreen', 'gold'][:len(gender_counts)]
            wedges, texts, autotexts = ax1.pie(gender_counts.values, labels=gender_counts.index, 
                                              autopct='%1.1f%%', colors=colors, startangle=90)
            ax1.set_title('Distribuição por Gênero', fontsize=14, fontweight='bold')
            
            # Gráfico de barras - taxa de readmissão por gênero
            ax2 = axes[1, col_idx]
            valid_genders = [g for g in gender_counts.index if g in gender_target.index]
            if valid_genders:
                bars2 = ax2.bar(valid_genders, 
                               [gender_target.loc[g, 'Taxa_Readmissao'] * 100 for g in valid_genders],
                               color=['lightcoral', 'lightskyblue', 'lightgreen'][:len(valid_genders)])
                ax2.set_title('Taxa de Readmissão por Gênero', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Gênero')
                ax2.set_ylabel('Taxa de Readmissão (%)')
                
                # Adicionar valores nas barras
                for i, g in enumerate(valid_genders):
                    v = gender_target.loc[g, 'Taxa_Readmissao'] * 100
                    ax2.text(i, v + 0.2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            col_idx += 1
        
        # Análise de raça
        if 'race' in cols_to_process and col_idx < n_cols:
            print(f"\nDistribuição por raça:")
            race_counts = self.df['race'].value_counts()
            race_target = self.df.groupby('race')['target'].agg(['count', 'mean']).round(3)
            race_target.columns = ['Total', 'Taxa_Readmissao']
            
            for race, count in race_counts.head(10).items():
                pct = (count / len(self.df)) * 100
                if race in race_target.index:
                    readmission_rate = race_target.loc[race, 'Taxa_Readmissao'] * 100
                    print(f"  {race}: {count:,} ({pct:.1f}%) - Taxa readmissão: {readmission_rate:.1f}%")
                else:
                    print(f"  {race}: {count:,} ({pct:.1f}%)")
            
            # Gráfico de barras horizontais - distribuição por raça (top 10)
            ax1 = axes[0, col_idx]
            top_races = race_counts.head(10)
            bars1 = ax1.barh(range(len(top_races)), top_races.values, color='lightgreen')
            ax1.set_title('Distribuição por Raça (Top 10)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Número de Pacientes')
            ax1.set_yticks(range(len(top_races)))
            ax1.set_yticklabels(top_races.index, fontsize=10)
            
            # Adicionar valores nas barras
            for i, v in enumerate(top_races.values):
                ax1.text(v + max(top_races.values)*0.01, i, f'{v:,}', 
                        va='center', ha='left', fontsize=9)
            
            # Gráfico de barras - taxa de readmissão por raça (top 10)
            ax2 = axes[1, col_idx]
            valid_races = [r for r in top_races.index if r in race_target.index]
            if valid_races:
                readmission_rates = [race_target.loc[r, 'Taxa_Readmissao'] * 100 for r in valid_races]
                bars2 = ax2.barh(range(len(valid_races)), readmission_rates, color='salmon')
                ax2.set_title('Taxa de Readmissão por Raça (Top 10)', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Taxa de Readmissão (%)')
                ax2.set_yticks(range(len(valid_races)))
                ax2.set_yticklabels(valid_races, fontsize=10)
                
                # Adicionar valores nas barras
                for i, v in enumerate(readmission_rates):
                    ax2.text(v + 0.1, i, f'{v:.1f}%', va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        self.save_figure(fig, 'demographic_analysis')
        
        # Estatísticas detalhadas
        print(f"\n📊 ESTATÍSTICAS DEMOGRÁFICAS:")
        if 'age' in self.df.columns:
            age_counts = self.df['age'].value_counts().sort_index()
            age_stats = self.df.groupby('age')['target'].agg(['count', 'mean'])
            most_common_age = age_counts.index[0]
            highest_risk_age = age_stats['mean'].idxmax()
            print(f"  • Faixa etária mais comum: {most_common_age}")
            print(f"  • Faixa etária com maior risco: {highest_risk_age} ({age_stats.loc[highest_risk_age, 'mean']*100:.1f}%)")
        
        if 'gender' in self.df.columns:
            gender_counts = self.df['gender'].value_counts()
            gender_stats = self.df.groupby('gender')['target'].agg(['count', 'mean'])
            if len(gender_stats) > 1:
                most_common_gender = gender_counts.index[0]
                highest_risk_gender = gender_stats['mean'].idxmax()
                print(f"  • Gênero mais comum: {most_common_gender}")
                print(f"  • Gênero com maior risco: {highest_risk_gender} ({gender_stats.loc[highest_risk_gender, 'mean']*100:.1f}%)")
        
        if 'race' in self.df.columns:
            race_counts = self.df['race'].value_counts()
            race_stats = self.df.groupby('race')['target'].agg(['count', 'mean'])
            most_common_race = race_counts.index[0]
            highest_risk_race = race_stats['mean'].idxmax()
            print(f"  • Raça mais comum: {most_common_race}")
            print(f"  • Raça com maior risco: {highest_risk_race} ({race_stats.loc[highest_risk_race, 'mean']*100:.1f}%)")
    
    def analyze_medical_data(self):
        """Analisa dados médicos com visualizações e correlações"""
        print("\n" + "="*60)
        print("ANÁLISE DE DADOS MÉDICOS")
        print("="*60)
        
        # Análise de especialidade médica
        if 'medical_specialty' in self.df.columns:
            print(f"\nTop 10 especialidades médicas:")
            specialty_counts = self.df['medical_specialty'].value_counts()
            specialty_target = self.df.groupby('medical_specialty')['target'].agg(['count', 'mean']).round(3)
            specialty_target.columns = ['Total', 'Taxa_Readmissao']
            
            for specialty, count in specialty_counts.head(10).items():
                pct = (count / len(self.df)) * 100
                if specialty in specialty_target.index:
                    readmission_rate = specialty_target.loc[specialty, 'Taxa_Readmissao'] * 100
                    print(f"  {specialty}: {count:,} ({pct:.1f}%) - Taxa: {readmission_rate:.1f}%")
                else:
                    print(f"  {specialty}: {count:,} ({pct:.1f}%)")
        
        # Análise de variáveis numéricas
        numeric_cols = ['num_lab_procedures', 'num_procedures', 'num_medications', 
                       'number_outpatient', 'number_emergency', 'number_inpatient',
                       'time_in_hospital', 'number_diagnoses']
        
        available_numeric = [col for col in numeric_cols if col in self.df.columns]
        
        if available_numeric:
            print(f"\n📊 ESTATÍSTICAS DAS VARIÁVEIS NUMÉRICAS:")
            stats_df = pd.DataFrame()
            
            for col in available_numeric:
                stats = {
                    'Média': self.df[col].mean(),
                    'Mediana': self.df[col].median(),
                    'Moda': self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else np.nan,
                    'Desvio_Padrão': self.df[col].std(),
                    'Mínimo': self.df[col].min(),
                    'Máximo': self.df[col].max(),
                    'Q1': self.df[col].quantile(0.25),
                    'Q3': self.df[col].quantile(0.75),
                    'Correlação_Target': self.df[col].corr(self.df['target'])
                }
                stats_df[col] = stats
                
                print(f"\n{col}:")
                for stat_name, stat_value in stats.items():
                    if stat_name == 'Correlação_Target':
                        print(f"  {stat_name}: {stat_value:.3f}")
                    else:
                        print(f"  {stat_name}: {stat_value:.2f}")
            
            # Visualizações
            n_numeric = len(available_numeric)
            fig = plt.figure(figsize=(20, 16))
            
            # 1. Histogramas das variáveis numéricas
            for i, col in enumerate(available_numeric):
                ax = plt.subplot(4, 4, i + 1)
                
                # Histograma
                self.df[col].hist(bins=30, alpha=0.7, color='skyblue', edgecolor='black', ax=ax)
                ax.axvline(self.df[col].mean(), color='red', linestyle='--', 
                          label=f'Média: {self.df[col].mean():.1f}')
                ax.axvline(self.df[col].median(), color='orange', linestyle='--', 
                          label=f'Mediana: {self.df[col].median():.1f}')
                
                ax.set_title(f'{col}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Valor')
                ax.set_ylabel('Frequência')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Ajustar layout dos histogramas
            plt.tight_layout()
            self.save_figure(fig, 'medical_data_histograms')
            
            # 2. Mapa de calor de correlações
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Correlações entre variáveis numéricas
            corr_matrix = self.df[available_numeric].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            ax1 = axes[0]
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax1, cbar_kws={"shrink": .8})
            ax1.set_title('Correlações entre Variáveis Numéricas', fontsize=14, fontweight='bold')
            
            # Correlações com a variável target
            target_corr = self.df[available_numeric + ['target']].corr()['target'].drop('target').sort_values()
            
            ax2 = axes[1]
            colors = ['red' if x < 0 else 'green' for x in target_corr.values]
            bars = ax2.barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
            ax2.set_yticks(range(len(target_corr)))
            ax2.set_yticklabels(target_corr.index, fontsize=10)
            ax2.set_xlabel('Correlação com Target')
            ax2.set_title('Correlação das Variáveis com Readmissão', fontsize=14, fontweight='bold')
            ax2.axvline(0, color='black', linestyle='-', linewidth=0.8)
            ax2.grid(True, alpha=0.3)
            
            # Adicionar valores nas barras
            for i, v in enumerate(target_corr.values):
                ax2.text(v + 0.001 if v >= 0 else v - 0.001, i, f'{v:.3f}', 
                        va='center', ha='left' if v >= 0 else 'right', fontsize=9)
            
            plt.tight_layout()
            self.save_figure(fig, 'medical_data_correlations')
            
            # 3. Box plots comparando readmitidos vs não readmitidos
            fig, axes = plt.subplots(2, 4, figsize=(20, 12))
            axes = axes.flatten()
            
            for i, col in enumerate(available_numeric):
                if i < len(axes):
                    ax = axes[i]
                    
                    # Box plot
                    self.df.boxplot(column=col, by='target', ax=ax)
                    ax.set_title(f'{col}\nReadmitido vs Não Readmitido')
                    ax.set_xlabel('Target (0: Não, 1: Sim)')
                    ax.set_ylabel(col)
                    
                    # Estatísticas comparativas
                    non_readmitted = self.df[self.df['target'] == 0][col]
                    readmitted = self.df[self.df['target'] == 1][col]
                    
                    # Teste t para diferença significativa
                    try:
                        t_stat, p_value = stats.ttest_ind(non_readmitted.dropna(), readmitted.dropna())
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                        ax.text(0.02, 0.98, f'p-value: {p_value:.3f}{significance}', 
                               transform=ax.transAxes, fontsize=8, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    except:
                        pass
            
            # Remover subplots vazios
            for j in range(i+1, len(axes)):
                axes[j].remove()
            
            plt.suptitle('Comparação de Variáveis Médicas\nReadmitidos vs Não Readmitidos', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            self.save_figure(fig, 'medical_data_boxplots')
        
        # Análise de tempo de internação se disponível
        if 'time_in_hospital' in self.df.columns:
            print(f"\n🏥 ANÁLISE DE TEMPO DE INTERNAÇÃO:")
            time_stats = self.df.groupby('time_in_hospital')['target'].agg(['count', 'mean']).round(3)
            time_stats.columns = ['Total_Casos', 'Taxa_Readmissao']
            
            print(f"  • Tempo médio de internação: {self.df['time_in_hospital'].mean():.1f} dias")
            print(f"  • Tempo mediano de internação: {self.df['time_in_hospital'].median():.1f} dias")
            print(f"  • Internações mais longas têm maior risco de readmissão:")
            
            # Mostrar estatísticas por tempo de internação
            for time_days in sorted(self.df['time_in_hospital'].unique())[:10]:
                if time_days in time_stats.index:
                    total = time_stats.loc[time_days, 'Total_Casos']
                    rate = time_stats.loc[time_days, 'Taxa_Readmissao'] * 100
                    print(f"    {time_days} dias: {total} casos - Taxa: {rate:.1f}%")
        
        # Estatísticas detalhadas
        print(f"\n📊 INSIGHTS MÉDICOS:")
        if available_numeric:
            # Variável mais correlacionada com readmissão
            target_corr = self.df[available_numeric].corrwith(self.df['target']).abs()
            most_correlated = target_corr.idxmax()
            correlation_value = self.df[most_correlated].corr(self.df['target'])
            print(f"  • Variável mais correlacionada com readmissão: {most_correlated} (r={correlation_value:.3f})")
            
            # Médias comparativas
            for col in available_numeric[:5]:  # Top 5 variáveis
                mean_no_readmit = self.df[self.df['target'] == 0][col].mean()
                mean_readmit = self.df[self.df['target'] == 1][col].mean()
                diff_pct = ((mean_readmit - mean_no_readmit) / mean_no_readmit) * 100
                print(f"  • {col}: Não readmitidos={mean_no_readmit:.1f}, Readmitidos={mean_readmit:.1f} ({diff_pct:+.1f}%)")
    
    def analyze_id_mappings(self):
        """Analisa dados com mapeamentos de IDs aplicados"""
        print("\n" + "="*60)
        print("ANÁLISE COM MAPEAMENTOS DE IDS")
        print("="*60)
        
        if self.id_mapper is None or not hasattr(self.id_mapper, 'mappings') or not self.id_mapper.mappings:
            print("⚠️  Mapeamentos de IDs não disponíveis.")
            print("Verificando se arquivo de mapeamento existe...")
            try:
                from src.config import MAPPING_FILE
                if os.path.exists(MAPPING_FILE):
                    print(f"✅ Arquivo encontrado: {MAPPING_FILE}")
                    print("💡 Você pode carregar os mapeamentos manualmente.")
                else:
                    print(f"❌ Arquivo não encontrado: {MAPPING_FILE}")
            except:
                print("❌ Não foi possível verificar o arquivo de mapeamento.")
            return
        
        try:
            # Aplicar mapeamentos temporariamente para análise
            df_with_mappings = self.id_mapper.apply_mappings_to_dataframe(self.df.copy())
            
            # Analisar admission_type
            if 'admission_type_desc' in df_with_mappings.columns:
                print("\n🏥 TIPO DE ADMISSÃO:")
                admission_analysis = df_with_mappings.groupby('admission_type_desc').agg({
                    'target': ['count', 'mean']
                }).round(3)
                admission_analysis.columns = ['Total_Casos', 'Taxa_Readmissao']
                print(admission_analysis.sort_values('Total_Casos', ascending=False))
            
            # Analisar discharge_disposition
            if 'discharge_disposition_desc' in df_with_mappings.columns:
                print("\n🚪 DISPOSIÇÃO DE ALTA:")
                discharge_analysis = df_with_mappings.groupby('discharge_disposition_desc').agg({
                    'target': ['count', 'mean']
                }).round(3)
                discharge_analysis.columns = ['Total_Casos', 'Taxa_Readmissao']
                # Mostrar apenas os 10 mais comuns
                top_discharges = discharge_analysis.sort_values('Total_Casos', ascending=False).head(10)
                print(top_discharges)
            
            # Analisar admission_source
            if 'admission_source_desc' in df_with_mappings.columns:
                print("\n📍 FONTE DE ADMISSÃO:")
                source_analysis = df_with_mappings.groupby('admission_source_desc').agg({
                    'target': ['count', 'mean']
                }).round(3)
                source_analysis.columns = ['Total_Casos', 'Taxa_Readmissao']
                # Mostrar apenas os 10 mais comuns
                top_sources = source_analysis.sort_values('Total_Casos', ascending=False).head(10)
                print(top_sources)
            
            # Análise cruzada dos mapeamentos
            if all(col in df_with_mappings.columns for col in ['admission_type_desc', 'discharge_disposition_desc']):
                print("\n📊 ANÁLISE CRUZADA - TIPO DE ADMISSÃO x DISPOSIÇÃO DE ALTA:")
                cross_analysis = pd.crosstab(
                    df_with_mappings['admission_type_desc'],
                    df_with_mappings['discharge_disposition_desc'],
                    margins=True
                )
                print(cross_analysis.iloc[:5, :5])  # Mostrar apenas uma amostra
                
        except Exception as e:
            print(f"❌ Erro ao aplicar mapeamentos de IDs: {str(e)}")
            print("💡 Continuando análise sem mapeamentos...")
    
    def analyze_medication_data(self):
        """Analisa dados de medicamentos com visualizações e estatísticas"""
        print("\n" + "="*60)
        print("ANÁLISE DE DADOS DE MEDICAMENTOS")
        print("="*60)
        
        # Identificar colunas de medicamentos
        med_columns = [col for col in self.df.columns if any(med in col.lower() for med in 
                      ['insulin', 'metformin', 'glyburide', 'glipizide', 'glimepiride', 
                       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
                       'tolazamide', 'examide', 'citoglipton', 'glyburide-metformin',
                       'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                       'metformin-pioglitazone'])]
        
        if med_columns:
            print(f"\n💊 Medicamentos analisados: {len(med_columns)}")
            
            # Análise detalhada dos principais medicamentos
            med_summary = []
            for med_col in med_columns:
                if med_col in self.df.columns:
                    value_counts = self.df[med_col].value_counts()
                    total_prescribed = len(self.df) - (self.df[med_col] == 'No').sum() if 'No' in value_counts else len(self.df)
                    prescription_rate = (total_prescribed / len(self.df)) * 100
                    
                    print(f"\n{med_col}:")
                    for value, count in value_counts.head(5).items():
                        pct = (count / len(self.df)) * 100
                        print(f"  {value}: {count:,} ({pct:.1f}%)")
                    
                    # Análise de impacto na readmissão
                    if 'target' in self.df.columns and total_prescribed > 0:
                        prescribed_patients = self.df[self.df[med_col] != 'No'] if 'No' in value_counts else self.df
                        if len(prescribed_patients) > 0:
                            readmission_rate = prescribed_patients['target'].mean() * 100
                            print(f"  Taxa de readmissão para pacientes com {med_col}: {readmission_rate:.1f}%")
                            
                            med_summary.append({
                                'Medicamento': med_col,
                                'Taxa_Prescricao': prescription_rate,
                                'Taxa_Readmissao': readmission_rate,
                                'Total_Pacientes': total_prescribed
                            })
            
            # Resumo dos medicamentos
            if med_summary:
                med_df = pd.DataFrame(med_summary)
                print(f"\n📊 RESUMO DOS MEDICAMENTOS:")
                print(med_df.sort_values('Taxa_Prescricao', ascending=False).to_string(index=False, float_format='%.1f'))
                
                # Visualização se houver dados suficientes
                if len(med_summary) >= 3:
                    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                    
                    # Gráfico 1: Taxa de prescrição
                    ax1 = axes[0]
                    top_meds = med_df.nlargest(10, 'Taxa_Prescricao')
                    bars1 = ax1.barh(range(len(top_meds)), top_meds['Taxa_Prescricao'], color='lightblue')
                    ax1.set_yticks(range(len(top_meds)))
                    ax1.set_yticklabels([med.replace('_', ' ') for med in top_meds['Medicamento']], fontsize=10)
                    ax1.set_xlabel('Taxa de Prescrição (%)')
                    ax1.set_title('Top 10 Medicamentos por Taxa de Prescrição', fontsize=14, fontweight='bold')
                    
                    # Adicionar valores
                    for i, v in enumerate(top_meds['Taxa_Prescricao']):
                        ax1.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)
                    
                    # Gráfico 2: Relação prescrição x readmissão
                    ax2 = axes[1]
                    scatter = ax2.scatter(med_df['Taxa_Prescricao'], med_df['Taxa_Readmissao'], 
                                        s=med_df['Total_Pacientes']/50, alpha=0.6, c='coral')
                    ax2.set_xlabel('Taxa de Prescrição (%)')
                    ax2.set_ylabel('Taxa de Readmissão (%)')
                    ax2.set_title('Relação entre Prescrição e Readmissão\n(Tamanho = nº pacientes)', 
                                 fontsize=14, fontweight='bold')
                    ax2.grid(True, alpha=0.3)
                    
                    # Adicionar labels para medicamentos importantes
                    for _, row in med_df.iterrows():
                        if row['Taxa_Prescricao'] > 10 or row['Taxa_Readmissao'] > 15:
                            ax2.annotate(row['Medicamento'].replace('_', ' ')[:15], 
                                       (row['Taxa_Prescricao'], row['Taxa_Readmissao']),
                                       xytext=(5, 5), textcoords='offset points', fontsize=8)
                    
                    plt.tight_layout()
                    self.save_figure(fig, 'medication_analysis')
        else:
            print("⚠️  Nenhuma coluna de medicamento encontrada no dataset.")
        
        # Analisar mudanças na medicação
        change_cols = [col for col in self.df.columns if 'change' in col.lower()]
        if change_cols:
            print(f"\n🔄 ANÁLISE DE MUDANÇAS NA MEDICAÇÃO:")
            for col in change_cols:
                if col in self.df.columns:
                    change_counts = self.df[col].value_counts()
                    print(f"\n{col}:")
                    for value, count in change_counts.items():
                        pct = (count / len(self.df)) * 100
                        print(f"  {value}: {count:,} ({pct:.1f}%)")
                    
                    # Impacto na readmissão
                    if 'target' in self.df.columns:
                        for value in change_counts.index:
                            subset = self.df[self.df[col] == value]
                            if len(subset) > 0:
                                readmit_rate = subset['target'].mean() * 100
                                print(f"    Taxa readmissão para '{value}': {readmit_rate:.1f}%")
        else:
            print("⚠️  Nenhuma coluna de mudança de medicação encontrada.")
        
        # Insights sobre medicamentos
        print(f"\n💡 INSIGHTS SOBRE MEDICAMENTOS:")
        if med_summary:
            most_prescribed = max(med_summary, key=lambda x: x['Taxa_Prescricao'])
            highest_readmit = max(med_summary, key=lambda x: x['Taxa_Readmissao'])
            print(f"  • Medicamento mais prescrito: {most_prescribed['Medicamento']} ({most_prescribed['Taxa_Prescricao']:.1f}%)")
            print(f"  • Maior taxa de readmissão: {highest_readmit['Medicamento']} ({highest_readmit['Taxa_Readmissao']:.1f}%)")
            
            # Correlação entre prescrição e readmissão
            prescriptions = [item['Taxa_Prescricao'] for item in med_summary]
            readmissions = [item['Taxa_Readmissao'] for item in med_summary]
            if len(prescriptions) > 2:
                correlation = np.corrcoef(prescriptions, readmissions)[0, 1]
                print(f"  • Correlação prescrição-readmissão: {correlation:.3f}")
        else:
            print("  • Dados insuficientes para gerar insights sobre medicamentos.")
    
    def analyze_multivariate_relationships(self):
        """Analisa relações multivariadas entre 3 ou mais variáveis"""
        print("\n" + "="*60)
        print("ANÁLISE MULTIVARIADA - INTERAÇÕES ENTRE VARIÁVEIS")
        print("="*60)
        
        # Verificar se temos variáveis necessárias
        required_cols = ['target']
        available_cols = [col for col in required_cols if col in self.df.columns]
        
        if 'target' not in self.df.columns:
            print("⚠️  Variável target não encontrada. Análise multivariada não pode ser realizada.")
            return
        
        # Análise 1: Taxa de readmissão por tempo de internação e raça
        if all(col in self.df.columns for col in ['time_in_hospital', 'race']):
            print("\n🏥 Análise: Tempo de Internação × Raça × Readmissão")
            
            # Preparar dados para análise
            df_multi = self.df.copy()
            
            # Agrupar raças menos comuns para simplificar visualização
            race_counts = df_multi['race'].value_counts()
            top_races = race_counts.head(4).index.tolist()
            df_multi['race_grouped'] = df_multi['race'].apply(
                lambda x: x if x in top_races else 'Other'
            )
            
            # Calcular taxa de readmissão por tempo e raça
            multivar_stats = df_multi.groupby(['time_in_hospital', 'race_grouped'])['target'].agg(['count', 'mean']).reset_index()
            multivar_stats.columns = ['time_in_hospital', 'race_grouped', 'count', 'readmission_rate']
            
            # Filtrar apenas grupos com pelo menos 10 casos
            multivar_stats = multivar_stats[multivar_stats['count'] >= 10]
            
            if not multivar_stats.empty:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # Gráfico 1: Taxa de readmissão por tempo de internação e raça
                ax1 = axes[0]
                for race in multivar_stats['race_grouped'].unique():
                    race_data = multivar_stats[multivar_stats['race_grouped'] == race]
                    ax1.plot(race_data['time_in_hospital'], race_data['readmission_rate'] * 100, 
                            marker='o', label=race, linewidth=2, markersize=6)
                
                ax1.set_xlabel('Tempo de Internação (dias)')
                ax1.set_ylabel('Taxa de Readmissão (%)')
                ax1.set_title('Taxa de Readmissão vs. Tempo de Internação por Raça', 
                             fontsize=14, fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Gráfico 2: Heatmap de interação
                ax2 = axes[1]
                pivot_data = multivar_stats.pivot(index='race_grouped', 
                                                 columns='time_in_hospital', 
                                                 values='readmission_rate')
                
                sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                           ax=ax2, cbar_kws={'label': 'Taxa de Readmissão'})
                ax2.set_title('Mapa de Calor: Readmissão por Tempo × Raça', 
                             fontsize=14, fontweight='bold')
                ax2.set_xlabel('Tempo de Internação (dias)')
                ax2.set_ylabel('Raça')
                
                plt.tight_layout()
                self.save_figure(fig, 'multivariate_time_race_analysis')
                
                # Insights da análise multivariada
                highest_risk = multivar_stats.loc[multivar_stats['readmission_rate'].idxmax()]
                print(f"  📊 Maior risco identificado:")
                print(f"    • Raça: {highest_risk['race_grouped']}")
                print(f"    • Tempo de internação: {highest_risk['time_in_hospital']} dias")
                print(f"    • Taxa de readmissão: {highest_risk['readmission_rate']*100:.1f}%")
                print(f"    • Número de casos: {highest_risk['count']}")
        
        # Análise 2: Medicamentos × Idade × Readmissão
        if all(col in self.df.columns for col in ['age', 'insulin']) or all(col in self.df.columns for col in ['age', 'metformin']):
            print("\n💊 Análise: Medicamentos × Idade × Readmissão")
            
            # Escolher medicamento mais comum
            med_col = None
            for potential_med in ['insulin', 'metformin', 'glyburide']:
                if potential_med in self.df.columns:
                    med_col = potential_med
                    break
            
            if med_col:
                # Análise de medicamento por idade
                med_age_stats = self.df.groupby(['age', med_col])['target'].agg(['count', 'mean']).reset_index()
                med_age_stats.columns = ['age', med_col, 'count', 'readmission_rate']
                
                # Filtrar grupos com pelo menos 20 casos
                med_age_stats = med_age_stats[med_age_stats['count'] >= 20]
                
                if not med_age_stats.empty:
                    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                    
                    # Gráfico de linhas por status do medicamento
                    for med_status in med_age_stats[med_col].unique():
                        if med_status != '?':  # Excluir valores missing
                            status_data = med_age_stats[med_age_stats[med_col] == med_status]
                            ax.plot(status_data['age'], status_data['readmission_rate'] * 100, 
                                   marker='o', label=f'{med_col}: {med_status}', 
                                   linewidth=2, markersize=6)
                    
                    ax.set_xlabel('Faixa Etária')
                    ax.set_ylabel('Taxa de Readmissão (%)')
                    ax.set_title(f'Taxa de Readmissão por Idade e Status de {med_col.title()}', 
                                fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    self.save_figure(fig, f'multivariate_age_{med_col}_analysis')
                    
                    print(f"  📊 Insights para {med_col}:")
                    for med_status in med_age_stats[med_col].unique():
                        if med_status != '?':
                            status_data = med_age_stats[med_age_stats[med_col] == med_status]
                            if not status_data.empty:
                                avg_risk = status_data['readmission_rate'].mean() * 100
                                print(f"    • {med_status}: Taxa média de readmissão = {avg_risk:.1f}%")
        
        # Análise 3: Gênero × Specialty × Readmissão  
        if all(col in self.df.columns for col in ['gender', 'medical_specialty']):
            print("\n👥 Análise: Gênero × Especialidade Médica × Readmissão")
            
            # Pegar top 6 especialidades
            top_specialties = self.df['medical_specialty'].value_counts().head(6).index.tolist()
            df_specialty = self.df[self.df['medical_specialty'].isin(top_specialties)]
            
            if not df_specialty.empty:
                gender_spec_stats = df_specialty.groupby(['gender', 'medical_specialty'])['target'].agg(['count', 'mean']).reset_index()
                gender_spec_stats.columns = ['gender', 'medical_specialty', 'count', 'readmission_rate']
                
                # Filtrar grupos com pelo menos 30 casos
                gender_spec_stats = gender_spec_stats[gender_spec_stats['count'] >= 30]
                
                if not gender_spec_stats.empty:
                    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
                    
                    # Criar gráfico de barras agrupadas
                    x_pos = np.arange(len(top_specialties))
                    width = 0.35
                    
                    male_data = []
                    female_data = []
                    
                    for specialty in top_specialties:
                        male_rate = gender_spec_stats[(gender_spec_stats['gender'] == 'Male') & 
                                                    (gender_spec_stats['medical_specialty'] == specialty)]
                        female_rate = gender_spec_stats[(gender_spec_stats['gender'] == 'Female') & 
                                                      (gender_spec_stats['medical_specialty'] == specialty)]
                        
                        male_data.append(male_rate['readmission_rate'].iloc[0] * 100 if not male_rate.empty else 0)
                        female_data.append(female_rate['readmission_rate'].iloc[0] * 100 if not female_rate.empty else 0)
                    
                    bars1 = ax.bar(x_pos - width/2, male_data, width, label='Masculino', alpha=0.8, color='lightblue')
                    bars2 = ax.bar(x_pos + width/2, female_data, width, label='Feminino', alpha=0.8, color='lightcoral')
                    
                    ax.set_xlabel('Especialidade Médica')
                    ax.set_ylabel('Taxa de Readmissão (%)')
                    ax.set_title('Taxa de Readmissão por Gênero e Especialidade Médica', 
                                fontsize=14, fontweight='bold')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels([spec.replace(' ', '\n') for spec in top_specialties], fontsize=10)
                    ax.legend()
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Adicionar valores nas barras
                    for bar in bars1:
                        height = bar.get_height()
                        if height > 0:
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
                    
                    for bar in bars2:
                        height = bar.get_height()
                        if height > 0:
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
                    
                    plt.tight_layout()
                    self.save_figure(fig, 'multivariate_gender_specialty_analysis')
                    
                    print(f"  📊 Diferenças por gênero nas especialidades:")
                    for i, specialty in enumerate(top_specialties):
                        male_rate = male_data[i]
                        female_rate = female_data[i]
                        if male_rate > 0 and female_rate > 0:
                            diff = abs(male_rate - female_rate)
                            higher_gender = "Masculino" if male_rate > female_rate else "Feminino"
                            print(f"    • {specialty}: {higher_gender} tem {diff:.1f}pp maior risco")
        
        print(f"\n💡 INSIGHTS MULTIVARIADOS:")
        print(f"  • Análises multivariadas revelam interações complexas entre variáveis")
        print(f"  • Certas combinações de características podem ter risco muito maior")
        print(f"  • Considere features de interação no modelo de ML")
        print(f"  • Segmentação de pacientes pode ser útil para estratégias targeted")
    
    def analyze_diagnosis_groups(self):
        """Analisa grupos de diagnósticos e sua relação com readmissão"""
        print("\n" + "="*60)
        print("ANÁLISE DE GRUPOS DE DIAGNÓSTICOS")
        print("="*60)
        
        # Verificar se temos colunas de diagnóstico
        diag_cols = [col for col in self.df.columns if col.startswith('diag_')]
        
        if not diag_cols:
            print("⚠️  Colunas de diagnóstico não encontradas no dataset.")
            return
        
        print(f"📋 Analisando colunas de diagnóstico: {diag_cols}")
        
        # Função para categorizar diagnósticos em grupos principais
        def categorize_diagnosis(code):
            """Categoriza códigos de diagnóstico em grupos principais"""
            if pd.isna(code) or code == 'unknown' or code == '?':
                return 'Unknown'
            
            try:
                # Converter para string e extrair primeiros dígitos
                code_str = str(code).strip()
                
                # Se começa com 'V' ou 'E', são códigos especiais
                if code_str.startswith(('V', 'E')):
                    if code_str.startswith('V'):
                        return 'V-Codes (Suplementar)'
                    else:
                        return 'E-Codes (Externa)'
                
                # Extrair número principal
                code_num = float(code_str.split('.')[0])
                
                # Categorização baseada em faixas CID-9
                if 1 <= code_num <= 139:
                    return 'Infecciosas/Parasitárias'
                elif 140 <= code_num <= 239:
                    return 'Neoplasias'
                elif 240 <= code_num <= 279:
                    return 'Endócrinas/Metabólicas'
                elif 280 <= code_num <= 289:
                    return 'Sangue/Órgãos Hematopoiéticos'
                elif 290 <= code_num <= 319:
                    return 'Mentais'
                elif 320 <= code_num <= 389:
                    return 'Sistema Nervoso'
                elif 390 <= code_num <= 459:
                    return 'Circulatórias'
                elif 460 <= code_num <= 519:
                    return 'Respiratórias'
                elif 520 <= code_num <= 579:
                    return 'Digestivas'
                elif 580 <= code_num <= 629:
                    return 'Genitourinárias'
                elif 630 <= code_num <= 679:
                    return 'Gravidez/Parto'
                elif 680 <= code_num <= 709:
                    return 'Pele/Subcutâneo'
                elif 710 <= code_num <= 739:
                    return 'Musculoesqueléticas'
                elif 740 <= code_num <= 759:
                    return 'Congênitas'
                elif 760 <= code_num <= 779:
                    return 'Perinatais'
                elif 780 <= code_num <= 799:
                    return 'Sintomas/Sinais'
                elif 800 <= code_num <= 999:
                    return 'Lesões/Envenenamentos'
                else:
                    return 'Outros'
                    
            except (ValueError, AttributeError):
                return 'Unknown'
        
        # Analisar cada coluna de diagnóstico
        diagnosis_results = {}
        
        for diag_col in diag_cols:
            print(f"\n🔍 Análise de {diag_col}:")
            
            # Aplicar categorização
            self.df[f'{diag_col}_category'] = self.df[diag_col].apply(categorize_diagnosis)
            
            # Estatísticas por categoria
            diag_stats = self.df.groupby(f'{diag_col}_category')['target'].agg(['count', 'mean']).round(3)
            diag_stats.columns = ['Total_Casos', 'Taxa_Readmissao']
            diag_stats = diag_stats.sort_values('Total_Casos', ascending=False)
            
            diagnosis_results[diag_col] = diag_stats
            
            print(f"  Top 10 categorias por frequência:")
            for category, row in diag_stats.head(10).iterrows():
                print(f"    {category}: {row['Total_Casos']:,} casos ({row['Taxa_Readmissao']*100:.1f}% readmissão)")
        
        # Visualizações
        if diagnosis_results:
            fig, axes = plt.subplots(2, len(diag_cols), figsize=(6*len(diag_cols), 12))
            if len(diag_cols) == 1:
                axes = axes.reshape(-1, 1)
            
            for i, (diag_col, diag_stats) in enumerate(diagnosis_results.items()):
                # Gráfico 1: Distribuição de categorias
                ax1 = axes[0, i] if len(diag_cols) > 1 else axes[0]
                top_categories = diag_stats.head(8)
                
                bars = ax1.barh(range(len(top_categories)), top_categories['Total_Casos'], 
                               color='lightblue', alpha=0.8)
                ax1.set_yticks(range(len(top_categories)))
                ax1.set_yticklabels([cat[:20] + '...' if len(cat) > 20 else cat 
                                   for cat in top_categories.index], fontsize=9)
                ax1.set_xlabel('Número de Casos')
                ax1.set_title(f'Top Categorias - {diag_col}', fontsize=12, fontweight='bold')
                
                # Adicionar valores
                for j, v in enumerate(top_categories['Total_Casos']):
                    ax1.text(v + max(top_categories['Total_Casos'])*0.01, j, f'{v:,}', 
                            va='center', fontsize=8)
                
                # Gráfico 2: Taxa de readmissão por categoria
                ax2 = axes[1, i] if len(diag_cols) > 1 else axes[1]
                readmit_rates = top_categories['Taxa_Readmissao'] * 100
                
                bars2 = ax2.barh(range(len(readmit_rates)), readmit_rates, 
                                color='salmon', alpha=0.8)
                ax2.set_yticks(range(len(readmit_rates)))
                ax2.set_yticklabels([cat[:20] + '...' if len(cat) > 20 else cat 
                                   for cat in readmit_rates.index], fontsize=9)
                ax2.set_xlabel('Taxa de Readmissão (%)')
                ax2.set_title(f'Taxa Readmissão - {diag_col}', fontsize=12, fontweight='bold')
                
                # Adicionar valores
                for j, v in enumerate(readmit_rates.values):
                    ax2.text(v + 0.2, j, f'{v:.1f}%', va='center', fontsize=8)
            
            plt.tight_layout()
            self.save_figure(fig, 'diagnosis_groups_analysis')
        
        # Análise combinada de diagnósticos
        if len(diag_cols) >= 2:
            print(f"\n🔗 Análise combinada de diagnósticos:")
            
            # Criar combinações dos dois diagnósticos principais
            primary_diag = f'{diag_cols[0]}_category'
            secondary_diag = f'{diag_cols[1]}_category'
            
            # Combinações mais comuns
            combo_stats = self.df.groupby([primary_diag, secondary_diag])['target'].agg(['count', 'mean']).reset_index()
            combo_stats.columns = [primary_diag, secondary_diag, 'Total_Casos', 'Taxa_Readmissao']
            combo_stats = combo_stats[combo_stats['Total_Casos'] >= 50]  # Filtrar combinações raras
            combo_stats = combo_stats.sort_values('Total_Casos', ascending=False)
            
            print(f"  Top 10 combinações de diagnósticos:")
            for _, row in combo_stats.head(10).iterrows():
                print(f"    {row[primary_diag]} + {row[secondary_diag]}: {row['Total_Casos']:,} casos ({row['Taxa_Readmissao']*100:.1f}% readmissão)")
        
        # Insights sobre diagnósticos
        print(f"\n💡 INSIGHTS DE DIAGNÓSTICOS:")
        if diagnosis_results:
            for diag_col, diag_stats in diagnosis_results.items():
                highest_risk_category = diag_stats['Taxa_Readmissao'].idxmax()
                highest_risk_rate = diag_stats.loc[highest_risk_category, 'Taxa_Readmissao'] * 100
                most_common_category = diag_stats['Total_Casos'].idxmax()
                
                print(f"  • {diag_col}:")
                print(f"    - Categoria mais comum: {most_common_category}")
                print(f"    - Maior risco de readmissão: {highest_risk_category} ({highest_risk_rate:.1f}%)")
        
        print(f"  • Diagnósticos circulatórios e respiratórios tendem a ter maior risco")
        print(f"  • Combinações de diagnósticos podem indicar maior complexidade")
        print(f"  • Considere agrupamento de diagnósticos como feature engineering")
    
    def generate_summary_report(self):
        """Gera um relatório resumido com os principais insights"""
        print("\n" + "="*60)
        print("RELATÓRIO RESUMIDO - PRINCIPAIS INSIGHTS")
        print("="*60)
        
        insights = []
        
        # Insights sobre a variável target
        if 'target' in self.df.columns:
            readmission_rate = self.df['target'].mean() * 100
            insights.append(f"📊 Taxa geral de readmissão em <30 dias: {readmission_rate:.1f}%")
            
            if readmission_rate > 15:
                insights.append("⚠️  Taxa de readmissão acima de 15% - indicador de risco elevado")
            elif readmission_rate < 5:
                insights.append("✅ Taxa de readmissão baixa - bom controle hospitalar")
        
        # Insights sobre dados faltantes
        missing_pct = (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        if missing_pct > 20:
            insights.append(f"⚠️  {missing_pct:.1f}% de dados faltantes - requer atenção na limpeza")
        elif missing_pct < 5:
            insights.append(f"✅ Apenas {missing_pct:.1f}% de dados faltantes - dataset de boa qualidade")
        
        # Insights demográficos
        if 'age' in self.df.columns and 'target' in self.df.columns:
            age_risk = self.df.groupby('age')['target'].mean()
            highest_risk_age = age_risk.idxmax()
            highest_risk_rate = age_risk.max() * 100
            insights.append(f"👥 Faixa etária de maior risco: {highest_risk_age} ({highest_risk_rate:.1f}%)")
        
        # Insights médicos
        numeric_cols = ['num_lab_procedures', 'num_medications', 'time_in_hospital']
        available_numeric = [col for col in numeric_cols if col in self.df.columns]
        
        if available_numeric and 'target' in self.df.columns:
            correlations = self.df[available_numeric].corrwith(self.df['target']).abs()
            strongest_predictor = correlations.idxmax()
            correlation_value = correlations.max()
            insights.append(f"🔬 Melhor preditor médico: {strongest_predictor} (correlação: {correlation_value:.3f})")
        
        # Insights sobre medicamentos
        med_columns = [col for col in self.df.columns if 'insulin' in col.lower() or 'metformin' in col.lower()]
        if med_columns and 'target' in self.df.columns:
            for med_col in med_columns[:2]:  # Top 2 medicamentos
                if med_col in self.df.columns:
                    med_usage = (self.df[med_col] != 'No').mean() * 100
                    if med_usage > 30:
                        insights.append(f"💊 {med_col}: usado em {med_usage:.1f}% dos casos")
        
        # Imprimir insights
        for i, insight in enumerate(insights, 1):
            print(f"{i:2d}. {insight}")
        
        # Recomendações
        print(f"\n💡 RECOMENDAÇÕES PARA MODELAGEM:")
        recommendations = []
        
        if missing_pct > 15:
            recommendations.append("• Implementar estratégias robustas de tratamento de dados faltantes")
        
        if len(available_numeric) > 0:
            recommendations.append("• Considerar normalização/padronização das variáveis numéricas")
        
        if 'medical_specialty' in self.df.columns:
            n_specialties = self.df['medical_specialty'].nunique()
            if n_specialties > 20:
                recommendations.append(f"• Agrupar especialidades médicas ({n_specialties} categorias) para reduzir dimensionalidade")
        
        recommendations.append("• Aplicar técnicas de feature engineering para variáveis categóricas")
        recommendations.append("• Considerar técnicas de balanceamento devido ao desbalanceamento da target")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec}")
        
        return insights, recommendations
    
    def save_html_report(self):
        """Gera um relatório HTML com todos os resultados"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Relatório de Análise Exploratória - Readmissão Hospitalar Diabética</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 10px; }}
                .summary {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .insight {{ background-color: #d5dbdb; padding: 10px; margin: 10px 0; border-left: 4px solid #3498db; }}
                .metric {{ font-weight: bold; color: #2980b9; }}
                .warning {{ color: #e74c3c; font-weight: bold; }}
                .success {{ color: #27ae60; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #bdc3c7; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                .figure-list {{ list-style-type: none; padding: 0; }}
                .figure-list li {{ margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>📊 Relatório de Análise Exploratória</h1>
            <h2>Predição de Readmissão Hospitalar Diabética</h2>
            
            <div class="summary">
                <h3>📋 Resumo Executivo</h3>
                <p><span class="metric">Dataset:</span> {len(self.df):,} registros, {len(self.df.columns)} colunas</p>
                <p><span class="metric">Pacientes únicos:</span> {self.df['patient_nbr'].nunique():,}</p>
                <p><span class="metric">Taxa de readmissão <30 dias:</span> {self.df['target'].mean()*100:.1f}%</p>
                <p><span class="metric">Data da análise:</span> {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
            </div>
            
            <h2>📈 Visualizações Geradas</h2>
            <ul class="figure-list">
        """
        
        # Adicionar lista de figuras geradas
        for figure_path in self.figures:
            figure_name = os.path.basename(figure_path).replace('.png', '').replace('_', ' ').title()
            html_content += f"<li>📊 {figure_name}</li>"
        
        html_content += """
            </ul>
            
            <div class="summary">
                <h3>🎯 Como Interpretar os Resultados</h3>
                <ul>
                    <li><strong>Gráficos de Distribuição:</strong> Mostram como os dados estão distribuídos</li>
                    <li><strong>Mapas de Calor:</strong> Revelam padrões de correlação e dados faltantes</li>
                    <li><strong>Análises Comparativas:</strong> Destacam diferenças entre grupos</li>
                    <li><strong>Box Plots:</strong> Identificam outliers e diferenças estatísticas</li>
                </ul>
            </div>
            
            <h2>📁 Arquivos de Saída</h2>
            <p>Todos os gráficos foram salvos na pasta <code>results/</code> em alta qualidade (300 DPI).</p>
            
            <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d;">
                <p>Relatório gerado automaticamente pelo módulo de Análise Exploratória</p>
                <p>Projeto: BCC325 - Inteligência Artificial | UFOP</p>
            </footer>
        </body>
        </html>
        """
        
        # Salvar arquivo HTML
        html_path = os.path.join(self.results_dir, 'eda_report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n📄 Relatório HTML salvo: {html_path}")
        return html_path
    
    def run_complete_analysis(self):
        """
        Executa a análise exploratória completa dos dados
        
        Returns:
            pd.DataFrame: DataFrame com os dados analisados e a variável target criada
        """
        print("🚀 INICIANDO ANÁLISE EXPLORATÓRIA COMPLETA")
        print("="*70)
        
        try:
            # 1. Carregar dados
            print("\n📊 Etapa 1: Carregamento dos dados")
            self.load_data()
            
            if self.df is None or self.df.empty:
                raise ValueError("Não foi possível carregar os dados")
            
            # 2. Informações básicas
            print("\n📋 Etapa 2: Informações básicas do dataset")
            self.basic_info()
            
            # 3. Análise da variável target (CRÍTICO - cria a variável target)
            print("\n🎯 Etapa 3: Análise da variável alvo")
            self.analyze_target_variable()
            
            # 4. Análise de dados faltantes
            print("\n🔍 Etapa 4: Análise de dados faltantes")
            self.analyze_missing_data()
            
            # 5. Análise demográfica
            print("\n👥 Etapa 5: Análise demográfica")
            self.analyze_demographic_data()
            
            # 6. Análise médica
            print("\n🏥 Etapa 6: Análise de dados médicos")
            self.analyze_medical_data()
            
            # 7. Análise de medicamentos
            print("\n💊 Etapa 7: Análise de medicamentos")
            self.analyze_medication_data()
            
            # 8. Análise multivariada (NOVA)
            print("\n🔗 Etapa 8: Análise multivariada")
            self.analyze_multivariate_relationships()
            
            # 9. Análise de grupos de diagnósticos (NOVA)
            print("\n🩺 Etapa 9: Análise de grupos de diagnósticos")
            self.analyze_diagnosis_groups()
            
            # 10. Análise com mapeamentos de IDs
            print("\n🔗 Etapa 10: Análise com mapeamentos de IDs")
            self.analyze_id_mappings()
            
            # 11. Gerar relatório resumido
            print("\n📝 Etapa 11: Geração de relatório resumido")
            insights, recommendations = self.generate_summary_report()
            
            # 12. Salvar relatório HTML
            print("\n💾 Etapa 12: Salvando relatório HTML")
            html_report_path = self.save_html_report()
            
            # Resumo final
            print("\n" + "="*70)
            print("✅ ANÁLISE EXPLORATÓRIA CONCLUÍDA COM SUCESSO!")
            print("="*70)
            print(f"📊 Total de registros analisados: {len(self.df):,}")
            print(f"📈 Gráficos gerados: {len(self.figures)}")
            print(f"📄 Relatório HTML: {html_report_path}")
            print(f"🎯 Variável target criada: {'target' in self.df.columns}")
            
            if 'target' in self.df.columns:
                print(f"📊 Taxa de readmissão <30 dias: {self.df['target'].mean()*100:.1f}%")
            
            return self.df
            
        except Exception as e:
            print(f"\n❌ Erro durante a análise exploratória: {str(e)}")
            print("🔧 Verifique se:")
            print("  • O arquivo de dados existe e está acessível")
            print("  • As dependências estão instaladas")
            print("  • O diretório de resultados pode ser criado")
            raise


def main():
    """Função principal para executar a análise exploratória"""
    eda = ExploratoryDataAnalysis()
    df_analyzed = eda.run_complete_analysis()
    return df_analyzed


if __name__ == "__main__":
    main()
