"""
Utilitário de Visualização Profissional para Análise de Dados Médicos

Este módulo implementa um sistema completo e padronizado de visualizações
para o sistema de predição de readmissão hospitalar diabética, oferecendo:

Funcionalidades de Visualização:
- Paleta de cores profissional e acessível
- Templates padronizados para diferentes tipos de gráficos
- Formatação automática de textos e labels
- Estilos consistentes em todos os gráficos
- Otimização automática de layout e espaçamento
- Suporte para gráficos médicos específicos
- Exportação em alta qualidade (300 DPI)
- Configurações responsivas para diferentes tamanhos
- Sistema de anotações e legendas inteligentes
- Prevenção automática de sobreposição de textos

Tipos de Gráficos Suportados:
- Gráficos de barras e histogramas médicos
- Matrizes de confusão profissionais
- Curvas ROC e Precision-Recall otimizadas
- Gráficos de importância de features
- Análises de correlação e heatmaps
- Distribuições e boxplots
- Gráficos de validação cruzada
- Métricas de performance

Autor: Thalles Felipe Rodrigues de Almeida Santos
Projeto: BCC325 - Inteligência Artificial - UFOP
Instituição: Universidade Federal de Ouro Preto (UFOP)
Disciplina: Inteligência Artificial
Professor: Jadson Castro Gertrudes
Data: Agosto 2025
"""

# Configurar matplotlib ANTES de qualquer import
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para evitar problemas de thread
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import warnings
from datetime import datetime
import os

# Configurações específicas para resolver problemas de thread no Windows
plt.ioff()  # Desabilitar modo interativo
matplotlib.rcParams['backend'] = 'Agg'  # Forçar backend Agg

# Suprimir warnings de matplotlib
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================================
# CONFIGURAÇÕES GLOBAIS DE VISUALIZAÇÃO PROFISSIONAL
# ============================================================================

# Paleta de cores profissional e acessível
PROFESSIONAL_COLORS = {
    'primary': '#2E4057',      # Azul escuro profissional
    'secondary': '#048A81',    # Verde-azulado médico
    'accent': '#F39C12',       # Laranja dourado para destaque
    'danger': '#E74C3C',       # Vermelho para alertas
    'success': '#27AE60',      # Verde para sucesso
    'warning': '#F39C12',      # Laranja para avisos
    'info': '#3498DB',         # Azul para informações
    'light': '#ECF0F1',        # Cinza claro para fundos
    'dark': '#2C3E50',         # Cinza escuro para textos
    'neutral': '#95A5A6'       # Cinza neutro
}

# Paleta de cores categóricas profissional
CATEGORICAL_PALETTE = [
    '#2E4057', '#048A81', '#F39C12', '#E74C3C', '#27AE60',
    '#3498DB', '#9B59B6', '#E67E22', '#1ABC9C', '#34495E',
    '#F1C40F', '#E91E63', '#00BCD4', '#FF9800', '#4CAF50'
]

# Paleta de cores sequencial médica
MEDICAL_SEQUENTIAL = ['#E8F4FD', '#BFDBF7', '#7BB8EB', '#4A90C2', '#2E4057']

# Configurações de fonte profissional
FONT_CONFIG = {
    'family': 'DejaVu Sans',
    'weight': 'normal',
    'size': {
        'small': 9,
        'medium': 11,
        'large': 13,
        'title': 16,
        'suptitle': 18
    }
}

# Configurações de figura
FIGURE_CONFIG = {
    'dpi': 300,
    'facecolor': 'white',
    'edgecolor': 'none',
    'bbox_inches': 'tight',
    'pad_inches': 0.2
}

# Configurações de grid profissional
GRID_CONFIG = {
    'alpha': 0.3,
    'linewidth': 0.5,
    'linestyle': '-',
    'color': '#BDC3C7'
}

# Configurações de layout e espaçamento
LAYOUT_CONFIG = {
    'hspace': 0.4,
    'wspace': 0.3,
    'top': 0.92,
    'bottom': 0.1,
    'left': 0.1,
    'right': 0.95
}

class ProfessionalVisualizer:
    """
    Classe principal para criação de visualizações profissionais padronizadas
    """
    
    def __init__(self, style='professional', save_dir='results'):
        """
        Inicializa o visualizador com configurações profissionais
        
        Args:
            style (str): Estilo base ('professional', 'medical', 'academic')
            save_dir (str): Diretório para salvar as figuras
        """
        self.style = style
        self.save_dir = save_dir
        self.setup_style()
        
        # Criar diretório se não existir
        os.makedirs(save_dir, exist_ok=True)
        
    def setup_style(self):
        """Configura o estilo global do matplotlib"""
        plt.style.use('default')  # Reset to default first
        
        # Configurações globais
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'none',
            
            # Fontes
            'font.family': FONT_CONFIG['family'],
            'font.size': FONT_CONFIG['size']['medium'],
            'axes.titlesize': FONT_CONFIG['size']['large'],
            'axes.labelsize': FONT_CONFIG['size']['medium'],
            'xtick.labelsize': FONT_CONFIG['size']['small'],
            'ytick.labelsize': FONT_CONFIG['size']['small'],
            'legend.fontsize': FONT_CONFIG['size']['small'],
            'figure.titlesize': FONT_CONFIG['size']['title'],
            
            # Cores
            'axes.edgecolor': PROFESSIONAL_COLORS['dark'],
            'axes.linewidth': 1.0,
            'xtick.color': PROFESSIONAL_COLORS['dark'],
            'ytick.color': PROFESSIONAL_COLORS['dark'],
            'text.color': PROFESSIONAL_COLORS['dark'],
            
            # Grid
            'axes.grid': True,
            'grid.alpha': GRID_CONFIG['alpha'],
            'grid.linewidth': GRID_CONFIG['linewidth'],
            'grid.color': GRID_CONFIG['color'],
            
            # Spines
            'axes.spines.top': False,
            'axes.spines.right': False,
            
            # DPI
            'figure.dpi': 100,  # Para exibição
            'savefig.dpi': FIGURE_CONFIG['dpi'],  # Para salvamento
        })

    def get_color_palette(self, n_colors=None):
        """
        Retorna uma paleta de cores profissionais
        
        Args:
            n_colors (int): Número de cores necessárias
            
        Returns:
            list: Lista de cores hexadecimais
        """
        if n_colors is None:
            return list(PROFESSIONAL_COLORS.values())
        
        # Cores base da paleta profissional
        base_colors = [
            PROFESSIONAL_COLORS['primary'],
            PROFESSIONAL_COLORS['secondary'], 
            PROFESSIONAL_COLORS['success'],
            PROFESSIONAL_COLORS['warning'],
            PROFESSIONAL_COLORS['accent'],
            '#28A745',  # Verde
            '#6610F2',  # Roxo
            '#E83E8C'   # Rosa
        ]
        
        # Se precisar de mais cores, repetir o ciclo
        if n_colors <= len(base_colors):
            return base_colors[:n_colors]
        else:
            # Repetir cores se necessário
            colors = base_colors * (n_colors // len(base_colors) + 1)
            return colors[:n_colors]

    def _apply_professional_style(self, ax):
        """
        Aplica estilo profissional a um eixo
        
        Args:
            ax: Eixo matplotlib
        """
        ax.grid(True, alpha=0.3, color=PROFESSIONAL_COLORS['grid'])
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(PROFESSIONAL_COLORS['text'])
        ax.spines['bottom'].set_color(PROFESSIONAL_COLORS['text'])
        ax.tick_params(colors=PROFESSIONAL_COLORS['text'])

    def create_figure(self, figsize=(12, 8), nrows=1, ncols=1, **kwargs):
        """
        Cria uma figura profissional com configurações padronizadas
        
        Args:
            figsize (tuple): Tamanho da figura
            nrows (int): Número de linhas de subplots
            ncols (int): Número de colunas de subplots
            **kwargs: Argumentos adicionais para plt.subplots
            
        Returns:
            fig, axes: Figura e eixos matplotlib
        """
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, 
                                facecolor='white', **kwargs)
        
        # Ajustar espaçamento
        if nrows > 1 or ncols > 1:
            plt.tight_layout(pad=3.0)
            
        return fig, axes

    def save_figure(self, fig, filename, title=None, subtitle=None):
        """
        Salva figura com configurações profissionais
        
        Args:
            fig: Figura matplotlib
            filename (str): Nome do arquivo (sem extensão)
            title (str): Título principal opcional
            subtitle (str): Subtítulo opcional
        """
        if title:
            fig.suptitle(title, fontsize=FONT_CONFIG['size']['suptitle'], 
                        fontweight='bold', y=0.98, color=PROFESSIONAL_COLORS['dark'])
        
        if subtitle:
            fig.text(0.5, 0.94, subtitle, ha='center', va='top', 
                    fontsize=FONT_CONFIG['size']['medium'], 
                    color=PROFESSIONAL_COLORS['neutral'])
        
        # Salvar com alta qualidade
        filepath = os.path.join(self.save_dir, f"{filename}.png")
        fig.savefig(filepath, **FIGURE_CONFIG)
        plt.close(fig)
        print(f"📊 Gráfico salvo: {filename}.png")
        return filepath

    def plot_confusion_matrix(self, cm, class_names=None, title="Matriz de Confusão", 
                            normalize=False, filename=None):
        """
        Cria matriz de confusão profissional
        
        Args:
            cm: Matriz de confusão
            class_names: Nomes das classes
            title: Título do gráfico
            normalize: Se deve normalizar os valores
            filename: Nome do arquivo para salvar
        """
        fig, ax = self.create_figure(figsize=(8, 6))
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            vmax = 1.0
        else:
            fmt = 'd'
            vmax = None
        
        # Criar heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmax=vmax)
        
        # Adicionar colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=FONT_CONFIG['size']['small'])
        
        # Configurar labels
        if class_names is None:
            class_names = [f'Classe {i}' for i in range(len(cm))]
        
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        
        # Rotacionar labels se necessário
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Adicionar valores nas células
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm[i, j]
                if normalize:
                    text = f'{value:.1%}'
                else:
                    text = f'{value:d}'
                
                ax.text(j, i, text, ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=FONT_CONFIG['size']['medium'],
                       fontweight='bold')
        
        ax.set_ylabel('Classe Real', fontweight='bold')
        ax.set_xlabel('Classe Predita', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        
        if filename:
            return self.save_figure(fig, filename)
        return fig, ax

    def plot_roc_curve(self, fpr, tpr, auc_score, title="Curva ROC", filename=None):
        """
        Cria curva ROC profissional
        
        Args:
            fpr: Taxa de falsos positivos
            tpr: Taxa de verdadeiros positivos
            auc_score: Score AUC
            title: Título do gráfico
            filename: Nome do arquivo para salvar
        """
        fig, ax = self.create_figure(figsize=(8, 8))
        
        # Plotar curva ROC
        ax.plot(fpr, tpr, color=PROFESSIONAL_COLORS['primary'], 
               linewidth=3, label=f'ROC (AUC = {auc_score:.3f})')
        
        # Linha diagonal (classificador aleatório)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, 
               label='Classificador Aleatório (AUC = 0.500)')
        
        # Configurações
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('Taxa de Falsos Positivos', fontweight='bold')
        ax.set_ylabel('Taxa de Verdadeiros Positivos', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        
        # Adicionar área sob a curva
        ax.fill_between(fpr, tpr, alpha=0.2, color=PROFESSIONAL_COLORS['primary'])
        
        # Legend
        ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
        
        # Grid profissional
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        if filename:
            return self.save_figure(fig, filename)
        return fig, ax

    def plot_feature_importance(self, feature_names, importance_values, 
                              title="Importância das Features", top_n=20, filename=None):
        """
        Cria gráfico de importância de features profissional
        
        Args:
            feature_names: Nomes das features
            importance_values: Valores de importância
            title: Título do gráfico
            top_n: Número de features a mostrar
            filename: Nome do arquivo para salvar
        """
        # Preparar dados
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=True).tail(top_n)
        
        # Criar figura
        fig, ax = self.create_figure(figsize=(12, max(6, top_n * 0.4)))
        
        # Criar cores degradê
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
        
        # Barras horizontais
        bars = ax.barh(range(len(df)), df['importance'], color=colors, 
                      edgecolor=PROFESSIONAL_COLORS['dark'], linewidth=0.5)
        
        # Configurar eixos
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['feature'], fontsize=FONT_CONFIG['size']['small'])
        ax.set_xlabel('Importância', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        
        # Adicionar valores nas barras
        for i, (bar, value) in enumerate(zip(bars, df['importance'])):
            width = bar.get_width()
            ax.text(width + max(df['importance']) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.4f}', ha='left', va='center', 
                   fontsize=FONT_CONFIG['size']['small'], fontweight='bold')
        
        # Ajustar layout
        plt.tight_layout()
        
        if filename:
            return self.save_figure(fig, filename)
        return fig, ax

    def plot_metrics_comparison(self, metrics_dict, title="Comparação de Métricas", filename=None):
        """
        Cria gráfico de comparação de métricas profissional
        
        Args:
            metrics_dict: Dicionário com métricas {modelo: {métrica: valor}}
            title: Título do gráfico
            filename: Nome do arquivo para salvar
        """
        # Preparar dados
        df = pd.DataFrame(metrics_dict).T
        metrics = df.columns.tolist()
        models = df.index.tolist()
        
        # Criar figura
        fig, ax = self.create_figure(figsize=(12, 8))
        
        # Configurar posições das barras
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        
        # Plotar barras para cada modelo
        for i, model in enumerate(models):
            values = df.loc[model].values
            bars = ax.bar(x + i * width, values, width, 
                         label=model, color=CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)],
                         edgecolor=PROFESSIONAL_COLORS['dark'], linewidth=0.5)
            
            # Adicionar valores nas barras
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom',
                       fontsize=FONT_CONFIG['size']['small'], fontweight='bold')
        
        # Configurações
        ax.set_xlabel('Métricas', fontweight='bold')
        ax.set_ylabel('Valores', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        
        # Configurar limites do eixo Y
        ax.set_ylim(0, min(1.1, max(df.max()) * 1.1))
        
        if filename:
            return self.save_figure(fig, filename)
        return fig, ax

    def plot_model_results_dashboard(self, metrics, cm, fpr, tpr, auc_score, 
                                   feature_names=None, feature_importance=None,
                                   model_name="Modelo", filename=None):
        """
        Cria dashboard completo com resultados do modelo
        
        Args:
            metrics: Dicionário com métricas
            cm: Matriz de confusão
            fpr, tpr: Dados da curva ROC
            auc_score: Score AUC
            feature_names: Nomes das features
            feature_importance: Importância das features
            model_name: Nome do modelo
            filename: Nome do arquivo para salvar
        """
        # Configurar subplot grid
        if feature_importance is not None:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        else:
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Métricas principais (canto superior esquerdo)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_metrics_bars(ax1, metrics)
        
        # 2. Matriz de confusão
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_confusion_matrix_subplot(ax2, cm)
        
        # 3. Curva ROC
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_roc_subplot(ax3, fpr, tpr, auc_score)
        
        # 4. Informações do modelo (texto)
        if feature_importance is not None:
            ax4 = fig.add_subplot(gs[0, 3])
        else:
            ax4 = fig.add_subplot(gs[1, :])
        self._plot_model_info(ax4, metrics, model_name)
        
        # 5. Feature importance (se disponível)
        if feature_importance is not None and feature_names is not None:
            ax5 = fig.add_subplot(gs[1:, :])
            self._plot_feature_importance_subplot(ax5, feature_names, feature_importance)
        
        # Título principal
        fig.suptitle(f'Dashboard de Resultados - {model_name}', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        if filename:
            return self.save_figure(fig, filename)
        return fig

    def _plot_metrics_bars(self, ax, metrics):
        """Subplot para métricas em barras"""
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax.bar(metric_names, metric_values, 
                     color=[PROFESSIONAL_COLORS['primary'], PROFESSIONAL_COLORS['secondary'],
                           PROFESSIONAL_COLORS['accent'], PROFESSIONAL_COLORS['success'],
                           PROFESSIONAL_COLORS['info']][:len(metric_names)],
                     edgecolor=PROFESSIONAL_COLORS['dark'], linewidth=1)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')
        
        ax.set_title('Métricas de Performance', fontweight='bold')
        ax.set_ylabel('Valor')
        ax.set_ylim(0, 1.1)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    def _plot_confusion_matrix_subplot(self, ax, cm):
        """Subplot para matriz de confusão"""
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        
        # Adicionar valores nas células
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i, j]}', ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=12, fontweight='bold')
        
        ax.set_title('Matriz de Confusão', fontweight='bold')
        ax.set_xlabel('Predito')
        ax.set_ylabel('Real')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Não Readmitido', 'Readmitido'])
        ax.set_yticklabels(['Não Readmitido', 'Readmitido'])

    def _plot_roc_subplot(self, ax, fpr, tpr, auc_score):
        """Subplot para curva ROC"""
        ax.plot(fpr, tpr, color=PROFESSIONAL_COLORS['primary'], 
               linewidth=3, label=f'ROC (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2)
        
        ax.fill_between(fpr, tpr, alpha=0.2, color=PROFESSIONAL_COLORS['primary'])
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Taxa de Falsos Positivos')
        ax.set_ylabel('Taxa de Verdadeiros Positivos')
        ax.set_title('Curva ROC', fontweight='bold')
        ax.legend()

    def _plot_model_info(self, ax, metrics, model_name):
        """Subplot para informações do modelo"""
        ax.axis('off')
        
        info_text = f"""
{model_name} - Resumo de Performance

Métricas Principais:
• Acurácia: {metrics.get('accuracy', 0):.1%}
• Precisão: {metrics.get('precision', 0):.1%}
• Recall: {metrics.get('recall', 0):.1%}
• F1-Score: {metrics.get('f1', 0):.1%}
• ROC-AUC: {metrics.get('roc_auc', 0):.3f}

Interpretação Clínica:
• Taxa de Verdadeiros Positivos: {metrics.get('recall', 0):.1%}
• Taxa de Falsos Positivos: {1 - metrics.get('precision', 1):.1%}
• Especificidade: {metrics.get('specificity', 0):.1%}

Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}
        """
        
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor=PROFESSIONAL_COLORS['light'],
                        edgecolor=PROFESSIONAL_COLORS['neutral'], alpha=0.8))

    def _plot_feature_importance_subplot(self, ax, feature_names, importance_values, top_n=15):
        """Subplot para importância de features"""
        # Preparar dados
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=True).tail(top_n)
        
        # Criar cores degradê
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
        
        # Barras horizontais
        bars = ax.barh(range(len(df)), df['importance'], color=colors,
                      edgecolor=PROFESSIONAL_COLORS['dark'], linewidth=0.5)
        
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['feature'], fontsize=9)
        ax.set_xlabel('Importância')
        ax.set_title(f'Top {top_n} Features Mais Importantes', fontweight='bold')
        
        # Adicionar valores
        for i, (bar, value) in enumerate(zip(bars, df['importance'])):
            width = bar.get_width()
            ax.text(width + max(df['importance']) * 0.01, 
                   bar.get_y() + bar.get_height()/2,
                   f'{value:.4f}', ha='left', va='center', 
                   fontsize=8, fontweight='bold')


# Funções utilitárias para compatibilidade com código existente
def create_professional_visualizer(save_dir='results'):
    """Cria uma instância do visualizador profissional"""
    return ProfessionalVisualizer(save_dir=save_dir)

def setup_professional_style():
    """Configura estilo profissional global"""
    visualizer = ProfessionalVisualizer()
    visualizer.setup_style()

# Configurar estilo automaticamente na importação
setup_professional_style()
