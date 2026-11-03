import os
import webbrowser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs


def plot_to_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False)


def salvar_html(html_parts, filename='NR-Curva-Carga-Baterias.html'):
    html_content = f'''<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="utf-8" />
    <script>{get_plotlyjs()}</script>
</head>
<body>
{''.join(html_parts)}
</body>
</html>'''
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    webbrowser.open(f'file://{os.path.abspath(filename)}')


ESTILOS = {
    'Sem perdas': dict(dash='solid', symbol='circle-open', size=6),
    'Com perdas': dict(dash='dash', symbol='triangle-up', size=6),
}


def plot_comparativo(df, y_col, titulo, y_label):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.08)
    series = {}
    for cenario, estilo in ESTILOS.items():
        dados = df.loc[df['Cenário'] == cenario].sort_values('Capacidade Bateria (p.u.)')
        series[cenario] = dados.set_index('Capacidade Bateria (p.u.)')[y_col]
        fig.add_trace(go.Scatter(
            x=dados['Capacidade Bateria (p.u.)'],
            y=dados[y_col],
            mode='lines+markers',
            name=cenario,
            line=dict(color='black', dash=estilo['dash']),
            marker=dict(symbol=estilo['symbol'], size=estilo['size'], line=dict(width=1.5)),
            connectgaps=False,
        ), row=1, col=1)

    dif = series['Com perdas'] - series['Sem perdas']
    fig.add_trace(go.Scatter(
        x=dif.index, y=dif.values,
        mode='lines+markers',
        name='Diferença (Com − Sem perdas)',
        line=dict(color='crimson'),
        marker=dict(symbol='diamond', size=5),
        connectgaps=False,
    ), row=2, col=1)

    fonte = dict(size=17, color='black', family='Arial Black')
    fig.update_xaxes(gridcolor='lightgrey', dtick=5)
    fig.update_xaxes(title='Capacidade Bateria (p.u.)', title_font=fonte, row=2, col=1)
    fig.update_yaxes(title=y_label, title_font=fonte, gridcolor='lightgrey', row=1, col=1)
    fig.update_yaxes(title='Diferença', title_font=fonte, gridcolor='lightgrey',
                     exponentformat='power', row=2, col=1)
    fig.update_layout(
        title=dict(text=titulo, x=0.5, y=0.97, xanchor='center',
                   font=dict(size=24, color='black', family='Arial Black')),
        height=800,
        showlegend=True,
        legend=dict(font=dict(size=15, family='Arial Black'),
                    orientation='h', x=0.5, xanchor='center', y=-0.12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=80, r=35, t=110, b=110),
    )
    return fig


def gera_gráficos_curva_new(df):
    colunas = {'Cenário', 'Capacidade Bateria (p.u.)',
               'Corte de Carga', 'Função Objetivo'}
    faltantes = colunas - set(df.columns)
    if faltantes:
        raise ValueError(f'Colunas ausentes no DataFrame: {sorted(faltantes)}')

    graficos = [
        (df, 'Corte de Carga',
         'Efeito da Capacidade da Bateria sobre o Corte de Carga',
         'Corte de Carga (MW)'),
        (df, 'Função Objetivo',
         'Efeito da Capacidade da Bateria sobre a Função Objetivo',
         'Função Objetivo (R$)'),
        (df.loc[df['Capacidade Bateria (p.u.)'] >= 68], 'Função Objetivo',
         'Efeito da Capacidade da Bateria sobre a Função Objetivo (a partir de 68)',
         'Função Objetivo (R$)'),
    ]
    html_parts = [
        plot_to_html(plot_comparativo(dados, coluna, titulo, eixo_y))
        for dados, coluna, titulo, eixo_y in graficos
    ]
    salvar_html(html_parts)