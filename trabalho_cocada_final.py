'''Trabalho final da disciplina Computação Científica e Análise de Dados do IC, da UFRJ.
Autor: Rafael Albuquerque.
Tema: Gayômetro de albuns musicais.

Resumo: O trabalho final foi em cima do algoritmo K-means, usado na clusterização. Foram escolhidos 3 albuns:
"Sour", da cantora Olivia Rodrigo. Um album mais melancólico, mais feminino.
"Rodeo", do rapper Travis Scott. Um álbum com uma letra mais pesada sobre vicios e excessos, que agrada o publico hetero.
"Renaissance", da diva pop Beyoncé. Um album com uma sonoridade mais dancante, com o ritmo House bem presente, o que agrada LGBTs.
Cada album é um vetor no R5, visto que só são necessárias as colunas Danceability, Energy, Valence, Acousticness e Tempo.
O K-means separou os três albuns citados em 3 clusters diferentes e calcula as distâncias entre os albuns e a entrada
fornecida pelo usuário.

Trabalho inspirado no trabalho "Qual música da Taylor Swift recomendaria para o Kanye West?" de Giovanna Lavouras.'''

import streamlit as st
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Configuração da página do Streamlit
st.set_page_config(page_title="Gayômetro - Clusterização Spotify", page_icon="🎵", layout="centered")
st.title("🎵 Gayômetro: Clusterizando Álbuns do Spotify")
st.write("Insira o nome de um álbum do catálogo para descobrir o quão gay ele é.")

'''1. FUNÇÃO COM CACHE: Carrega, une e limpa as bases de dados apenas uma vez. Reuni duas bases de dados em uma só 
porque uma tinha poucos dados e a outra tinha dados mais atualizados'''
@st.cache_data
def carregar_e_unificar_dados():
    # Dataset 1
    df1 = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "maharshipandya/-spotify-tracks-dataset",
        "dataset.csv",
    )
    
    # Dataset 2
    df2 = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "solomonameh/spotify-music-dataset",
        "high_popularity_spotify_data.csv",
    )
    
    '''Aqui são as colunas usadas na clusterização. Esses dados foram extraídos da API do spotify e são cruciais
    para calcular as diferenças entre os mesmos.'''
    colunas_necessarias = ['album_name', 'danceability', 'energy', 'valence', 'tempo', 'acousticness']
    
    '''Padronização do nome da coluna de álbum no df2 pois tinha uma coluna
    na segunda base de dados que tinha a coluna referente ao nome do album com nome diferente.'''
    if 'track_album_name' in df2.columns:
        df2 = df2.rename(columns={'track_album_name': 'album_name'})
    elif 'Track Album Name' in df2.columns:
        df2 = df2.rename(columns={'Track Album Name': 'album_name'})
        
    '''Deixa apenas as colunas necessárias e o restante tira para otimização do programa.'''
    df1_limpo = df1[colunas_necessarias].dropna()
    df2_limpo = df2[colunas_necessarias].dropna()
    
    '''Une ambas as bases de dados.'''
    df_completo = pd.concat([df1_limpo, df2_limpo], ignore_index=True).drop_duplicates()
    
    ''''Nessa parte as instruções apagarão "albuns" que só possuem uma música.
    Como não sabemos quantas músicas os albuns base terão os guardaremos.'''
    meus_albuns_base = ['SOUR', 'RENAISSANCE', 'Rodeo (Expanded Edition)']
    
    # 2. Filtro de singles (>= 2 faixas)
    contagem_por_album = df_completo['album_name'].value_counts()
    albuns_validos = contagem_por_album[contagem_por_album >= 2].index
    
    # 3. A Regra: Fica quem passou no teste de 2 faixas OU ( | ) quem tem imunidade
    df_albuns_reais = df_completo[
        df_completo['album_name'].isin(albuns_validos) | 
        df_completo['album_name'].isin(meus_albuns_base)
    ]
    
    return df_albuns_reais

'''Executa o carregamento com um aviso visual na tela.
Inicio do site.'''
with st.spinner("Processando base de dados unificada... Aguarde."):
    df_catalogo = carregar_e_unificar_dados()

with st.expander("📚 Ver catálogo completo de álbuns"):
    # Usamos o df_catalogo, que é a variável que existe aqui fora!
    todos_os_albuns = sorted(df_catalogo['album_name'].unique())
    
    st.write(f"Temos **{len(todos_os_albuns)}** álbuns disponíveis para teste.")
    
    st.dataframe(pd.DataFrame(todos_os_albuns, columns=["Nome do Álbum"]), use_container_width=True)

'''TREINAMENTO DO MODELO (Roda instantaneamente em segundo plano). Aqui a matemática acontece.'''
colunas_matematicas = ['danceability', 'energy', 'valence', 'tempo', 'acousticness']
meus_albuns = ['SOUR', 'RENAISSANCE', 'Rodeo (Expanded Edition)']

df_filtrado = df_catalogo[df_catalogo['album_name'].isin(meus_albuns)]
medias_dos_albuns = df_filtrado[['album_name'] + colunas_matematicas].groupby('album_name').mean()

'''Ajuste de escala e treinamento do K-Means'''
scaler = StandardScaler()
dados_nivelados = scaler.fit_transform(medias_dos_albuns)
df_nivelado = pd.DataFrame(dados_nivelados, columns=medias_dos_albuns.columns, index=medias_dos_albuns.index)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters_gerados = kmeans.fit_predict(df_nivelado)
medias_dos_albuns['Grupo_Atribuido'] = clusters_gerados

# Mapeamento inteligente dos resultados para exibição na interface
mapeamento_perfis = {}
for alb, grupo in zip(medias_dos_albuns.index, medias_dos_albuns['Grupo_Atribuido']):
    if alb == 'RENAISSANCE':
        mapeamento_perfis[grupo] = "Público Gay 🏳️‍🌈"
    elif alb == 'Rodeo (Expanded Edition)':
        mapeamento_perfis[grupo] = "Público Hétero 🤠"
    elif alb == 'SOUR':
        mapeamento_perfis[grupo] = "Vibe Menina 💔"

'''Exibe os grupos de referência na barra lateral para consulta do usuário'''
st.sidebar.header("📌 Grupos de Referência do Modelo")
for alb, grupo in zip(medias_dos_albuns.index, medias_dos_albuns['Grupo_Atribuido']):
    st.sidebar.markdown(f"**Grupo {grupo}:** {alb}")

# 3. INTERFACE DE INTERAÇÃO DO USUÁRIO
st.subheader("🔍 Escolha o Álbum para Teste")
album_teste = st.text_input("Digite o nome exato do álbum:", "Future Nostalgia")

if st.button("Executar Predict"):
    '''Busca o álbum ignorando diferenças de maiúsculas/minúsculas'''
    df_teste = df_catalogo[df_catalogo['album_name'].str.lower() == album_teste.strip().lower()]
    
    if df_teste.empty:
        st.error("Álbum não encontrado na base unificada. Tente ajustar o nome.")
        
        '''Sistema simples de busca aproximada para sugerir nomes corretos'''
        sugestoes = df_catalogo[df_catalogo['album_name'].str.contains(album_teste, case=False, na=False)]['album_name'].unique()
        if len(sugestoes) > 0:
            st.info(f"Você quis dizer: {', '.join(sugestoes[:4])}?")
    else:
        '''Extração das características e predição espacial'''
        medias_teste = df_teste[colunas_matematicas].mean().to_frame().T
        medias_teste.index = [df_teste['album_name'].iloc[0]]
        
        dados_teste_nivelados = scaler.transform(medias_teste)
        resultado_cluster = kmeans.predict(dados_teste_nivelados)[0]
        nome_confirmado = medias_teste.index[0]
        
        # Resultados visuais na tela
        st.success(f"### Veredito para: {nome_confirmado}")
        st.metric(label="Cluster Atribuído", value=f"Grupo {resultado_cluster}")
        
        vibe_detectada = mapeamento_perfis.get(resultado_cluster, "Perfil não mapeado.")
        st.markdown(f"A vibe predominante calculada pelo algoritmo é: **{vibe_detectada}**")

        if vibe_detectada == "Público Gay 🏳️‍🌈":
            st.image("https://i.imgur.com/FBxLCIK.png", width=1200)
            
        elif vibe_detectada == "Público Hétero 🤠":
            st.image("https://i.imgur.com/zVVZ7S3.png", width=1200)
            
        elif vibe_detectada == "Vibe Menina 💔":
            st.image("https://i.imgur.com/nQ86aCg.png", width=1200)
        
        # Renderização da tabela de características numéricas médias do álbum avaliado
        st.markdown("#### 📊 Atributos Matemáticos Médios do Álbum")
        st.dataframe(medias_teste)