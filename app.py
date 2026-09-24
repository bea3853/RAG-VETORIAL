import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import AzureOpenAI, APIError
import streamlit as st
import pandas as pd
from pypdf import PdfReader

import chromadb
from sentence_transformers import SentenceTransformer

# Carrega variáveis locais se existirem (para execução local com .env)
load_dotenv(override=True)

# Compatibilidade entre st.secrets e os nomes de variáveis de ambiente solicitados
def get_env_variable(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

AZURE_ENDPOINT = get_env_variable("ENDPOINT")
AZURE_API_KEY = get_env_variable("API_KEY")
AZURE_API_VERSION = get_env_variable("API_VERSION", "2025-04-01-preview")
DEPLOYMENT_NAME = get_env_variable("GPT5_MODEL")

# Inicialização do Cliente Azure OpenAI
client = None
if AZURE_ENDPOINT and AZURE_API_KEY:
    try:
        client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION
        )
    except Exception:
        pass

# Carrega o modelo de embeddings local (leve e eficiente para CPU/GPU)
@st.cache_resource
def carregar_modelo_embedding():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Inicializa o banco vetorial local (ChromaDB em memória)
@st.cache_resource
def inicializar_banco_vetorial():
    chroma_client = chromadb.Client()
    # Cria ou obtém uma coleção para o RAG
    collection = chroma_client.get_or_create_collection(name="base_conhecimento_rag")
    return collection

def fatiar_texto(texto, tamanho_chunk=500, sobreposicao=50):
    """Divide textos longos em pedaços (chunks) menores para indexação vetorial."""
    chunks = []
    for i in range(0, len(texto), tamanho_chunk - sobreposicao):
        chunk = texto[i:i + tamanho_chunk]
        if len(chunk.strip()) > 20:  # Ignora pedaços muito vazios
            chunks.append(chunk)
    return chunks

@st.cache_data(ttl=3600)
def processar_e_indexar_fontes():
    """Realiza o scraping do site e a leitura dos arquivos locais, fatiando e populando o ChromaDB."""
    collection = inicializar_banco_vetorial()
    embed_model = carregar_modelo_embedding()
    
    # Limpa dados anteriores da coleção para evitar duplicação em cache recarregado
    existing = collection.get()
    if existing and existing['ids']:
        collection.delete(ids=existing['ids'])

    documentos_totais = []
    
    # 1. Processamento do Site
    url = "https://gratuitos.netlify.app/"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.extract()
        texto_site = soup.get_text(separator="\n")
        linhas = [linha.strip() for linha in texto_site.splitlines() if linha.strip()]
        texto_site_limpo = "\n".join(linhas)
        
        for i, chunk in enumerate(fatiar_texto(texto_site_limpo)):
            documentos_totais.append({
                "id": f"site_chunk_{i}",
                "text": chunk,
                "source": "Site Institucional"
            })
    except Exception as e:
        print(f"Erro ao acessar o site: {e}")

    # 2. Processamento do CSV
    try:
        if os.path.exists("dados.csv"):
            df = pd.read_csv("dados.csv")
            # Converte cada linha do DataFrame em um texto estruturado indexável
            for index, row in df.iterrows():
                row_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
                documentos_totais.append({
                    "id": f"csv_row_{index}",
                    "text": f"Dado Tabular (CSV): {row_text}",
                    "source": "Arquivo CSV"
                })
    except Exception as e:
        print(f"Erro ao ler CSV: {e}")

    # 3. Processamento do PDF
    try:
        if os.path.exists("escola.pdf"):
            reader = PdfReader("escola.pdf")
            texto_pdf_completo = ""
            for pagina in reader.pages:
                t_pag = pagina.extract_text()
                if t_pag:
                    texto_pdf_completo += t_pag + "\n"
            
            for i, chunk in enumerate(fatiar_texto(texto_pdf_completo)):
                documentos_totais.append({
                    "id": f"pdf_chunk_{i}",
                    "text": chunk,
                    "source": "Documento PDF"
                })
    except Exception as e:
        print(f"Erro ao ler PDF: {e}")

    # Se houver documentos, gera embeddings e insere no ChromaDB
    if documentos_totais:
        ids = [doc["id"] for doc in documentos_totais]
        texts = [doc["text"] for doc in documentos_totais]
        metadatas = [{"source": doc["source"]} for doc in documentos_totais]
        
        embeddings = embed_model.encode(texts).tolist()
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

def recuperar_contexto_relevante(pergunta, n_resultados=3):
    """Busca os trechos mais semanticamente similares à pergunta no banco vetorial."""
    collection = inicializar_banco_vetorial()
    embed_model = carregar_modelo_embedding()
    
    pergunta_embedding = embed_model.encode([pergunta]).tolist()
    
    resultados = collection.query(
        query_embeddings=pergunta_embedding,
        n_results=n_resultados
    )
    
    contexto_formatado = ""
    if resultados and resultados['documents'] and resultados['documents'][0]:
        for i, (doc, meta) in enumerate(zip(resultados['documents'][0], resultados['metadatas'][0])):
            contexto_formatado += f"\n--- [FONTE: {meta['source']}] ---\n{doc}\n"
            
    return contexto_formatado

def validar_configuracao():
    faltando = []
    if not AZURE_ENDPOINT:
        faltando.append("ENDPOINT")
    if not AZURE_API_KEY:
        faltando.append("API_KEY")
    if not DEPLOYMENT_NAME:
        faltando.append("GPT5_MODEL")
    return faltando

def main():
    st.set_page_config(page_title="Consulta de Unidades (RAG Vetorial)", page_icon="🏫", layout="centered")
    
    st.title("🏫 Consulta de Unidades e Cursos (RAG Vetorial)")
    st.write("Digite sua dúvida abaixo para consultar as informações indexadas na base de conhecimento semântica.")

    faltando = validar_configuracao()
    if faltando:
        st.error(f"Erro de configuração: As seguintes variáveis não foram definidas no ambiente ou secrets: {', '.join(faltando)}")
        return

    # Processa e indexa as fontes automaticamente na primeira execução
    with st.spinner("Indexando fontes de dados (Site, CSV, PDF)..."):
        processar_e_indexar_fontes()

    # Interface de Entrada do Aluno
    with st.form(key="form_pergunta"):
        pergunta = st.text_input("Qual a sua dúvida ou unidade que deseja buscar?", placeholder="Ex: Qual o endereço da unidade mais próxima?")
        botao_enviar = st.form_submit_button("Buscar Unidade / Perguntar")

    if botao_enviar:
        if not pergunta.strip():
            st.warning("Por favor, digite uma pergunta válida.")
            return

        if not client:
            st.error("Cliente Azure OpenAI não inicializado corretamente.")
            return

        # Recupera apenas os trechos estritamente relevantes via Busca Vetorial
        with st.spinner("Buscando informações relevantes na base vetorial..."):
            dados_recuperados = recuperar_contexto_relevante(pergunta, n_resultados=4)

        if not dados_recuperados.strip():
            dados_recuperados = "Nenhuma informação relevante encontrada nas fontes indexadas."

        prompt = f"""Você é um assistente educacional prestativo. Use apenas os dados extraídos das fontes abaixo para responder de forma clara, objetiva e educativa:

Trechos recuperados da base de conhecimento:
{dados_recuperados}
"""

        try:
            with st.spinner("Gerando resposta com IA..."):
                response = client.chat.completions.create(
                    model=DEPLOYMENT_NAME,
                    messages=[
                        {
                            "role": "system",
                            "content": prompt,
                        },
                        {
                            "role": "user",
                            "content": pergunta
                        }
                    ]
                )
                resposta_texto = response.choices[0].message.content
                
                st.markdown("### Resposta:")
                st.success(resposta_texto)
                
                with st.expander("Ver trechos recuperados da base vetorial (Contexto)"):
                    st.text(dados_recuperados)

        except APIError as e:
            st.error(f"Erro na API do Azure: {e}")
        except Exception as e:
            st.error(f"Erro inesperado: {e}")

if __name__ == "__main__":
    main()


    # pip install streamlit requests beautifulsoup4 python-dotenv openai pandas pypdf chromadb sentence-transformers onnxruntime
    
    # python -m pip install --upgrade 
    # python -m pip install --upgrade pip setuptools wheel
