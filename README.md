# 🎬 Seleção Inteligente de Takes com YOLOv10

Sistema web automático de seleção de takes em vídeos, usando detecção de objetos com YOLOv10 e alinhamento com roteiro (.srt). **Funciona com qualquer produto** — análise genérica baseada em métricas de qualidade visual.

## ✨ Funcionalidades

- 📄 Upload de roteiro `.srt` + múltiplos vídeos `.mp4`
- ✂️ Detecção automática de cortes secos
- 🔍 Análise de qualidade por frame (nitidez, bordas, detecção YOLO)
- 🎯 Alinhamento takes ↔ segmentos do roteiro
- 📦 Exportação em ZIP + relatório CSV
- 👀 Preview dos melhores takes na própria interface

---

## 🚀 Como rodar

### 📍 Opção A: Streamlit Community Cloud (grátis, mais fácil)

1. **Crie um repositório no GitHub** e faça upload de todos os arquivos deste projeto.
2. Acesse [share.streamlit.io](https://share.streamlit.io).
3. Clique em **"New app"** → selecione seu repositório → arquivo: `app.py` → branch: `main`.
4. Clique em **"Deploy"**. Em ~3 minutos sua aplicação estará no ar com URL pública.

> ⚠️ **Limites do plano grátis:**
> - ~1 GB de RAM (use vídeos curtos para teste)
> - Sem GPU (YOLO roda em CPU)
> - App "dorme" após inatividade (primeira chamada demora)

### 📍 Opção B: Hugging Face Spaces (suporta GPU paga)

1. Crie um Space em [huggingface.co/new-space](https://huggingface.co/new-space).
2. Tipo: **Streamlit**. Hardware: CPU básico (grátis) ou T4 small (GPU paga).
3. Faça upload dos arquivos (ou conecte ao GitHub).
4. Renomeie `packages.txt` para `apt.txt` se necessário (HF usa esse nome).

### 📍 Opção C: Railway / Render / Fly.io

Plataformas pagas com mais memória e CPU. Use os mesmos arquivos. Geralmente exige:

```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

### 📍 Opção D: Local

```bash
# 1. Clone o repositório
git clone <seu-repo>
cd video_take_selector

# 2. Crie ambiente virtual
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate    # Windows

# 3. Instale dependências do sistema (macOS)
brew install ffmpeg

# Linux:
# sudo apt-get install ffmpeg libsm6 libxext6 libgl1

# 4. Instale dependências Python
pip install -r requirements.txt

# 5. Rode o app
streamlit run app.py
```

Acesse: http://localhost:8501

---

## 📁 Estrutura do Projeto

```
video_take_selector/
├── app.py                  # Aplicação principal
├── requirements.txt        # Dependências Python
├── packages.txt            # Dependências do sistema (apt)
├── .gitignore
├── README.md
└── .streamlit/
    └── config.toml         # Configurações do Streamlit
```

---

## ⚙️ Configurações ajustáveis (sidebar)

| Parâmetro | Descrição | Default |
|-----------|-----------|---------|
| **Modelo YOLO** | Caminho do modelo (`.pt`). Baixado automaticamente. | `yolov10n.pt` |
| **Threshold de corte seco** | Sensibilidade para detectar cortes. ↑ menos cortes | 27 |
| **Padding entre takes** | Margem de segurança (segundos) | 0.1 |
| **Frames por take** | Quantos frames analisar por take | 12 |
| **Máx. takes por segmento** | Quantos takes selecionar por linha do roteiro | 5 |
| **Threads paralelas** | Workers do ThreadPoolExecutor (1 = mais seguro) | 1 |

---

## 🐛 Troubleshooting

### "missing ScriptRunContext!" no terminal
Você rodou com `python app.py`. Use `streamlit run app.py`.

### Erro `ImportError: cannot import name 'VideoFileClip'`
Versão do MoviePy incompatível. O `requirements.txt` fixa `moviepy==1.0.3` para evitar isso.

### App fica lento ou trava no Streamlit Cloud
Memória insuficiente. Reduza:
- Número de vídeos enviados simultaneamente
- Resolução dos vídeos (faça downscale antes de subir)
- Slider "Threads paralelas" para `1`

### YOLO não baixa o modelo
Em ambientes sem internet liberada, faça upload manual do `yolov10n.pt` para a raiz do projeto e referencie no campo "Modelo YOLO".

---

## 📊 Como funciona

1. **Detecção de cortes**: compara diferença média entre frames consecutivos.
2. **Análise de qualidade**: para cada take, extrai N frames e calcula:
   - **Nitidez** (variância do Laplaciano)
   - **Densidade de bordas** no centro do frame
   - **Detecção YOLO**: confiança, tamanho da bbox, classificação de ângulo (close-up, lateral, pessoa, geral)
3. **Pontuação ponderada**: combina as métricas em um score único.
4. **Alinhamento com roteiro**: para cada linha do `.srt`, filtra candidatos por duração compatível e detecção de produto, e seleciona os top-N por pontuação.
5. **Export**: cada take vira um `.mp4` separado, nomeado por `{ordem}.{posicao}_{video}_{contagem}.mp4`.

---

## 📝 Licença

Uso livre para projetos pessoais e comerciais.

---

**Feito com ❤️ usando Streamlit, OpenCV, MoviePy e Ultralytics YOLOv10.**
