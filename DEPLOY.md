# 🚀 DEPLOY RÁPIDO – Streamlit Community Cloud (5 minutos)

Esse é o caminho mais simples para ter sua app na web **grátis**.

## ✅ Pré-requisitos

- Conta no GitHub (grátis): https://github.com/signup
- Conta no Streamlit (grátis, faz login com GitHub): https://share.streamlit.io

## 📋 Passo a passo

### 1️⃣ Criar repositório no GitHub

1. Acesse: https://github.com/new
2. Nome do repositório: `video-take-selector` (ou outro nome)
3. Deixe **Público** (necessário para o plano grátis)
4. Clique em **"Create repository"**

### 2️⃣ Subir os arquivos

**Modo fácil (interface web do GitHub):**

1. No repositório recém-criado, clique em **"uploading an existing file"**
2. Arraste TODOS os arquivos desta pasta (`app.py`, `requirements.txt`, `packages.txt`, `README.md`, `.gitignore`)
3. ⚠️ A pasta `.streamlit/` também precisa subir junto. Se o GitHub não aceitar arrastar pasta, faça assim:
   - Clique em "Add file" → "Create new file"
   - No campo nome, digite: `.streamlit/config.toml`
   - Cole o conteúdo do arquivo
   - Commit
4. Clique em **"Commit changes"**

**Modo terminal (mais técnico):**

```bash
cd /caminho/para/video_take_selector
git init
git add .
git commit -m "Deploy inicial"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/video-take-selector.git
git push -u origin main
```

### 3️⃣ Deploy no Streamlit Cloud

1. Acesse: https://share.streamlit.io
2. Faça login com GitHub
3. Clique em **"New app"** (canto superior direito)
4. Preencha:
   - **Repository**: `SEU_USUARIO/video-take-selector`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Clique em **"Deploy!"**

⏳ Aguarde **3-5 minutos** para a instalação das dependências (o YOLO + PyTorch são pesados).

### 4️⃣ Pronto!

Sua app terá uma URL pública como:
```
https://SEU_USUARIO-video-take-selector-app-xxxxx.streamlit.app
```

Compartilhe com quem quiser!

---

## 🔄 Atualizando a aplicação

Qualquer mudança que você fizer no GitHub é **automaticamente refletida** no Streamlit Cloud em ~1 minuto. Basta dar commit/push.

---

## ⚡ Dicas para melhor desempenho no Cloud

1. **Use vídeos curtos** (< 1 min) para teste — o servidor grátis tem só 1 GB de RAM.
2. **Reduza a resolução dos vídeos** antes de subir (1080p → 720p ou 480p).
3. **Mantenha "Threads paralelas" em 1** na sidebar.
4. Se travar, recarregue a página e clique em **"Limpar arquivos temporários"**.

---

## 🆘 Se algo der errado

### Erro durante o build (Streamlit Cloud)

Veja o log clicando em **"Manage app"** → aba **"Logs"**. Erros mais comuns:

- **`ffmpeg: command not found`** → confirme que o `packages.txt` está na raiz do repo.
- **`OutOfMemoryError`** → vídeo muito grande. Reduza tamanho.
- **`Module not found`** → falta no `requirements.txt`. Adicione e refaça push.

### App carrega mas não processa

- Confira se o `.srt` está bem formado.
- Tente com **1 vídeo curto** primeiro para validar o fluxo.

---

## 💡 Alternativas se precisar de mais poder

| Plataforma | Custo | RAM | GPU | Quando usar |
|------------|-------|-----|-----|-------------|
| **Streamlit Cloud** | Grátis | 1 GB | ❌ | Demos e protótipos |
| **HuggingFace Spaces** | Grátis (CPU) / pago (GPU) | 16 GB | ✅ pago | Apps de ML em produção |
| **Railway** | $5+/mês | 8 GB+ | ❌ | Apps com tráfego médio |
| **Render** | $7+/mês | Variável | ❌ | Apps profissionais |
| **AWS/GCP** | Variável | Ilimitado | ✅ | Produção empresarial |

Para o seu caso (YOLO + processamento de vídeo), **HuggingFace Spaces com GPU T4** é a melhor relação custo-benefício se o uso for intenso.
