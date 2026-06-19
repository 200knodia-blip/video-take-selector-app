"""
🎬 Seleção Inteligente de Takes com YOLOv10
Versão otimizada para deploy web (Streamlit Community Cloud, HuggingFace Spaces, etc.)
"""

import streamlit as st
import os
import gc
import cv2
import numpy as np
import srt
import pandas as pd
import zipfile
import tempfile
import shutil
from datetime import datetime
from moviepy.video.io.VideoFileClip import VideoFileClip
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

# ====================== CONFIG DA PÁGINA ======================
st.set_page_config(
    page_title="🎬 Seleção Inteligente de Takes",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Sistema de seleção automática de takes com YOLOv10. "
                 "Recomendado vídeos de até 200MB cada para melhor desempenho."
    }
)

st.title("🎬 Sistema Web de Seleção de Takes com YOLOv10")
st.markdown("**Seleção genérica e automática** – Funciona com qualquer produto")

# Aviso de limitações em ambiente web
with st.expander("⚠️ Limitações do ambiente web (clique para ver)", expanded=False):
    st.markdown("""
    - **Memória limitada** em servidores grátis (~1 GB). Use vídeos curtos para teste.
    - **Sem GPU**: o processamento YOLO roda em CPU, então pode demorar.
    - **Tamanho de upload**: até 1 GB por arquivo (configurável em `.streamlit/config.toml`).
    - **Tempo de inatividade**: apps no Streamlit Cloud "dormem" após inatividade — primeira chamada pode demorar.
    """)

# ====================== ESTADO INICIAL ======================
DEFAULTS = {
    "threshold_corte": 27,
    "padding": 0.1,
    "frames_por_take": 12,
    "max_takes_por_segmento": 5,
    "max_workers": 1,  # Em web/CPU, paralelismo alto piora desempenho
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ====================== CARREGAMENTO DO MODELO (cacheado) ======================
@st.cache_resource(show_spinner="Carregando modelo YOLOv10...")
def carregar_modelo_yolo(model_path: str):
    """Carrega o modelo YOLO uma única vez por sessão do servidor."""
    return YOLO(model_path)

# ====================== PASTAS TEMPORÁRIAS ======================
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp(prefix="takes_")
    for sub in ("uploads", "takes_selecionados", "relatorios"):
        os.makedirs(os.path.join(st.session_state.temp_dir, sub), exist_ok=True)

UPLOAD_DIR = os.path.join(st.session_state.temp_dir, "uploads")
TAKES_DIR = os.path.join(st.session_state.temp_dir, "takes_selecionados")
REPORT_DIR = os.path.join(st.session_state.temp_dir, "relatorios")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("⚙️ Configurações")

    model_path = st.text_input(
        "📌 Modelo YOLO",
        value="yolov10n.pt",
        help="Use 'yolov10n.pt' (nano - mais leve) ou 'yolov8n.pt'. O Ultralytics baixa automaticamente."
    )

    st.session_state.threshold_corte = st.slider(
        "Threshold de corte seco", 20, 40, st.session_state.threshold_corte
    )
    st.session_state.padding = st.slider(
        "Padding entre takes (segundos)", 0.0, 0.5, st.session_state.padding, step=0.05
    )
    st.session_state.frames_por_take = st.slider(
        "Frames analisados por take", 6, 20, st.session_state.frames_por_take
    )
    st.session_state.max_takes_por_segmento = st.slider(
        "Máx. takes por segmento", 3, 10, st.session_state.max_takes_por_segmento
    )
    st.session_state.max_workers = st.slider(
        "Threads paralelas",
        1, 4, st.session_state.max_workers,
        help="Em servidor com pouca RAM, mantenha em 1."
    )

    st.divider()
    uploaded_srt = st.file_uploader("📄 Upload do roteiro.srt", type=["srt"])
    uploaded_videos = st.file_uploader(
        "🎥 Upload dos vídeos (.mp4)",
        type=["mp4"],
        accept_multiple_files=True
    )

    if uploaded_videos:
        total_mb = sum(v.size for v in uploaded_videos) / (1024 * 1024)
        st.caption(f"📦 Total: **{total_mb:.1f} MB** em {len(uploaded_videos)} vídeo(s)")
        if total_mb > 500:
            st.warning("⚠️ Mais de 500 MB pode estourar a memória em servidores grátis.")

    if st.button("🚀 INICIAR ANÁLISE", type="primary", use_container_width=True):
        if not uploaded_srt or not uploaded_videos:
            st.error("❌ Envie o .srt e pelo menos um vídeo!")
        else:
            st.session_state.run_analysis = True
            st.session_state.uploaded_srt = uploaded_srt
            st.session_state.uploaded_videos = uploaded_videos

    st.divider()
    if st.button("🧹 Limpar arquivos temporários", use_container_width=True):
        try:
            shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
            del st.session_state["temp_dir"]
            gc.collect()
            st.success("Limpo! Recarregue a página.")
        except Exception as e:
            st.error(f"Erro ao limpar: {e}")

# Carrega modelo (cacheado entre reruns)
model_yolo = carregar_modelo_yolo(model_path)

# ====================== HELPERS ======================
def _subclip(clip, inicio, fim):
    """Compatível com MoviePy 1.x (subclip) e 2.x (subclipped)."""
    fn = getattr(clip, "subclipped", None) or getattr(clip, "subclip", None)
    if fn is None:
        raise AttributeError("VideoFileClip não possui método subclip/subclipped")
    return fn(inicio, fim)

# ====================== FUNÇÕES ======================
def carregar_roteiro(srt_bytes):
    """Lê o .srt e retorna lista de segmentos."""
    with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as tmp:
        tmp.write(srt_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, encoding="utf-8") as f:
            subs = list(srt.parse(f.read()))
    finally:
        os.unlink(tmp_path)
    return [{
        "ordem": i + 1,
        "inicio": sub.start.total_seconds(),
        "fim": sub.end.total_seconds(),
        "texto": sub.content.strip(),
        "duracao": sub.end.total_seconds() - sub.start.total_seconds()
    } for i, sub in enumerate(subs)]

def detectar_cortes_secos(video_path, threshold, padding):
    """Detecta cortes secos comparando diferença média entre frames."""
    cap = cv2.VideoCapture(video_path)
    cortes = []
    frame_anterior = None
    ultimo_corte = 0.0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_anterior is not None:
                diff = cv2.absdiff(frame, frame_anterior)
                pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if np.mean(diff) > threshold and (pos - ultimo_corte) > 0.5:
                    corte_ajustado = round(pos - padding, 2)
                    if corte_ajustado > ultimo_corte:
                        cortes.append(corte_ajustado)
                        ultimo_corte = corte_ajustado
            frame_anterior = frame.copy()
    finally:
        cap.release()
    return sorted(list(set(cortes)))

def analisar_frame_com_yolo(frame):
    """Roda YOLO no frame e extrai métricas de qualidade."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    h, w = gray.shape
    center_crop = gray[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
    edges = cv2.Canny(center_crop, 80, 180)
    edge_density = np.mean(edges > 0)

    results = model_yolo(frame, verbose=False)[0]
    boxes = results.boxes
    product_detected = False
    person_detected = False
    max_conf = 0.0
    bbox_size_ratio = 0.0

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf > max_conf:
            max_conf = conf
        if results.names[cls_id] == "person":
            person_detected = True
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bbox_area = (x2 - x1) * (y2 - y1)
        frame_area = w * h
        ratio = bbox_area / frame_area
        if ratio > 0.08 and 0.3 < (x1 + x2)/2/w < 0.7 and 0.3 < (y1 + y2)/2/h < 0.7:
            product_detected = True
            bbox_size_ratio = ratio

    if product_detected and bbox_size_ratio > 0.25:
        angulo = "close-up"
    elif product_detected and bbox_size_ratio > 0.12:
        angulo = "lateral"
    elif person_detected:
        angulo = "pessoa_usando"
    else:
        angulo = "geral"

    return {
        "nitidez": laplacian_var,
        "brilho": brightness,
        "densidade_bordas_centro": edge_density,
        "angulo_yolo": angulo,
        "conf_yolo": round(max_conf, 4),
        "bbox_ratio": round(bbox_size_ratio, 4),
        "product_detected": product_detected,
        "person_detected": person_detected
    }

def classificar_take_avancado_yolo(clip, inicio, fim, video_path, frames_por_take):
    if fim - inicio < 0.5:
        return None
    duracao_take = fim - inicio
    step = max(1, int(duracao_take * clip.fps / frames_por_take))
    frames_analisados = []
    for i in range(0, int(duracao_take * clip.fps), step):
        try:
            frame = clip.get_frame(inicio + i / clip.fps)
            metrics = analisar_frame_com_yolo(frame)
            frames_analisados.append(metrics)
        except Exception:
            continue
    if not frames_analisados:
        return None

    avg_nitidez = np.mean([m["nitidez"] for m in frames_analisados])
    avg_densidade = np.mean([m["densidade_bordas_centro"] for m in frames_analisados])
    avg_conf_yolo = np.mean([m["conf_yolo"] for m in frames_analisados])
    avg_bbox_ratio = np.mean([m["bbox_ratio"] for m in frames_analisados])
    angulos = [m["angulo_yolo"] for m in frames_analisados]
    angulo_final = max(set(angulos), key=angulos.count)

    pontuacao = (
        (avg_nitidez / 2000 * 0.25) +
        (avg_densidade * 0.20) +
        (avg_bbox_ratio * 0.25) +
        (avg_conf_yolo * 0.20) +
        (1 if angulo_final == "close-up" else 0.6) * 0.10
    )

    return {
        "video": video_path,
        "inicio": round(inicio, 2),
        "fim": round(fim, 2),
        "duracao": round(duracao_take, 2),
        "angulo": angulo_final,
        "pontuacao": round(pontuacao, 4),
        "conf_yolo_media": round(avg_conf_yolo, 4),
        "bbox_ratio_media": round(avg_bbox_ratio, 4),
        "product_detected": any(m["product_detected"] for m in frames_analisados),
        "frames_analisados": len(frames_analisados)
    }

def processar_video(video_path, threshold, padding, frames_por_take):
    """
    Roda em thread separada — NÃO chama Streamlit aqui.
    Retorna dict com takes e eventuais erros.
    """
    clip = None
    try:
        clip = VideoFileClip(video_path)
        cortes = detectar_cortes_secos(video_path, threshold, padding)
        cortes_completos = [0.0] + cortes + [clip.duration]
        takes = []
        for i in range(len(cortes_completos) - 1):
            inicio = cortes_completos[i]
            fim = cortes_completos[i + 1]
            take = classificar_take_avancado_yolo(clip, inicio, fim, video_path, frames_por_take)
            if take:
                takes.append(take)
        return {"video": video_path, "takes": takes, "erro": None}
    except Exception as e:
        return {"video": video_path, "takes": [], "erro": str(e)}
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception:
                pass

def alinhar_e_selecionar(todos_takes, segmentos, max_takes):
    alinhamento = []
    takes_usados = set()
    for seg in segmentos:
        candidatos = [t for t in todos_takes if id(t) not in takes_usados]
        dur_seg = seg["duracao"]

        candidatos_filtrados = []
        for t in candidatos:
            if t.get("product_detected", False) and t.get("bbox_ratio_media", 0) > 0.12:
                candidatos_filtrados.append(t)
            elif 0.6 * dur_seg <= t.get("duracao", 0) <= 1.4 * dur_seg:
                candidatos_filtrados.append(t)

        if not candidatos_filtrados:
            candidatos_filtrados = sorted(candidatos, key=lambda x: x.get("pontuacao", 0), reverse=True)

        candidatos_filtrados.sort(key=lambda x: x.get("pontuacao", 0), reverse=True)
        selecionados = candidatos_filtrados[:max_takes]

        for pos, take in enumerate(selecionados):
            takes_usados.add(id(take))
            alinhamento.append({
                "ordem_roteiro": seg["ordem"],
                "posicao_qualidade": pos + 1,
                "video_origem": os.path.basename(take["video"]),
                "contagem_video": f"take_{pos + 1}",
                "take": take,
                "texto_roteiro": seg["texto"]
            })
    return alinhamento

def exportar_take_web(item, output_dir):
    """Executa na thread principal — pode usar st.error com segurança."""
    clip = None
    sub = None
    try:
        take = item["take"]
        nome_arquivo = (
            f"{item['ordem_roteiro']}.{item['posicao_qualidade']}_"
            f"{item['video_origem']}_{item['contagem_video']}.mp4"
        )
        caminho_saida = os.path.join(output_dir, nome_arquivo)

        # Margem de segurança de 0.3s no início
        inicio_com_trim = take["inicio"] + 0.3
        fim_com_trim = take["fim"]

        clip = VideoFileClip(take["video"])
        sub = _subclip(clip, inicio_com_trim, fim_com_trim)
        sub.write_videofile(
            caminho_saida,
            codec="libx264",
            audio=False,
            preset="fast",
            ffmpeg_params=["-crf", "18"],
            logger=None,  # silencia ffmpeg em logs do Streamlit
        )
    except Exception as e:
        st.error(f"Erro ao exportar take {item.get('ordem_roteiro', '???')}: {e}")
    finally:
        for obj in (sub, clip):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass

# ====================== EXECUÇÃO ======================
if (
    st.session_state.get("run_analysis", False)
    and st.session_state.get("uploaded_srt")
    and st.session_state.get("uploaded_videos")
):
    with st.spinner("Processando todos os vídeos com YOLOv10..."):
        # Salva arquivos enviados
        srt_path = os.path.join(UPLOAD_DIR, "roteiro.srt")
        with open(srt_path, "wb") as f:
            f.write(st.session_state.uploaded_srt.getbuffer())

        video_paths = []
        for vid in st.session_state.uploaded_videos:
            path = os.path.join(UPLOAD_DIR, vid.name)
            with open(path, "wb") as f:
                f.write(vid.getbuffer())
            video_paths.append(path)

        segmentos = carregar_roteiro(st.session_state.uploaded_srt.getbuffer())

        threshold = st.session_state.threshold_corte
        padding = st.session_state.padding
        frames_por_take = st.session_state.frames_por_take
        max_takes = st.session_state.max_takes_por_segmento
        max_workers = st.session_state.max_workers

        st.info(f"📊 {len(segmentos)} segmentos no roteiro · {len(video_paths)} vídeo(s) para processar")

        progress_bar = st.progress(0, text="Iniciando análise...")
        todos_takes = []

        # Processa vídeos (paralelo controlado)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            resultados = list(executor.map(
                lambda p: processar_video(p, threshold, padding, frames_por_take),
                video_paths
            ))

        # Trata erros na main thread
        for i, resultado in enumerate(resultados):
            if resultado["erro"]:
                st.error(f"Erro em {os.path.basename(resultado['video'])}: {resultado['erro']}")
            todos_takes.extend(resultado["takes"])
            progress_bar.progress(
                (i + 1) / len(video_paths),
                text=f"Vídeos analisados: {i+1}/{len(video_paths)}"
            )

        progress_bar.empty()
        st.write(f"**Total de takes detectados:** {len(todos_takes)}")

        if not todos_takes:
            st.error("Nenhum take foi detectado. Verifique os parâmetros ou os vídeos.")
            st.session_state.run_analysis = False
            st.stop()

        alinhamento_final = alinhar_e_selecionar(todos_takes, segmentos, max_takes)
        st.write(f"**Takes selecionados para exportação:** {len(alinhamento_final)}")

        # Exporta takes
        export_progress = st.progress(0, text="Exportando takes...")
        for idx, item in enumerate(alinhamento_final):
            exportar_take_web(item, TAKES_DIR)
            export_progress.progress(
                (idx + 1) / len(alinhamento_final),
                text=f"Exportando: {idx+1}/{len(alinhamento_final)}"
            )
        export_progress.empty()

        # Libera memória
        gc.collect()

        # ZIP
        zip_path = os.path.join(REPORT_DIR, f"takes_{datetime.now().strftime('%Y%m%d_%H%M')}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(TAKES_DIR):
                for file in files:
                    zipf.write(os.path.join(root, file), file)

        # CSV
        df = pd.DataFrame([{
            "ordem": item["ordem_roteiro"],
            "posicao": item["posicao_qualidade"],
            "video": item["video_origem"],
            "take": item["contagem_video"],
            "angulo_yolo": item["take"]["angulo"],
            "inicio": item["take"]["inicio"],
            "fim": item["take"]["fim"],
            "duracao": item["take"]["duracao"],
            "pontuacao": item["take"]["pontuacao"],
            "conf_yolo_media": item["take"]["conf_yolo_media"],
            "bbox_ratio_media": item["take"]["bbox_ratio_media"],
            "product_detected": item["take"]["product_detected"],
            "frames_analisados": item["take"]["frames_analisados"],
            "texto_roteiro": item["texto_roteiro"]
        } for item in alinhamento_final])
        csv_path = os.path.join(REPORT_DIR, "relatorio_selecao.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")

        st.success(f"✅ Análise concluída! {len(alinhamento_final)} takes selecionados.")

        col1, col2 = st.columns(2)
        with col1:
            with open(zip_path, "rb") as f:
                st.download_button(
                    "⬇️ Baixar TODOS os Takes (ZIP)",
                    f,
                    "takes_selecionados.zip",
                    use_container_width=True
                )
        with col2:
            with open(csv_path, "rb") as f:
                st.download_button(
                    "⬇️ Baixar Relatório CSV",
                    f,
                    "relatorio_selecao.csv",
                    use_container_width=True
                )

        st.subheader("👀 Preview dos melhores takes")
        for item in alinhamento_final[:6]:
            video_file = os.path.join(
                TAKES_DIR,
                f"{item['ordem_roteiro']}.{item['posicao_qualidade']}_"
                f"{item['video_origem']}_{item['contagem_video']}.mp4"
            )
            if os.path.exists(video_file):
                st.video(video_file)
                st.caption(
                    f"Ordem {item['ordem_roteiro']} • {item['take']['angulo']} • "
                    f"Pontuação {item['take']['pontuacao']}"
                )

    st.session_state.run_analysis = False

st.caption("Sistema genérico – trim de 0.3s no início de cada take · Pronto para web ✨")
