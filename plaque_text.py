import streamlit as st
import os
import cv2
import xml.etree.ElementTree as ET
import pandas as pd
import zipfile
import tempfile
import google.generativeai as genai
from PIL import Image
import io
import base64
from pathlib import Path
import time

def parse_yolo_annotation(txt_path, img_width, img_height):
    """
    Parse un fichier .txt au format YOLO et retourne les coordonnées des bounding boxes
    Format YOLO: class_id x_center y_center width height (valeurs normalisées 0-1)
    """
    boxes = []
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convertir les coordonnées normalisées en pixels
                    x_center_px = x_center * img_width
                    y_center_px = y_center * img_height
                    width_px = width * img_width
                    height_px = height * img_height
                    
                    # Calculer les coordonnées des coins
                    xmin = int(x_center_px - width_px/2)
                    ymin = int(y_center_px - height_px/2)
                    xmax = int(x_center_px + width_px/2)
                    ymax = int(y_center_px + height_px/2)
                    
                    boxes.append({
                        'class_id': class_id,
                        'xmin': max(0, xmin),
                        'ymin': max(0, ymin),
                        'xmax': min(img_width, xmax),
                        'ymax': min(img_height, ymax)
                    })
    except Exception as e:
        print(f"Erreur lors du parsing de {txt_path}: {e}")
    return boxes

def extract_plates_from_image(image_path, txt_path, output_dir, base_name):
    """
    Extrait les plaques d'une image en utilisant les annotations YOLO (.txt)
    """
    image = cv2.imread(image_path)
    if image is None:
        return []

    img_height, img_width = image.shape[:2]

    try:
        boxes = parse_yolo_annotation(txt_path, img_width, img_height)
    except Exception as e:
        st.warning(f"Impossible de parser l'annotation pour {os.path.basename(image_path)}: {e}")
        return []

    extracted_plates = []
    for i, box in enumerate(boxes):
        plate_roi = image[box['ymin']:box['ymax'], box['xmin']:box['xmax']]
        
        if plate_roi.size > 0:
            plate_filename = f"{base_name}_{i:03d}.jpg"
            plate_path = os.path.join(output_dir, plate_filename)
            cv2.imwrite(plate_path, plate_roi)
            extracted_plates.append({
                'filename': plate_filename,
                'path': plate_path,
                'box': box
            })
    return extracted_plates

def get_text_from_gemini(image_path, api_key):
    """
    Utilise l'API Gemini pour extraire le texte d'une image de plaque
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        img = Image.open(image_path)
        
        prompt = """
        Cette image contient une plaque d'immatriculation. 
        Extrais UNIQUEMENT le texte/numéros visibles sur la plaque.
        Réponds avec seulement les caractères que tu vois, sans espaces ni formatage supplémentaire.
        Si tu ne peux pas lire clairement, réponds "ILLEGIBLE".
        """
        
        response = model.generate_content([prompt, img])
        text = response.text.strip().upper().replace(" ", "")
        
        return text if text else "ILLEGIBLE"
        
    except Exception as e:
        st.error(f"Erreur avec l'API Gemini: {str(e)}")
        return "ERROR"

def process_dataset(images_folder, annotations_folder, api_key, progress_callback=None):
    """
    Traite l'ensemble du dataset : extraction + OCR
    """
    plates_output_dir = tempfile.mkdtemp(prefix="extracted_plates_")
    results = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    # --- DÉBUT DE LA CORRECTION ---
    # L'indentation de cette boucle a été corrigée.
    for file in os.listdir(images_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            base_name = os.path.splitext(file)[0]
            txt_file = base_name + '.txt'
            txt_path = os.path.join(annotations_folder, txt_file)
            
            if os.path.exists(txt_path):
                image_files.append((file, txt_file))
    # --- FIN DE LA CORRECTION ---

    total_files = len(image_files)
    if total_files == 0:
        return [], plates_output_dir

    plate_counter = 1
    for idx, (image_file, txt_file) in enumerate(image_files):
        if progress_callback:
            progress_callback(idx / total_files)
        
        image_path = os.path.join(images_folder, image_file)
        txt_path = os.path.join(annotations_folder, txt_file)
        
        base_name = f"plaque{plate_counter:04d}"
        extracted_plates = extract_plates_from_image(
            image_path, txt_path, plates_output_dir, base_name
        )
        
        for plate_info in extracted_plates:
            plate_text = get_text_from_gemini(plate_info['path'], api_key)
            results.append({
                'nom_plaque': plate_info['filename'],
                'texte_detecte': plate_text,
                'image_source': image_file
            })
            plate_counter += 1
            
    # Mettre à jour la barre de progression à la fin
    if progress_callback:
        progress_callback(1.0)

    return results, plates_output_dir

def create_excel_file(results):
    """
    Crée un fichier Excel avec les résultats
    """
    df = pd.DataFrame(results)
    excel_path = os.path.join(tempfile.gettempdir(), f"dataset_ocr_{int(time.time())}.xlsx")
    df.to_excel(excel_path, index=False)
    return excel_path

def main():
    st.set_page_config(
        page_title="Générateur Dataset OCR",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Générateur de Dataset OCR avec Gemini (Format YOLO)")
    st.markdown("---")

    st.markdown("""
    ### 📋 Instructions :
    1. **Entrez votre clé API Gemini**
    2. **Uploadez vos images** (dossier ZIP des frames extraites)  
    3. **Uploadez vos annotations YOLO** (dossier ZIP des fichiers .txt au format YOLO)
    4. **Lancez le traitement** pour générer le dataset OCR

    ⚠️ **Format YOLO requis :** `0 0.507414 0.693981 0.240527 0.028704`
    """)

    with st.sidebar:
        st.header("🔑 Configuration API")
        api_key = st.text_input(
            "Clé API Gemini",
            type="password",
            help="Obtenez votre clé sur https://makersuite.google.com/app/apikey"
         )
        st.markdown("---")
        st.markdown("""
        ### 📝 Format YOLO
        Chaque fichier .txt doit contenir :
        ```
        0 x_center y_center width height
        ```
        """)
        if api_key:
            st.success("✅ Clé API configurée")
        else:
            st.warning("⚠️ Clé API requise")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📁 Images")
        images_zip = st.file_uploader(
            "Uploadez le ZIP contenant les images",
            type=['zip'],
            key="images"
        )
    with col2:
        st.subheader("🏷️ Annotations YOLO")
        annotations_zip = st.file_uploader(
            "Uploadez le ZIP contenant les fichiers .txt (YOLO)",
            type=['zip'],
            key="annotations",
            help="Format: 0 x_center y_center width height"
        )

    if api_key and images_zip and annotations_zip:
        if st.button("🚀 Générer le Dataset OCR", type="primary", use_container_width=True):
            images_temp_dir = tempfile.mkdtemp(prefix="images_")
            annotations_temp_dir = tempfile.mkdtemp(prefix="annotations_")
            
            try:
                with st.spinner("📦 Extraction des fichiers..."):
                    with zipfile.ZipFile(images_zip, 'r') as zip_ref:
                        zip_ref.extractall(images_temp_dir)
                    with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
                        zip_ref.extractall(annotations_temp_dir)
                
                st.info("🤖 Traitement avec Gemini en cours...")
                progress_bar = st.progress(0)
                
                def update_progress(progress):
                    progress_bar.progress(progress)
                
                results, plates_dir = process_dataset(
                    images_temp_dir, 
                    annotations_temp_dir, 
                    api_key,
                    update_progress
                )
                
                if results:
                    st.success(f"✅ {len(results)} plaques traitées avec succès!")
                    st.subheader("📊 Résultats du traitement")
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                    
                    # ... (le reste de la logique d'affichage est identique)

                else:
                    st.error("❌ Aucune plaque n'a pu être extraite. Vérifiez que les noms des images et des fichiers .txt correspondent et que les annotations ne sont pas vides.")
            
            except Exception as e:
                st.error(f"❌ Erreur lors du traitement: {str(e)}")
    else:
        missing_items = []
        if not api_key: missing_items.append("🔑 Clé API Gemini")
        if not images_zip: missing_items.append("📁 Fichier ZIP des images")
        if not annotations_zip: missing_items.append("🏷️ Fichier ZIP des annotations")
        
        if missing_items:
            st.warning(f"⚠️ Éléments manquants: {', '.join(missing_items)}")

# --- BONNE PRATIQUE ---
# S'assure que le code n'est exécuté que si le script est le programme principal.
if __name__ == "__main__":
    main()
