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
    # Lire l'image
    image = cv2.imread(image_path)
    if image is None:
        return []
    
    img_height, img_width = image.shape[:2]
    
    # Parser le fichier YOLO
    try:
        boxes = parse_yolo_annotation(txt_path, img_width, img_height)
    except:
        return []
    
    extracted_plates = []
    
    for i, box in enumerate(boxes):
        # Extraire la région de la plaque
        plate_roi = image[box['ymin']:box['ymax'], box['xmin']:box['xmax']]
        
        if plate_roi.size > 0:
            # Générer le nom du fichier de la plaque
            plate_filename = f"{base_name}_{i:03d}.jpg"
            plate_path = os.path.join(output_dir, plate_filename)
            
            # Sauvegarder la plaque
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
        # Configurer l'API Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Charger l'image
        img = Image.open(image_path)
        
        # Prompt optimisé pour les plaques d'immatriculation
        prompt = """
        Cette image contient une plaque d'immatriculation. 
        Extrais UNIQUEMENT le texte/numéros visibles sur la plaque.
        Réponds avec seulement les caractères que tu vois, sans espaces ni formatage supplémentaire.
        Si tu ne peux pas lire clairement, réponds "ILLEGIBLE".
        """
        
        response = model.generate_content([prompt, img])
        
        # Nettoyer la réponse
        text = response.text.strip().upper()
        text = ''.join(text.split())  # Supprimer tous les espaces
        
        return text if text else "ILLEGIBLE"
        
    except Exception as e:
        st.error(f"Erreur avec l'API Gemini: {str(e)}")
        return "ERROR"

def process_dataset(images_folder, annotations_folder, api_key, progress_callback=None):
    """
    Traite l'ensemble du dataset : extraction + OCR
    """
    # Créer le dossier de sortie pour les plaques
    plates_output_dir = tempfile.mkdtemp(prefix="extracted_plates_")
    
    # Liste pour stocker les résultats
    results = []
    
    # Obtenir la liste des fichiers images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
            for file in os.listdir(images_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            # Vérifier qu'il existe un fichier .txt correspondant
            base_name = os.path.splitext(file)[0]
            txt_file = base_name + '.txt'
            txt_path = os.path.join(annotations_folder, txt_file)
            
            if os.path.exists(txt_path):
                image_files.append((file, txt_file))
    
    total_files = len(image_files)
    if total_files == 0:
        return [], plates_output_dir
    
    plate_counter = 1
    
    for idx, (image_file, txt_file) in enumerate(image_files):
        if progress_callback:
            progress_callback(idx / total_files)
        
        image_path = os.path.join(images_folder, image_file)
        txt_path = os.path.join(annotations_folder, txt_file)
        
        # Extraire les plaques de cette image
        base_name = f"plaque{plate_counter:04d}"
        extracted_plates = extract_plates_from_image(
            image_path, txt_path, plates_output_dir, base_name
        )
        
        # Pour chaque plaque extraite, obtenir le texte via Gemini
        for plate_info in extracted_plates:
            plate_text = get_text_from_gemini(plate_info['path'], api_key)
            
            results.append({
                'nom_plaque': plate_info['filename'],
                'texte_detecte': plate_text,
                'image_source': image_file
            })
            
            plate_counter += 1
    
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
    
    # Configuration API
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
        
        **Exemple :**
        ```
        0 0.507414 0.693981 0.240527 0.028704
        0 0.312456 0.445123 0.180234 0.032145
        ```
        
        - `0` = ID de classe (plaque)
        - Coordonnées normalisées (0-1)
        """)
        
        if api_key:
            st.success("✅ Clé API configurée")
        else:
            st.warning("⚠️ Clé API requise")
    
    # Zone d'upload
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
    
    # Vérifier que tout est prêt
    if api_key and images_zip and annotations_zip:
        if st.button("🚀 Générer le Dataset OCR", type="primary", use_container_width=True):
            
            # Créer des dossiers temporaires
            images_temp_dir = tempfile.mkdtemp(prefix="images_")
            annotations_temp_dir = tempfile.mkdtemp(prefix="annotations_")
            
            try:
                # Extraire les fichiers ZIP
                with st.spinner("📦 Extraction des fichiers..."):
                    # Extraire les images
                    with zipfile.ZipFile(images_zip, 'r') as zip_ref:
                        zip_ref.extractall(images_temp_dir)
                    
                    # Extraire les annotations
                    with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
                        zip_ref.extractall(annotations_temp_dir)
                
                # Traitement du dataset
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
                    
                    # Afficher les résultats
                    st.subheader("📊 Résultats du traitement")
                    
                    # Créer le DataFrame pour affichage
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                    
                    # Statistiques
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🔢 Total plaques", len(results))
                    with col2:
                        legible_count = len([r for r in results if r['texte_detecte'] not in ['ILLEGIBLE', 'ERROR']])
                        st.metric("✅ Lisibles", legible_count)
                    with col3:
                        illegible_count = len(results) - legible_count
                        st.metric("❌ Non lisibles", illegible_count)
                    
                    # Aperçu de quelques plaques extraites
                    st.subheader("🖼️ Aperçu des plaques extraites")
                    
                    plate_files = [f for f in os.listdir(plates_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    
                    if len(plate_files) > 0:
                        cols = st.columns(min(4, len(plate_files)))
                        for i, plate_file in enumerate(plate_files[:4]):
                            with cols[i]:
                                plate_path = os.path.join(plates_dir, plate_file)
                                st.image(plate_path, caption=plate_file, use_column_width=True)
                                
                                # Afficher le texte détecté correspondant
                                for result in results:
                                    if result['nom_plaque'] == plate_file:
                                        st.code(result['texte_detecte'])
                                        break
                    
                    # Créer les fichiers de téléchargement
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Excel avec les résultats
                        excel_path = create_excel_file(results)
                        with open(excel_path, 'rb') as excel_file:
                            st.download_button(
                                label="📥 Télécharger Excel",
                                data=excel_file.read(),
                                file_name=f"dataset_ocr_{int(time.time())}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                type="primary"
                            )
                    
                    with col2:
                        # ZIP avec toutes les images de plaques
                        plates_zip_path = os.path.join(tempfile.gettempdir(), f"plates_{int(time.time())}.zip")
                        with zipfile.ZipFile(plates_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                            for root, dirs, files in os.walk(plates_dir):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    arcname = os.path.relpath(file_path, plates_dir)
                                    zipf.write(file_path, arcname)
                        
                        with open(plates_zip_path, 'rb') as zip_file:
                            st.download_button(
                                label="📥 Télécharger Images",
                                data=zip_file.read(),
                                file_name=f"plates_images_{int(time.time())}.zip",
                                mime="application/zip",
                                type="primary"
                            )
                    
                    st.markdown("---")
                    st.markdown("""
                    ### 🎯 Dataset prêt pour l'entraînement !
                    - **Excel :** Contient les correspondances image ↔ texte
                    - **Images :** Toutes les plaques découpées prêtes pour PaddleOCR
                    
                    Vous pouvez maintenant utiliser ces données pour entraîner votre modèle OCR personnalisé.
                    """)
                
                else:
                    st.error("❌ Aucune plaque n'a pu être extraite. Vérifiez vos fichiers d'annotation.")
            
            except Exception as e:
                st.error(f"❌ Erreur lors du traitement: {str(e)}")
    
    else:
        # Instructions d'utilisation
        missing_items = []
        if not api_key:
            missing_items.append("🔑 Clé API Gemini")
        if not images_zip:
            missing_items.append("📁 Fichier ZIP des images")
        if not annotations_zip:
            missing_items.append("🏷️ Fichier ZIP des annotations (.txt YOLO)")
        
        if missing_items:
            st.warning(f"⚠️ Éléments manquants: {', '.join(missing_items)}")
        
        st.markdown("""
        <div style="text-align: center; padding: 50px; background-color: #f0f2f6; border-radius: 10px; margin: 20px 0;">
            <h3>📋 Instructions détaillées</h3>
            <p><b>1.</b> Obtenez une clé API Gemini (gratuite) sur <a href="https://makersuite.google.com/app/apikey">Google AI Studio</a></p>
            <p><b>2.</b> Annotez vos images avec <b>LabelImg en mode YOLO</b> (format .txt)</p>
            <p><b>3.</b> Créez deux fichiers ZIP : un pour les images, un pour les annotations .txt</p>
            <p><b>4.</b> Uploadez les fichiers et lancez le traitement</p>
            <hr>
            <h4>⚙️ Configuration LabelImg pour YOLO :</h4>
            <p>• Ouvrir LabelImg → View → Change Save Dir</p>
            <p>• View → Change Default Save Annotation to YOLO</p>
            <p>• Créer une classe "plaque" (ID = 0)</p>
            <p>• Annoter et sauvegarder au format .txt</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
