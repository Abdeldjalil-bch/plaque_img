import streamlit as st
import os
import cv2
import pandas as pd
import zipfile
import tempfile
import google.generativeai as genai
from PIL import Image
import time

# --- FONCTIONS DE TRAITEMENT ---

def parse_yolo_annotation(txt_path, img_width, img_height):
    """
    Parse un fichier .txt au format YOLO et retourne les coordonnées des bounding boxes.
    Format YOLO: class_id x_center y_center width height (valeurs normalisées 0-1).
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
                    # On s'assure de ne pas planter si la ligne est malformée
                    try:
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
                        
                        # Calculer les coordonnées des coins (xmin, ymin, xmax, ymax)
                        xmin = int(x_center_px - width_px / 2)
                        ymin = int(y_center_px - height_px / 2)
                        xmax = int(x_center_px + width_px / 2)
                        ymax = int(y_center_px + height_px / 2)
                        
                        boxes.append({
                            'class_id': class_id,
                            'xmin': max(0, xmin),
                            'ymin': max(0, ymin),
                            'xmax': min(img_width, xmax),
                            'ymax': min(img_height, ymax)
                        })
                    except (ValueError, IndexError):
                        print(f"Ligne ignorée dans {txt_path}: format invalide.")
    except Exception as e:
        print(f"Erreur lors du parsing de {txt_path}: {e}")
    return boxes

def extract_plates_from_image(image_path, txt_path, output_dir, base_name_prefix):
    """
    Extrait les plaques d'une image en utilisant les annotations YOLO (.txt).
    """
    image = cv2.imread(image_path)
    if image is None:
        st.warning(f"Impossible de lire l'image : {os.path.basename(image_path)}")
        return []

    img_height, img_width = image.shape[:2]
    boxes = parse_yolo_annotation(txt_path, img_width, img_height)

    if not boxes:
        return []

    extracted_plates = []
    for i, box in enumerate(boxes):
        # Découper la plaque de l'image originale
        plate_roi = image[box['ymin']:box['ymax'], box['xmin']:box['xmax']]
        
        if plate_roi.size > 0:
            # Générer un nom de fichier unique pour la plaque extraite
            plate_filename = f"{base_name_prefix}_{i:03d}.jpg"
            plate_path = os.path.join(output_dir, plate_filename)
            
            # Sauvegarder l'image de la plaque
            cv2.imwrite(plate_path, plate_roi)
            extracted_plates.append({
                'filename': plate_filename,
                'path': plate_path,
                'box': box
            })
    return extracted_plates

def get_text_from_gemini(image_path, api_key):
    """
    Utilise l'API Gemini pour extraire le texte d'une image de plaque.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        img = Image.open(image_path)
        
        prompt = """
        Analyse cette image d'une plaque d'immatriculation.
        Extrais UNIQUEMENT les lettres et les chiffres visibles sur la plaque.
        Ta réponse doit contenir uniquement les caractères de la plaque, sans aucun mot, espace ou explication.
        Si la plaque est illisible ou vide, réponds "ILLEGIBLE".
        """
        
        response = model.generate_content([prompt, img], request_options={"timeout": 60})
        
        # Nettoyage de la réponse pour ne garder que les caractères alphanumériques
        text = response.text.strip().upper()
        text = ''.join(filter(str.isalnum, text))
        
        return text if text else "ILLEGIBLE"
        
    except Exception as e:
        st.error(f"Erreur avec l'API Gemini : {str(e)}")
        return "ERROR"

def process_dataset(images_folder, annotations_folder, api_key, progress_callback=None):
    """
    Traite l'ensemble du dataset : trouve les correspondances, extrait les plaques et lance l'OCR.
    """
    plates_output_dir = tempfile.mkdtemp(prefix="extracted_plates_")
    results = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    # ==================================================================
    # DÉBUT DE LA CORRECTION DE L'INDENTATION
    # La boucle 'for' est maintenant correctement alignée avec le reste du code de la fonction.
    # ==================================================================
    for file in os.listdir(images_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            base_name = os.path.splitext(file)[0]
            txt_file = base_name + '.txt'
            txt_path = os.path.join(annotations_folder, txt_file)
            
            # On vérifie que le fichier d'annotation correspondant existe
            if os.path.exists(txt_path):
                image_files.append((file, txt_file))

    total_files = len(image_files)
    if total_files == 0:
        return [], plates_output_dir

    plate_counter = 1
    for idx, (image_file, txt_file) in enumerate(image_files):
        if progress_callback:
            progress_callback((idx + 1) / total_files)
        
        image_path = os.path.join(images_folder, image_file)
        txt_path = os.path.join(annotations_folder, txt_file)
        
        # Utiliser un préfixe basé sur le compteur pour garantir l'unicité
        base_name_prefix = f"plaque{plate_counter:04d}"
        extracted_plates = extract_plates_from_image(
            image_path, txt_path, plates_output_dir, base_name_prefix
        )
        
        for plate_info in extracted_plates:
            plate_text = get_text_from_gemini(plate_info['path'], api_key)
            
            results.append({
                'nom_plaque': plate_info['filename'],
                'texte_detecte': plate_text,
                'image_source': image_file
            })
        
        # Incrémenter le compteur principal seulement si des plaques ont été trouvées
        if extracted_plates:
            plate_counter += 1

    return results, plates_output_dir

def create_excel_file(results):
    """
    Crée un fichier Excel en mémoire avec les résultats.
    """
    df = pd.DataFrame(results)
    output = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    df.to_excel(output.name, index=False)
    return output.name

def create_zip_file(folder_path):
    """
    Crée une archive ZIP en mémoire à partir d'un dossier.
    """
    zip_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    with zipfile.ZipFile(zip_path.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
    return zip_path.name

# --- INTERFACE STREAMLIT ---

def main():
    st.set_page_config(
        page_title="Générateur Dataset OCR",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Générateur de Dataset OCR avec Gemini (Format YOLO)")
    st.markdown("---")

    # Colonne de configuration à gauche
    with st.sidebar:
        st.header("🔑 Configuration")
        api_key = st.text_input(
            "Clé API Gemini",
            type="password",
            help="Obtenez votre clé sur https://makersuite.google.com/app/apikey"
         )
        
        if api_key:
            st.success("✅ Clé API configurée")
        else:
            st.warning("⚠️ Clé API requise pour continuer")
        
        st.markdown("---")
        st.info("Cette application extrait les plaques d'immatriculation de vos images, utilise l'IA de Google pour lire le texte, et génère un dataset prêt à l'emploi.")

    # Zone principale
    st.header("1. Uploadez vos fichiers")
    col1, col2 = st.columns(2)
    with col1:
        images_zip = st.file_uploader(
            "Uploadez un fichier ZIP contenant vos images",
            type=['zip'],
            key="images"
        )
    with col2:
        annotations_zip = st.file_uploader(
            "Uploadez un fichier ZIP contenant vos annotations (.txt)",
            type=['zip'],
            key="annotations",
            help="Les noms des fichiers .txt doivent correspondre aux noms des images."
        )

    st.markdown("---")

    # Bouton de lancement et traitement
    if st.button("🚀 Lancer le traitement", type="primary", use_container_width=True, disabled=not (api_key and images_zip and annotations_zip)):
        
        images_temp_dir = tempfile.mkdtemp(prefix="images_")
        annotations_temp_dir = tempfile.mkdtemp(prefix="annotations_")
        
        try:
            with st.spinner("📦 Extraction des fichiers..."):
                with zipfile.ZipFile(images_zip, 'r') as zip_ref:
                    zip_ref.extractall(images_temp_dir)
                with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
                    zip_ref.extractall(annotations_temp_dir)
            
            st.info("🤖 Traitement en cours... L'appel à l'API Gemini peut prendre du temps.")
            progress_bar = st.progress(0, text="Analyse des images...")
            
            def update_progress(progress):
                progress_bar.progress(progress, text=f"Analyse des images... {int(progress*100)}%")
            
            results, plates_dir = process_dataset(
                images_temp_dir, 
                annotations_temp_dir, 
                api_key,
                update_progress
            )
            
            if results:
                st.success(f"✅ Traitement terminé ! {len(results)} plaques ont été analysées.")
                
                # Affichage des résultats
                st.header("📊 Résultats")
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                # Création des fichiers pour le téléchargement
                with st.spinner("Génération des fichiers de téléchargement..."):
                    excel_path = create_excel_file(results)
                    plates_zip_path = create_zip_file(plates_dir)

                st.header("📥 Téléchargements")
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    with open(excel_path, 'rb') as f:
                        st.download_button(
                            label="📥 Télécharger le fichier Excel",
                            data=f,
                            file_name="dataset_ocr_resultats.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                with dl_col2:
                    with open(plates_zip_path, 'rb') as f:
                        st.download_button(
                            label="📥 Télécharger les images des plaques",
                            data=f,
                            file_name="dataset_images_plaques.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
            else:
                st.error("❌ Aucune plaque n'a pu être extraite. Vérifiez que les noms de vos images et de vos fichiers .txt correspondent et que les fichiers d'annotation ne sont pas vides.")
        
        except Exception as e:
            st.error(f"❌ Une erreur inattendue est survenue : {str(e)}")
            # Affiche plus de détails pour le débogage
            st.exception(e)

    else:
        if not (api_key and images_zip and annotations_zip):
            st.warning("Veuillez fournir la clé API et les deux fichiers ZIP pour activer le traitement.")

# ==================================================================
# BONNE PRATIQUE :
# S'assure que la fonction main() est appelée uniquement lorsque le script est exécuté directement.
# ==================================================================
if __name__ == "__main__":
    main()
