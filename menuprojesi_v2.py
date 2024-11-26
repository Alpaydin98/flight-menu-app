import base64
import streamlit as st
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from pdf2image import convert_from_path
from PIL import Image
import anthropic
import openai
import tempfile
import time
import json
import re

from dotenv import load_dotenv
import os

# Streamlit Secrets ile çevre değişkenlerini alıyoruz
openai.api_key = st.secrets["OPENAI_API_KEY"]
AZURE_OCR_ENDPOINT = st.secrets["AZURE_OCR_ENDPOINT"]
AZURE_OCR_KEY = st.secrets["AZURE_OCR_KEY"]
anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]


# Streamlit Başlık ve Ayarlar
cv_client = ComputerVisionClient(AZURE_OCR_ENDPOINT, CognitiveServicesCredentials(AZURE_OCR_KEY))
st.set_page_config(page_title="Dinamik Menü Analiz ve Seçim", page_icon="✈️", layout="wide")
# Claude Entegrasyonu
client = anthropic.Client(api_key=anthropic_api_key)



# OCR İşlevi
def azure_ocr(image_path):
    try:
        with open(image_path, "rb") as image_stream:
            read_response = cv_client.read_in_stream(image_stream, raw=True)
        operation_location = read_response.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]

        while True:
            result = cv_client.get_read_result(operation_id)
            if result.status not in [OperationStatusCodes.not_started, OperationStatusCodes.running]:
                break
            time.sleep(1)

        extracted_text = ""
        if result.status == OperationStatusCodes.succeeded:
            for page in result.analyze_result.read_results:
                for line in page.lines:
                    extracted_text += line.text + "\n"
        return extracted_text
    except Exception as e:
        st.error(f"OCR işlemi sırasında bir hata oluştu: {e}")
        return ""

# Prompt Oluşturma (Pattern Kuralları Dahil)
def create_pattern_prompt(menu_text):
    pattern_prompt = """
    Sana vereceğim her bir menüde hem İngilizce hem Türkçe versiyonu olacak şekilde inişten önce ve inişte sonra olarak iki menü var. İki menü için de ayrı ayrı patternları incelerken İngilizce versiyonu hesaba katabilirsin. Burada bazı bağlaçlar var ve bu bağlaçlar hangi besinleri aynı anda seçip seçemeyeceğimizi belirtiyor. Senin işini kolaylaştırmak ve olayı sana anlatmak adına bazı patternları sana örnek olarak vereceğim. Gereksiz sözcükleri silebilirsin.
    

Pattern 1: 

besin1
or
besin2
and/or
besin3

Yukarıdakine benzer bir pattern varsa burada besin 1 veya 2'den sadece biri seçilecek ve bunlardan bağımsız olarak besin 3 tek başına alınabilecektir.

Konuyla ilgili sana örnek vereyim:
Örnek1:
Lütfen seçim yapınız:
künefe
or
kebap
and/or
çorba
(geri kalan besiinler önemli değil.)

Yukarıda verdiğim örnekte künefe veya kebaptan biri seçilebilir onlardan bağımsız çorba da seçilebilir. Burada "lütfen seçim yapınız" gibi seçim yaptırmaya yönelik bir yazı olmasına da gerek yok.Bağımsız seçilen ürünleri type'ı optional olacak şekilde ayrı göster.

Pattern 2:

Lütfen seçim yapınız gibi bir ibare varsa "or" genel bir menüyü ya da yemek grubunu birbirinden ayırabilir. Eğer bir menüyü ayırıyorsa "or" kelimesinin önünde ve ardında iki farklı menü vardır ve bizden bu menüden birini seçmemiz isteniyordur. Bu menülerde de kesinlikle bir tatlı bulunur.

Örnek 1: 
Lütfen seçim yapınız:
makarna
fasulye yemeği 
künefe 
or 
pilav 
köfte 
sütlaç

Olması gereken çıktı:
Menü 1: makarna, fasulye yemeği, künefe Menü 2: pilav, köfte, sütlaç

Örnek 2:

domatesli makarna 
bezelye yemeği
puding 
or 
erişte makarnası 
nohut yemeği 
profiterol

Olması gereken çıktı:
Grup 1: domatesli makarna, bezelye yemeği, puding  Grup 2: erişte makarnası, nohut yemeği, profiterol

Pattern 3
Eğer "or" öncesindeki grup menüde tatlı yoksa muhtemelen tatlıyı sonda bu iki seçenek için de sunmuşlardır. Bunu her iki grup menüye de dahil edebilirsin. Çünkü bu tür seçimlerde tatlı almak zorunludur.

Örnek:
Lütfen seçim yapınız:
Domatesli makarna
Nohut Yemeği
or
Dana bonfile
Buharda pilav
Puding

Olması gereken çıktı:
Grup 1: Domatesli makarna, Nohut Yemeği, puding  Grup 2: Dana bonfile, Buharda pilav, puding

Pattern 4 :

Eğer ilgili kelime küçük harflerden de oluşuyorsa onun içinde bulunduğu cümle; 

-gereksiz bir ifade olabilir 
veya 
-besini tarif eden bir yazı olabilir 

Dolayısıyla küçük harf kullanılmış kelimeleri ve bağlı olduğu cümleyi teker teker ayrıca değerlendirmen lazım. Besini tarif eden bir yazı ise tarif ettiği besinin hemen yanına  parantez içerisinde yazdır. Türkçe tarifse besinin Türkçe adının yanına, İngilizce tarifse besinin İngilizce adının yanına yazacaksın. Eğer tarif değilse gereksiz bir ifade olarak kabul edebilirsin.

Örnek:
DANA STROGANOFF
BEEF STROGANOFF
Sote kabak ve kırmızı biber, patates graten
sautéed zucchini and red pepper, potatoes gratin


Olması gereken çıktı:
DANA STROGANOFF (Sote kabak ve kırmızı biber, patates graten)
BEEF STROGANOFF (sautéed zucchini and red pepper, potatoes gratin)

Pattern 5:

Eğer ilgili besin aralarında and, or veya seçim yapınız gibi seçim yapmaya teşvik edici bir cümle yoksa yoksa ilgili bağlama göre o besinlerin hepsi seçilir. 

Örnek: 
besin 1 
besin 2 
besin 3 
besin 4 
. . 
besin 5 

burada besin 1 besin 2 besin 3 .... tüm besinler seçilebilir.

Her bir kategori için JSON formatında şöyle bir çıktı oluştur ve bunu iki dil için de yap:
İnişten önce yazısından önce menü 1, sonraki yazılar menü 2 olacak.
    
     "menü 1  / menü 2": {
        "Besin seçimi / Seçenek Seçimi": [
            {
                "dil": "Türkçe",
                "name": "Başlangıçlar veya Seçenek 1 (Eğer besin seçimi yapacaksak kategori adı, menü veya besin grubu seçimi yapacaksak seçenek numarasını yazdır)",
                "type": "optional/single/multiple (Ürün bağımsız seçilebiliyorsa optional yap.)",
                "items": [
                    "KARİDESLİ SEBZE SALATASI",
                    "MÜTEBBEL" (Hangi ürünlerden seçim yapılabiliyorsa onu yazdır.)
                ],
                "rules": "Her iki başlangıç da seçilebilir"
            },
        
            
        
            
            
    Eğer menüler yada besin grubu arasında seçim yapılacaksa Seçenek Seçimi eğer Besin seçimi arasında seçim yapılacaksa Besin seçimi keyine koy. Bağımsız seçenekler type optional olacak şekilde  ayrı bir json oluştur.Ekmek vb. pastane ürünleri bazen verilebiliyor. Ona da dikkat et.Ayrıca bazen dilenilen zamanda ikramda bulunulabiliyor. Onları da name kısmına "Dilediğiniz zaman" yazarak verebilirsin.
    Lütfen bana Türkçe bir dict format olarak döndür.
    Pattern 2 ve 3 karıştırılabilir iyi analiz et.
    Tüm patternları göz önünde bulundur.
    Şimdi aşağıdaki menüyü analiz et:
    """
    return pattern_prompt + menu_text


import json
import re

import re
import json

import re
import json


# Menü Analizi
def analyze_menu_with_openai(text):
    
    try:
        prompt = create_pattern_prompt(text)
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            temperature=0.5,  
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
       
        
     # Convert response to string and extract JSON
        response_text = str(response.content)
        # OpenAI Prompt
        # Prompt
        prompt = f"""
        Read the raw response I sent you, and using the relevant food names from the menu below, create dict formats in both Turkish and English one below the other.
        
        This is the menu:
        
        {text}
        
        This is the Turkish output:
        
        {response.content}
        
        Bunu hem Türkçe hem de İngilizce versiyonlarını yalnızca bir JSON formatında olacak şekilde  language key'i de oluşturarak bana ver.
        
        Örnek:
            
        {{
    "Turkish": {{
        "menu 1": {{
            "Seçenek Seçimi": [
                {{
                    "name": "Seçenek 1",
                    "type": "single",
                    "items": [
                        "HUMUS",
                        "MEVSİM SALATASI",
                        "DANA STROGANOFF (Sote kabak ve kırmızı biber, patates graten)",
                        "ÇİKOLATALI KEK"
                    ],
                    "rules": "Bu menüden tüm ürünler birlikte seçilmelidir"
                }},
                {{
                    "name": "Seçenek 2",
                    "type": "single",
                    "items": [
                        "HAVUÇLU ERİŞTE SALATASI",
                        "MEVSİM SALATASI",
                        "KESTANELİ TAVUK YAHNİ (Çin lahanası, sebzeli erişte)",
                        "HİNDİSTAN CEVİZİ SÜTLÜ VE MEYVELİ SAGO PUDING"
                    ],
                    "rules": "Bu menüden tüm ürünler birlikte seçilmelidir"
                }}
            ]
        }}
    }},
    "English": {{
        "menu 1": {{
            "Option Selection": [
                {{
                    "name": "Option 1",
                    "type": "single",
                    "items": [
                        "HUMMUS",
                        "GARDEN FRESH SALAD",
                        "BEEF STROGANOFF (sautéed zucchini and red pepper, potatoes gratin)",
                        "CHOCOLATE TRUFFLE CAKE"
                    ],
                    "rules": "All items in this menu must be selected together"
                }},
                {{
                    "name": "Option 2",
                    "type": "single",
                    "items": [
                        "NOODLE SALAD WITH CARROT",
                        "GARDEN FRESH SALAD",
                        "CHICKEN STEW WITH CHESTNUTS (kale, vegetable egg noodle)",
                        "FRUIT & COCONUT MILK SAGO PUDDING"
                    ],
                    "rules": "All items in this menu must be selected together"
                }}
            ]
        }}
    }}
}}
        
        """
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
            {"role": "system", "content": "You are an assistant that extracts and formats JSON data."},
            {"role": "user", "content": prompt}
        ],
            max_tokens=2000,
            temperature=0.2,
        )
        categorized_text = response['choices'][0]['message']['content']
        cleaned_text = re.sub(r'^```json\n|```$', '', categorized_text.strip(), flags=re.MULTILINE)
        menu_dict = json.loads(cleaned_text)
        create_menu_ui(menu_dict)
        
            
    except Exception as e:
        st.error(f"Menu analysis error: {e}")
        st.write("Response object:", response)
        return None

import streamlit as st
import json

# Dinamik Menü UI Fonksiyonu
def create_menu_ui(menu_data):
    st.title("Dinamik Menü ve Chatbot")

    # Dil Seçimi
    language_options = list(menu_data.keys())
    selected_language = st.selectbox("Lütfen bir dil seçin:", language_options)
    selected_language_data = menu_data[selected_language]

    # Menü Seçimi
    menu_options = list(selected_language_data.keys())
    selected_menu = st.selectbox("Lütfen bir menü seçin:", menu_options)
    selected_menu_data = selected_language_data[selected_menu]

    # Seçim Türünü Dinamik Olarak Belirle (Büyük/Küçük Harf Duyarlılığını Kaldır)
    possible_keys = ["besin seçimi", "seçenek seçimi", "food selection", "option selection"]
    selection_type = None
    for key in possible_keys:
        for actual_key in selected_menu_data.keys():
            if actual_key.lower() == key:
                selection_type = actual_key  # Gerçek anahtarı seçiyoruz
                break
        if selection_type:
            break

    if not selection_type:
        st.error("Uygun bir seçim türü bulunamadı!")
        return

    # Seçimler için veri işleme
    selections = {}
    for category in selected_menu_data[selection_type]:
        st.subheader(category["name"])
        items = category["items"]

        if category["type"] == "single":
            selected_item = st.radio(f"{category['name']} seçiniz:", items)
            selections[category["name"]] = selected_item
        elif category["type"] == "multiple":
            selected_items = []
            for item in items:
                if st.checkbox(item):
                    selected_items.append(item)
            selections[category["name"]] = selected_items
        elif category["type"] == "optional":
            optional_items = []
            for item in items:
                if st.checkbox(item):
                    optional_items.append(item)
            selections[category["name"]] = optional_items

    # Kullanıcı Seçimlerini Göster
    st.write("### Seçimleriniz:")
    for category, items in selections.items():
        if isinstance(items, list):
            st.write(f"**{category}:** {', '.join(items) if items else 'Seçilmedi'}")
        else:
            st.write(f"**{category}:** {items if items else 'Seçilmedi'}")

    # Chatbot Entegrasyonu
    st.subheader("Chatbot'a Sorular Sorun")
    user_message = st.text_input("Sorunuzu yazın:")
    
    if user_message:
        # Prompt oluşturma
        prompt = f"""
        Kullanıcı seçimleri:
        {selections}
    
        Kullanıcının sorusu:
        {user_message}
    
        Lütfen seçimlere dayanarak ve kullanıcının sorusunu dikkate alarak uygun bir cevap verin.
        """
        # OpenAI API çağrısı
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant that provides information based on menu selections."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.5,
        )
        chatbot_response = response['choices'][0]['message']['content']
        st.write(f"**Chatbot Cevabı:** {chatbot_response}")

# Camera input with fullscreen support (HTML/JS integration)
def camera_input_with_fullscreen():
    camera_html = """
    <div style="text-align: center;">
        <video id="video" autoplay style="width: 100%; max-height: 80vh;"></video>
        <button id="capture" style="margin-top: 20px; padding: 10px 20px; font-size: 18px;">Fotoğraf Çek</button>
        <canvas id="canvas" style="display: none;"></canvas>
    </div>
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const captureButton = document.getElementById('capture');

        // Start the camera
        navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
            video.srcObject = stream;
        });

        // Capture photo
        captureButton.addEventListener('click', () => {
            const context = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
            const imageData = canvas.toDataURL('image/jpeg');
            const link = document.createElement('a');
            link.href = imageData;
            link.download = 'captured_photo.jpg';
            link.click();
        });
    </script>
    """
    components.html(camera_html, height=600)




# Photo upload or capture section
st.subheader("Fotoğraf Yükle veya Mobil Kameranızı Kullanarak Çekin")

# 1. File Upload
uploaded_file = st.file_uploader("PDF veya Görüntü Dosyanızı Yükleyin", type=["pdf", "png", "jpg", "jpeg"])

# 2. Camera Capture Button
camera_triggered = st.button("📷 Fotoğraf Çek")
if camera_triggered:
    st.info("Kameranızı kullanarak fotoğraf çekmek için izin verin.")
    camera_input_with_fullscreen()

# Process images
images = []
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_pdf_path = temp_file.name
        pdf_pages = convert_from_path(temp_pdf_path, dpi=300)
        images.extend(pdf_pages)
    elif uploaded_file.type in ["image/png", "image/jpeg"]:
        image = Image.open(uploaded_file)
        images.append(image)

# OCR and output display
if images:
    st.subheader("OCR İşlemi ve Menü Analizi")
    extracted_text = ""
    for img in images:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image_file:
            img.save(temp_image_file.name)
            extracted_text += azure_ocr(temp_image_file.name) + "\n"

    if extracted_text:
        st.success("OCR işlemi tamamlandı!")
        st.text_area("OCR Çıktısı", value=extracted_text, height=300)
else:
    st.info("Lütfen bir dosya yükleyin veya '📷 Fotoğraf Çek' butonunu kullanarak fotoğraf çekin.")
