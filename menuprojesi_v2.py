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
        
        if 'menu_data' in st.session_state and not st.session_state.get('new_file_uploaded', False):
            return st.session_state.menu_data
        
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
            ],
            "Besin Seçimi": [
                {{
                    "name": "Başlangıçlar",
                    "type": "multiple",
                    "items": [
                        "NİSUAZ SALATA",
                        "MOZZARELLA VE IZGARA SEBZELER"
                    ],
                    "rules": "Her iki başlangıç da seçilebilir"
                }}
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
        st.write(categorized_text)
        menu_dict = json.loads(cleaned_text)
        st.session_state.new_file_uploaded = False  # Yeni dosya işlenmiş kabul edilir
        st.session_state.menu_data = menu_dict  # Menü verisini kaydet
        return menu_dict
    except Exception as e:
        st.error(f"Menü analizi sırasında bir hata oluştu: {e}")
        return {}
        
       

import streamlit as st
import json

import streamlit as st
from typing import Dict, List, Union

def render_option_based_menu(menu_data: Dict, key: str) -> Dict:
    """
    Seçenek bazlı menüyü render eder
    """
    selections = {}
    options = menu_data[key]
    option_names = [option["name"] for option in options]

    st.subheader("Menü Seçimi")
    selected_option = st.radio("Lütfen bir seçenek seçin:", option_names)

    if selected_option:
        selected_items = next(option["items"] for option in options if option["name"] == selected_option)
        st.write("### Seçilen Menü İçeriği:")
        for item in selected_items:
            st.write(f"- {item}")
        selections[selected_option] = selected_items

    return selections

def render_food_category(category: Dict, category_name: str) -> Union[List, str, Dict]:
    """
    Tekil bir yemek kategorisini render eder
    """
    st.subheader(category_name)
    st.caption(f"*{category['rules']}*")
    
    if category["type"] == "single":
        return render_single_selection(category, category_name)
    elif category["type"] in ["multiple", "optional"]:
        return render_multiple_selection(category, category_name)
    
    return None

def render_single_selection(category: Dict, category_name: str) -> Union[str, Dict]:
    """
    Tekli seçim tipindeki kategorileri render eder
    """
    # Eğer items içinde dict varsa ve birden fazla seçenek varsa
    if isinstance(category["items"][0], dict):
        options_dict = {}
        for item_dict in category["items"]:
            options_dict.update(item_dict)
        
        selected_option = st.radio(
            "Lütfen bir seçenek seçin:",
            list(options_dict.keys()),
            key=f"{category_name}_option"
        )
        
        st.write(f"**{selected_option} içeriği:**")
        for item in options_dict[selected_option]:
            st.write(f"- {item}")
        return {selected_option: options_dict[selected_option]}
    
    return st.radio(
        f"Lütfen bir {category_name.lower()} seçin:", 
        category["items"],
        key=f"{category_name}_single"
    )
    

def render_multiple_selection(category: Dict, category_name: str) -> List:
    """
    Çoklu seçim tipindeki kategorileri render eder
    """
    selected_items = []
    for item in category["items"]:
        if st.checkbox(item, key=f"{category_name}_{item}"):
            selected_items.append(item)
    return selected_items

def display_selections(selections: Dict):
    """
    Kullanıcı seçimlerini görüntüler
    """
    st.write("### Seçimleriniz:")
    for category, items in selections.items():
        if isinstance(items, dict):
            for option, selected_items in items.items():
                st.write(f"**{category} - {option}:** {', '.join(selected_items)}")
        elif isinstance(items, list):
            st.write(f"**{category}:** {', '.join(items) if items else 'Seçilmedi'}")
        else:
            st.write(f"**{category}:** {items if items else 'Seçilmedi'}")

def create_menu_ui(menu_data: Dict):
    """
    Ana menü UI bileşenini oluşturur
    """
    # Oturum durumunu kullanarak menü verilerini bir kez sakla
    if 'menu_data' not in st.session_state:
        st.session_state.menu_data = menu_data

    st.title("Dinamik Menü ve Chatbot")
    
    # Dil Seçimi
    language_options = list(st.session_state.menu_data.keys())
    selected_language = st.selectbox("Lütfen bir dil seçin:", language_options)
    selected_language_data = st.session_state.menu_data[selected_language]
    
    # Menü Seçimi
    menu_options = list(selected_language_data.keys())
    selected_menu = st.selectbox("Lütfen bir menü seçin:", menu_options)
    selected_menu_data = selected_language_data[selected_menu]
    
    # Menü tipini belirle
    menu_type_map = {
        "option": ("Seçenek Seçimi", "Option Selection"),
        "food": ("Besin seçimi", "Food Selection")
    }
    
    menu_type = None
    selected_key = None
    
    for m_type, keys in menu_type_map.items():
        for key in keys:
            if key in selected_menu_data:
                menu_type = m_type
                selected_key = key
                break
        if menu_type:
            break
    
    if not menu_type:
        st.error("Uygun bir menü türü bulunamadı!")
        return
    
    selections = {}
    
    # Menü tipine göre render
    if menu_type == "option":
        selections = render_option_based_menu(selected_menu_data, selected_key)
    else:  # food type
        categories = selected_menu_data[selected_key]
        for category in categories:
            result = render_food_category(category, category["name"])
            if result:  # Boş olmayan sonuçları ekle
                selections[category["name"]] = result
    
    # Seçimleri görüntüle
    if selections:
        display_selections(selections)
    
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
        try:
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
        except Exception as e:
            st.error(f"Chatbot cevabı alınamadı: {e}")



# Streamlit session_state kullanımı: Dosya yükleme durumu
if "menu_data" not in st.session_state:
    st.session_state.menu_data = None
if "new_file_uploaded" not in st.session_state:
    st.session_state.new_file_uploaded = False
if "capture_photo" not in st.session_state:
    st.session_state.capture_photo = False  # Kamera ekranını tetiklemek için flag

# Fotoğraf Yükleme ve Çekim Butonları
st.header("Fotoğraf Yükleme veya Çekim")

# 1. Dosya Yükleme
uploaded_file = st.file_uploader("PDF veya Görüntü Dosyanızı Yükleyin", type=["pdf", "png", "jpg", "jpeg"])

# 2. Fotoğraf Çekim
if st.button("Fotoğraf Çek"):
    st.session_state.capture_photo = True

# Eğer "Fotoğraf Çek" butonuna tıklanmışsa kamera girişini göster
if st.session_state.capture_photo:
    camera_photo = st.camera_input("Fotoğraf Çek")
    if camera_photo:
        st.session_state.new_file_uploaded = True
        st.session_state.capture_photo = False  # Kamera ekranını kapatmak için

# Görsellerin Toplanması
images = []

if uploaded_file:
    if "last_uploaded_file_name" not in st.session_state or st.session_state.last_uploaded_file_name != uploaded_file.name:
        st.session_state.new_file_uploaded = True
        st.session_state.last_uploaded_file_name = uploaded_file.name

    if uploaded_file.type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_pdf_path = temp_file.name
        images.append(temp_pdf_path)
    elif uploaded_file.type in ["image/png", "image/jpeg"]:
        image = Image.open(uploaded_file)
        st.image(image, caption="Yüklenen Görüntü", use_column_width=True)
        images.append(image)

if "camera_photo" in locals() and camera_photo:
    try:
        image = Image.open(camera_photo)
        st.image(image, caption="Kameradan Çekilen Görüntü", use_column_width=True)
        images.append(image)
    except Exception as e:
        st.error(f"Kameradan alınan görüntü işlenemedi: {e}")

# OCR İşlemi
if images and st.session_state.new_file_uploaded:
    st.subheader("OCR İşlemi Başlatılıyor")
    extracted_text = ""

    for img in images:
        if isinstance(img, str):
            extracted_text += azure_ocr(img) + "\n"
        else:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image_file:
                    img.save(temp_image_file.name)
                    extracted_text += azure_ocr(temp_image_file.name) + "\n"
            except Exception as e:
                st.error(f"Görüntü OCR işleminde hata oluştu: {e}")

    if extracted_text.strip():
        st.success("OCR işlemi başarıyla tamamlandı!")

        # Menü Analizi
        st.subheader("Menü Analizi")
        retry_count = 0
        max_retries = 3
        menu_analysis = None
        while retry_count < max_retries:
            try:
                menu_analysis = analyze_menu_with_openai(extracted_text)
                if menu_analysis:
                    st.success("Menü başarıyla analiz edildi!")
                    st.session_state.menu_data = menu_analysis
                    st.session_state.new_file_uploaded = False
                    break
            except Exception as e:
                retry_count += 1
                st.warning(f"Menü analizi sırasında hata oluştu. Tekrar deneniyor ({retry_count}/{max_retries})...")
        if not menu_analysis:
            st.error("Menü analizi başarısız oldu. Lütfen dosyanızı kontrol edin.")
    else:
        st.error("OCR işlemi başarısız. Lütfen dosyanızı kontrol edin.")

# Menü UI
if st.session_state.menu_data:
    create_menu_ui(st.session_state.menu_data)
else:
    st.info("Lütfen bir dosya yükleyin veya kamerayla fotoğraf çekin.")

