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
   Örnek:
       
"Türkçe" : {{
    "menü 1  / menü 2": {{
        "Besin seçimi ": [
            {{
                "name": "Kategori adı (Eğer besin seçimi yapacaksak kategori adı, menü veya besin grubu seçimi yapacaksak seçenek numarasını yazdır)",
                "type": "optional/single/multiple (Ürün bağımsız seçilebiliyorsa optional yap.)",
                "items": [
                    "KARİDESLİ SEBZE SALATASI",
                    "MÜTEBBEL" (Hangi ürünlerden seçim yapılabiliyorsa onu yazdır.)
                ],
                "rules": "Her iki başlangıç da seçilebilir"
            }}
                "name": "Seçenek",
                "type": "seçenek" (Ürünler toplu halde seçenek olarak sunuluyorsa type seçenek yap.)",
                "items": [
                    "Seçenek 1": [
                                "Salata",
                                "IZGARA TAVUK GÖĞUS",
                                "DOMATES SOSLU PENNE MAKARNA"(Sote kabak, baharatlı tereyağı sos)
                            ],
                    "Seçenek 2": [
                                "SEBZELİ BASA BALIĞI",
                                "Buharda pilav"
                            ]
                ],
                "rules": "Seçeneklerden biri seçilebilir."
    }}    

 }}
            
            veya
          
        "Türkçe" : {{
    "menü 1  / menü 2": {{
                "name": "Seçenek",
                "type": "seçenek" (Ürünler toplu halde seçenek olarak sunuluyorsa type seçenek yap.)",
                "items": [
                    "Seçenek 1": [
                                "Salata",
                                "IZGARA TAVUK GÖĞUS",
                                "DOMATES SOSLU PENNE MAKARNA"(Sote kabak, baharatlı tereyağı sos)
                            ],
                    "Seçenek 2": [
                                "SEBZELİ BASA BALIĞI",
                                "Buharda pilav"
                            ]
                ],
                "rules": "Seçeneklerden biri seçilebilir."
    }}    

 }}
        
        veya hiç seçenek olarak da sunulmayabilir.Birden fazla optional da olabilir.
        
           
    Eğer seçenekler arası seçim yapılacaksa type seçenek olacak items içerine seçenek 1 ve seçenek 2 koy ve içine o seçeneğin ürünlerini koy. Bağımsız seçenekler type optional olacak şekilde  ayrı bir json oluştur.Ekmek vb. pastane ürünleri bazen verilebiliyor. Ona da dikkat et.Ayrıca bazen dilenilen zamanda ikramda bulunulabiliyor. Onları da name kısmına "Dilediğiniz zaman" yazarak verebilirsin.
    Lütfen bana Türkçe bir dict format olarak döndür.
    Pattern 2 ve 3 karıştırılabilir iyi analiz et.
    Tüm patternları göz önünde bulundur.
    Şimdi aşağıdaki menüyü analiz et:
    """
    return pattern_prompt + menu_text


import json
import re


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
       
        Bunu hem Türkçe hem de İngilizce versiyonlarını yalnızca bir JSON formatında olacak şekilde oluşturarak bana ver.
        
        İstenilen format:
        
        "Turkish" : {{
    "menü 1  / menü 2": {{
        "Besin seçimi ": [
            {{
                "name": "Başlangıçlar veya Seçenek 1 (Eğer besin seçimi yapacaksak kategori adı, menü veya besin grubu seçimi yapacaksak seçenek numarasını yazdır)",
                "type": "optional/single/multiple (Ürün bağımsız seçilebiliyorsa optional yap.)",
                "items": [
                    "KARİDESLİ SEBZE SALATASI",
                    "MÜTEBBEL" (Hangi ürünlerden seçim yapılabiliyorsa onu yazdır.)
                ],
                "rules": "Her iki başlangıç da seçilebilir"
            }}
                "name": "Başlangıçlar veya Seçenek 1 (Eğer besin seçimi yapacaksak kategori adı, menü veya besin grubu seçimi yapacaksak seçenek numarasını yazdır)",
                "type": "seçenek",
                "items": [
                    "Seçenek 1": ["Salata",
                                "IZGARA TAVUK GÖĞUS",
                                "DOMATES SOSLU PENNE MAKARNA"(Sote kabak, baharatlı tereyağı sos)
                            ],
                            "Seçenek 2": [
                                "SEBZELİ BASA BALIĞI",
                                "Buharda pilav"
                            ]
                ],
                "rules": "Her iki başlangıç da seçilebilir"
    }}    

 }}
            
            altına İngilizcesini yap
        
        "English" : {{.....
            
            }}
       
    
    
    Fakat şuna dikkat et: her iki dilde aynı çıktı,format almak istiyorum.
        """
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
            {"role": "system", "content": "You are an assistant that extracts and formats JSON data."},
            {"role": "user", "content": prompt}
        ],
            max_tokens=2000,
            temperature=0.2,
        )
        categorized_text = response.choices[0].message.content
        cleaned_text = re.sub(r'^```json\n|```$', '', categorized_text.strip(), flags=re.MULTILINE)
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


def create_menu_ui(menu_data: Dict):
    # Arayüz Başlığı
    st.title("Menü Seçim Arayüzü")
    
    # Dil seçimi
    language = st.selectbox("Dil Seçin / Choose Language", list(menu_data.keys()))
    
    # Seçilen dile göre menü yükleme
    menu = menu_data[language]
    
    # Menü seçimi
    menu_selection = st.selectbox("Bir Menü Seçin / Choose a Menu", list(menu.keys()))
    selected_menu = menu[menu_selection]
    
    # Seçimleri saklamak için bir dictionary
    selections = {}
    
    # Menü bölümlerini dinamik olarak işleme
    for section_key, section_content in selected_menu.items():
        st.header(section_key)  # Bölüm başlığı
    
        for section in section_content:
            st.subheader(section["name"])
            st.write(section["rules"])
    
            # "Seçenek" veya "Option" tipi (iç içe yapı)
            if section["type"] in ["seçenek", "choice", "option"] and isinstance(section["items"], dict):
                options = list(section["items"].keys())
                selected_option = st.radio(f"{section['name']} için bir seçim yapın:", options)
    
                # Seçilen seçeneği ve içeriğini sakla
                selections[section["name"]] = {
                    "type": "seçenek",
                    "selected": selected_option,
                    "items": section["items"][selected_option]
                }
    
            # Çoklu seçim (Multiple)
            elif section["type"] == "multiple":
                selected_items = st.multiselect(
                    f"{section['name']} için seçim yapın:", section["items"]
                )
                selections[section["name"]] = {
                    "type": "multiple",
                    "items": selected_items
                }
    
            # Tekli seçim (Single)
            elif section["type"] == "single":
                selected_item = st.radio(
                    f"{section['name']} için bir seçim yapın:", section["items"]
                )
                selections[section["name"]] = {
                    "type": "single",
                    "item": selected_item
                }
    
            # İsteğe bağlı seçim (Optional)
            elif section["type"] == "optional":
                selected_items = []
                for item in section["items"]:
                    if st.checkbox(item):
                        selected_items.append(item)
                selections[section["name"]] = {
                    "type": "optional",
                    "items": selected_items
                }
    
            # Desteklenmeyen tipler için bir uyarı
            else:
                st.warning(f"{section['name']} için geçersiz seçim tipi: {section['type']}")
    
    # Seçimlerin Özeti
    st.subheader("Seçim Özeti")
    for name, value in selections.items():
        st.markdown(f"### {name}")
        if value["type"] == "seçenek":
            st.write(f"**Seçilen Seçenek:** {value['selected']}")
            for item in value["items"]:
                st.write(f"- {item}")
        elif value["type"] in ["multiple", "optional"]:
            if value["items"]:
                for item in value["items"]:
                    st.write(f"- {item}")
            else:
                st.write("Hiçbir şey seçilmedi.")
        elif value["type"] == "single":
            st.write(f"**Seçilen:** {value['item']}")
        else:
            st.write("Geçersiz veri.")
    
    # Chatbot Entegrasyonu
    st.subheader("Chatbot'a Sorular Sorun")
    # Benzersiz bir key, dil ve menü seçimlerine göre oluşturulur
    unique_key = f"chatbot_input_{language}_{menu_selection}"
    user_message = st.text_input("Sorunuzu yazın:", key=unique_key)
    
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
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an assistant that provides information based on menu selections."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.5,
            )
            chatbot_response = response.choices[0].message.content
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
