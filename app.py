import pandas as pd
from flask import Flask, render_template, request, redirect, flash
import requests
import json
import chardet
import csv

#bu kısım flask için web uygulaması açılsın diye
app = Flask(__name__)
app.secret_key = "çokgizli_anahtar"

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read(10000)
    result = chardet.detect(rawdata)
    encoding = result.get("encoding", "utf-8")
    if not encoding or encoding.lower() in ["ascii", "utf-16", "utf-32"]:
        encoding = "ISO-8859-1"
    print(f"{file_path} - Detected encoding: {encoding}")
    return encoding

high_end_gpus = (
    "Yüksek performanslı ekran kartları şunlardır: Nvidia RTX 3050, 3060, 4050, 4060, 4050 Ti, 2070, 2080. "
    "Laptop önerisi yapılırken bu ekran kartlarına sahip modellere öncelik verilmelidir."
)


cache = {}
datasets = {
    "mobilephone1": pd.read_csv("mobilesdataset.csv", encoding=detect_encoding("mobilesdataset.csv"), on_bad_lines='skip'),
    "tablet": pd.read_csv("gunceltablet.csv", encoding=detect_encoding("gunceltablet.csv"), on_bad_lines='skip'),
    "laptop_highend": pd.read_csv("laptops.csv", encoding=detect_encoding("laptops.csv"), on_bad_lines='skip'),
    "tech_turkiye": pd.read_csv("tehnology.csv", encoding=detect_encoding("tehnology.csv"), on_bad_lines='skip'),
    "headphone": pd.read_csv("headphones.csv", encoding=detect_encoding("headphones.csv"), on_bad_lines='skip'),
    "keyboard": pd.read_csv("keyboard.csv", encoding=detect_encoding("keyboard.csv"), on_bad_lines='skip'),
    "mice": pd.read_csv("mice.csv", encoding=detect_encoding("mice.csv"), on_bad_lines='skip'),
    "smart_home": pd.read_csv("smart_home.csv", encoding=detect_encoding("smart_home.csv"), on_bad_lines='skip'),
    "smartwatches": pd.read_csv("smartwatches.csv", encoding=detect_encoding("smartwatches.csv"), on_bad_lines='skip'),
    "printers": pd.read_csv("printers.csv", encoding=detect_encoding("printers.csv"), on_bad_lines='skip'),
    "webcams": pd.read_csv("webcams.csv", encoding=detect_encoding("webcams.csv"), on_bad_lines='skip'),
    "monitors": pd.read_csv("monitors.csv", encoding=detect_encoding("monitors.csv"), on_bad_lines='skip'),

}

def match_profession_to_needs(profession, needs):
    profession_needs_map = {
        "Fotoğrafçı": ["kamera", "tripod", "lens"],
        "Grafik Tasarımcı": ["laptop", "çizim tableti", "monitör"],
        "Web Geliştirici": ["laptop", "ekran kartı", "SSD disk"],
        "Doktor": ["telefon", "tablet"],
        "Müzisyen": ["mikrofon", "kulaklık", "ses kartı"],
    }

    user_keywords = [word.strip().lower() for word in needs.split(',') if word.strip()]
    mapped_keywords = profession_needs_map.get(profession, [])
    combined_needs = list(set(user_keywords + mapped_keywords))
    return combined_needs

category_synonyms = {
    "telefon": ["telefon", "smartphone", "cell phone"],
    "laptop": ["laptop", "notebook", "ultrabook"],
    "tablet": ["tablet", "ipad", "android tablet"],
    "kulaklık": ["kulaklık", "headphone", "earbud", "earphone"],
    "kamera": ["camera", "kamera", "dslr"],
}

def validate_input(value):
    if not value or value.strip() == "":
        return "Bu alan boş bırakılamaz"
    if value.lower() == "bütçe":
        try:
            float(value)
        except ValueError:
            return "Bütçe geçerli bir sayı olmalıdır"
    return None

def get_relevant_data(profession, needs, budget):
    relevant_products = []
    matched_needs = match_profession_to_needs(profession, needs)
    for category, df in datasets.items():
        if 'Price' in df.columns and 'category' in df.columns:
            try:
                df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
                for need in matched_needs:
                    alternatives = category_synonyms.get(need.lower(), [need])
                    for alt in alternatives:
                        filtered_df = df[
                            (df['Price'] <= float(budget)) &
                            (df['category'].str.contains(alt, case=False, na=False))
                        ]
                        relevant_products.extend(filtered_df.dropna(subset=['Price']).to_dict(orient="records"))
            except Exception as e:
                print(f"{category} verisinde filtreleme hatası: {e}")
    return relevant_products

def format_product_info(item):
    name = item.get('name', 'Ürün')
    price = item.get('Price', 'Fiyat Yok')
    description = item.get('description', 'Açıklama bulunamadı')

    fields_to_check = [
        'ram', 'RAM', 'RAM (GB)', 'memory',
        'işlemci', 'işlemci modeli', 'processor', 'cpu',
        'ekran kartı', 'ekran kartı modeli', 'gpu', 'graphics',
        'depola', 'storage', 'disk', 'SSD', 'HDD'
    ]

    details = []
    for field in fields_to_check:
        value = item.get(field)
        if value and str(value).strip() and str(value).lower() != "nan":
            details.append(f"{field}: {value}")

    if not description or description == 'Açıklama bulunamadı':
        description = f"{name} için açıklama mevcut değil."

    detail_text = "\n   - " + "\n   - ".join(details) if details else ""
    return f"{name} - {description} - {price} USD{detail_text}"

def ask_model(prompt, model="llama3.1:8b"):
    url = "http://127.0.0.1:11434/api/chat"
    headers = {"Content-Type": "application/json"}

    system_messages = [
        {
            "role": "system",
            "content": (
                "Sen, kullanıcıların mesleği, ihtiyaçları ve bütçesi doğrultusunda en uygun teknolojik ürünleri öneren uzman bir danışmansın. Kullanıcı sana mesleğini, teknolojik ihtiyaçlarını ve bütçesini belirtecek. Sen bu bilgilere dayanarak güncel, kaliteli ve fiyat-performans dengesi iyi olan teknolojik ürünleri önereceksin. "
                "Önerileri sistemine aynı kriterlere benzeyen promptlar girildiğinde önerilerini çeşitlendir"
                f"{high_end_gpus} "
                " 1. Laptop önerilerinde, özellikle 2024-2025 döneminin popüler ekran kartlarını dikkate alacaksın: Nvidia RTX 3050, RTX 4050, RTX 4060, RTX 4070 modellerini içeren güncel ve performans açısından dengeli laptopları önereceksin."
                "2. Telefon önerilerinde, son çıkan ve piyasada talep gören modelleri tercih edeceksin. Örneğin iPhone 15, iPhone 16, iPhone 16 Pro Max, Samsung S25 Ultra, Vivo, Xiaomi ve Huawei'nin 2024-2025 güncel modellerini dikkate alacaksın.ve önerilerini çeşitlendirmeye dikkat et "
                "3. Önerilerini verirken ürünlerin temel teknik özelliklerine, performansına, fiyat-performans dengesine ve kullanıcının meslek ve ihtiyaçlarına uygunluğuna dikkat edeceksin."
                "Yanıtını sadece numaralı madde listesi (1., 2., 3.) şeklinde alt alta olarak ver. "
                "Kullanıcı iyi bir tablet istediğinde ona apple ipad pro, samsung galaxy tab s9 ultra, huawei matepad 11,lenovo tab11, xiaomi redmi pad se 8 tarzı güçlü ve güncel tabletlerden öner. "
                "Öneriler dışında başka hiçbir şey yazma"
                "Girilen bütçenin üstünde ürün önerme"
                "Öneriler sırasında kullanıcı tarafından girilen meslek seçimine dikkat ederek öner eğer kendi mesleğine uygun bir ürün önerisi istiyorsa profesyonel modeller öner."
                "Eğer kullanıcı birden fazla teknolojik ürüne ihtiyacı varsa(örn: fotoğraf makinası ve telefon) mesleğine göre iki ürünü birleştirip beraber set olarak önerebilirsin , yeterli özelliklere sahip tek bir ürün de önerebilirsin"
                "Bütçe yüksek olunca olmayan özellikler ekleme doğruluktan sapma "
                "Her öneride mutlaka şunları belirt:\n"
                "- Ürün adı\n"
                "- Kısa açıklama (neden önerildiği)\n"
                "- Fiyat (USD cinsinden)\n"
                "- Detaylı teknik özellikler:\n"
                "   - Eğer laptop öneriyorsan: RAM (GB), işlemci modeli, ekran kartı modeli, depolama türü ve miktarı\n"
                "   - Eğer telefon öneriyorsan: Depolama, RAM, işlemci, ekran boyutu, batarya, kamera\n"
                "   - Eğer tablet öneriyorsan: RAM, depolama, ekran boyutu, işlemci, kullanım amacı\n"
            )
        }
    ]

    user_message = {"role": "user", "content": prompt}
    messages = system_messages + [user_message]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "top_p": 0.8
    }

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True)
        full_response = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    part_data = json.loads(line)
                    content = part_data.get("message", {}).get("content", "")
                    full_response += content
                except json.JSONDecodeError:
                    continue
        return full_response
    except Exception as e:
        return f"Hata oluştu: {str(e)}"

def get_cached_response(prompt):
    return cache.get(prompt)

def set_cached_response(prompt, response):
    cache[prompt] = response

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/compare', methods=['POST'])
def compare():
    selected_items = request.form.getlist('selected_products')  # checkboxtan  bu kısım geliyo
    if not selected_items:
        flash("Lütfen karşılaştırmak için en az bir ürün seçin.", "warning")
        return redirect(request.referrer or '/')

    prompt = f"""
Aşağıda seçilen teknolojik ürünleri detaylıca karşılaştır:
{chr(10).join(selected_items)}

Karşılaştırma sonucunda şunlara mutlaka değin:
- Ürünlerin Güçlü Yönleri ve Zayıf Yönleri Nelerdir
- Hangi Kullanıcıya Hitap Ediyor
- Fiyat-Performans Değerlendirmesi
- Hangi Meslek İçin Daha Uygun
bu 4 kritere göre kıyasla
Karşılaştırmayı madde madde yap.
"""
    response = ask_model(prompt)
    return render_template('comparison_result.html', response=response)

@app.route('/process', methods=['POST'])
def process():
    profession = request.form.get('profession')
    needs = request.form.get('needs')
    budget = request.form.get('budget')

    errors = []
    for field_name, field_value in zip(["Meslek", "İhtiyaçlar", "Bütçe"], [profession, needs, budget]):
        error = validate_input(field_value)
        if error:
            errors.append(f"{field_name}: {error}")

    if errors:
        return render_template('index.html', errors=errors)

    relevant_data = get_relevant_data(profession, needs, budget)
    relevant_info = "\n".join([format_product_info(item) for item in relevant_data]) or "Uygun ürün bulunamadı."

    prompt = f"""
Meslek: {profession}
İhtiyaçlar: {needs}
Bütçe: {budget} USD

Kullanıcı sadece {needs} ile ilgili ürün istiyor ve öneri sırasında {profession} kısmını da göz önünde bulundur. Lütfen yalnızca bu kategorilere odaklan. Örneğin sadece telefon istenmişse laptop önerme. Ama birden fazla ürüne ihityacı varsa bunu kombin şeklinde verebilirsin.

Aşağıdaki ürünleri göz önünde bulundurarak, 8 öneri oluştur:

Uygun ürünler:
{relevant_info}
"""

    cached_response = get_cached_response(prompt)
    if cached_response:
        return render_template('result.html', response=cached_response)

    model_response = ask_model(prompt)
    set_cached_response(prompt, model_response)

    return render_template('result.html', response=model_response)

@app.route('/feedback', methods=['POST'])
def feedback():
    product_name = request.form.get('product_name')
    feedback_type = request.form.get('feedback')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('feedback.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([product_name, feedback_type, timestamp])
    flash("Geri bildiriminiz için teşekkür ederiz!", "success")
    return redirect(request.referrer or '/')

if __name__ == '__main__':
        app.run(debug=True)
