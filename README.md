This project is a Flask-based web application that recommends technological products based on the user's profession, needs, and budget. It also allows users to compare selected products after the recommendations are generated.

🚀 Features
User input: profession, needs, and budget

Support for both predefined and custom professions

Integration with LLaMA 3.1 8B via Ollama

Interactive product comparison feature

Clean and user-friendly interface

🛠 Technologies Used
Python + Flask

HTML + CSS + JavaScript

LLaMA 3.1 (via Ollama)

CSV datasets (phones, laptops, tablets, headphones, etc.)

⚙️ Installation & Setup

Clone the repository:
git clone https://github.com/yourusername/project-name.git
cd project-name

Install Python dependencies:
pip install -r requirements.txt

Install Ollama and pull the LLaMA 3.1 model:
ollama pull llama3:8b

Start Ollama in the terminal (keep it running in the background):
ollama start

Run the Flask application:
python app.py

Open your browser and go to:
cpp
http://127.0.0.1:5000/

📁 Datasets
You can find product datasets in the /datasets directory.

🧪 Example Usage
Fill out the form with your profession, needs, and budget.

Receive 8 product recommendations tailored to your input.

Select any two or more products to view a detailed comparison.

🔮 Future Work
Add real-time product data via external APIs

Include more product categories and filters

Improve the comparison UI with charts and visuals

Incorporate user feedback for more accurate recommendations
