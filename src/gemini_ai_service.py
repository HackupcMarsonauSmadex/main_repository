import os
import json
from google import genai
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'utils', 'ranges.json')

def analyze_full_campaign(campaign_text, creative_texts, image_files):
    client = genai.Client(api_key=os.getenv("GEMINI_API_TOKEN"))
    
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        return {"error": f"Config no trobat a: {CONFIG_PATH}"}, False

    content_parts = [f"CAMPAIGN CONTEXT: {campaign_text}"]
    for i in range(6):
        content_parts.append(f"--- SLOT {i+1} ---")
        if creative_texts[i]: content_parts.append(f"Text: {creative_texts[i]}")
        if image_files[i]: content_parts.append(Image.open(image_files[i]))

    # PROMPT SENCER
    prompt = f"""
    Ets un extractor de dades AdTech per a Smadex. El teu objectiu és omplir el JSON amb la màxima precisió possible.

    ### INSTRUCCIONS:
    1. **Certesa o null**: Si no pots determinar una dada amb certesa absoluta, posa exactament null. NO inventis.
    2. **Verticals**: Identifica el vertical de cada slot de forma independent.
    3. **Slots**: Genera EXACTAMENT 6 objectes a la llista 'creatives'.
    4. **Idiomes**: Codi ISO 2 lletres [es, ca, ja...].
    5. **Booleans**: 1, 0 o null.

    ### VALORS PERMESOS:
    - Verticals: {config['verticals']}
    - KPI Goals: {config['kpi_goals']}

    ### ESTRUCTURA DE SORTIDA (JSON PUR):
    {{
      "campaign": {{
        "vertical": null, "kpi_goal": null, "advertiser_name": null, "app_name": null,
        "objective": null, "daily_budget_usd": null, "countries": null
      }},
      "creatives": [
        {{
          "vertical": null, "headline": null, "subhead": null, "cta_text": null,
          "has_gameplay": null, "text_density": null, "copy_length_chars": null,
          "language": null, "dominant_color": null
        }}
      ]
    }}
    Respon EXCLUSIVAMENT amb el JSON pur.
    """
    content_parts.append(prompt)

    try:
        response = client.models.generate_content(model="gemini-flash-latest", contents=content_parts)
        data = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
        
        avis_borrat = False
        v_camp = data['campaign'].get('vertical')

        # LÒGICA DE POLICIA (PYTHON):
        # Si el vertical de la creativa no coincideix amb el de la campanya, la borrem.
        if v_camp:
            for idx, c in enumerate(data['creatives']):
                v_creative = c.get('vertical')
                if v_creative and v_creative != v_camp:
                    # BORRAT PER INCONSISTÈNCIA
                    empty_slot = {key: None for key in c.keys()}
                    empty_slot['vertical'] = v_camp # Posem el vertical de la campanya com vas demanar
                    data['creatives'][idx] = empty_slot
                    avis_borrat = True
        
        return data, avis_borrat
    except Exception as e:
        return {"error": str(e)}, False