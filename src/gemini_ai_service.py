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
        if creative_texts[i]:
            content_parts.append(f"Text: {creative_texts[i]}")
        if image_files[i]:
            content_parts.append(Image.open(image_files[i]))

    prompt = f"""
    Ets un extractor de dades AdTech per a Smadex. El teu objectiu és omplir el JSON amb la màxima precisió possible.
    ### INSTRUCCIONS:
    1. **Certesa o null**: Si no pots determinar una dada amb certesa absoluta, posa exactament null. NO inventis.
    2. **Verticals**: Identifica el vertical de cada slot de forma independent.
    3. **Slots**: Genera EXACTAMENT 6 objectes a la llista 'creatives'.
    4. **Idiomes**: Codi ISO 2 lletres [es, ca, ja...].
    5. **Booleans**: 1, 0 o null.
    6. **Fija los valores de text_density y area a null quan analitztis les descripcions**
    7. **Cuando definas la campaign configuration no pongas nada en los campos de creative, y viceversa, menos el vertical**
    8. **Cuando obtengas un dominant_color dale un nombre genérico (red, blue, green...) en vez de un código hexadecimal**    
    ### VALORS PERMESOS:
    - Verticals: {config['verticals']}
    - KPI Goals: {list(config['kpi_goals'].keys())}
        ### ESTRUCTURA DE SORTIDA (JSON PUR, sense cap text addicional):
{{
      "campaign": {{
        "vertical": null,
        "kpi_goal": null,
        "advertiser_name": null,
        "app_name": null,
        "objective": null,
        "daily_budget_usd": null,
        "countries": null
      }},
      "creatives": [
        {{
          "vertical": null,
          "format": null,
          "language": null,
          "theme": null,
          "hook_type": null,
          "dominant_color": null,
          "emotional_tone": null,
          "advertiser_name": null,
          "app_name": null,
          "cta_text": null,
          "headline": null,
          "subhead": null,
          "has_price": null,
          "has_discount_badge": null,
          "has_gameplay": null,
          "has_ugc_style": null,
          "text_density": null,
          "copy_length_chars": null,
          "faces_count": null,
          "product_count": null,
          "duration_sec": null,
          "total_days_active": null,
          "total_spend_usd": null,
          "area": null
        }}
      ]
    }}
    Respon EXCLUSIVAMENT amb el JSON pur. Cap text abans ni després.
    """

    content_parts.append(prompt)

    try:
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=content_parts
        )
        data = json.loads(
            response.text.strip()
            .replace("```json", "")
            .replace("```", "")
        )

        avis_borrat = False
        v_camp = data['campaign'].get('vertical')

        #If the vertical of the campaign is defined, all creatives must match it. If not, we will erase the creative data but keep the vertical to allow partial analysis
        if v_camp:
            for idx, c in enumerate(data['creatives']):
                v_creative = c.get('vertical')
                if v_creative and v_creative != v_camp:
                    empty_slot = {key: None for key in c.keys()}
                    empty_slot['vertical'] = v_camp
                    data['creatives'][idx] = empty_slot
                    avis_borrat = True

        return data, avis_borrat

    except Exception as e:
        return {"error": str(e)}, False