from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import sys
import time
import os

app = Flask(__name__)

# Global variables for model and tokenizer
tokenizer = None
model = None
device = None

def initialize_model():
    global tokenizer, model, device
    if tokenizer is not None and model is not None:
        return

    print("Attempting to initialize model...")
    HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN", "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")
    if not HUGGINGFACE_TOKEN:
        print("Error: HF_TOKEN environment variable not set or invalid.", file=sys.stderr)
        return

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if device.type == "cuda":
            print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            torch.cuda.empty_cache()

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        print(f"Attempting to load tokenizer from meta-llama/Llama-3.1-8B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", use_auth_token=HUGGINGFACE_TOKEN)
        print(f"Attempting to load model from meta-llama/Llama-3.1-8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            use_auth_token=HUGGINGFACE_TOKEN,
            quantization_config=quantization_config
        ).to(device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        print("LLaMA-3.1-8B-Instruct model and tokenizer initialized successfully with 4-bit quantization on GPU.")
    except Exception as e:
        print(f"Error initializing LLaMA-3.1-8B-Instruct: {e}", file=sys.stderr)
        tokenizer = None
        model = None

initialize_model()

def generate_itinerary(city, days, budget, interests):
    if not tokenizer or not model:
        print("Model or tokenizer not initialized.", file=sys.stderr)
        return None, 0, []

    print(f"Starting itinerary generation for {city} with {days} days, ${budget} budget, interests: {interests}")
    prompt = (
        f"Generate a detailed {days}-day trip itinerary for {city} with a ${budget} budget for activities and dining only, focusing on a balanced mix of {', '.join(interests)} experiences as of March 2025. "
        f"Structure the output as follows:\n"
        f"- Overview: A brief paragraph (2-3 sentences) describing the itinerary's focus and appeal for travelers interested in {', '.join(interests)}, mentioning how it fits within the ${budget} budget for activities and dining.\n"
        f"- Day X: [Day Name] (repeat for each day from 1 to {days})\n"
        f"  - Morning:\n"
        f"    - HH:MM AM/PM: Activity at Specific {city} Location ($Cost)\n"
        f"  - Afternoon:\n"
        f"    - HH:MM AM/PM: Activity at Specific {city} Location ($Cost)\n"
        f"  - Evening:\n"
        f"    - HH:MM AM/PM: Activity at Specific {city} Location ($Cost)\n"
        f"- Budget Overview: A section estimating costs with approximate ranges in USD for Accommodation, Meals, Sightseeing & Tours, Transport & Miscellaneous, and a Total Estimated Cost range that fits within ${budget} for activities and dining, with additional notes for accommodation and transport as separate considerations.\n"
        f"Ensure each day includes exactly one activity per time slot (Morning, Afternoon, Evening) with specific times, locations, and costs, and the total cost of activities and dining is strictly within ${budget}. "
        f"Reflect all selected interests, use realistic {city} locations with approximate costs in USD, and stop after generating exactly {days} days. "
        f"Use this exact format for each entry: - HH:MM AM/PM: Activity at Specific {city} Location ($Cost). "
        f"Do not include moderation labels, advice beyond the example, or unrelated topics. "
        f"Example:\n"
        f"- Overview: This itinerary is tailored to travelers interested in both history and food. It features guided tours of Beijing’s most iconic historical sites and immersive culinary experiences ranging from local markets to trendy dining spots. With a $500 budget for activities and dining, you can enjoy museum visits and meals with room for accommodation and transport.\n"
        f"- Day 1: Arrival & Historic Landmarks\n"
        f"  - Morning:\n"
        f"    - 8:30 AM: Breakfast at Café am Neuen See ($8)\n"
        f"  - Afternoon:\n"
        f"    - 12:00 PM: Visit Beijing Wall Memorial ($0)\n"
        f"  - Evening:\n"
        f"    - 6:00 PM: Dinner at Vau ($25)\n"
        f"- Budget Overview:\n"
        f"  - Accommodation: ~$210–$300 (for 3 nights at a budget hotel, separate from activity budget)\n"
        f"  - Meals: ~$120–$150 (approx. $40–$50 per day, within activity budget)\n"
        f"  - Sightseeing & Tours: ~$70–$100 (within activity budget)\n"
        f"  - Transport & Miscellaneous: ~$30–$50 (separate from activity budget)\n"
        f"  - Total Estimated Cost: Approximately $430–$500 (activities and dining within $500, plus accommodation and transport)\n"
        f"Start the itinerary:\n- Overview:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=800)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        try:
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=600 * days,
                do_sample=True,
                top_k=15,
                top_p=0.9,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        except RuntimeError as e:
            print(f"CUDA error during generation: {e}", file=sys.stderr)
            return None, 0, []

    prompt_len = input_ids.shape[-1]
    raw_output = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    total_cost = 0
    markers = []

    # Hardcoded coordinates for Beijing locations (latitude, longitude)
    location_coords = {
        "Forbidden City": [39.9151, 116.3972],
        "Temple of Heaven": [39.8822, 116.4066],
        "Da Dong Roast Duck": [39.9337, 116.4194],  # Approximate location
        "Summer Palace": [39.9991, 116.2752],
        "Fragrant Hills Park": [39.9894, 116.1859],
        "Siji Minfu": [39.9145, 116.4108],  # Approximate location
        "Ming Tombs": [40.2500, 116.2167],
        "Panjiayuan Antique Market": [39.8762, 116.4814],
        "Quanjude Roast Duck": [39.9057, 116.3972]  # Approximate location
    }

    # Parse the itinerary to extract locations for mapping
    lines = raw_output.split('\n')
    current_day = None
    current_time_slot = None
    for line in lines:
        if line.strip():
            if line.startswith('- Day'):
                current_day = line.strip()
            elif 'Morning:' in line or 'Afternoon:' in line or 'Evening:' in line:
                current_time_slot = line.strip()
            elif line.startswith('- ') and current_day and current_time_slot:
                match = re.match(r'- (\d{1,2}:\d{2} [AP]M): (.*?) \(\$(\d+\.?\d*)\)', line.strip())
                if match:
                    time = match.group(1)
                    location = match.group(2).split(' at ')[-1] if ' at ' in match.group(2) else match.group(2)
                    cost = match.group(3)
                    if location in location_coords:
                        markers.append({
                            'day': current_day,
                            'time_slot': current_time_slot,
                            'time': time,
                            'location': location,
                            'lat': location_coords[location][0],
                            'lng': location_coords[location][1],
                            'cost': cost
                        })
            match_cost = re.match(r'.*\(([$]\d+\.?\d*)\)', line)
            if match_cost:
                try:
                    cost = float(match_cost.group(1).replace('$', ''))
                    total_cost += cost
                except ValueError:
                    pass

    if total_cost > budget:
        lines = raw_output.split('\n')
        adjusted_lines = []
        current_cost = 0
        for line in lines:
            if line.startswith('- ') and current_cost < budget:
                match = re.match(r'.*\(([$]\d+\.?\d*)\)', line)
                if match:
                    cost = float(match.group(1).replace('$', ''))
                    if current_cost + cost <= budget:
                        adjusted_lines.append(line)
                        current_cost += cost
                    else:
                        adjusted_lines.append(line.replace(f"(${cost})", "(Removed to fit budget)"))
                else:
                    adjusted_lines.append(line)
            else:
                adjusted_lines.append(line)
        raw_output = '\n'.join(adjusted_lines)
        total_cost = current_cost

    print(f"Raw Output: {raw_output}")
    print(f"Adjusted Total Cost: {total_cost}")
    print(f"Markers: {markers}")
    return raw_output, total_cost, markers

@app.route('/', methods=['GET', 'POST'])
def index():
    print(f"Received request: method={request.method}, path={request.path}, form={request.form}")
    if request.method == 'POST':
        print("Processing POST request...")
        city = request.form.get('city', 'Beijing')
        days = int(request.form.get('days', 3))
        budget = float(request.form.get('budget', 500))
        interests = request.form.getlist('interests') or ['food', 'history']
        print(f"POST data - City: {city}, Days: {days}, Budget: {budget}, Interests: {interests}")

        itinerary = ""
        total_cost = 0
        markers = []

        if tokenizer and model:
            print(f"Calling generate_itinerary with city={city}, days={days}, budget={budget}, interests={interests}")
            try:
                start_time = time.time()
                raw_output, total_cost, markers = generate_itinerary(city, days, budget, interests)
                end_time = time.time()
                print(f"Generation completed in {end_time - start_time:.2f} seconds.")
                print(f"Final Raw Output: {raw_output}")

                if raw_output is None:
                    itinerary = "\n".join([f"Day {d}\n- Failed to generate itinerary due to CUDA error." for d in range(1, days + 1)])
                else:
                    itinerary = raw_output
            except Exception as e:
                print(f"Generation error: {e}", file=sys.stderr)
                itinerary = "\n".join([f"Day {d}\n- Error generating itinerary: {str(e)}" for d in range(1, days + 1)])
        else:
            itinerary = "\n".join([f"Day {d}\n- Model initialization failed." for d in range(1, days + 1)])

        print(f"Returning itinerary: {itinerary}, Total Cost: {total_cost}, Markers: {markers}")
        return render_template('index.html', itinerary=itinerary, total_cost=total_cost, markers=markers, city=city, days=days, budget=budget, interests=interests)

    print("Rendering GET request for index page")
    return render_template('index.html', itinerary="", total_cost=0, markers=[], city="Beijing", days=3, budget=500, interests=['food', 'history'])

if __name__ == '__main__':
    app.run(debug=True)