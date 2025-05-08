# export_prompts.py
import sqlite3
import random

def generate_variations(base_prompt, flow_data):
    """Generate variations of the base prompt with flow information"""
    src_ip = flow_data["src_ip"]
    dst_ip = flow_data["dst_ip"]
    protocol = flow_data["protocol"]
    flow_id_info = f"src_ip:{src_ip}, dst_ip:{dst_ip}, protocol:{protocol}"
    
    variations = [
        f"{base_prompt} [traffic Details: {flow_id_info}]",
        f"{base_prompt} with network traffic parameters: {flow_id_info}",
        f"Simulate the following stream: {flow_id_info}. {base_prompt}",
        f"{base_prompt}. Details of the packet stream: {flow_id_info}",
        f"Based on the configuration ({flow_id_info}), {base_prompt}",
        f"Given the network configuration ({flow_id_info}), {base_prompt}",
        f"Generate a simulation using these network settings: {flow_id_info} → {base_prompt}",
    ]
    
    # Add more diverse variations
    additional_variations = [
        f"Create network traffic that simulates {base_prompt}",
        f"Generate a packet stream for {base_prompt}",
        f"What would the traffic look like for {base_prompt}?",
        f"Show me packets representing {base_prompt}",
        f"Emulate network behavior for {base_prompt}",
        f"Craft a packet sequence that would occur when {base_prompt}"
    ]
    
    return variations + additional_variations

def export_prompts_from_db(db_path="traffic_data.db", output_file="all_prompts_new2.txt", include_variations=True, max_variations_per_prompt=4):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    with open(output_file, "w", encoding="utf-8") as f:
        if include_variations:
            cursor.execute("SELECT * FROM flows")
            for flow in cursor:
                base_prompt = flow["prompt"]
                
                # Write the original prompt
                if base_prompt:
                    f.write(base_prompt.strip().replace('\n', ' ') + "\n")
                
                # Generate and write variations
                variations = generate_variations(base_prompt, flow)
                selected_variations = random.sample(variations, min(max_variations_per_prompt, len(variations)))
                
                for var in selected_variations:
                    f.write(var.strip().replace('\n', ' ') + "\n")
        else:
            # Original behavior - just export the base prompts
            cursor.execute("SELECT prompt FROM flows")
            for row in cursor:
                prompt_text = row[0]
                if prompt_text:
                    f.write(prompt_text.strip().replace('\n', ' ') + "\n")
    
    conn.close()
    print(f"Exported prompts to {output_file}")

if __name__ == "__main__":
    export_prompts_from_db(include_variations=True, max_variations_per_prompt=5)
