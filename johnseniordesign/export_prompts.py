# export_prompts.py
import sqlite3

def export_prompts_from_db(db_path="traffic_data.db", output_file="all_prompts_new.txt"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT prompt FROM flows")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for row in cursor:
            prompt_text = row[0]
            if prompt_text:
                f.write(prompt_text.strip().replace('\n', ' ') + "\n")
    
    conn.close()
    print(f"Exported prompts to {output_file}")

if __name__ == "__main__":
    export_prompts_from_db()
