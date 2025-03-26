import csv

# Define input CSV file and output TXT file
input_csv = r"C:\Users\gurum\Documents\police\ChatBot\train.csv"
output_txt = ".\Data.txt"

# Read CSV and write to TXT
with open(input_csv, "r", encoding="utf-8") as csv_file, open(output_txt, "w", encoding="utf-8") as txt_file:
    reader = csv.reader(csv_file)
    
    for row in reader:
        if len(row) >= 2:  # Ensure there are at least two columns
            question, answer = row[0], row[1]
            txt_file.write(f"Q: {question}\nA: {answer}\n\n")

print(f"Conversion completed: {output_txt}")
