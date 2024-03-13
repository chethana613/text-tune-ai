import csv

# Audio file path
audio_file_path = "audio_output/tuneai.wav"

# CSV file path
csv_file_path = "audiClips.csv"

# Open the CSV file in append mode
with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    # Add a row to the CSV file with the audio file path
    writer.writerow([audio_file_path])