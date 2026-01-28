import os

file_path = r'd:\Projects\Speech-Enhancement\firmware\esp32-rnnoise\components\rnnoise\rnnoise_data.c'

replacements = {
    '"conv2_weights_float"': 'NULL',
    '"gru1_input_weights_float"': 'NULL',
    '"gru1_recurrent_weights_float"': 'NULL',
    '"gru2_input_weights_float"': 'NULL',
    '"gru2_recurrent_weights_float"': 'NULL',
    '"gru3_input_weights_float"': 'NULL',
    '"gru3_recurrent_weights_float"': 'NULL'
}

print(f"Reading file: {file_path}")
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

count = 0
for old, new in replacements.items():
    if old in content:
        content = content.replace(old, new)
        count += 1
        print(f"Replaced {old} -> {new}")
    else:
        print(f"Warning: Could not find {old}")

if count > 0:
    print(f"Writing {count} changes back to file...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Done!")
else:
    print("No changes made. Please check if the file was already patched.")
