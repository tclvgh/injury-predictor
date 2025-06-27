import csv
import os

def convert_name_format(input_file, output_file):
    """
    Convert names from 'First_Name Last_Name' format to 'Last_name, First_Name' format
    in a CSV file.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output CSV file
    """
    # Ensure the data directory exists for the output file
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the input CSV file and write to the output CSV file
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write the header row
        header = next(reader)
        writer.writerow(header)
        
        # Process each row
        for row in reader:
            if len(row) >= 1:  # Ensure there's at least one column (name)
                name = row[0]
                # Split the name into first and last name
                name_parts = name.split(' ', 1)  # Split on the first space only
                
                if len(name_parts) == 2:
                    first_name, last_name = name_parts
                    # Convert to 'Last_name, First_Name' format
                    new_name = f"{last_name}, {first_name}"
                    row[0] = new_name
                
            writer.writerow(row)
    
    print(f"Conversion complete. Output saved to {output_file}")

if __name__ == "__main__":
    input_file = "data/2020to2024injuries.csv"
    output_file = "data/2020to2024injuries_name_converted.csv"
    
    convert_name_format(input_file, output_file)