import pandas as pd
import re
import sys

# Define a dictionary to map month abbreviations to month numbers
month_map = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
    'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12',
    'January': '01', 'February': '02', 'March': '03', 'April': '04', 'May': '05', 'June': '06',
    'July': '07', 'August': '08', 'September': '09', 'October': '10', 'November': '11', 'December': '12'
}

def convert_date(date_str):
    # If the date is already in MM/DD/YYYY format, return it as is
    if re.match(r'\d{1,2}/\d{1,2}/\d{4}', date_str):
        return date_str

    # Match patterns like "Apr '23", "Apr' '24", "April '20", "Jul' 22", etc.
    pattern = r"([A-Za-z]+)['']?\s?['']?(\d{2})"
    match = re.match(pattern, date_str)

    if match:
        month_str = match.group(1)
        year_str = match.group(2)

        # Get the month number
        month_num = month_map.get(month_str, '01')  # Default to January if not found

        # Determine the century (assuming 20xx for years)
        year = f"20{year_str}"

        # Return the date in MM/01/YYYY format
        return f"{month_num}/01/{year}"

    # If no match, print debug info
    if "'" in date_str or "'" in date_str:
        print(f"DEBUG: Failed to match: {date_str}", file=sys.stderr)

    # If no match, return the original string
    return date_str

# Read the input CSV file
input_file = 'data\\injuries.csv'
output_file = 'data\\injuries_date_converted.csv'

try:
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Apply the date conversion to the injury_date column
    df['injury_date'] = df['injury_date'].apply(convert_date)

    # Save the processed data to the output file
    df.to_csv(output_file, index=False)

    print(f"Conversion complete. Processed data saved to {output_file}")

    # Print some statistics
    total_rows = len(df)
    print(f"Total rows processed: {total_rows}")

except Exception as e:
    print(f"Error: {e}")
