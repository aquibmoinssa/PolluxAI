import io
import pandas as pd

def extract_table_from_response(gpt_response):
    # Split the response into lines
    lines = gpt_response.strip().split("\n")
    
    # Find where the table starts and ends (based on the presence of pipes `|` and at least 3 columns)
    table_lines = [line for line in lines if '|' in line and len(line.split('|')) > 3]
    
    # If no table is found, return None or an empty string
    if not table_lines:
        return None
    
    # Find the first and last index of the table lines
    first_table_index = lines.index(table_lines[0])
    last_table_index = lines.index(table_lines[-1])
    
    # Extract only the table part
    table_text = lines[first_table_index:last_table_index + 1]
    
    return table_text

def gpt_response_to_dataframe(gpt_response):
    # Extract the table text from the GPT response
    table_lines = extract_table_from_response(gpt_response)
    
    # If no table found, return an empty DataFrame
    if table_lines is None or len(table_lines) == 0:
        return pd.DataFrame()

    # Find the header and row separator (assume it's a line with dashes like |---|)
    try:
        # The separator line (contains dashes separating headers and rows)
        sep_line_index = next(i for i, line in enumerate(table_lines) if set(line.strip()) == {'|', '-'})
    except StopIteration:
        # If no separator line is found, return an empty DataFrame
        return pd.DataFrame()

    # Extract headers (the line before the separator) and rows (lines after the separator)
    headers = [h.strip() for h in table_lines[sep_line_index - 1].split('|')[1:-1]]
    
    # Extract rows (each line after the separator)
    rows = [
        [cell.strip() for cell in row.split('|')[1:-1]]
        for row in table_lines[sep_line_index + 1:]
    ]

    # Create DataFrame
    df = pd.DataFrame(rows, columns=headers)
    return df