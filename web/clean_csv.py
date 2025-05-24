#!/usr/bin/env python3
"""
Clean the CSV file by removing duplicate columns
"""

def clean_csv_file():
    with open('data_latih_2.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    for line in lines:
        # Split by semicolon
        parts = line.strip().split(';')
        # Take only the first half (remove duplicates)
        unique_parts = parts[:len(parts)//2]
        # Join back with semicolon
        cleaned_line = ';'.join(unique_parts) + '\n'
        cleaned_lines.append(cleaned_line)
    
    # Write cleaned file
    with open('data_latih_2_clean.csv', 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)
    
    print(f"Cleaned CSV file created: data_latih_2_clean.csv")
    print(f"Original lines: {len(lines)}")
    print(f"Cleaned lines: {len(cleaned_lines)}")
    
    # Show first few lines
    print("\nFirst few lines of cleaned file:")
    for i, line in enumerate(cleaned_lines[:5]):
        print(f"{i}: {line.strip()}")

if __name__ == "__main__":
    clean_csv_file()
