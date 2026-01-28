#!/usr/bin/env python3
"""
20260113: fh.py - Fix ENVI Header files

Finds all .hdr files in the present directory that have corresponding .bin files.
- Removes whitespace in braced records (compacts opening/closing braces)
- Calculates actual number of bands from .bin file size
- Updates band names field to match actual band count
- Saves backup as .hdr.bak before modifying
"""

import os
import re
import shutil
from misc import hdr_fn
from pathlib import Path


def get_bin_file_bands(bin_path, samples, lines):
    """Calculate number of bands from .bin file size assuming 32-bit floats."""
    file_size = os.path.getsize(bin_path)
    bytes_per_band = samples * lines * 4  # 4 bytes per float32
    if bytes_per_band == 0:
        return 0
    return file_size // bytes_per_band


def parse_hdr_file(hdr_path):
    """Parse an ENVI header file into a list of (key, value) tuples preserving order."""
    with open(hdr_path, 'r') as f:
        content = f.read()
    
    records = []
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i]

        if line.strip() == '':
            i += 1
            continue
        
        # Check if this is the ENVI header marker
        if line.strip() == 'ENVI':
            records.append(('ENVI', None))
            i += 1
            continue
        
        # Check for key = value pattern
        match = re.match(r'^(\s*)([^=]+?)\s*=\s*(.*)', line)
        if match:
            indent = match.group(1)
            key = match.group(2).strip()
            value_start = match.group(3)
            
            # Check if value contains opening brace
            if '{' in value_start:
                # Collect full value including multiline braced content
                full_value = value_start
                brace_count = value_start.count('{') - value_start.count('}')
                
                while brace_count > 0 and i + 1 < len(lines):
                    i += 1
                    full_value += '\n' + lines[i]
                    brace_count += lines[i].count('{') - lines[i].count('}')
                
                records.append((key, full_value))
            else:
                records.append((key, value_start))
        elif line.strip():
            # Non-empty line without = sign, keep as-is
            records.append((None, line))
        else:
            # Empty line
            records.append((None, ''))
        
        i += 1
    
    return records


def compact_braced_value(value):
    """
    Compact a braced value by removing whitespace after { and before }.
    Keeps line breaks after commas for multiline values.
    """
    if '{' not in value:
        return value
    
    # Find the content between braces
    match = re.match(r'^(\{)\s*(.*?)\s*(\})$', value, re.DOTALL)
    if not match:
        # Try to handle case where { might have whitespace before content
        match = re.match(r'^(\{)\s*\n?(.*?)\s*\n?\s*(\})$', value, re.DOTALL)
    
    if match:
        opening = match.group(1)
        content = match.group(2)
        closing = match.group(3)
        
        # Clean up the content - remove leading/trailing whitespace from each line
        # but preserve the structure
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped:
                cleaned_lines.append(stripped)
        
        if len(cleaned_lines) == 1:
            # Single line - compact fully
            return '{' + cleaned_lines[0] + '}'
        else:
            # Multiline - keep line breaks but compact
            return '{' + cleaned_lines[0] + '\n' + '\n'.join(cleaned_lines[1:]) + '}'
    
    return value


def parse_band_names(value):
    """Parse band names from a braced value."""
    # Remove braces and split by comma
    match = re.match(r'^\{(.*)\}$', value, re.DOTALL)
    if not match:
        return []
    
    content = match.group(1)
    # Split by comma, handling potential newlines
    names = []
    for part in content.split(','):
        name = part.strip()
        if name:
            names.append(name)
    return names


def format_band_names(names):
    """Format band names into the proper multiline format."""
    if len(names) == 0:
        return '{}'
    elif len(names) == 1:
        return '{' + names[0] + '}'
    else:
        result = '{' + names[0] + ','
        for name in names[1:-1]:
            result += '\n' + name + ','
        result += '\n' + names[-1] + '}'
        return result


def process_hdr_file(hdr_path, bin_path):
    """Process a single .hdr file."""
    print(f"Processing: {hdr_path}")
    
    # Parse the header file
    records = parse_hdr_file(hdr_path)
    
    # Extract samples, lines, and bands values
    samples = None
    lines_val = None
    bands = None
    band_names_idx = None
    band_names = []
    
    for i, (key, value) in enumerate(records):
        if key == 'samples':
            samples = int(value.strip())
        elif key == 'lines':
            lines_val = int(value.strip())
        elif key == 'bands':
            bands = int(value.strip())
        elif key == 'band names':
            band_names_idx = i
            band_names = parse_band_names(compact_braced_value(value))
    
    if samples is None or lines_val is None:
        print(f"  Warning: Could not find samples or lines in {hdr_path}")
        return False
    
    # Calculate actual bands from .bin file
    actual_bands = get_bin_file_bands(bin_path, samples, lines_val)
    print(f"  samples={samples}, lines={lines_val}, header bands={bands}, actual bands={actual_bands}")
    
    # Create backup
    backup_path = str(hdr_path) + '.bak'
    shutil.copy2(hdr_path, backup_path)
    print(f"  Backup saved to: {backup_path}")
    
    # Update records
    new_records = []
    bands_updated = False
    band_names_updated = False
    
    for i, (key, value) in enumerate(records):
        if key == 'ENVI':
            new_records.append(('ENVI', None))
        elif key is None:
            new_records.append((None, value))
        elif key == 'bands':
            # Update bands count if different
            if actual_bands > 0 and actual_bands != bands:
                new_records.append((key, str(actual_bands)))
                bands_updated = True
                print(f"  Updated bands: {bands} -> {actual_bands}")
            else:
                new_records.append((key, value.strip()))
        elif key == 'band names':
            # Will handle after loop to ensure correct band count
            band_names_updated = True
            # Update band names if needed
            current_names = band_names.copy()
            if actual_bands > len(current_names):
                # Add generic names for missing bands
                for j in range(len(current_names) + 1, actual_bands + 1):
                    current_names.append(f'band {j}')
                print(f"  Added {actual_bands - len(band_names)} generic band name(s)")
            elif actual_bands < len(current_names) and actual_bands > 0:
                # Truncate to actual bands
                current_names = current_names[:actual_bands]
                print(f"  Truncated band names to {actual_bands}")
            
            new_records.append((key, format_band_names(current_names)))
        else:
            # Compact any braced values
            if value and '{' in value:
                new_records.append((key, compact_braced_value(value)))
            else:
                new_records.append((key, value.strip() if value else value))
    
    # If no band names field existed, add one
    if not band_names_updated and actual_bands > 0:
        # Insert band names after bands field or at end
        insert_idx = len(new_records)
        for i, (key, value) in enumerate(new_records):
            if key == 'bands':
                insert_idx = i + 1
                break
        
        generic_names = [f'band {j}' for j in range(1, actual_bands + 1)]
        new_records.insert(insert_idx, ('band names', format_band_names(generic_names)))
        print(f"  Added band names field with {actual_bands} generic names")
    
    # Write updated header
    with open(hdr_path, 'w') as f:
        for key, value in new_records:
            if key == 'ENVI':
                f.write('ENVI\n')
            elif key is None:
                f.write(value + '\n')
            else:
                f.write(f'{key} = {value}\n')
    
    print(f"  Updated: {hdr_path}")
    return True


def main():
    """Main function to process all .hdr files in current directory."""
    
    hdr_files = None
    args = sys.argv
    if os.path.exists(args[1]) and args[1].endswith('.hdr'):
        hdr_files = [args[1]]
    else:
        cwd = Path('.')
        hdr_files = list(cwd.glob('*.hdr'))
    
    if not hdr_files:
        print("No .hdr files found in current directory.")
        return
    
    processed = 0
    skipped = 0
    
    for hdr_path in hdr_files:
        # Skip backup files
        if str(hdr_path).endswith('.hdr.bak'):
            continue
            
        # Check for corresponding .bin file
        bin_path = hdr_path.with_suffix('.bin')
        
        if not bin_path.exists():
            print(f"Skipping {hdr_path}: No corresponding .bin file")
            skipped += 1
            continue
        
        try:
            if process_hdr_file(hdr_path, bin_path):
                processed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"Error processing {hdr_path}: {e}")
            skipped += 1
    
    print(f"\nDone. Processed: {processed}, Skipped: {skipped}")


if __name__ == '__main__':
    main()
