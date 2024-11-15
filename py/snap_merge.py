'''20240820: merge the bands of input images (of which there are N) together into an output.
SNAP "beam/dimap" format assumed
'''
import os
import sys
import subprocess
import xml.etree.ElementTree as ET

def get_band_names(dimap_file):
    # Extracts band names from a BEAM/DIMAP XML file.
    tree = ET.parse(dimap_file)
    root = tree.getroot()
    
    # Find all bands
    bands = []
    for band in root.findall(".//Band"):
        band_name = band.get("name")
        bands.append(band_name)
    
    return bands

def create_band_math_expression(band_lists):
    # Create an expression for band math. This example adds bands from multiple images.
    expressions = []
    num_bands = len(band_lists[0])
    for i in range(num_bands):
        terms = [f'B{i+1}(source{j+1})' for j in range(len(band_lists))]
        # Example operation: addition of corresponding bands
        expressions.append(' + '.join(terms))
    return ' '.join(expressions)

def main():
    # File paths
    input_images = ['image1.dim', 'image2.dim', 'image3.dim']  # Add more images as needed
    output_image = 'output_image.tif'

    # Paths to the DIMAP XML files
    xml_images = [img.replace('.dim', '.xml') for img in input_images]

    # Check if all XML files exist
    for xml_file in xml_images:
        if not os.path.exists(xml_file):
            print(f"Error: DIMAP XML file {xml_file} not found.")
            return

    # Get band names
    band_lists = []
    for xml_file in xml_images:
        bands = get_band_names(xml_file)
        band_lists.append(bands)

    # Verify that all images have the same number of bands
    num_bands = len(band_lists[0])
    if not all(len(bands) == num_bands for bands in band_lists):
        print("Error: The number of bands in the images does not match.")
        return

    # Create band math expression
    band_math_expression = create_band_math_expression(band_lists)

    # Construct the GPT command
    sources = ' '.join(f'-Ssource{i+1}={img}' for i, img in enumerate(input_images))
    gpt_command = (
        f'gpt BandMath {sources} '
        f'-Pexpression="{band_math_expression}" -t {output_image}'
    )

    # Print the GPT command to verify
    print(f'Generated GPT Command: {gpt_command}')

    # Execute the GPT command
    try:
        subprocess.run(gpt_command, shell=True, check=True)
        print(f'Successfully executed: {gpt_command}')
    except subprocess.CalledProcessError as e:
        print(f'Error occurred: {e}')

if __name__ == '__main__':
    main()

