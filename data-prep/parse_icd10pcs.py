import xml.etree.ElementTree as ET
import json
import itertools
from tqdm import tqdm

def get_axis_info(axis_element):
    """Extracts all labels from an axis element."""
    info = {}
    for label in axis_element.findall('label'):
        info[label.get('code')] = label.text
    return info

def parse_pcs_tables(xml_file, json_file):
    """
    Parses the ICD-10-PCS XML tables, generates all possible codes,
    and saves them to a JSON file.
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return
    except FileNotFoundError:
        print(f"Error: XML file not found at {xml_file}")
        return

    all_codes = []
    
    pcs_tables = root.findall('pcsTable')
    
    print(f"Found {len(pcs_tables)} PCS tables to process.")

    for table in tqdm(pcs_tables, desc="Processing PCS Tables"):
        # Extract axis information for the first 3 characters, which are fixed per table
        axis1 = table.find("axis[@pos='1']/label")
        axis2 = table.find("axis[@pos='2']/label")
        axis3 = table.find("axis[@pos='3']")
        
        section_code = axis1.get('code')
        section_title = axis1.text
        
        body_system_code = axis2.get('code')
        body_system_title = axis2.text
        
        operation_code = axis3.find('label').get('code')
        operation_title = axis3.find('label').text
        operation_definition = axis3.find('definition').text if axis3.find('definition') is not None else ""

        # Process each row in the table to generate codes for the last 4 characters
        for row in table.findall('pcsRow'):
            axis4_info = get_axis_info(row.find("axis[@pos='4']"))
            axis5_info = get_axis_info(row.find("axis[@pos='5']"))
            axis6_info = get_axis_info(row.find("axis[@pos='6']"))
            axis7_info = get_axis_info(row.find("axis[@pos='7']"))

            # Generate all combinations for the last 4 characters
            # The keys of the info dicts are the character codes
            character_combinations = itertools.product(
                axis4_info.keys(),
                axis5_info.keys(),
                axis6_info.keys(),
                axis7_info.keys()
            )

            for combo in character_combinations:
                char4, char5, char6, char7 = combo
                
                full_code = f"{section_code}{body_system_code}{operation_code}{char4}{char5}{char6}{char7}"
                
                body_part_title = axis4_info[char4]
                approach_title = axis5_info[char5]
                device_title = axis6_info[char6]
                qualifier_title = axis7_info[char7]

                # Construct the rich text for semantic search
                rich_text = (
                    f"This is an ICD-10-PCS code for a procedure in the '{section_title}' section (Code: {section_code}), "
                    f"involving the '{body_system_title}' body system (Code: {body_system_code}). "
                    f"The operation is '{operation_title}' (Code: {operation_code}), which is defined as: {operation_definition}. "
                    f"The specific body part is '{body_part_title}' (Code: {char4}). "
                    f"The procedure is performed using a '{approach_title}' approach (Code: {char5}). "
                    f"The device is recorded as '{device_title}' (Code: {char6}), "
                    f"and the qualifier is '{qualifier_title}' (Code: {char7})."
                )

                code_entry = {
                    'code': full_code,
                    'section_code': section_code,
                    'section': section_title,
                    'body_system_code': body_system_code,
                    'body_system': body_system_title,
                    'operation_code': operation_code,
                    'operation': operation_title,
                    'operation_definition': operation_definition,
                    'body_part_code': char4,
                    'body_part': body_part_title,
                    'approach_code': char5,
                    'approach': approach_title,
                    'device_code': char6,
                    'device': device_title,
                    'qualifier_code': char7,
                    'qualifier': qualifier_title,
                    'rich_text': rich_text
                }
                all_codes.append(code_entry)

    print(f"Generated a total of {len(all_codes)} codes.")

    # Write to JSON file
    try:
        with open(json_file, 'w') as f:
            json.dump(all_codes, f, indent=2)
        print(f"Successfully wrote {len(all_codes)} codes to {json_file}")
    except IOError as e:
        print(f"Error writing to JSON file: {e}")


if __name__ == '__main__':
    XML_FILE_PATH = 'raw-data/PCS/icd10pcs_tables_2026.xml'
    JSON_OUTPUT_PATH = 'data-prep/icd10pcs_parsed.json'
    parse_pcs_tables(XML_FILE_PATH, JSON_OUTPUT_PATH) 