import pdfplumber
import pandas as pd
import re
import json
import hashlib
import os
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
import boto3
import tempfile
import psycopg2
import pytz
from datetime import datetime
import string
import io

def main2(config):

    class PDFProcessorA:
        def __init__(self):
            # Load environment variables
            load_dotenv()
            self.google_api_key = os.getenv("GOOGLE_API_KEY")
            if not self.google_api_key:
                raise ValueError("API key not found. Make sure you have set GOOGLE_API_KEY in your .env file.")
            
            # Define section patterns
            self.section_patterns = {
                'A': [
                    r'SECTION\s+A\s*[-:]?\s*GENERAL\s+DISCLOSURES',
                    r'Section\s+A\s*[-:]?\s*General\s+Disclosures',
                    r'SECTION\s+A',
                    r'Section\s+A'
                ],
                'B': [
                    r'SECTION\s+B\s*[-:]?\s*MANAGEMENT\s+AND\s+PROCESS\s+DISCLOSURES',
                    r'Section\s+B\s*[-:]?\s*Management\s+and\s+Process\s+Disclosures',
                    r'SECTION\s+B',
                    r'Section\s+B'
                ],
                'C': [
                    r'SECTION\s+C\s*[-:]?\s*PRINCIPLE\s+WISE\s+PERFORMANCE\s+DISCLOSURE',
                    r'Section\s+C\s*[-:]?\s*Principle\s+Wise\s+Performance\s+Disclosure',
                    r'SECTION\s+C',
                    r'Section\s+C'
                ]
            }
            
            # Specific Roman numeral patterns
            self.roman_numerals = [
                r"i\.?",
                r"ii\.?",
                r"iii\.?",
                r"iv\.?",
                r"v\.?",
                r"vi\.?",
                r"vii\.?"
            ]
            
            # Initialize start_page_number as None, will be set when processing a PDF
            self.start_page_number = None
            
            # Use roman numerals as heading patterns
            self.heading_patterns = self.roman_numerals

            # Add mapping dictionary for Section A subsection descriptive names
            self.section_a_heading_names = {
                'i': 'i Details of the entity',
                'ii': 'ii Products_Services',
                'iii': 'iii Operations',
                'iv': 'iv Employees',
                'v': 'v Holding_Subsidiary and Associate Companies',
                'vi': 'vi CSR Details',
                'vii': 'vii Transparency and Disclosures Compliances'
            }

        def find_start_page(self, pdf_path):
            """
            Automatically find the starting page that contains both 
            "BUSINESS RESPONSIBILITY AND SUSTAINABILITY REPORT" and "Section A: General Disclosures".
            
            Args:
                pdf_path: Path to the PDF file
                
            Returns:
                int: The page number (1-indexed) where the BRSR report starts
            """
            print("\nSearching for BRSR start page...")
            
            with pdfplumber.open(pdf_path) as pdf:
                # Create patterns for the text we're looking for
                brsr_pattern = r"(?i)BUSINESS\s+RESPONSIBILITY\s+(AND|&)\s+SUSTAINABILITY\s+REPORT"
                section_a_pattern = r"(?i)Section\s+A\s*[-:]?\s*General\s+Disclosure[s]?"
                
                # Set batch size to process multiple pages at once for efficiency
                batch_size = 50
                total_pages = len(pdf.pages)
                
                # Process pages in batches to improve performance
                for batch_start in range(0, total_pages, batch_size):
                    batch_end = min(batch_start + batch_size, total_pages)
                    print(f"Scanning pages {batch_start + 1} to {batch_end}...")
                    
                    for page_num in range(batch_start, batch_end):
                        page = pdf.pages[page_num]
                        text = page.extract_text()
                        
                        if not text:
                            continue
                        
                        # Check if both patterns are on the same page
                        if (re.search(brsr_pattern, text, re.IGNORECASE) and 
                            re.search(section_a_pattern, text, re.IGNORECASE)):
                            print(f"Found BRSR start on page {page_num + 1}")
                            return page_num + 1  # Return 1-indexed page number
                        
                        # If only the BRSR pattern is found, check the next few pages for Section A
                        if re.search(brsr_pattern, text, re.IGNORECASE):
                            print(f"Found BRSR title on page {page_num + 1}, checking for Section A...")
                            # Check next 5 pages for Section A
                            for next_page_num in range(page_num + 1, min(page_num + 6, total_pages)):
                                next_page = pdf.pages[next_page_num]
                                next_text = next_page.extract_text()
                                if next_text and re.search(section_a_pattern, next_text, re.IGNORECASE):
                                    print(f"Found Section A on page {next_page_num + 1}")
                                    return page_num + 1  # Return the BRSR title page
                
                # If not found, return a default value or raise an exception
                print("Warning: Could not find BRSR start page. Using default page 1.")
                return 1

        def get_user_section_choice(self):
            """Ask user which section to process."""
            while True:
                choice = "A"
                if choice in self.section_patterns:
                    return choice
                print("Invalid choice. Please enter A, B, or C.")

        def detect_headings(self, text, page_num):
            """Enhanced heading detection focusing on Roman numerals."""
            headings = []
            lines = text.split('\n')
            
            print(f"\nScanning page {page_num + 1} for Roman numeral headings...")
            
            for line in lines:
                line = line.strip()
                if line:  # Only process non-empty lines
                    # Convert to lowercase for consistent matching
                    line_lower = line.lower()
                    
                    # Try to find Roman numerals at the start of the line
                    for pattern in self.heading_patterns:
                        if re.match(pattern, line_lower):
                            # Get the full line as heading
                            print(f"Found heading: {line}")
                            headings.append(line)
                            break
            
            return headings
        
        def process_section_headings(self, pdf_path, output_dir):
            """Process all headings within a chosen section."""
            section_choice = self.get_user_section_choice()
            self.current_section = section_choice
            
            print(f"\nProcessing SECTION {section_choice}")
            
            # Find the start page if it's not already set
            if self.start_page_number is None:
                self.start_page_number = self.find_start_page(pdf_path)
            
            section_dir = os.path.join(output_dir, f"SECTION_{section_choice}")
            os.makedirs(section_dir, exist_ok=True)
            
            with pdfplumber.open(pdf_path) as pdf:
                # Ensure the PDF has enough pages
                if len(pdf.pages) < self.start_page_number:
                    print(f"PDF has fewer than {self.start_page_number} pages!")
                    return None
                    
                # Convert to 0-based index
                start_page = self.start_page_number - 1
                end_page = len(pdf.pages) - 1
                start_y = None
                
                # Find start position on page 106
                page = pdf.pages[start_page]
                text = page.extract_text()
                words = page.extract_words()
                
                # Find section boundaries
                self.current_start_patterns = self.section_patterns[section_choice]
                next_section = chr(ord(section_choice) + 1)
                self.current_end_patterns = (
                    self.section_patterns[next_section] if next_section in self.section_patterns 
                    else [r"SECTION\s+[" + next_section + r"-Z]"]
                )
                
                for pattern in self.current_start_patterns:
                    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                    if match:
                        matched_text = match.group(0)
                        first_word = matched_text.split()[0]
                        for word in words:
                            if first_word.lower() in word['text'].lower():
                                start_y = word['top']
                                print(f"Found section start '{matched_text}' on page {start_page + 1} at y-position: {start_y}")
                                break
                    if start_y is not None:
                        break
                
                if start_y is None:
                    print(f"Could not find SECTION {section_choice} on page {self.start_page_number}!")
                    return None
                
                # Find the next section (e.g., Section B) boundary
                section_b_page = None
                section_b_y = None
                
                for page_num in range(start_page, end_page + 1):
                    if section_b_page is not None:
                        break
                        
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    words = page.extract_words()
                    
                    for pattern in self.current_end_patterns:
                        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                        if match:
                            matched_text = match.group(0)
                            first_word = matched_text.split()[0]
                            for word in words:
                                if first_word.lower() in word['text'].lower():
                                    section_b_page = page_num
                                    section_b_y = word['top']
                                    print(f"Found next section '{matched_text}' on page {page_num + 1} at y-position: {section_b_y}")
                                    break
                        if section_b_page is not None:
                            break
                
                print(f"\nSection boundaries: Pages {start_page + 1} to {section_b_page + 1 if section_b_page is not None else end_page + 1}")
                
                # Define Roman numeral patterns with word boundaries
                roman_patterns = [
                    (r"\bi\.?", "i"),
                    (r"\bii\.?", "ii"),
                    (r"\biii\.?", "iii"),
                    (r"\biv\.?", "iv"),
                    (r"\bv\.?", "v"),
                    (r"\bvi\.?", "vi"),
                    (r"\bvii\.?", "vii")
                ]
                
                # Find Roman numeral headings sequentially
                heading_locations = []
                current_pattern_index = 0
                vii_found = False
                vii_page = None
                vii_y = None
                
                print("\nScanning for Roman numeral headings sequentially...")
                
                while current_pattern_index < len(roman_patterns):
                    pattern, numeral = roman_patterns[current_pattern_index]
                    found_current = False
                    
                    search_start_page = heading_locations[-1]['page'] if heading_locations else start_page
                    
                    for page_num in range(search_start_page, end_page + 1):
                        if found_current:
                            break
                        
                        # Don't search beyond Section B boundary if found
                        if section_b_page is not None and page_num > section_b_page:
                            break
                        
                        page = pdf.pages[page_num]
                        text = page.extract_text()
                        words = page.extract_words()
                        
                        matches = re.finditer(pattern, text.lower())
                        for match in matches:
                            match_start = match.start()
                            match_end = match.end()
                            
                            if match_start > 0 and text[match_start-1:match_start].lower() in 'iv':
                                continue
                                
                            match_text = match.group(0)
                            for word in words:
                                if match_text in word['text'].lower():
                                    if current_pattern_index > 0:
                                        prev_heading = heading_locations[-1]
                                        if page_num < prev_heading['page'] or \
                                        (page_num == prev_heading['page'] and word['top'] <= prev_heading['y']):
                                            continue
                                    
                                    # Don't add if after Section B boundary
                                    if section_b_page is not None and (page_num > section_b_page or 
                                    (page_num == section_b_page and word['top'] >= section_b_y)):
                                        continue
                                    
                                    heading_locations.append({
                                        'numeral': numeral,
                                        'page': page_num,
                                        'y': word['top'],
                                        'pattern': pattern
                                    })
                                    print(f"Found heading {numeral} on page {page_num + 1} at y={word['top']}")
                                    
                                    # If this is 'vii', mark it as the end point
                                    if numeral == 'vii':
                                        vii_found = True
                                        vii_page = page_num
                                        vii_y = word['top']
                                        print(f"Found 'vii.' - marking as end point on page {page_num + 1} at y={word['top']}")
                                    
                                    found_current = True
                                    break
                            
                            if found_current:
                                break
                    
                    if found_current:
                        current_pattern_index += 1
                        if vii_found and current_pattern_index >= len(roman_patterns):
                            print("Found 'vii.' - stopping heading detection")
                            break
                    else:
                        if current_pattern_index == 0:
                            print("Could not find starting pattern 'i.'!")
                            return None
                        else:
                            print(f"No more patterns found after {roman_patterns[current_pattern_index-1][1]}.")
                            break
                
                if not heading_locations:
                    print("No Roman numeral headings found in section!")
                    return None
                
                # Process tables between each pair of headings
                all_results = {}
                
                for i in range(len(heading_locations)):
                    current_heading = heading_locations[i]
                    next_heading = heading_locations[i + 1] if i + 1 < len(heading_locations) else None
                    
                    print(f"\nProcessing subsection {current_heading['numeral'].upper()}")
                    
                    # Use descriptive names for Section A
                    if self.current_section == 'A' and current_heading['numeral'] in self.section_a_heading_names:
                        descriptive_name = self.section_a_heading_names[current_heading['numeral']]
                        heading_dir = os.path.join(section_dir, descriptive_name)
                        folder_name = descriptive_name
                    else:
                        heading_dir = os.path.join(section_dir, f"subsection_{current_heading['numeral']}")
                        
                    os.makedirs(heading_dir, exist_ok=True)
                    
                    # Set boundaries for table extraction
                    self.current_heading_start = {
                        'page': current_heading['page'],
                        'y': current_heading['y']
                    }
                    
                    if next_heading:
                        self.current_heading_end = {
                            'page': next_heading['page'],
                            'y': next_heading['y']
                        }
                    elif current_heading['numeral'] == 'vii':
                        # For 'vii', process until the next section boundary (Section B)
                        if section_b_page is not None:
                            self.current_heading_end = {
                                'page': section_b_page,
                                'y': section_b_y
                            }
                            print(f"Processing 'vii' until Section B on page {section_b_page + 1} at y={section_b_y}")
                        else:
                            # If Section B not found, process until end of document
                            self.current_heading_end = {
                                'page': end_page,
                                'y': float('inf')
                            }
                            print(f"Processing 'vii' until end of document (page {end_page + 1})")
                    else:
                        # If next_heading is None and this is not 'vii', 
                        # process until the end of current page or until Section B
                        if section_b_page is not None:
                            if current_heading['page'] < section_b_page:
                                self.current_heading_end = {
                                    'page': current_heading['page'],
                                    'y': float('inf')
                                }
                            else:
                                self.current_heading_end = {
                                    'page': section_b_page,
                                    'y': section_b_y
                                }
                        else:
                            self.current_heading_end = {
                                'page': current_heading['page'],
                                'y': float('inf')
                            }
                    
                    # Extract and process tables
                    tables = self.extract_tables_from_pdf(pdf)
                    if tables:
                        results = self.process_tables(tables, heading_dir)
                        all_results[f"subsection_{current_heading['numeral']}"] = {
                            'heading': current_heading['numeral'].upper(),
                            'tables': results
                        }
                        print(f"Processed {len(results)} tables under subsection {current_heading['numeral'].upper()}")
                    else:
                        print(f"No tables found under subsection {current_heading['numeral'].upper()}")
                        all_results[f"subsection_{current_heading['numeral']}"] = {
                            'heading': current_heading['numeral'].upper(),
                            'tables': []
                        }
                
                # Save section summary
                summary_path = os.path.join(section_dir, "section_summary.json")
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2)
                
                return all_results

        def process_all_sections(self, pdf_path, output_dir):
            """Process all sections sequentially."""
            print("\nStarting sequential processing of all sections...")
            
            # Create main output directory
            os.makedirs(output_dir, exist_ok=True)
            
            all_results = {}
            
            # Process each section
            for section in ['A', 'B', 'C']:
                print(f"\n{'='*50}")
                print(f"Processing SECTION {section}")
                print(f"{'='*50}")
                
                # Create section-specific output directory
                section_dir = os.path.join(output_dir, f"SECTION_{section}")
                os.makedirs(section_dir, exist_ok=True)
                
                # Set patterns for current section
                self.current_start_patterns = self.section_patterns[section]
                next_section = chr(ord(section) + 1)
                self.current_end_patterns = (
                    self.section_patterns[next_section] if next_section in self.section_patterns 
                    else self.roman_numerals
                )
                
                # Process the section
                try:
                    with pdfplumber.open(pdf_path) as pdf:
                        # Extract tables from the section
                        tables = self.extract_tables_from_pdf(pdf)
                        if tables:
                            # Process the tables
                            results = self.process_tables(tables, section_dir)
                            all_results[f"SECTION_{section}"] = results
                            print(f"\nSuccessfully processed {len(results)} tables in SECTION {section}")
                        else:
                            print(f"\nNo tables found in SECTION {section}")
                            all_results[f"SECTION_{section}"] = []
                except Exception as e:
                    print(f"\nError processing SECTION {section}: {str(e)}")
                    all_results[f"SECTION_{section}"] = []
            
            # Save overall processing summary
            summary_path = os.path.join(output_dir, "overall_processing_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nOverall processing summary saved to: {summary_path}")
            
            return all_results

        def get_section_patterns(self):
            """
            Ask user for section to extract and create appropriate patterns.
            Returns tuple of (start_pattern, end_patterns)
            """
            # Ask user for section name
            section_name = input("\nWhich section do you want to extract? ").strip()

            # Helper function to create pattern variations
            def create_pattern_variations(text):
                # Split the text into words
                words = text.split()
                
                # Create word boundary pattern with flexible whitespace
                word_boundary_pattern = r'\s+'.join(rf'\b{re.escape(word)}\b' for word in words)
                
                # Create patterns with different cases and formats
                patterns = [
                    word_boundary_pattern,  # Word boundaries with flexible spacing
                    r'\s+'.join(re.escape(word) for word in words),  # Exact match with flexible spacing
                    r'\s+'.join(re.escape(word.upper()) for word in words),  # All caps with flexible spacing
                    r'\s+'.join(re.escape(word.lower()) for word in words),  # All lowercase with flexible spacing
                    rf'\d+\.\s*{word_boundary_pattern}',  # Numbered sections
                    rf'[A-Z]\.\s*{word_boundary_pattern}',  # Letter-based sections
                    # Add patterns for handling hyphenated variations
                    word_boundary_pattern.replace(r'\s+', '-'),  # Hyphenated version
                    word_boundary_pattern.replace(r'\s+', '_'),  # Underscore version
                ]
                
                # Add patterns for common heading formats
                heading_variations = [
                    f"(?i)(?:SECTION|Section)\\s*\\d*\\s*[-:]?\\s*{word_boundary_pattern}",  # Section X: Pattern
                    f"(?i)(?:PART|Part)\\s*\\d*\\s*[-:]?\\s*{word_boundary_pattern}",  # Part X: Pattern
                    f"(?i)\\d+\\.\\d+\\s*{word_boundary_pattern}",  # 1.1 Pattern
                    f"(?i)[A-Z]\\)\\s*{word_boundary_pattern}",  # A) Pattern
                ]
                
                patterns.extend(heading_variations)
                return patterns
            
            # Generate patterns
            start_patterns = create_pattern_variations(section_name)
            
            print("\nSearching for section:", section_name)
            print("\nGenerated patterns:")
            for i, pattern in enumerate(start_patterns):
                print(f"{i+1}. {pattern}")
                
            return start_patterns, self.roman_numerals

        def detect_section_boundaries(self, pdf):
            """Detect the start and end pages and positions of the target section."""
            start_page = None
            start_y = None
            end_page = None
            end_y = None
            in_section = False
            
            print("\nScanning document for section boundaries...")
            
            # Start scanning from page 1
            for page_num in range(len(pdf.pages)):
                page = pdf.pages[page_num]
                text = page.extract_text()
                if text is None:
                    continue
                    
                print(f"\nAnalyzing page {page_num + 1}")
                
                # If we haven't found the section start yet
                if not in_section:
                    for pattern in self.current_start_patterns:
                        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                        if match:
                            start_page = page_num
                            words = page.extract_words()
                            matched_text = match.group(0)
                            first_word = matched_text.split()[0]
                            for word in words:
                                if first_word.lower() in word['text'].lower():
                                    start_y = word['top']
                                    print(f"Found section start '{matched_text}' on page {page_num + 1} at y-position: {start_y}")
                                    in_section = True
                                    break
                        if in_section:
                            break
                
                # If we're in the section, look for the end
                elif in_section:
                    for pattern in self.current_end_patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            end_page = page_num
                            words = page.extract_words()
                            for word in words:
                                if re.search(pattern, word['text'], re.IGNORECASE):
                                    end_y = word['top']
                                    print(f"Found section end on page {page_num + 1} at y-position: {end_y}")
                                    return start_page, end_page, start_y, end_y
            
            # If we found start but no end, use last page
            if start_page is not None:
                end_page = len(pdf.pages) - 1
                end_y = float('inf')
                print(f"No explicit section end found, using last page ({end_page + 1})")
            
            return start_page, end_page, start_y, end_y

        def extract_tables_from_pdf(self, pdf):
            """Extract tables from PDF within the current heading boundaries."""
            start_page = self.current_heading_start['page']
            start_y = self.current_heading_start['y']
            end_page = self.current_heading_end['page']
            end_y = self.current_heading_end['y']
            
            print(f"Extracting tables between: Page {start_page + 1}, y={start_y} and Page {end_page + 1}, y={end_y}")
            
            all_tables = []
            current_table = []
            table_count = 0
            
            for page_num in range(start_page, end_page + 1):
                page = pdf.pages[page_num]
                
                # Extract tables from the current page
                tables = page.extract_tables()
                if not tables:
                    tables = page.extract_tables(table_settings={
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "intersection_y_tolerance": 10
                    })
                
                if tables:
                    # Get table positions
                    table_positions = page.find_tables()
                    
                    # Make sure we have positions for all tables
                    if len(table_positions) != len(tables):
                        print(f"Warning: Found {len(tables)} tables but only {len(table_positions)} positions on page {page_num + 1}")
                        # Fall back to a simpler approach if positions don't match
                        for table in tables:
                            # Skip empty tables
                            if not any(any(cell for cell in row) for row in table):
                                continue
                                
                            cleaned_table = [[self.clean_cell_content(cell) for cell in row] for row in table]
                            df = pd.DataFrame(cleaned_table)
                            df = df.dropna(how='all').dropna(axis=1, how='all')
                            
                            if not df.empty and page_num != end_page:  # Include all tables except on end page
                                if current_table and self.is_table_continuation(current_table[-1], df):
                                    current_table.append(df)
                                else:
                                    if current_table:
                                        processed_table = pd.concat(current_table, ignore_index=True)
                                        all_tables.append(processed_table)
                                        table_count += 1
                                    current_table = [df]
                        continue  # Skip the position-based logic below
                    
                    for table_idx, (table, table_pos) in enumerate(zip(tables, table_positions)):
                        # Skip empty tables
                        if not any(any(cell for cell in row) for row in table):
                            continue
                            
                        table_top = table_pos.bbox[1]
                        table_bottom = table_pos.bbox[3]
                        
                        print(f"Table {table_idx} on page {page_num + 1}: top={table_top}, bottom={table_bottom}")
                        
                        # Check if table is within heading boundaries
                        include_table = False
                        
                        # Special case for tables that span the boundary exactly
                        if page_num == start_page and page_num == end_page:
                            # Both start and end on same page
                            if table_top >= start_y and table_bottom <= end_y:
                                include_table = True
                                print(f"Including table: Same-page boundary check passed")
                            else:
                                print(f"Excluding table: Outside bounds on single boundary page")
                        elif page_num == start_page:
                            # First page of section - table must start after heading
                            if table_top >= start_y:
                                include_table = True
                                print(f"Including table: Start-page check passed (y={table_top} >= {start_y})")
                            else:
                                print(f"Excluding table: Before start on start page (y={table_top} < {start_y})")
                        elif page_num == end_page:
                            # Last page of section - table must end before next heading
                            if table_top < end_y:  # The top of the table should be before the next heading
                                include_table = True
                                print(f"Including table: End-page check passed (y={table_top} < {end_y})")
                            else:
                                print(f"Excluding table: After end on end page (y={table_top} >= {end_y})")
                        else:
                            # Middle pages - include all tables
                            include_table = True
                            print(f"Including table: On middle page")
                        
                        if include_table:
                            # Process table
                            cleaned_table = [[self.clean_cell_content(cell) for cell in row] for row in table]
                            df = pd.DataFrame(cleaned_table)
                            df = df.dropna(how='all').dropna(axis=1, how='all')
                            
                            if not df.empty:
                                # Add page and position info for debugging
                                df.attrs['page_num'] = page_num + 1
                                df.attrs['y_pos'] = table_top
                                
                                if current_table and self.is_table_continuation(current_table[-1], df):
                                    print(f"Detected table continuation")
                                    current_table.append(df)
                                else:
                                    if current_table:
                                        processed_table = pd.concat(current_table, ignore_index=True)
                                        all_tables.append(processed_table)
                                        table_count += 1
                                        print(f"Added complete table #{table_count}")
                                    current_table = [df]
                                    print(f"Started new table")
            
            # Process any remaining table
            if current_table:
                processed_table = pd.concat(current_table, ignore_index=True)
                all_tables.append(processed_table)
                table_count += 1
                print(f"Added final table #{table_count}")
            
            print(f"Extracted {table_count} tables for this subsection")
            return all_tables if all_tables else None
            
        def is_table_continuation(self, prev_df, curr_df):
            """
            Determine if the current table is a continuation of the previous table.
            """
            # Check if number of columns match
            if len(prev_df.columns) != len(curr_df.columns):
                return False
            
            try:
                # Convert column labels to strings and compare
                prev_headers = [str(col).lower().strip() for col in prev_df.columns]
                curr_headers = [str(col).lower().strip() for col in curr_df.columns]
                
                # Check if headers are identical or very similar
                header_similarity = sum(p == c for p, c in zip(prev_headers, curr_headers))
                header_match = header_similarity / len(prev_headers) > 0.8
                
                # Check if current table starts with numeric data (likely continuation)
                first_row_numeric = all(
                    str(x).replace('.', '').replace('%', '').strip().isdigit() 
                    for x in curr_df.iloc[0] 
                    if pd.notna(x) and str(x).strip()
                )
                
                # Check for similar data patterns in last row of prev and first row of current
                prev_last_row = prev_df.iloc[-1]
                curr_first_row = curr_df.iloc[0]
                
                pattern_matches = 0
                total_checks = 0
                
                for i in range(len(prev_df.columns)):
                    prev_val = str(prev_last_row.iloc[i]).strip() if pd.notna(prev_last_row.iloc[i]) else ''
                    curr_val = str(curr_first_row.iloc[i]).strip() if pd.notna(curr_first_row.iloc[i]) else ''
                    
                    if prev_val and curr_val:  # Only check non-empty values
                        total_checks += 1
                        # Check if both are numeric or both are text
                        prev_is_numeric = prev_val.replace('.', '').replace('%', '').isdigit()
                        curr_is_numeric = curr_val.replace('.', '').replace('%', '').isdigit()
                        if prev_is_numeric == curr_is_numeric:
                            pattern_matches += 1
                
                pattern_similarity = pattern_matches / total_checks if total_checks > 0 else 0
                
                # Return True if either condition is met
                return header_match or (first_row_numeric and pattern_similarity > 0.8)
                    
            except Exception as e:
                print(f"Warning: Error comparing tables: {str(e)}")
                return False

        def process_tables(self, tables, output_dir):
            """Process multiple tables and store only the original CSV and decoded JSON."""
            processed_results = []

            for idx, df in enumerate(tables, 1):
                print(f"\nProcessing table {idx}...")

                # Save original CSV
                base_name = f"table_{idx}"
                csv_path = os.path.join(output_dir, f"{base_name}_original.csv")
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"Original table {idx} saved to: {csv_path}")

                # Create hash mapping
                hash_mapping = self.create_hash_mapping(df)

                # Encode data (but do NOT save it to a file)
                encoded_df, _ = self.encode_csv_data(df)

                # Preprocess for Gemini (but do NOT save it to a file)
                preprocessed_df = self.preprocess_csv_for_gemini(encoded_df)

                # Process with Gemini
                processed_json_path = self.process_csv_with_gemini(preprocessed_df, output_dir, idx)

                if processed_json_path:
                    try:
                        with open(processed_json_path, 'r', encoding='utf-8') as f:
                            processed_json = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse Gemini output as JSON for table {idx}: {e}")
                        continue

                    # Decode JSON using hash mapping
                    decoded_json = self.decode_json_data(processed_json, hash_mapping)

                    # Save only decoded JSON
                    decoded_json_path = os.path.join(output_dir, f"{base_name}_decoded.json")
                    with open(decoded_json_path, 'w', encoding='utf-8') as f:
                        json.dump(decoded_json, f, indent=2, ensure_ascii=False)

                    print(f"Decoded JSON saved to: {decoded_json_path}")

                    # Delete intermediate files if they were mistakenly saved elsewhere
                    for file_path in [
                        os.path.join(output_dir, f"{base_name}_encoded.csv"),
                        os.path.join(output_dir, f"{base_name}_preprocessed.csv"),
                        processed_json_path  # Gemini output file
                    ]:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            print(f"Deleted: {file_path}")

                    processed_results.append({
                        'table_number': idx,
                        'original_csv': csv_path,
                        'decoded_json': decoded_json_path
                    })
                else:
                    print(f"Skipping decoding for table {idx} due to invalid Gemini output.")

            return processed_results
        
        def preprocess_csv_for_gemini(self, df):
            """
            Preprocess CSV by consolidating related rows into single entries
            before sending to Gemini.
            """
            # Create a new DataFrame to store consolidated data
            consolidated_rows = []
            current_entry = None
            columns = df.columns.tolist()
            
            # Function to create a new entry dictionary
            def create_entry_dict():
                return {col: [] for col in columns}
            
            for idx, row in df.iterrows():
                # Check if this is a new main entry (has an S. No. or first column value)
                if pd.notna(row[0]) and str(row[0]).strip():
                    # Save previous entry if exists
                    if current_entry:
                        # Clean up empty lists and convert single-item lists to values
                        cleaned_entry = {}
                        for col, values in current_entry.items():
                            if values:
                                cleaned_entry[col] = " ".join(values)  # Join list into space-separated string
                            else:
                                cleaned_entry[col] = ''
                        consolidated_rows.append(cleaned_entry)
                    
                    # Start new entry
                    current_entry = create_entry_dict()
                    # Add values from current row
                    for col in columns:
                        if pd.notna(row[col]) and str(row[col]).strip():
                            current_entry[col].append(str(row[col]).strip())
                
                # This is a continuation row
                elif current_entry is not None:
                    # Add non-empty values to their respective columns
                    for col in columns:
                        if pd.notna(row[col]) and str(row[col]).strip():
                            current_entry[col].append(str(row[col]).strip())
            
            # Don't forget to add the last entry
            if current_entry:
                cleaned_entry = {}
                for col, values in current_entry.items():
                    if values:
                        cleaned_entry[col] = " ".join(values)  # Join list into space-separated string
                    else:
                        cleaned_entry[col] = ''
                consolidated_rows.append(cleaned_entry)
            
            # Convert consolidated data back to DataFrame
            consolidated_df = pd.DataFrame(consolidated_rows)
            
            return consolidated_df


        def process_csv_with_gemini(self, df, output_dir, table_idx):
            """Process CSV data directly using Gemini Pro and save the output."""
            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.3,
                google_api_key=self.google_api_key
            )

            # Convert DataFrame to string representation
            csv_str = df.to_csv(index=False)
            
            prompt_template = """
            Restructure the following encoded csv file into a proper nested JSON format using original headers and values:
            {context}

            Goal: Create a hierarchical JSON that preserves original structure and relationships.
            Make sure no column names and values are left from csv file in nested json.
            Structure the output as a nested JSON with a "Data" array containing row objects.
            Ensure the output starts with '{{' and ends with '}}'.
            Important: Do NOT include markdown code blocks or backticks in your response. Just return the raw JSON.
            """

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context"]
            )

            docs = [Document(page_content=csv_str)]
            chain = load_qa_chain(
                llm=model,
                chain_type="stuff",
                prompt=prompt,
                document_variable_name="context"
            )

            # Run Gemini processing
            print("\nProcessing with Gemini...")
            response = chain.run(docs)
            
            print("\n===== Raw Response from Gemini =====\n")
            #print(response)
            print("\n====================================\n")

            # Save the raw response to a file
            raw_output_path = os.path.join(output_dir, f"table_{table_idx}_gemini_output.json")
            with open(raw_output_path, 'w', encoding='utf-8') as f:
                f.write(response)
            
            print(f"Raw Gemini output saved to: {raw_output_path}")

            # Ensure response is in JSON format
            try:
                json.loads(response)
                return raw_output_path
            except json.JSONDecodeError as e:
                print("Error parsing JSON:", e)
                return None

        def process_pdf(self, pdf_path, output_dir):
            """Main function to process PDF and generate all outputs."""
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract tables from PDF
            print("Extracting tables from PDF...")
            tables = self.extract_tables_from_pdf(pdf_path)
            if tables is None:
                return None
            
            # Process each table
            results = self.process_tables(tables, output_dir)
            
            # Save processing summary
            summary_path = os.path.join(output_dir, "processing_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"\nProcessing summary saved to: {summary_path}")
            
            return results
        
        def is_header_like(self, value):
            """
            Determine if a value looks like a header or label rather than data.
            """
            # Convert to string and clean
            value = str(value).strip()
            
            # Patterns that suggest a header/label
            header_patterns = [
                r'^[A-Z\s]{3,}$',  # All caps text
                r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$',  # Title case words
                r'.*[:()].*',  # Contains colon or parentheses
                r'.*[/\\].*',  # Contains slashes
                r'^\d+\.\s+\w+',  # Numbered items
                r'.+\s+of\s+.+',  # Descriptive phrases with 'of'
                r'^(Total|Average|Mean|Sum|Count)\b'  # Common header words
            ]
            
            return any(bool(re.match(pattern, value)) for pattern in header_patterns)

        def hash_six_digits(self, value):
            """
            Generate a 6-digit hash for a given string value.
            Carefully preserves percentage signs and other suffixes in the output.
            """
            if pd.isna(value) or str(value).strip() == '':
                return ''
                
            value_str = str(value).strip()
            
            # Special handling for percentage values
            percentage_match = re.match(r'^([-+]?\d*\.?\d+)\s*%\s*$', value_str)
            if percentage_match:
                number_part = percentage_match.group(1)
                # Generate hash for the number part and append '%'
                hashed = hashlib.sha256(number_part.encode()).hexdigest()[:6]
                return f"{hashed}%"
            
            # Handle other numeric values with potential suffixes
            match = re.match(r'^([-+]?\d*\.?\d+)(\s*[A-Za-z%]+\s*)?$', value_str)
            if match:
                number_part, suffix = match.groups()
                suffix = suffix if suffix else ''
                # Generate hash for the number part only
                hashed = hashlib.sha256(number_part.encode()).hexdigest()[:6]
                return f"{hashed}{suffix}"
            
            return hashlib.sha256(value_str.encode()).hexdigest()[:6]

        def detect_table_boundaries(self, data):
            """
            Detect the start and end of the actual table content.
            Returns tuple: (header_start, data_start, data_end) indices
            """
            header_start = 0
            data_start = 0
            data_end = len(data)
            
            for i, row in data.iterrows():
                if row.notna().sum() > 0:
                    header_start = i
                    break
            
            for i, row in data.iloc[header_start:].iterrows():
                if row.notna().sum() > len(row) * 0.5:
                    data_start = i + 1
                    break
            
            for i in range(len(data) - 1, data_start, -1):
                if data.iloc[i].notna().sum() > 0:
                    data_end = i + 1
                    break
                    
            return header_start, data_start, data_end

        def should_hash_column(self, column_data, column_name, index):
            """
            Determine if a column's data should be hashed based on its content.
            Returns False for header/label columns.
            """
            # Don't hash the first column (usually contains labels/categories)
            if index == 0:
                return False
                
            # Only consider non-empty values
            non_empty_values = [
                x for x in column_data 
                if pd.notna(x) and str(x).strip() != ''
            ]
            
            if not non_empty_values:  # Skip entirely empty columns
                return False
                
            # Check if column contains mostly numeric data
            numeric_count = sum(
                1 for x in non_empty_values 
                if (
                    str(x).replace('.', '').isdigit() or 
                    str(x).replace('.', '').replace('%', '').isdigit()
                )
            )
            
            numeric_ratio = numeric_count / len(non_empty_values) if non_empty_values else 0
            
            # Return True if more than 50% of non-empty values are numeric
            return numeric_ratio > 0.5
        
        def encode_csv_data(self, data):
            """
            Encode only numeric data inside the table while preserving headers and text content.
            Works with any table structure without hardcoding column names.
            Returns encoded DataFrame and data boundaries.
            """
            header_start, data_start, data_end = self.detect_table_boundaries(data)
            result = data.copy()
            
            # Process each row after the header section
            for row_idx in range(data_start, data_end):
                row = result.iloc[row_idx]
                for col_idx, value in enumerate(row):
                    if pd.notna(value) and str(value).strip():
                        # Check if the value is numeric or percentage
                        value_str = str(value).strip()
                        
                        # Skip if the value appears to be a header or label
                        if self.is_header_like(value_str):
                            continue
                            
                        # Clean the value for numeric checking
                        cleaned_value = value_str.replace(',', '')  # Remove commas from numbers
                        
                        # Check if it's a numeric value
                        is_numeric = (
                            cleaned_value.replace('.', '').isdigit() or  # Pure numbers
                            cleaned_value.endswith('%') and cleaned_value[:-1].replace('.', '').isdigit() or  # Percentages
                            bool(re.match(r'^\d+\.?\d*$', cleaned_value))  # Decimal numbers
                        )
                        
                        if is_numeric:
                            result.iloc[row_idx, col_idx] = self.hash_six_digits(value_str)
            
            print(f"\nTable content encoded from row {data_start} to {data_end}")
            return result, (data_start, data_end)
        
        def create_hash_mapping(self, df):
            """
            Create a mapping of hash values to original values from the DataFrame.
            Enhanced to properly handle percentage values and other numeric suffixes.
            """
            hash_map = {}
            
            def process_value(value):
                if pd.isna(value):
                    return
                
                value_str = str(value).strip()
                if not value_str:
                    return
                    
                # Handle percentage values
                percentage_match = re.match(r'^([-+]?\d*\.?\d+)\s*%\s*$', value_str)
                if percentage_match:
                    number_part = percentage_match.group(1)
                    hash_value = hashlib.sha256(number_part.encode()).hexdigest()[:6]
                    hash_map[f"{hash_value}%"] = value_str
                    return
                    
                # Handle other numeric values with suffixes
                match = re.match(r'^([-+]?\d*\.?\d+)(\s*[A-Za-z%]+\s*)?$', value_str)
                if match:
                    number_part, suffix = match.groups()
                    suffix = suffix if suffix else ''
                    hash_value = hashlib.sha256(number_part.encode()).hexdigest()[:6]
                    hash_map[f"{hash_value}{suffix}"] = value_str
                    return
                    
                # Handle regular values
                hash_value = hashlib.sha256(value_str.encode()).hexdigest()[:6]
                hash_map[hash_value] = value_str

            # Process all values in the DataFrame
            for column in df.columns:
                for value in df[column]:
                    process_value(value)
            
            return hash_map
        
        def decode_json_data(self, json_data, hash_mapping):
            """
            Replace hash values in JSON with original values using the mapping.
            Enhanced to handle percentage values and other numeric suffixes.
            """
            if isinstance(json_data, dict):
                return {k: self.decode_json_data(v, hash_mapping) for k, v in json_data.items()}
            elif isinstance(json_data, list):
                return [self.decode_json_data(item, hash_mapping) for item in json_data]
            elif isinstance(json_data, str):
                words = json_data.split()
                decoded_words = []
                
                for word in words:
                    # Try to match the word exactly in the hash mapping
                    if word in hash_mapping:
                        decoded_words.append(hash_mapping[word])
                        continue
                    
                    # Check if it's a hashed value with a suffix
                    match = re.match(r'^([a-f0-9]{6})([A-Za-z%]+)?$', word)
                    if match:
                        hash_part, suffix = match.groups()
                        suffix = suffix if suffix else ''
                        full_hash = f"{hash_part}{suffix}"
                        if full_hash in hash_mapping:
                            decoded_words.append(hash_mapping[full_hash])
                        else:
                            decoded_words.append(word)  # Keep original if no match
                    else:
                        decoded_words.append(word)  # Keep original if no match
                        
                return " ".join(decoded_words)
            else:
                return json_data

        def clean_cell_content(self, cell):
            """Clean cell content while preserving newlines within cells."""
            if cell is None:
                return ""
            cell = str(cell)
            lines = cell.split('\n')
            lines = [line.strip() for line in lines if line.strip()]
            return '\n'.join(lines)
        
    class PDFProcessorB:
        def __init__(self):
            # Load environment variables
            load_dotenv()
            self.google_api_key = os.getenv("GOOGLE_API_KEY")
            if not self.google_api_key:
                raise ValueError("API key not found. Make sure you have set GOOGLE_API_KEY in your .env file.")
            
            # Define section patterns
            self.section_patterns = {
                'A': [
                    r'SECTION\s+A\s*[-:]?\s*GENERAL\s+DISCLOSURES',
                    r'Section\s+A\s*[-:]?\s*General\s+Disclosures',
                    r'SECTION\s+A',
                    r'Section\s+A'
                ],
                'B': [
                    r'SECTION\s+B\s*[-:]?\s*MANAGEMENT\s+AND\s+PROCESS\s+DISCLOSURES',
                    r'Section\s+B\s*[-:]?\s*Management\s+and\s+Process\s+Disclosures',
                    r'SECTION\s+B',
                    r'Section\s+B'
                ],
                'C': [
                    r'SECTION\s+C\s*[-:]?\s*PRINCIPLE\s+WISE\s+PERFORMANCE\s+DISCLOSURE',
                    r'Section\s+C\s*[-:]?\s*Principle\s+Wise\s+Performance\s+Disclosure'
                ]
            }
            
            # Specific Roman numeral patterns
            self.roman_numerals = [
                r"i\.",
                r"ii\.",
                r"iii\.",
                r"iv\.",
                r"v\.",
                r"vi\.",
                r"vii\."
            ]
            
            self.start_page_number = None  # Will be set when process_section_headings is called
            # Use roman numerals as heading patterns
            self.heading_patterns = self.roman_numerals

        def get_user_section_choice(self):
            """Ask user which section to process."""
            while True:
                choice = "B"
                if choice in self.section_patterns:
                    return choice
                print("Invalid choice. Please enter A, B, or C.")

        def detect_start_page(self, pdf_path):
            """
            Detect the page number containing "SECTION B: MANAGEMENT AND PROCESS DISCLOSURES"
            Returns the 1-based page number if found, otherwise returns the default page number
            Processes pages in batches of 50 for improved efficiency
            """
            default_page = 49  # Default fallback if section not found
            
            # Define patterns to search for Section B
            section_b_patterns = [
                r'SECTION\s+B\s*[-:]?\s*MANAGEMENT\s+AND\s+PROCESS\s+DISCLOSURES',
                r'Section\s+B\s*[-:]?\s*Management\s+and\s+Process\s+Disclosures',
                r'SECTION\s+B', 
                r'Section\s+B'
            ]
            
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    print("\nScanning document for SECTION B...")
                    total_pages = len(pdf.pages)
                    
                    # Process in batches of 50 pages
                    batch_size = 50
                    for batch_start in range(0, total_pages, batch_size):
                        batch_end = min(batch_start + batch_size, total_pages)
                        print(f"Processing batch: pages {batch_start + 1} to {batch_end}")
                        
                        # Process each page in the current batch
                        for page_num in range(batch_start, batch_end):
                            page = pdf.pages[page_num]
                            text = page.extract_text()
                            
                            if text is None:
                                continue
                            
                            for pattern in section_b_patterns:
                                if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                                    # Found section B on this page (add 1 for 1-based page number)
                                    print(f"Found SECTION B on page {page_num + 1}")
                                    return page_num + 1  # Return 1-based page number
                    
                    print(f"SECTION B not found, using default page {default_page}")
                    return default_page
            
            except Exception as e:
                print(f"Error scanning for SECTION B: {str(e)}")
                print(f"Using default page {default_page}")
                return default_page

        def detect_headings(self, text, page_num):
            """Enhanced heading detection focusing on Roman numerals."""
            headings = []
            lines = text.split('\n')
            
            print(f"\nScanning page {page_num + 1} for Roman numeral headings...")
            
            for line in lines:
                line = line.strip()
                if line:  # Only process non-empty lines
                    # Convert to lowercase for consistent matching
                    line_lower = line.lower()
                    
                    # Try to find Roman numerals at the start of the line
                    for pattern in self.heading_patterns:
                        if re.match(pattern, line_lower):
                            # Get the full line as heading
                            print(f"Found heading: {line}")
                            headings.append(line)
                            break
            
            return headings
        
        def process_section_headings(self, pdf_path, output_dir):
            """Process all headings within a chosen section."""
            # Detect the start page number for Section B
            self.start_page_number = self.detect_start_page(pdf_path)
            
            section_choice = self.get_user_section_choice()
            self.current_section = section_choice
            
            print(f"\nProcessing SECTION {section_choice} starting from page {self.start_page_number}")
            
            section_dir = os.path.join(output_dir, f"SECTION_{section_choice}")
            os.makedirs(section_dir, exist_ok=True)
            
            with pdfplumber.open(pdf_path) as pdf:
                # Ensure the PDF has enough pages
                if len(pdf.pages) < self.start_page_number:
                    print(f"PDF has fewer than {self.start_page_number} pages!")
                    return None
                    
                # Convert to 0-based index
                start_page = self.start_page_number - 1
                end_page = len(pdf.pages) - 1
                start_y = None
                
                # Find start position on page 106
                page = pdf.pages[start_page]
                text = page.extract_text()
                words = page.extract_words()
                
                # Find section boundaries
                self.current_start_patterns = self.section_patterns[section_choice]
                next_section = chr(ord(section_choice) + 1)
                self.current_end_patterns = (
                    self.section_patterns[next_section] if next_section in self.section_patterns 
                    else [r"SECTION\s+[" + next_section + r"-Z]"]
                )
                
                for pattern in self.current_start_patterns:
                    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                    if match:
                        matched_text = match.group(0)
                        first_word = matched_text.split()[0]
                        for word in words:
                            if first_word.lower() in word['text'].lower():
                                start_y = word['top']
                                print(f"Found section start '{matched_text}' on page {start_page + 1} at y-position: {start_y}")
                                break
                    if start_y is not None:
                        break
                
                if start_y is None:
                    print(f"Could not find SECTION {section_choice} on page {self.start_page_number}!")
                    return None
                    
                # Find end position (next section)
                end_y = None
                end_page = None
                
                # Search for the next section (Section C if current is Section B)
                for page_num in range(start_page, len(pdf.pages)):
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    words = page.extract_words()
                    
                    # Check if this is the end pattern (next section)
                    for pattern in self.current_end_patterns:
                        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                        if match:
                            matched_text = match.group(0)
                            first_word = matched_text.split()[0]
                            for word in words:
                                if first_word.lower() in word['text'].lower():
                                    end_y = word['top']
                                    end_page = page_num
                                    print(f"Found section end '{matched_text}' on page {page_num + 1} at y-position: {end_y}")
                                    break
                        if end_y is not None:
                            break
                    if end_y is not None:
                        break
                
                # If no end found, set end as the last page
                if end_y is None:
                    end_page = len(pdf.pages) - 1
                    end_y = float('inf')
                    print(f"No explicit section end found, using last page ({end_page + 1})")
                    
                print(f"\nSection boundaries: Pages {start_page + 1} to {end_page + 1}")
                
                all_results = {}
                
                # Special handling for section B (no subsections) - process whole section at once
                if section_choice == 'B':
                    print("\nProcessing Section B (no subsections detected)")
                    
                    # Set boundaries for table extraction
                    self.current_heading_start = {
                        'page': start_page,
                        'y': start_y
                    }
                    
                    self.current_heading_end = {
                        'page': end_page,
                        'y': end_y
                    }
                    
                    # Extract and process tables
                    tables = self.extract_tables_from_pdf(pdf)
                    if tables:
                        results = self.process_tables(tables, section_dir)
                        all_results["section_B_tables"] = {
                            'heading': 'SECTION B',
                            'tables': results
                        }
                        print(f"Processed {len(results)} tables under SECTION B")
                    else:
                        print("No tables found under SECTION B")
                        all_results["section_B_tables"] = {
                            'heading': 'SECTION B',
                            'tables': []
                        }
                        
                else:
                    # Original logic for sections with subsections (like A and C)
                    # Define Roman numeral patterns with word boundaries
                    roman_patterns = [
                        (r"\bi\.", "i"),
                        (r"\bii\.", "ii"),
                        (r"\biii\.", "iii"),
                        (r"\biv\.", "iv"),
                        (r"\bv\.", "v"),
                        (r"\bvi\.", "vi"),
                        (r"\bvii\.", "vii")
                    ]
                    
                    # Find Roman numeral headings sequentially
                    heading_locations = []
                    current_pattern_index = 0
                    vii_found = False
                    vii_page = None
                    vii_y = None
                    
                    print("\nScanning for Roman numeral headings sequentially...")
                    
                    while current_pattern_index < len(roman_patterns):
                        pattern, numeral = roman_patterns[current_pattern_index]
                        found_current = False
                        
                        search_start_page = heading_locations[-1]['page'] if heading_locations else start_page
                        
                        for page_num in range(search_start_page, end_page + 1):
                            if found_current:
                                break
                                
                            page = pdf.pages[page_num]
                            text = page.extract_text()
                            words = page.extract_words()
                            
                            matches = re.finditer(pattern, text.lower())
                            for match in matches:
                                match_start = match.start()
                                match_end = match.end()
                                
                                if match_start > 0 and text[match_start-1:match_start].lower() in 'iv':
                                    continue
                                    
                                match_text = match.group(0)
                                for word in words:
                                    if match_text in word['text'].lower():
                                        if current_pattern_index > 0:
                                            prev_heading = heading_locations[-1]
                                            if page_num < prev_heading['page'] or \
                                            (page_num == prev_heading['page'] and word['top'] <= prev_heading['y']):
                                                continue
                                        
                                        heading_locations.append({
                                            'numeral': numeral,
                                            'page': page_num,
                                            'y': word['top'],
                                            'pattern': pattern
                                        })
                                        print(f"Found heading {numeral} on page {page_num + 1} at y={word['top']}")
                                        
                                        # If this is 'vii', mark it as the end point
                                        if numeral == 'vii':
                                            vii_found = True
                                            vii_page = page_num
                                            vii_y = word['top']
                                            print(f"Found 'vii.' - marking as end point on page {page_num + 1} at y={word['top']}")
                                        
                                        found_current = True
                                        break
                                
                                if found_current:
                                    break
                        
                        if found_current:
                            current_pattern_index += 1
                            if vii_found:
                                print("Found 'vii.' - stopping heading detection")
                                break
                        else:
                            if current_pattern_index == 0:
                                print("Could not find starting pattern 'i.'!")
                                return None
                            else:
                                print(f"No more patterns found after {roman_patterns[current_pattern_index-1][1]}.")
                                break
                    
                    if not heading_locations:
                        print("No Roman numeral headings found in section!")
                        return None
                    
                    # Process tables between each pair of headings
                    for i in range(len(heading_locations)):
                        current_heading = heading_locations[i]
                        next_heading = heading_locations[i + 1] if i + 1 < len(heading_locations) else None
                        
                        print(f"\nProcessing subsection {current_heading['numeral'].upper()}")
                        
                        heading_dir = os.path.join(section_dir, f"subsection_{current_heading['numeral']}")
                        os.makedirs(heading_dir, exist_ok=True)
                        
                        # Set boundaries for table extraction
                        self.current_heading_start = {
                            'page': current_heading['page'],
                            'y': current_heading['y']
                        }
                        
                        if next_heading:
                            self.current_heading_end = {
                                'page': next_heading['page'],
                                'y': next_heading['y']
                            }
                        elif vii_found and current_heading['numeral'] == 'vii':
                            # If this is 'vii', only process until the end of its section
                            self.current_heading_end = {
                                'page': vii_page,
                                'y': float('inf')  # Process until end of the vi page
                            }
                        elif vii_found:
                            # If vi was found but this is an earlier heading,
                            # process until the next heading or vi
                            next_vii_heading = next(
                                (h for h in heading_locations if h['numeral'] == 'vii'),
                                None
                            )
                            if next_vii_heading:
                                self.current_heading_end = {
                                    'page': next_vii_heading['page'],
                                    'y': next_vii_heading['y']
                                }
                            else:
                                # Fallback in case vi isn't found in heading_locations
                                self.current_heading_end = {
                                    'page': vii_page,
                                    'y': vii_y
                                }
                        else:
                            # If vi wasn't found, process until end of current page
                            self.current_heading_end = {
                                'page': current_heading['page'],
                                'y': float('inf')
                            }
                        
                        # Skip if this heading comes after vi
                        if vii_found and current_heading['numeral'] > 'vii':
                            print(f"Skipping subsection {current_heading['numeral']} as it comes after vii")
                            continue
                        
                        # Extract and process tables
                        tables = self.extract_tables_from_pdf(pdf)
                        if tables:
                            results = self.process_tables(tables, heading_dir)
                            all_results[f"subsection_{current_heading['numeral']}"] = {
                                'heading': current_heading['numeral'].upper(),
                                'tables': results
                            }
                            print(f"Processed {len(results)} tables under subsection {current_heading['numeral'].upper()}")
                        else:
                            print(f"No tables found under subsection {current_heading['numeral'].upper()}")
                            all_results[f"subsection_{current_heading['numeral']}"] = {
                                'heading': current_heading['numeral'].upper(),
                                'tables': []
                            }
                
                # Save section summary
                summary_path = os.path.join(section_dir, "section_summary.json")
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2)
                
                return all_results

        def process_all_sections(self, pdf_path, output_dir):
            """Process all sections sequentially."""
            print("\nStarting sequential processing of all sections...")
            
            # Create main output directory
            os.makedirs(output_dir, exist_ok=True)
            
            all_results = {}
            
            # Process each section
            for section in ['A', 'B', 'C']:
                print(f"\n{'='*50}")
                print(f"Processing SECTION {section}")
                print(f"{'='*50}")
                
                # Create section-specific output directory
                section_dir = os.path.join(output_dir, f"SECTION_{section}")
                os.makedirs(section_dir, exist_ok=True)
                
                # Set patterns for current section
                self.current_start_patterns = self.section_patterns[section]
                next_section = chr(ord(section) + 1)
                self.current_end_patterns = (
                    self.section_patterns[next_section] if next_section in self.section_patterns 
                    else self.roman_numerals
                )
                
                # Process the section
                try:
                    with pdfplumber.open(pdf_path) as pdf:
                        # Extract tables from the section
                        tables = self.extract_tables_from_pdf(pdf)
                        if tables:
                            # Process the tables
                            results = self.process_tables(tables, section_dir)
                            all_results[f"SECTION_{section}"] = results
                            print(f"\nSuccessfully processed {len(results)} tables in SECTION {section}")
                        else:
                            print(f"\nNo tables found in SECTION {section}")
                            all_results[f"SECTION_{section}"] = []
                except Exception as e:
                    print(f"\nError processing SECTION {section}: {str(e)}")
                    all_results[f"SECTION_{section}"] = []
            
            # Save overall processing summary
            summary_path = os.path.join(output_dir, "overall_processing_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nOverall processing summary saved to: {summary_path}")
            
            return all_results

        def get_section_patterns(self):
            """
            Ask user for section to extract and create appropriate patterns.
            Returns tuple of (start_pattern, end_patterns)
            """
            # Ask user for section name
            section_name = input("\nWhich section do you want to extract? ").strip()

            # Helper function to create pattern variations
            def create_pattern_variations(text):
                # Split the text into words
                words = text.split()
                
                # Create word boundary pattern with flexible whitespace
                word_boundary_pattern = r'\s+'.join(rf'\b{re.escape(word)}\b' for word in words)
                
                # Create patterns with different cases and formats
                patterns = [
                    word_boundary_pattern,  # Word boundaries with flexible spacing
                    r'\s+'.join(re.escape(word) for word in words),  # Exact match with flexible spacing
                    r'\s+'.join(re.escape(word.upper()) for word in words),  # All caps with flexible spacing
                    r'\s+'.join(re.escape(word.lower()) for word in words),  # All lowercase with flexible spacing
                    rf'\d+\.\s*{word_boundary_pattern}',  # Numbered sections
                    rf'[A-Z]\.\s*{word_boundary_pattern}',  # Letter-based sections
                    # Add patterns for handling hyphenated variations
                    word_boundary_pattern.replace(r'\s+', '-'),  # Hyphenated version
                    word_boundary_pattern.replace(r'\s+', '_'),  # Underscore version
                ]
                
                # Add patterns for common heading formats
                heading_variations = [
                    f"(?i)(?:SECTION|Section)\\s*\\d*\\s*[-:]?\\s*{word_boundary_pattern}",  # Section X: Pattern
                    f"(?i)(?:PART|Part)\\s*\\d*\\s*[-:]?\\s*{word_boundary_pattern}",  # Part X: Pattern
                    f"(?i)\\d+\\.\\d+\\s*{word_boundary_pattern}",  # 1.1 Pattern
                    f"(?i)[A-Z]\\)\\s*{word_boundary_pattern}",  # A) Pattern
                ]
                
                patterns.extend(heading_variations)
                return patterns
            
            # Generate patterns
            start_patterns = create_pattern_variations(section_name)
            
            print("\nSearching for section:", section_name)
            print("\nGenerated patterns:")
            for i, pattern in enumerate(start_patterns):
                print(f"{i+1}. {pattern}")
                
            return start_patterns, self.roman_numerals

        def detect_section_boundaries(self, pdf):
            """Detect the start and end pages and positions of the target section."""
            start_page = None
            start_y = None
            end_page = None
            end_y = None
            in_section = False
            
            print("\nScanning document for section boundaries...")
            
            # Start scanning from page 1
            for page_num in range(len(pdf.pages)):
                page = pdf.pages[page_num]
                text = page.extract_text()
                if text is None:
                    continue
                    
                print(f"\nAnalyzing page {page_num + 1}")
                
                # If we haven't found the section start yet
                if not in_section:
                    for pattern in self.current_start_patterns:
                        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                        if match:
                            start_page = page_num
                            words = page.extract_words()
                            matched_text = match.group(0)
                            first_word = matched_text.split()[0]
                            for word in words:
                                if first_word.lower() in word['text'].lower():
                                    start_y = word['top']
                                    print(f"Found section start '{matched_text}' on page {page_num + 1} at y-position: {start_y}")
                                    in_section = True
                                    break
                        if in_section:
                            break
                
                # If we're in the section, look for the end
                elif in_section:
                    for pattern in self.current_end_patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            end_page = page_num
                            words = page.extract_words()
                            for word in words:
                                if re.search(pattern, word['text'], re.IGNORECASE):
                                    end_y = word['top']
                                    print(f"Found section end on page {page_num + 1} at y-position: {end_y}")
                                    return start_page, end_page, start_y, end_y
            
            # If we found start but no end, use last page
            if start_page is not None:
                end_page = len(pdf.pages) - 1
                end_y = float('inf')
                print(f"No explicit section end found, using last page ({end_page + 1})")
            
            return start_page, end_page, start_y, end_y

        def extract_tables_from_pdf(self, pdf):
            """Modified to use heading boundaries."""
            start_page = self.current_heading_start['page']
            end_page = self.current_heading_end['page']
            start_y = self.current_heading_start['y']
            end_y = self.current_heading_end['y']
            
            all_tables = []
            current_table = []
            table_count = 0
            
            for page_num in range(start_page, end_page + 1):
                page = pdf.pages[page_num]
                
                # Extract tables from the current page
                tables = page.extract_tables()
                if not tables:
                    tables = page.extract_tables(table_settings={
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "intersection_y_tolerance": 10
                    })
                
                if tables:
                    table_positions = page.find_tables()
                    
                    for table_idx, (table, table_pos) in enumerate(zip(tables, table_positions)):
                        table_top = table_pos.bbox[1]
                        table_bottom = table_pos.bbox[3]
                        
                        # Check if table is within heading boundaries
                        include_table = False
                        
                        if page_num == start_page:
                            if table_top > start_y:
                                include_table = True
                        elif page_num == end_page and end_y is not None:
                            if table_bottom < end_y:
                                include_table = True
                        else:
                            include_table = True
                        
                        if include_table:
                            # Process table as before
                            cleaned_table = [[self.clean_cell_content(cell) for cell in row] for row in table]
                            df = pd.DataFrame(cleaned_table)
                            df = df.dropna(how='all').dropna(axis=1, how='all')
                            
                            if not df.empty:
                                if current_table and self.is_table_continuation(current_table[-1], df):
                                    current_table.append(df)
                                else:
                                    if current_table:
                                        processed_table = pd.concat(current_table, ignore_index=True)
                                        all_tables.append(processed_table)
                                        table_count += 1
                                    current_table = [df]
            
            # Process any remaining table
            if current_table:
                processed_table = pd.concat(current_table, ignore_index=True)
                all_tables.append(processed_table)
                table_count += 1
            
            return all_tables if all_tables else None
            
        def is_table_continuation(self, prev_df, curr_df):
            """
            Determine if the current table is a continuation of the previous table.
            """
            # Check if number of columns match
            if len(prev_df.columns) != len(curr_df.columns):
                return False
            
            try:
                # Convert column labels to strings and compare
                prev_headers = [str(col).lower().strip() for col in prev_df.columns]
                curr_headers = [str(col).lower().strip() for col in curr_df.columns]
                
                # Check if headers are identical or very similar
                header_similarity = sum(p == c for p, c in zip(prev_headers, curr_headers))
                header_match = header_similarity / len(prev_headers) > 0.8
                
                # Check if current table starts with numeric data (likely continuation)
                first_row_numeric = all(
                    str(x).replace('.', '').replace('%', '').strip().isdigit() 
                    for x in curr_df.iloc[0] 
                    if pd.notna(x) and str(x).strip()
                )
                
                # Check for similar data patterns in last row of prev and first row of current
                prev_last_row = prev_df.iloc[-1]
                curr_first_row = curr_df.iloc[0]
                
                pattern_matches = 0
                total_checks = 0
                
                for i in range(len(prev_df.columns)):
                    prev_val = str(prev_last_row.iloc[i]).strip() if pd.notna(prev_last_row.iloc[i]) else ''
                    curr_val = str(curr_first_row.iloc[i]).strip() if pd.notna(curr_first_row.iloc[i]) else ''
                    
                    if prev_val and curr_val:  # Only check non-empty values
                        total_checks += 1
                        # Check if both are numeric or both are text
                        prev_is_numeric = prev_val.replace('.', '').replace('%', '').isdigit()
                        curr_is_numeric = curr_val.replace('.', '').replace('%', '').isdigit()
                        if prev_is_numeric == curr_is_numeric:
                            pattern_matches += 1
                
                pattern_similarity = pattern_matches / total_checks if total_checks > 0 else 0
                
                # Return True if either condition is met
                return header_match or (first_row_numeric and pattern_similarity > 0.8)
                    
            except Exception as e:
                print(f"Warning: Error comparing tables: {str(e)}")
                return False

        def process_tables(self, tables, output_dir):
            """Process multiple tables and store only the original CSV and decoded JSON."""
            processed_results = []

            for idx, df in enumerate(tables, 1):
                print(f"\nProcessing table {idx}...")

                # Save original CSV
                base_name = f"table_{idx}"
                csv_path = os.path.join(output_dir, f"{base_name}_original.csv")
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"Original table {idx} saved to: {csv_path}")

                # Create hash mapping
                hash_mapping = self.create_hash_mapping(df)

                # Encode data (but do NOT save it to a file)
                encoded_df, _ = self.encode_csv_data(df)

                # Preprocess for Gemini (but do NOT save it to a file)
                preprocessed_df = self.preprocess_csv_for_gemini(encoded_df)

                # Process with Gemini
                processed_json_path = self.process_csv_with_gemini(preprocessed_df, output_dir, idx)

                if processed_json_path:
                    try:
                        with open(processed_json_path, 'r', encoding='utf-8') as f:
                            processed_json = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse Gemini output as JSON for table {idx}: {e}")
                        continue

                    # Decode JSON using hash mapping
                    decoded_json = self.decode_json_data(processed_json, hash_mapping)

                    # Save only decoded JSON
                    decoded_json_path = os.path.join(output_dir, f"{base_name}_decoded.json")
                    with open(decoded_json_path, 'w', encoding='utf-8') as f:
                        json.dump(decoded_json, f, indent=2, ensure_ascii=False)

                    print(f"Decoded JSON saved to: {decoded_json_path}")

                    # Delete intermediate files if they were mistakenly saved elsewhere
                    for file_path in [
                        os.path.join(output_dir, f"{base_name}_encoded.csv"),
                        os.path.join(output_dir, f"{base_name}_preprocessed.csv"),
                        processed_json_path  # Gemini output file
                    ]:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            print(f"Deleted: {file_path}")

                    processed_results.append({
                        'table_number': idx,
                        'original_csv': csv_path,
                        'decoded_json': decoded_json_path
                    })
                else:
                    print(f"Skipping decoding for table {idx} due to invalid Gemini output.")

            return processed_results
        
        def preprocess_csv_for_gemini(self, df):
            """
            Preprocess CSV by consolidating related rows into single entries
            before sending to Gemini.
            """
            # Create a new DataFrame to store consolidated data
            consolidated_rows = []
            current_entry = None
            columns = df.columns.tolist()
            
            # Function to create a new entry dictionary
            def create_entry_dict():
                return {col: [] for col in columns}
            
            for idx, row in df.iterrows():
                # Check if this is a new main entry (has an S. No. or first column value)
                if pd.notna(row[0]) and str(row[0]).strip():
                    # Save previous entry if exists
                    if current_entry:
                        # Clean up empty lists and convert single-item lists to values
                        cleaned_entry = {}
                        for col, values in current_entry.items():
                            if values:
                                cleaned_entry[col] = " ".join(values)  # Join list into space-separated string
                            else:
                                cleaned_entry[col] = ''
                        consolidated_rows.append(cleaned_entry)
                    
                    # Start new entry
                    current_entry = create_entry_dict()
                    # Add values from current row
                    for col in columns:
                        if pd.notna(row[col]) and str(row[col]).strip():
                            current_entry[col].append(str(row[col]).strip())
                
                # This is a continuation row
                elif current_entry is not None:
                    # Add non-empty values to their respective columns
                    for col in columns:
                        if pd.notna(row[col]) and str(row[col]).strip():
                            current_entry[col].append(str(row[col]).strip())
            
            # Don't forget to add the last entry
            if current_entry:
                cleaned_entry = {}
                for col, values in current_entry.items():
                    if values:
                        cleaned_entry[col] = " ".join(values)  # Join list into space-separated string
                    else:
                        cleaned_entry[col] = ''
                consolidated_rows.append(cleaned_entry)
            
            # Convert consolidated data back to DataFrame
            consolidated_df = pd.DataFrame(consolidated_rows)
            
            return consolidated_df


        def process_csv_with_gemini(self, df, output_dir, table_idx):
            """Process CSV data directly using Gemini Pro and save the output."""
            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.3,
                google_api_key=self.google_api_key
            )

            # Convert DataFrame to string representation
            csv_str = df.to_csv(index=False)
            
            prompt_template = """
            Restructure the following encoded csv file into a proper nested JSON format using original headers and values:
            {context}

            Goal: Create a hierarchical JSON that preserves original structure and relationships.
            Make sure no column names and values are left from csv file in nested json.
            Structure the output as a nested JSON with a "Data" array containing row objects.
            Ensure the output starts with '{{' and ends with '}}'.
            Important: Do NOT include markdown code blocks or backticks in your response. Just return the raw JSON.
            """

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context"]
            )

            docs = [Document(page_content=csv_str)]
            chain = load_qa_chain(
                llm=model,
                chain_type="stuff",
                prompt=prompt,
                document_variable_name="context"
            )

            # Run Gemini processing
            print("\nProcessing with Gemini...")
            response = chain.run(docs)
            
            print("\n===== Raw Response from Gemini =====\n")
            #print(response)
            print("\n====================================\n")

            # Save the raw response to a file
            raw_output_path = os.path.join(output_dir, f"table_{table_idx}_gemini_output.json")
            with open(raw_output_path, 'w', encoding='utf-8') as f:
                f.write(response)
            
            print(f"Raw Gemini output saved to: {raw_output_path}")

            # Ensure response is in JSON format
            try:
                json.loads(response)
                return raw_output_path
            except json.JSONDecodeError as e:
                print("Error parsing JSON:", e)
                return None

        def process_pdf(self, pdf_path, output_dir):
            """Main function to process PDF and generate all outputs."""
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract tables from PDF
            print("Extracting tables from PDF...")
            tables = self.extract_tables_from_pdf(pdf_path)
            if tables is None:
                return None
            
            # Process each table
            results = self.process_tables(tables, output_dir)
            
            # Save processing summary
            summary_path = os.path.join(output_dir, "processing_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"\nProcessing summary saved to: {summary_path}")
            
            return results
        
        def is_header_like(self, value):
            """
            Determine if a value looks like a header or label rather than data.
            """
            # Convert to string and clean
            value = str(value).strip()
            
            # Patterns that suggest a header/label
            header_patterns = [
                r'^[A-Z\s]{3,}$',  # All caps text
                r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$',  # Title case words
                r'.*[:()].*',  # Contains colon or parentheses
                r'.*[/\\].*',  # Contains slashes
                r'^\d+\.\s+\w+',  # Numbered items
                r'.+\s+of\s+.+',  # Descriptive phrases with 'of'
                r'^(Total|Average|Mean|Sum|Count)\b'  # Common header words
            ]
            
            return any(bool(re.match(pattern, value)) for pattern in header_patterns)

        def hash_six_digits(self, value):
            """
            Generate a 6-digit hash for a given string value.
            Carefully preserves percentage signs and other suffixes in the output.
            """
            if pd.isna(value) or str(value).strip() == '':
                return ''
                
            value_str = str(value).strip()
            
            # Special handling for percentage values
            percentage_match = re.match(r'^([-+]?\d*\.?\d+)\s*%\s*$', value_str)
            if percentage_match:
                number_part = percentage_match.group(1)
                # Generate hash for the number part and append '%'
                hashed = hashlib.sha256(number_part.encode()).hexdigest()[:6]
                return f"{hashed}%"
            
            # Handle other numeric values with potential suffixes
            match = re.match(r'^([-+]?\d*\.?\d+)(\s*[A-Za-z%]+\s*)?$', value_str)
            if match:
                number_part, suffix = match.groups()
                suffix = suffix if suffix else ''
                # Generate hash for the number part only
                hashed = hashlib.sha256(number_part.encode()).hexdigest()[:6]
                return f"{hashed}{suffix}"
            
            return hashlib.sha256(value_str.encode()).hexdigest()[:6]

        def detect_table_boundaries(self, data):
            """
            Detect the start and end of the actual table content.
            Returns tuple: (header_start, data_start, data_end) indices
            """
            header_start = 0
            data_start = 0
            data_end = len(data)
            
            for i, row in data.iterrows():
                if row.notna().sum() > 0:
                    header_start = i
                    break
            
            for i, row in data.iloc[header_start:].iterrows():
                if row.notna().sum() > len(row) * 0.5:
                    data_start = i + 1
                    break
            
            for i in range(len(data) - 1, data_start, -1):
                if data.iloc[i].notna().sum() > 0:
                    data_end = i + 1
                    break
                    
            return header_start, data_start, data_end

        def should_hash_column(self, column_data, column_name, index):
            """
            Determine if a column's data should be hashed based on its content.
            Returns False for header/label columns.
            """
            # Don't hash the first column (usually contains labels/categories)
            if index == 0:
                return False
                
            # Only consider non-empty values
            non_empty_values = [
                x for x in column_data 
                if pd.notna(x) and str(x).strip() != ''
            ]
            
            if not non_empty_values:  # Skip entirely empty columns
                return False
                
            # Check if column contains mostly numeric data
            numeric_count = sum(
                1 for x in non_empty_values 
                if (
                    str(x).replace('.', '').isdigit() or 
                    str(x).replace('.', '').replace('%', '').isdigit()
                )
            )
            
            numeric_ratio = numeric_count / len(non_empty_values) if non_empty_values else 0
            
            # Return True if more than 50% of non-empty values are numeric
            return numeric_ratio > 0.5
        
        def encode_csv_data(self, data):
            """
            Encode only numeric data inside the table while preserving headers and text content.
            Works with any table structure without hardcoding column names.
            Returns encoded DataFrame and data boundaries.
            """
            header_start, data_start, data_end = self.detect_table_boundaries(data)
            result = data.copy()
            
            # Process each row after the header section
            for row_idx in range(data_start, data_end):
                row = result.iloc[row_idx]
                for col_idx, value in enumerate(row):
                    if pd.notna(value) and str(value).strip():
                        # Check if the value is numeric or percentage
                        value_str = str(value).strip()
                        
                        # Skip if the value appears to be a header or label
                        if self.is_header_like(value_str):
                            continue
                            
                        # Clean the value for numeric checking
                        cleaned_value = value_str.replace(',', '')  # Remove commas from numbers
                        
                        # Check if it's a numeric value
                        is_numeric = (
                            cleaned_value.replace('.', '').isdigit() or  # Pure numbers
                            cleaned_value.endswith('%') and cleaned_value[:-1].replace('.', '').isdigit() or  # Percentages
                            bool(re.match(r'^\d+\.?\d*$', cleaned_value))  # Decimal numbers
                        )
                        
                        if is_numeric:
                            result.iloc[row_idx, col_idx] = self.hash_six_digits(value_str)
            
            print(f"\nTable content encoded from row {data_start} to {data_end}")
            return result, (data_start, data_end)
        
        def create_hash_mapping(self, df):
            """
            Create a mapping of hash values to original values from the DataFrame.
            Enhanced to properly handle percentage values and other numeric suffixes.
            """
            hash_map = {}
            
            def process_value(value):
                if pd.isna(value):
                    return
                
                value_str = str(value).strip()
                if not value_str:
                    return
                    
                # Handle percentage values
                percentage_match = re.match(r'^([-+]?\d*\.?\d+)\s*%\s*$', value_str)
                if percentage_match:
                    number_part = percentage_match.group(1)
                    hash_value = hashlib.sha256(number_part.encode()).hexdigest()[:6]
                    hash_map[f"{hash_value}%"] = value_str
                    return
                    
                # Handle other numeric values with suffixes
                match = re.match(r'^([-+]?\d*\.?\d+)(\s*[A-Za-z%]+\s*)?$', value_str)
                if match:
                    number_part, suffix = match.groups()
                    suffix = suffix if suffix else ''
                    hash_value = hashlib.sha256(number_part.encode()).hexdigest()[:6]
                    hash_map[f"{hash_value}{suffix}"] = value_str
                    return
                    
                # Handle regular values
                hash_value = hashlib.sha256(value_str.encode()).hexdigest()[:6]
                hash_map[hash_value] = value_str

            # Process all values in the DataFrame
            for column in df.columns:
                for value in df[column]:
                    process_value(value)
            
            return hash_map
        
        def decode_json_data(self, json_data, hash_mapping):
            """
            Replace hash values in JSON with original values using the mapping.
            Enhanced to handle percentage values and other numeric suffixes.
            """
            if isinstance(json_data, dict):
                return {k: self.decode_json_data(v, hash_mapping) for k, v in json_data.items()}
            elif isinstance(json_data, list):
                return [self.decode_json_data(item, hash_mapping) for item in json_data]
            elif isinstance(json_data, str):
                words = json_data.split()
                decoded_words = []
                
                for word in words:
                    # Try to match the word exactly in the hash mapping
                    if word in hash_mapping:
                        decoded_words.append(hash_mapping[word])
                        continue
                    
                    # Check if it's a hashed value with a suffix
                    match = re.match(r'^([a-f0-9]{6})([A-Za-z%]+)?$', word)
                    if match:
                        hash_part, suffix = match.groups()
                        suffix = suffix if suffix else ''
                        full_hash = f"{hash_part}{suffix}"
                        if full_hash in hash_mapping:
                            decoded_words.append(hash_mapping[full_hash])
                        else:
                            decoded_words.append(word)  # Keep original if no match
                    else:
                        decoded_words.append(word)  # Keep original if no match
                        
                return " ".join(decoded_words)
            else:
                return json_data

        def clean_cell_content(self, cell):
            """Clean cell content while preserving newlines within cells."""
            if cell is None:
                return ""
            cell = str(cell)
            lines = cell.split('\n')
            lines = [line.strip() for line in lines if line.strip()]
            return '\n'.join(lines)
        
    class PDFProcessorC:
        def __init__(self):
            # Load environment variables
            load_dotenv()
            self.google_api_key = os.getenv("GOOGLE_API_KEY")
            if not self.google_api_key:
                raise ValueError("API key not found. Make sure you have set GOOGLE_API_KEY in your .env file.")
            
            # More robust patterns for Principles
            self.principle_patterns = [
                r"(?:PRINCIPLE|Principle|SEBI\s+PRINCIPLE|SEBI\s+Principle)\s*1[\s\.:]*(?:Businesses\s+should\s+conduct\s+and\s+govern\s+themselves\s+with\s+integrity)?",
                r"(?:PRINCIPLE|Principle|SEBI\s+PRINCIPLE|SEBI\s+Principle)\s*2[\s\.:]*(?:Businesses\s+should\s+provide\s+goods\s+and\s+services\s+that\s+are\s+safe\s+and\s+contribute\s+to\s+sustainability)?",
                r"(?:PRINCIPLE|Principle|SEBI\s+PRINCIPLE|SEBI\s+Principle)\s*3[\s\.:]*",
                r"(?:PRINCIPLE|Principle|SEBI\s+PRINCIPLE|SEBI\s+Principle)\s*4[\s\.:]*",
                r"(?:PRINCIPLE|Principle|SEBI\s+PRINCIPLE|SEBI\s+Principle)\s*5[\s\.:]*",
                r"(?:PRINCIPLE|Principle|SEBI\s+PRINCIPLE|SEBI\s+Principle)\s*6[\s\.:]*",
                r"(?:PRINCIPLE|Principle|SEBI\s+PRINCIPLE|SEBI\s+Principle)\s*7[\s\.:]*",
                r"(?:PRINCIPLE|Principle|SEBI\s+PRINCIPLE|SEBI\s+Principle)\s*8[\s\.:]*",
                r"(?:PRINCIPLE|Principle|SEBI\s+PRINCIPLE|SEBI\s+Principle)\s*9[\s\.:]*"
            ]

            # Define section patterns
            self.section_patterns = [
                r"(?i)Section\s+C\s*[:-]?\s*Principle-wise\s+Disclosures",
                r"(?i)SECTION\s+C\s*[:-]?\s*PRINCIPLE\s+WISE\s+PERFORMANCE\s+DISCLOSURE"
            ]
            
            self.principle1_patterns = [
                r"(?i)PRINCIPLE\s*1\s*[:.]?\s*Businesses\s+should\s+conduct\s+and\s+govern\s+themselves\s+with\s+integrity",
                r"(?i)PRINCIPLE\s*1"
            ]

            self.current_start_patterns = None
            self.current_end_patterns = None

        def create_principle_patterns(self, principle_num):
            """
            Create patterns for a specific principle.
            Returns tuple of (start_pattern, end_patterns)
            """
            # Create start patterns for the specified principle
            start_patterns = [
                rf"(?:PRINCIPLE|Principle|SEBI\s+PRINCIPLE|SEBI\s+Principle)\s*{principle_num}[\s\.:]*",
                rf"(?i)principle\s*{principle_num}[\s\.:]*",
                rf"(?i)sebi\s+principle\s*{principle_num}[\s\.:]*"
            ]
            
            # For end patterns, use the pattern of the next principle
            # If current principle is 9, use a custom end pattern
            end_principle = principle_num + 1 if principle_num < 9 else None
            
            if end_principle:
                end_patterns = [
                    rf"(?:PRINCIPLE|Principle|SEBI\s+PRINCIPLE|SEBI\s+Principle)\s*{end_principle}[\s\.:]*",
                    rf"(?i)principle\s*{end_principle}[\s\.:]*",
                    rf"(?i)sebi\s+principle\s*{end_principle}[\s\.:]*"
                ]
            else:
                # For Principle 9, use a pattern that matches the end of the section or document
                end_patterns = [
                    r"(?i)Section\s+D",
                    r"(?i)SECTION\s+D",
                    r"(?i)Annexure",
                    r"(?i)Appendix"
                ]
            
            print(f"\nPatterns for Principle {principle_num}:")
            print(f"Start patterns: {start_patterns}")
            print(f"End patterns: {end_patterns}")
            
            return start_patterns, end_patterns
        
        def find_start_page(self, pdf_path):
            """
            Automatically find the starting page that contains both 
            "Section C: Principle-wise Disclosures" and "Principle 1".
            """
            print("\nSearching for Section C start page...")
            
            with pdfplumber.open(pdf_path) as pdf:
                batch_size = 50
                total_pages = len(pdf.pages)
                
                for batch_start in range(0, total_pages, batch_size):
                    batch_end = min(batch_start + batch_size, total_pages)
                    print(f"Scanning pages {batch_start + 1} to {batch_end}...")
                    
                    for page_num in range(batch_start, batch_end):
                        page = pdf.pages[page_num]
                        text = page.extract_text()
                        if not text:
                            continue
                        
                        if (any(re.search(pattern, text, re.IGNORECASE) for pattern in self.section_patterns) and 
                            re.search(self.principle1_patterns[0], text, re.IGNORECASE)):
                            print(f"Found Section C start on page {page_num + 1}")
                            return page_num + 1  
                
                print("Warning: Could not find Section C start page. Using default page 1.")
                return 1

        def detect_section_boundaries(self, pdf, principle_num, start_scan_page):
            """
            Detect the start and end pages and positions of the target section.
            Returns more detailed boundary information.
            """
            # Get section patterns for the specific principle
            self.current_start_patterns, self.current_end_patterns = self.create_principle_patterns(principle_num)
            
            section_text = []
            start_page = None
            start_y = None
            end_page = None
            end_y = None
            in_section = False
            next_principle_position = None  # Track next principle start if found
            
            print(f"\nScanning document for Principle {principle_num} boundaries starting from page {start_scan_page + 1}...")
            
            for page_num in range(start_scan_page, len(pdf.pages)):
                page = pdf.pages[page_num]
                text = page.extract_text()
                if text is None:
                    continue
                    
                print(f"\nAnalyzing page {page_num + 1}")
                
                # If we haven't found the section start yet
                if not in_section:
                    for pattern in self.current_start_patterns:
                        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                        if match:
                            start_page = page_num
                            # Find the y-position of the section header
                            words = page.extract_words()
                            matched_text = match.group(0)
                            
                            # Look for the first word of the matched text
                            first_word = matched_text.split()[0]
                            for word in words:
                                if first_word.lower() in word['text'].lower():
                                    start_y = word['top']  # Use top position
                                    print(f"Found section start '{matched_text}' on page {page_num + 1}")
                                    in_section = True
                                    
                                    # Special case for Principle 9
                                    if principle_num == 9:
                                        # Set end page to be 4 pages after start page
                                        end_page = min(page_num + 4, len(pdf.pages) - 1)
                                        end_y = float('inf')  # Take the entire page
                                        print(f"Principle 9 detected. Setting end page to {end_page + 1}")
                                        return start_page, end_page, start_y, end_y, None
                                    break
                        if in_section:
                            break
                
                # If we're in the section, look for the end
                if in_section:
                    # Print the text being searched (for debugging)
                    print(f"Searching for end pattern on page {page_num + 1}")
                    
                    # First, check if next principle begins on this page
                    next_principle = principle_num + 1
                    if next_principle <= 9:  # Only for principles 1-8
                        next_principle_patterns = [
                            rf"(?:PRINCIPLE|Principle|SEBI\s+PRINCIPLE|SEBI\s+Principle)\s*{next_principle}[\s\.:]*",
                            rf"(?i)principle\s*{next_principle}[\s\.:]*",
                            rf"(?i)sebi\s+principle\s*{next_principle}[\s\.:]*"
                        ]
                        
                        for pattern in next_principle_patterns:
                            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                            if match:
                                end_page = page_num
                                matched_end_text = match.group(0)
                                words = page.extract_words()
                                
                                # Find the word containing "Principle"
                                for word in words:
                                    if 'principle' in word['text'].lower():
                                        end_y = word['top']
                                        next_principle_position = (page_num, word['top'])
                                        print(f"Found next principle '{matched_end_text}' on page {page_num + 1}")
                                        # Return with the position of the next principle
                                        return start_page, end_page, start_y, end_y, next_principle_position
                    
                    # If no next principle on this page, check for standard end patterns
                    for pattern in self.current_end_patterns:
                        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                        if match:
                            end_page = page_num
                            matched_end_text = match.group(0)
                            words = page.extract_words()
                            
                            # Find the word containing "Principle"
                            for word in words:
                                if 'principle' in word['text'].lower():
                                    end_y = word['top']
                                    print(f"Found section end '{matched_end_text}' on page {page_num + 1}")
                                    # Return immediately when end pattern is found
                                    return start_page, end_page, start_y, end_y, None
            
            # If we found start but no end, use last page
            if start_page is not None:
                end_page = len(pdf.pages) - 1
                end_y = float('inf')
                print(f"No explicit section end found, using last page ({end_page + 1})")
            
            return start_page, end_page, start_y, end_y, None

        def extract_tables_from_pdf(self, pdf, principle_num, start_scan_page):
            """Extract tables from the target section of the PDF."""
            print(f"\nExtracting tables for Principle {principle_num} starting from page {start_scan_page + 1}")
            
            # Get more detailed boundary information for the specific principle
            start_page, end_page, start_y, end_y, next_principle_position = self.detect_section_boundaries(pdf, principle_num, start_scan_page)
            if start_page is None:
                print(f"No section start found for Principle {principle_num}.")
                return None, start_scan_page
            
            print(f"\nExtracting tables from Principle {principle_num} section (Pages {start_page + 1} to {end_page + 1})")
            
            # Initialize variables for table tracking
            all_tables = []
            current_table = []
            table_count = 0
            
            # Only process pages within the detected section boundaries
            for page_num in range(start_page, end_page + 1):
                page = pdf.pages[page_num]
                
                # Extract tables from the current page
                tables = page.extract_tables()
                if not tables:
                    tables = page.extract_tables(table_settings={
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "intersection_y_tolerance": 10
                    })
                
                if tables:
                    print(f"Found {len(tables)} potential tables on page {page_num + 1}")
                    table_positions = page.find_tables()
                    
                    for table_idx, (table, table_pos) in enumerate(zip(tables, table_positions)):
                        table_top = table_pos.bbox[1]  # y-coordinate of table top
                        table_bottom = table_pos.bbox[3]  # y-coordinate of table bottom
                        
                        # Check if table is within section boundaries
                        include_table = False
                        
                        if page_num == start_page:
                            if table_top > start_y:
                                include_table = True
                        elif page_num == end_page and end_y != float('inf'):
                            if table_bottom < end_y:
                                include_table = True
                        else:
                            include_table = True
                        
                        if include_table:
                            print(f"Processing table {table_idx + 1} on page {page_num + 1}")
                            
                            # Clean and validate table
                            cleaned_table = [[self.clean_cell_content(cell) for cell in row] for row in table]
                            df = pd.DataFrame(cleaned_table)
                            
                            # Remove empty rows and columns
                            df = df.dropna(how='all').dropna(axis=1, how='all')
                            
                            if not df.empty:
                                if current_table and self.is_table_continuation(current_table[-1], df):
                                    current_table.append(df)
                                else:
                                    if current_table:
                                        processed_table = pd.concat(current_table, ignore_index=True)
                                        all_tables.append(processed_table)
                                        table_count += 1
                                    current_table = [df]
            
            # Process any remaining table
            if current_table:
                processed_table = pd.concat(current_table, ignore_index=True)
                all_tables.append(processed_table)
                table_count += 1
            
            print(f"\nExtracted {table_count} tables from Principle {principle_num} section")
            
            # If we found a next principle, use its position as the starting point
            next_scan_page = end_page
            
            # If we found the next principle on the same page, use that page again
            if next_principle_position:
                next_scan_page = next_principle_position[0]  # Return the page number where next principle was found
            
            return all_tables if all_tables else None, next_scan_page
            
        def is_table_continuation(self, prev_df, curr_df):
            """
            Determine if the current table is a continuation of the previous table.
            """
            # Check if number of columns match
            if len(prev_df.columns) != len(curr_df.columns):
                return False
            
            try:
                # Convert column labels to strings and compare
                prev_headers = [str(col).lower().strip() for col in prev_df.columns]
                curr_headers = [str(col).lower().strip() for col in curr_df.columns]
                
                # Check if headers are identical or very similar
                header_similarity = sum(p == c for p, c in zip(prev_headers, curr_headers))
                header_match = header_similarity / len(prev_headers) > 0.8
                
                # Check if current table starts with numeric data (likely continuation)
                first_row_numeric = all(
                    str(x).replace('.', '').replace('%', '').strip().isdigit() 
                    for x in curr_df.iloc[0] 
                    if pd.notna(x) and str(x).strip()
                )
                
                # Check for similar data patterns in last row of prev and first row of current
                prev_last_row = prev_df.iloc[-1]
                curr_first_row = curr_df.iloc[0]
                
                pattern_matches = 0
                total_checks = 0
                
                for i in range(len(prev_df.columns)):
                    prev_val = str(prev_last_row.iloc[i]).strip() if pd.notna(prev_last_row.iloc[i]) else ''
                    curr_val = str(curr_first_row.iloc[i]).strip() if pd.notna(curr_first_row.iloc[i]) else ''
                    
                    if prev_val and curr_val:  # Only check non-empty values
                        total_checks += 1
                        # Check if both are numeric or both are text
                        prev_is_numeric = prev_val.replace('.', '').replace('%', '').isdigit()
                        curr_is_numeric = curr_val.replace('.', '').replace('%', '').isdigit()
                        if prev_is_numeric == curr_is_numeric:
                            pattern_matches += 1
                
                pattern_similarity = pattern_matches / total_checks if total_checks > 0 else 0
                
                # Return True if either condition is met
                return header_match or (first_row_numeric and pattern_similarity > 0.8)
                    
            except Exception as e:
                print(f"Warning: Error comparing tables: {str(e)}")
                return False

        def process_tables(self, tables, output_dir, principle_num):
            """Process multiple tables and store only the original CSV and decoded JSON."""
            processed_results = []

            for idx, df in enumerate(tables, 1):
                print(f"\nProcessing table {idx} for Principle {principle_num}...")

                # Save original CSV
                base_name = f"table_{idx}"
                csv_path = os.path.join(output_dir, f"{base_name}_original.csv")
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"Original table {idx} saved to: {csv_path}")

                # Create hash mapping
                hash_mapping = self.create_hash_mapping(df)

                # Encode data (without saving)
                encoded_df, _ = self.encode_csv_data(df)

                # Preprocess for Gemini (without saving)
                preprocessed_df = self.preprocess_csv_for_gemini(encoded_df)

                # Process with Gemini
                processed_json_path = self.process_csv_with_gemini(preprocessed_df, output_dir, idx)

                if processed_json_path:
                    try:
                        with open(processed_json_path, 'r', encoding='utf-8') as f:
                            processed_json = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse Gemini output as JSON for table {idx}: {e}")
                        continue

                    # Decode JSON using hash mapping
                    decoded_json = self.decode_json_data(processed_json, hash_mapping)

                    # Save only decoded JSON
                    decoded_json_path = os.path.join(output_dir, f"{base_name}_decoded.json")
                    with open(decoded_json_path, 'w', encoding='utf-8') as f:
                        json.dump(decoded_json, f, indent=2, ensure_ascii=False)

                    print(f"Decoded JSON saved to: {decoded_json_path}")

                    # Delete intermediate files if they were mistakenly saved
                    for file_path in [
                        os.path.join(output_dir, f"{base_name}_encoded.csv"),
                        os.path.join(output_dir, f"{base_name}_preprocessed.csv"),
                        processed_json_path  # Gemini output file
                    ]:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            print(f"Deleted: {file_path}")

                    processed_results.append({
                        'principle': principle_num,
                        'table_number': idx,
                        'original_csv': csv_path,
                        'decoded_json': decoded_json_path
                    })
                else:
                    print(f"Skipping decoding for table {idx} due to invalid Gemini output.")

            return processed_results
        
        def preprocess_csv_for_gemini(self, df):
            """
            Preprocess CSV by consolidating related rows into single entries
            before sending to Gemini.
            """
            # Create a new DataFrame to store consolidated data
            consolidated_rows = []
            current_entry = None
            columns = df.columns.tolist()
            
            # Function to create a new entry dictionary
            def create_entry_dict():
                return {col: [] for col in columns}
            
            for idx, row in df.iterrows():
                # Check if this is a new main entry (has an S. No. or first column value)
                if pd.notna(row[0]) and str(row[0]).strip():
                    # Save previous entry if exists
                    if current_entry:
                        # Clean up empty lists and convert single-item lists to values
                        cleaned_entry = {}
                        for col, values in current_entry.items():
                            if values:
                                cleaned_entry[col] = " ".join(values)  # Join list into space-separated string
                            else:
                                cleaned_entry[col] = ''
                        consolidated_rows.append(cleaned_entry)
                    
                    # Start new entry
                    current_entry = create_entry_dict()
                    # Add values from current row
                    for col in columns:
                        if pd.notna(row[col]) and str(row[col]).strip():
                            current_entry[col].append(str(row[col]).strip())
                
                # This is a continuation row
                elif current_entry is not None:
                    # Add non-empty values to their respective columns
                    for col in columns:
                        if pd.notna(row[col]) and str(row[col]).strip():
                            current_entry[col].append(str(row[col]).strip())
            
            # Don't forget to add the last entry
            if current_entry:
                cleaned_entry = {}
                for col, values in current_entry.items():
                    if values:
                        cleaned_entry[col] = " ".join(values)  # Join list into space-separated string
                    else:
                        cleaned_entry[col] = ''
                consolidated_rows.append(cleaned_entry)
            
            # Convert consolidated data back to DataFrame
            consolidated_df = pd.DataFrame(consolidated_rows)
            
            return consolidated_df


        def process_csv_with_gemini(self, df, output_dir, table_idx):
            """Process CSV data directly using Gemini Pro and save the output."""
            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.3,
                google_api_key=self.google_api_key
            )

            # Convert DataFrame to string representation
            csv_str = df.to_csv(index=False)
            
            prompt_template = """
            Restructure the following encoded csv file into a proper nested JSON format using original headers and values:
            {context}

            Goal: Create a hierarchical JSON that preserves original structure and relationships.
            Make sure no column names and values are left from csv file in nested json.
            Structure the output as a nested JSON with a "Data" array containing row objects.
            Ensure the output starts with '{{' and ends with '}}'.
            Important: Do NOT include markdown code blocks or backticks in your response. Just return the raw JSON.
            """

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context"]
            )

            docs = [Document(page_content=csv_str)]
            chain = load_qa_chain(
                llm=model,
                chain_type="stuff",
                prompt=prompt,
                document_variable_name="context"
            )

            # Run Gemini processing
            print("\nProcessing with Gemini...")
            response = chain.run(docs)
            
            print("\n===== Raw Response from Gemini =====\n")
            #print(response)
            print("\n====================================\n")

            # Save the raw response to a file
            raw_output_path = os.path.join(output_dir, f"table_{table_idx}_gemini_output.json")
            with open(raw_output_path, 'w', encoding='utf-8') as f:
                f.write(response)
            
            print(f"Raw Gemini output saved to: {raw_output_path}")

            # Ensure response is in JSON format
            try:
                json.loads(response)
                return raw_output_path
            except json.JSONDecodeError as e:
                print("Error parsing JSON:", e)
                return None

        def process_pdf(self, pdf_path, output_dir):
            """Main function to process PDF and extract all principles."""
            # Create main output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            all_results = []
            
            # Find Section C start page once
            with pdfplumber.open(pdf_path) as pdf:
                section_c_start_page = self.find_start_page(pdf_path)
                print(f"Section C starts on page {section_c_start_page}")
                
                # Track the last end page to use as start for next principle
                last_end_page = section_c_start_page - 1  # Start from the Section C page
                
                # Process each principle from 1 to 9
                for principle_num in range(1, 10):
                    print(f"\n{'='*50}")
                    print(f"Processing Principle {principle_num}")
                    print(f"{'='*50}")
                    
                    # Create principle-specific folder
                    principle_folder = f"Principle_{principle_num}"
                    principle_output_path = os.path.join(output_dir, principle_folder)
                    os.makedirs(principle_output_path, exist_ok=True)
                    
                    # Extract tables for this principle starting from last position
                    tables, next_scan_page = self.extract_tables_from_pdf(pdf, principle_num, last_end_page)
                    
                    if tables:
                        # Process tables for this principle
                        results = self.process_tables(tables, principle_output_path, principle_num)
                        
                        if results:
                            # Save principle-specific summary
                            summary_path = os.path.join(principle_output_path, f"principle_{principle_num}_summary.json")
                            with open(summary_path, 'w', encoding='utf-8') as f:
                                json.dump(results, f, indent=2)
                            print(f"\nPrinciple {principle_num} summary saved to: {summary_path}")
                            
                            # Add to overall results
                            all_results.extend(results)
                        else:
                            print(f"\nNo tables successfully processed for Principle {principle_num}")
                    else:
                        print(f"\nNo tables found for Principle {principle_num}")
                    
                    # Update last_end_page for next principle
                    last_end_page = next_scan_page
            
            # Save overall processing summary
            if all_results:
                overall_summary_path = os.path.join(output_dir, "all_principles_summary.json")
                with open(overall_summary_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2)
                print(f"\nOverall processing summary saved to: {overall_summary_path}")
                
                print(f"\nProcessing completed! Extracted tables for {len(set(r['principle'] for r in all_results))} principles.")
                return all_results
            else:
                print("\nNo principles were successfully processed.")
                return None
        
        def is_header_row(self, row_values):
            """
            Determine if a row looks like a header row based on patterns.
            """
            if not row_values:
                return False
                
            # Common header patterns
            header_indicators = [
                lambda x: x.endswith(':'),  # Ends with colon
                lambda x: x.isupper(),  # All caps
                lambda x: len(x.split()) > 3,  # Longer descriptive text
                lambda x: any(char in x for char in '()/\\'),  # Contains special characters
                lambda x: bool(re.search(r'[A-Z][a-z]+(/|\s|$)', x))  # Title case words
            ]
            
            # Check how many values in the row match header patterns
            header_matches = sum(
                1 for value in row_values
                if any(indicator(value) for indicator in header_indicators)
            )
            
            # Consider it a header if a significant portion matches header patterns
            return header_matches / len(row_values) > 0.3
        
        def is_header_like(self, value):
            """
            Determine if a value looks like a header or label rather than data.
            """
            # Convert to string and clean
            value = str(value).strip()
            
            # Patterns that suggest a header/label
            header_patterns = [
                r'^[A-Z\s]{3,}$',  # All caps text
                r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$',  # Title case words
                r'.*[:()].*',  # Contains colon or parentheses
                r'.*[/\\].*',  # Contains slashes
                r'^\d+\.\s+\w+',  # Numbered items
                r'.+\s+of\s+.+',  # Descriptive phrases with 'of'
                r'^(Total|Average|Mean|Sum|Count)\b'  # Common header words
            ]
            
            return any(bool(re.match(pattern, value)) for pattern in header_patterns)

        def hash_six_digits(self, value):
            """
            Generate a 6-digit hash for a given string value.
            Carefully preserves percentage signs and other suffixes in the output.
            """
            if pd.isna(value) or str(value).strip() == '':
                return ''
                
            value_str = str(value).strip()
            
            # Special handling for percentage values
            percentage_match = re.match(r'^([-+]?\d*\.?\d+)\s*%\s*$', value_str)
            if percentage_match:
                number_part = percentage_match.group(1)
                # Generate hash for the number part and append '%'
                hashed = hashlib.sha256(number_part.encode()).hexdigest()[:6]
                return f"{hashed}%"
            
            # Handle other numeric values with potential suffixes
            match = re.match(r'^([-+]?\d*\.?\d+)(\s*[A-Za-z%]+\s*)?$', value_str)
            if match:
                number_part, suffix = match.groups()
                suffix = suffix if suffix else ''
                # Generate hash for the number part only
                hashed = hashlib.sha256(number_part.encode()).hexdigest()[:6]
                return f"{hashed}{suffix}"
            
            return hashlib.sha256(value_str.encode()).hexdigest()[:6]

        def detect_table_boundaries(self, data):
            """
            Detect the start and end of the actual table content dynamically.
            Uses pattern recognition instead of hardcoded headers.
            Returns tuple: (header_start, data_start, data_end) indices
            """
            header_start = 0
            data_start = 0
            data_end = len(data)
            
            # Find first non-empty row (content starts)
            for i, row in data.iterrows():
                if row.notna().sum() > 0:
                    header_start = i
                    break
            
            # Find where actual data starts (after header section)
            # Look for patterns that indicate header rows have ended
            header_pattern_count = 0
            max_header_rows = 5  # Maximum reasonable number of header rows
            
            for i, row in data.iloc[header_start:].iterrows():
                row_values = [str(x).strip() for x in row if pd.notna(x) and str(x).strip()]
                
                if not row_values:  # Skip empty rows
                    continue
                    
                # Check if this row looks like a header
                if self.is_header_row(row_values):
                    header_pattern_count += 1
                    if header_pattern_count >= max_header_rows:
                        data_start = i + 1
                        break
                else:
                    # If we find a non-header row after some headers, that's our data start
                    if header_pattern_count > 0:
                        data_start = i
                        break
            
            # If we haven't found a clear data start, use a reasonable default
            if data_start <= header_start:
                data_start = header_start + min(max_header_rows, 3)
            
            # Find where data ends (last non-empty row)
            for i in range(len(data) - 1, data_start - 1, -1):
                if data.iloc[i].notna().sum() > 0:
                    data_end = i + 1
                    break
            
            return header_start, data_start, data_end
        
        def should_hash_column(self, column_data, column_name, index):
            """
            Determine if a column's data should be hashed based on its content.
            Returns False for header/label columns.
            """
            # Don't hash the first column (usually contains labels/categories)
            if index == 0:
                return False
                
            # Only consider non-empty values
            non_empty_values = [
                x for x in column_data 
                if pd.notna(x) and str(x).strip() != ''
            ]
            
            if not non_empty_values:  # Skip entirely empty columns
                return False
                
            # Check if column contains mostly numeric data
            numeric_count = sum(
                1 for x in non_empty_values 
                if (
                    str(x).replace('.', '').isdigit() or 
                    str(x).replace('.', '').replace('%', '').isdigit()
                )
            )
            
            numeric_ratio = numeric_count / len(non_empty_values) if non_empty_values else 0
            
            # Return True if more than 50% of non-empty values are numeric
            return numeric_ratio > 0.5
        
        def encode_csv_data(self, data):
            """
            Encode only numeric data inside the table while preserving headers and text content.
            Works with any table structure without hardcoding column names.
            Returns encoded DataFrame and data boundaries.
            """
            header_start, data_start, data_end = self.detect_table_boundaries(data)
            result = data.copy()
            
            # Process each row after the header section
            for row_idx in range(data_start, data_end):
                row = result.iloc[row_idx]
                for col_idx, value in enumerate(row):
                    if pd.notna(value) and str(value).strip():
                        # Check if the value is numeric or percentage
                        value_str = str(value).strip()
                        
                        # Skip if the value appears to be a header or label
                        if self.is_header_like(value_str):
                            continue
                            
                        # Clean the value for numeric checking
                        cleaned_value = value_str.replace(',', '')  # Remove commas from numbers
                        
                        # Check if it's a numeric value
                        is_numeric = (
                            cleaned_value.replace('.', '').isdigit() or  # Pure numbers
                            cleaned_value.endswith('%') and cleaned_value[:-1].replace('.', '').isdigit() or  # Percentages
                            bool(re.match(r'^\d+\.?\d*$', cleaned_value))  # Decimal numbers
                        )
                        
                        if is_numeric:
                            result.iloc[row_idx, col_idx] = self.hash_six_digits(value_str)
            
            print(f"\nTable content encoded from row {data_start} to {data_end}")
            return result, (data_start, data_end)
        
        def create_hash_mapping(self, df):
            """
            Create a mapping of hash values to original values from the DataFrame.
            Enhanced to properly handle percentage values and other numeric suffixes.
            """
            hash_map = {}
            
            def process_value(value):
                if pd.isna(value):
                    return
                
                value_str = str(value).strip()
                if not value_str:
                    return
                    
                # Handle percentage values
                percentage_match = re.match(r'^([-+]?\d*\.?\d+)\s*%\s*$', value_str)
                if percentage_match:
                    number_part = percentage_match.group(1)
                    hash_value = hashlib.sha256(number_part.encode()).hexdigest()[:6]
                    hash_map[f"{hash_value}%"] = value_str
                    return
                    
                # Handle other numeric values with suffixes
                match = re.match(r'^([-+]?\d*\.?\d+)(\s*[A-Za-z%]+\s*)?$', value_str)
                if match:
                    number_part, suffix = match.groups()
                    suffix = suffix if suffix else ''
                    hash_value = hashlib.sha256(number_part.encode()).hexdigest()[:6]
                    hash_map[f"{hash_value}{suffix}"] = value_str
                    return
                    
                # Handle regular values
                hash_value = hashlib.sha256(value_str.encode()).hexdigest()[:6]
                hash_map[hash_value] = value_str

            # Process all values in the DataFrame
            for column in df.columns:
                for value in df[column]:
                    process_value(value)
            
            return hash_map
        
        def decode_json_data(self, json_data, hash_mapping):
            """
            Replace hash values in JSON with original values using the mapping.
            Enhanced to handle percentage values and other numeric suffixes.
            """
            if isinstance(json_data, dict):
                return {k: self.decode_json_data(v, hash_mapping) for k, v in json_data.items()}
            elif isinstance(json_data, list):
                return [self.decode_json_data(item, hash_mapping) for item in json_data]
            elif isinstance(json_data, str):
                words = json_data.split()
                decoded_words = []
                
                for word in words:
                    # Try to match the word exactly in the hash mapping
                    if word in hash_mapping:
                        decoded_words.append(hash_mapping[word])
                        continue
                    
                    # Check if it's a hashed value with a suffix
                    match = re.match(r'^([a-f0-9]{6})([A-Za-z%]+)?$', word)
                    if match:
                        hash_part, suffix = match.groups()
                        suffix = suffix if suffix else ''
                        full_hash = f"{hash_part}{suffix}"
                        if full_hash in hash_mapping:
                            decoded_words.append(hash_mapping[full_hash])
                        else:
                            decoded_words.append(word)  # Keep original if no match
                    else:
                        decoded_words.append(word)  # Keep original if no match
                        
                return " ".join(decoded_words)
            else:
                return json_data

        def clean_cell_content(self, cell):
            """Clean cell content while preserving newlines within cells."""
            if cell is None:
                return ""
            cell = str(cell)
            lines = cell.split('\n')
            lines = [line.strip() for line in lines if line.strip()]
            return '\n'.join(lines)

    class PDFProcessorDB:
        """Processor for handling database insertion from JSON files stored in S3."""

        def __init__(self):
            self.db_host = config.get("DB_HOST")
            self.db_name = config.get("DB_NAME")
            self.db_user = config.get("DB_USER")
            self.db_password = config.get("DB_PASSWORD")
            self.db_port = config.get("DB_PORT")
            
            # Initialize S3 Client
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION")
            )

        def connect_to_db(self):
            """Establishes connection to the PostgreSQL database."""
            try:
                conn = psycopg2.connect(
                    host=self.db_host,
                    dbname=self.db_name,
                    user=self.db_user,
                    password=self.db_password,
                    port=self.db_port
                )
                return conn
            except Exception as e:
                print(f"❌ Database connection error: {e}")
                return None

        def get_next_id(self, cursor):
            """Fetches the next sequential 6-character alphanumeric ID from the database."""
            cursor.execute('SELECT MAX(id) FROM "FileRecord";')
            max_id = cursor.fetchone()[0]

            if not max_id or not isinstance(max_id, str): 
                return "10000A"

            try:
                num_part = int(max_id[:-1])  
                letter_part = max_id[-1]  

                if letter_part not in string.ascii_uppercase:
                    return "10000A"

                if letter_part == 'Z':  
                    num_part += 1
                    next_letter = 'A'
                else:
                    next_letter = chr(ord(letter_part) + 1)

                return f"{num_part}{next_letter}"

            except Exception as e:
                print(f"Error in get_next_id(): {e}")
                return "10000A"

        def process_pdf(self, pdf_s3_path, local_output_dir):
            """Fetch JSON files from S3 and insert them into PostgreSQL."""
            IST = pytz.timezone('Asia/Kolkata')

            # Extract AWS_BUCKET_NAME dynamically from pdf_s3_path
            AWS_BUCKET_NAME = pdf_s3_path.split("/")[2]

            # Corrected output directory (without `s3://` prefix)
            s3_directory = "/".join(pdf_s3_path.split("/")[3:]).replace("brsr_input", "brsr_output").rsplit("/", 1)[0] + "/"

            conn = self.connect_to_db()
            if conn is None:
                print("❌ Exiting due to database connection failure.")
                return None  # Return None to maintain consistency with other processors

            try:
                cur = conn.cursor()
                processed_count = 0

                # List objects in the specified S3 directory
                response = self.s3_client.list_objects_v2(Bucket=AWS_BUCKET_NAME, Prefix=s3_directory)

                if 'Contents' not in response:
                    print(f" No files found in S3 directory: {s3_directory}")
                    return None

                json_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('decoded.json')]

                if not json_files:
                    print(f" No 'decoded.json' files found in S3: {s3_directory}")
                    return None

                for s3_key in json_files:
                    try:
                        path_parts = s3_key.split("/")

                        # Identify section and subsection
                        if "SECTION_A" in path_parts:
                            base_section = "SECTION A: GENERAL DISCLOSURES"
                        elif "SECTION_B" in path_parts:
                            base_section = "SECTION B: MANAGEMENT AND PROCESS DISCLOSURES"
                        elif "SECTION_C" in path_parts:
                            base_section = "SECTION C: PRINCIPLE-WISE PERFORMANCE DISCLOSURE"
                        else:
                            base_section = "SECTION C: PRINCIPLE-WISE PERFORMANCE DISCLOSURE"

                        subsection_name = path_parts[-2] if len(path_parts) > 3 else "Unknown Subsection"
                        table_name = f"{base_section} - {subsection_name}"

                        print(f"📂 Processing file from S3: {s3_key}")
                        print(f"Detected base section: {base_section}")
                        print(f"Final table name: {table_name}")

                        # Download JSON file from S3
                        file_obj = self.s3_client.get_object(Bucket=AWS_BUCKET_NAME, Key=s3_key)
                        json_data = json.load(io.BytesIO(file_obj['Body'].read()))

                        unique_id = self.get_next_id(cur)
                        current_timestamp = datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                        # Updated file_link and record_type to match the PDF's S3 path
                        file_link = f"s3://{AWS_BUCKET_NAME}/{pdf_s3_path}"
                        record_type = f"{os.path.basename(pdf_s3_path).replace('.pdf', '')} Annual Report"

                        insert_query = """
                        INSERT INTO "FileRecord" (id, data, "tableName", timestamp, "fileLink", "recordType")
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """
                        cur.execute(insert_query, (unique_id, json.dumps(json_data), table_name, current_timestamp, file_link, record_type))
                        conn.commit()

                        processed_count += 1
                        print(f"✅ Successfully inserted data from {s3_key}")
                        print(f"   Inserted timestamp: {current_timestamp}")
                        print(f"   Table name used: {table_name}")
                        print(f"   Assigned ID: {unique_id}")
                        print(f"   File Link: {file_link}")
                        print(f"   Record Type: {record_type}")
                        print("-" * 50)

                    except json.JSONDecodeError:
                        print(f" Error: {s3_key} is not a valid JSON file")
                    except Exception as e:
                        print(f" Error processing {s3_key}: {e}")
                        conn.rollback()

                print(f"✅ Total files processed from S3: {processed_count}")
                return processed_count  # Maintain consistency in returning data

            except Exception as e:
                print(f" Database processing error: {e}")
            finally:
                if conn:
                    cur.close()
                    conn.close()
                    print(" Database connection closed.")
        
    # Load environment variables from .env file
    load_dotenv()

    # Extract dynamic values from `config`
    pdf_s3_path = config.get("pdf_s3_path")

    # Extract AWS_BUCKET_NAME dynamically
    AWS_BUCKET_NAME = pdf_s3_path.split("/")[2]

    # Derive dynamic S3 Output Path
    output_s3_dir = "/".join(pdf_s3_path.split("/")[3:]).replace("brsr_input", "brsr_output").rsplit("/", 1)[0] + "/"

    pdf_filename = os.path.basename(pdf_s3_path)
    local_pdf_path = os.path.join(tempfile.gettempdir(), pdf_filename)
    local_output_dir = os.path.join(tempfile.gettempdir(), "brsr_output")

    os.makedirs(local_output_dir, exist_ok=True)

    # Download PDF from S3
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )

    print(f" Downloading {pdf_s3_path} from S3 to {local_pdf_path}...")
    s3_client.download_file(AWS_BUCKET_NAME, "/".join(pdf_s3_path.split("/")[3:]), local_pdf_path)
    print(" Download complete!")

    def upload_directory_to_s3(local_directory, s3_bucket, s3_prefix):
        """Uploads the entire local directory to S3 recursively."""
        for root, _, files in os.walk(local_directory):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_directory)  # Maintain relative structure
                s3_path = os.path.join(s3_prefix, relative_path).replace("\\", "/")  # Ensure correct S3 format

                print(f"📤 Uploading {local_path} → s3://{s3_bucket}/{s3_path} ...")
                s3_client.upload_file(local_path, s3_bucket, s3_path)

        print("✅ All results uploaded to S3 successfully!")

    try:
        # Extract dynamic values from `config`
        pdf_s3_path = config.get("pdf_s3_path")

        # Extract AWS_BUCKET_NAME dynamically
        AWS_BUCKET_NAME = pdf_s3_path.split("/")[2]

        # Derive dynamic S3 Output Path
        output_s3_dir = "/".join(pdf_s3_path.split("/")[3:]).replace("brsr_input", "brsr_output").rsplit("/", 1)[0] + "/"

        # Print S3 output directory
        print(f" Dynamically generated S3 output directory: s3://{AWS_BUCKET_NAME}/{output_s3_dir}")

        # Extract the PDF filename dynamically
        pdf_filename = os.path.basename(pdf_s3_path)

        # Temporary local paths
        local_pdf_path = os.path.join(tempfile.gettempdir(), pdf_filename)
        local_output_dir = os.path.join(tempfile.gettempdir(), "brsr_output")  # Local processing directory

        # Ensure local output directory exists
        os.makedirs(local_output_dir, exist_ok=True)

        # Download PDF from S3
        print(f" Downloading {pdf_s3_path} from S3 to {local_pdf_path}...")
        s3_client.download_file(AWS_BUCKET_NAME, "/".join(pdf_s3_path.split("/")[3:]), local_pdf_path)
        print(" Download complete!")

        # Initialize and run processor
        processor = PDFProcessorA()

        try:
            print(f"🚀 Processing PDF and saving results locally: {local_output_dir}...")
            
            # Process PDF and save results locally
            results = processor.process_section_headings(local_pdf_path, local_output_dir)

            if results:
                total_tables = sum(len(tables) for tables in results.values())
                print(f"\n🎉 Processing completed successfully! {total_tables} tables extracted.")

                # Upload processed results to S3
                print(f"📤 Uploading processed results to S3: {output_s3_dir} ...")
                upload_directory_to_s3(local_output_dir, AWS_BUCKET_NAME, output_s3_dir)

            else:
                print("\n Processing failed!")

        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")

        # Initialize and run processor
        processor = PDFProcessorB()

        try:
            print(f"🚀 Processing PDF and saving results locally: {local_output_dir}...")
            
            # Process PDF and save results locally
            results = processor.process_section_headings(local_pdf_path, local_output_dir)

            if results:
                total_tables = sum(len(tables) for tables in results.values())
                print(f"\n🎉 Processing completed successfully! {total_tables} tables extracted.")

                # Upload processed results to S3
                print(f"📤 Uploading processed results to S3: {output_s3_dir} ...")
                upload_directory_to_s3(local_output_dir, AWS_BUCKET_NAME, output_s3_dir)

            else:
                print("\n Processing failed!")

        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")

        # Initialize and run processor
        processor = PDFProcessorC()

        try:
            print(f"🚀 Processing PDF and saving results locally: {local_output_dir}...")
            
            # Process PDF and save results locally
            results = processor.process_pdf(local_pdf_path, local_output_dir)

            if results:
                # Handle both dict and list results
                if isinstance(results, dict):
                    total_tables = sum(len(tables) for tables in results.values())
                elif isinstance(results, list):
                    total_tables = len(results)  # Count total tables if results is a list
                else:
                    total_tables = 0  # Default case if results is unexpected
                
                print(f"\n Processing completed successfully! {total_tables} tables extracted.")

                # Upload processed results to S3
                print(f"📤 Uploading processed results to S3: {output_s3_dir} ...")
                upload_directory_to_s3(local_output_dir, AWS_BUCKET_NAME, output_s3_dir)

            else:
                print("\n⚠️ Processing failed!")

        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")

        # Initialize and run processor
        processor = PDFProcessorDB()

        try:
            print(f"🚀 Processing JSON data and inserting into database...")
            
            # Process JSON and insert into DB
            results = processor.process_pdf(pdf_s3_path, local_output_dir)

            if results:
                print(f"\n🎉 Database processing completed successfully! {results} records inserted.")

            else:
                print("\n⚠️ Database processing failed!")

        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
    
    except:
        print("no")

