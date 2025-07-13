import json
import logging
import re
from pathlib import Path
from typing import List, Dict
import tempfile

class TestProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_questions = []
        self.correct_answers = []
        self.agent_answers = []
        
    def extract_questions_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract questions from practice test PDF using markdown conversion"""
        self.logger.info(f"Extracting questions from {pdf_path}")
        
        try:
            # Convert PDF to markdown first
            markdown_text = self._convert_pdf_to_markdown(pdf_path)
            
            if not markdown_text.strip():
                self.logger.error("No text extracted from PDF")
                return []
            
            # Parse questions from markdown
            questions = self._parse_questions_from_markdown(markdown_text)
            self.logger.info(f"Extracted {len(questions)} questions")
            return questions
            
        except Exception as e:
            self.logger.error(f"Error extracting questions: {e}")
            return []
    
    def extract_answers_from_pdf(self, pdf_path: str) -> List[str]:
        """Extract correct answers from answer key PDF using markdown conversion"""
        self.logger.info(f"Extracting answers from {pdf_path}")
        
        try:
            # Convert PDF to markdown first
            markdown_text = self._convert_pdf_to_markdown(pdf_path)
            
            if not markdown_text.strip():
                self.logger.error("No text extracted from answer key PDF")
                return []
            
            # Parse answers from markdown
            answers = self._parse_answers_from_markdown(markdown_text)
            self.logger.info(f"Extracted {len(answers)} answers")
            return answers
            
        except Exception as e:
            self.logger.error(f"Error extracting answers: {e}")
            return []
    
    def _convert_pdf_to_markdown(self, pdf_path: str) -> str:
        """Convert PDF to markdown text using existing conversion logic"""
        try:
            # Use the convert_pdf_to_markdown function from ConvertPDF2md
            import pdfplumber
            
            markdown_output = ""
            
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        markdown_output += text + "\n\n"
            
            return markdown_output
            
        except Exception as e:
            self.logger.error(f"Error converting PDF to markdown: {e}")
            return ""
    
    def _parse_questions_from_markdown(self, markdown_text: str) -> List[Dict]:
        """Parse questions from markdown text"""
        questions = []
        
        # Split text into sections - look for question patterns
        lines = markdown_text.split('\n')
        current_question = None
        current_options = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check if line starts with a number (question)
            question_match = re.match(r'^(\d+)\.\s*(.*)', line)
            if question_match:
                # Save previous question if exists
                if current_question:
                    questions.append({
                        'question_number': current_question['number'],
                        'question': current_question['text'],
                        'options': current_options.copy()
                    })
                
                # Start new question
                current_question = {
                    'number': int(question_match.group(1)),
                    'text': question_match.group(2).strip()
                }
                current_options = []
                continue
            
            # Check if line is an option (A, B, C, D)
            option_match = re.match(r'^([A-D])\.\s*(.*)', line)
            if option_match and current_question:
                option_letter = option_match.group(1)
                option_text = option_match.group(2).strip()
                current_options.append(f"{option_letter}. {option_text}")
                continue
            
            # If we have a current question and this line doesn't match option pattern,
            # it might be continuation of question text
            if current_question and not option_match:
                # Only add if it's not obviously an option without proper format
                if not re.match(r'^[A-D]\s', line):
                    current_question['text'] += " " + line
        
        # Don't forget the last question
        if current_question:
            questions.append({
                'question_number': current_question['number'],
                'question': current_question['text'],
                'options': current_options.copy()
            })
        
        return questions
    
    def _parse_answers_from_markdown(self, markdown_text: str) -> List[str]:
        """Parse answers from markdown text"""
        answers = []
        
        # Split text into lines for processing
        lines = markdown_text.split('\n')
        
        # Pattern to match answer key format
        answer_patterns = [
            r'^(\d+)\.\s*([A-D])',  # Format: "1. A"
            r'^(\d+)\)\s*([A-D])',  # Format: "1) A"
            r'^Question\s*(\d+):\s*([A-D])',  # Format: "Question 1: A"
            r'^(\d+)\s*-\s*([A-D])',  # Format: "1 - A"
            r'^(\d+)\s+([A-D])',  # Format: "1 A"
        ]
        
        answer_dict = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try each pattern
            for pattern in answer_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    question_num = int(match.group(1))
                    answer_letter = match.group(2).upper()
                    answer_dict[question_num] = answer_letter
                    break
        
        # Convert dict to ordered list
        if answer_dict:
            max_question = max(answer_dict.keys())
            for i in range(1, max_question + 1):
                answers.append(answer_dict.get(i, 'A'))  # Default to A if missing
        
        # If no structured answers found, try to extract from text patterns
        if not answers:
            # Look for answer key sections
            answer_key_section = ""
            capture_answers = False
            
            for line in lines:
                if re.search(r'answer\s*key|answers|solution', line, re.IGNORECASE):
                    capture_answers = True
                    continue
                
                if capture_answers:
                    answer_key_section += line + " "
            
            # Extract letters from answer key section
            if answer_key_section:
                letter_matches = re.findall(r'\b([A-D])\b', answer_key_section)
                if len(letter_matches) > 5:  # Reasonable number of answers
                    answers = letter_matches[:100]  # Limit to reasonable number
        
        return answers
    
    def save_extracted_data(self, questions: List[Dict], answers: List[str], output_dir: str = "temp_test_data"):
        """Save extracted questions and answers for debugging"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save questions
        with open(output_path / "extracted_questions.json", 'w') as f:
            json.dump(questions, f, indent=2)
        
        # Save answers
        with open(output_path / "extracted_answers.json", 'w') as f:
            json.dump(answers, f, indent=2)
        
        # Save formatted view
        with open(output_path / "formatted_test.md", 'w') as f:
            f.write("# Extracted Test Data\n\n")
            f.write("## Questions\n\n")
            for i, q in enumerate(questions):
                f.write(f"### Question {q['question_number']}\n")
                f.write(f"{q['question']}\n\n")
                for option in q['options']:
                    f.write(f"- {option}\n")
                f.write(f"\n**Correct Answer**: {answers[i] if i < len(answers) else 'N/A'}\n\n")
        
        self.logger.info(f"Extracted data saved to {output_path}")