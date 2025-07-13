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
        """Parse answers from markdown text - improved version"""
        answers = []
        
        # Split text into lines for processing
        lines = markdown_text.split('\n')
        
        # Method 1: Look for "Correct Answer" pattern (for options 2 & 3)
        answer_dict = {}
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for "Correct Answer" pattern
            if line.startswith("**Correct Answer**:"):
                # Extract the answer letter
                answer_match = re.search(r'\*\*Correct Answer\*\*:\s*([A-D])', line)
                if answer_match:
                    answer_letter = answer_match.group(1)
                    
                    # Find the corresponding question number by looking backwards
                    question_num = None
                    for j in range(i-1, max(0, i-20), -1):  # Look back up to 20 lines
                        prev_line = lines[j].strip()
                        question_match = re.search(r'### Question (\d+)', prev_line)
                        if question_match:
                            question_num = int(question_match.group(1))
                            break
                    
                    if question_num:
                        answer_dict[question_num] = answer_letter
        
        # If we found answers using the "Correct Answer" pattern, use those
        if answer_dict:
            # Convert dict to ordered list
            max_question = max(answer_dict.keys()) if answer_dict else 0
            for i in range(1, max_question + 1):
                if i in answer_dict:
                    answers.append(answer_dict[i])
                else:
                    self.logger.warning(f"Missing answer for question {i}")
                    answers.append("A")  # Default fallback
            
            self.logger.info(f"Extracted {len(answers)} answers using 'Correct Answer' pattern")
            return answers
        
        # Method 2: Look for simple answer key pattern (for option 1)
        answer_patterns = [
            r'^(\d+)\.\s*([A-D])',            # 1. A
            r'^(\d+)\)\s*([A-D])',            # 1) A
            r'^Question\s*(\d+):\s*([A-D])',  # Question 1: A
            r'^(\d+)\s*-\s*([A-D])',          # 1 - A
            r'^(\d+)\s+([A-D])',              # 1 A
        ]
        
        answer_dict = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in answer_patterns:
                match = re.search(pattern, line)
                if match:
                    question_num = int(match.group(1))
                    answer_letter = match.group(2)
                    answer_dict[question_num] = answer_letter
                    break
        
        # Convert dict to ordered list
        if answer_dict:
            max_question = max(answer_dict.keys()) if answer_dict else 0
            for i in range(1, max_question + 1):
                if i in answer_dict:
                    answers.append(answer_dict[i])
                else:
                    self.logger.warning(f"Missing answer for question {i}")
                    answers.append("A")  # Default fallback
            
            self.logger.info(f"Extracted {len(answers)} answers using simple pattern")
            return answers
        
        # Method 3: Fallback - look for any A, B, C, D pattern (last resort)
        if not answers:
            self.logger.warning("No structured answers found, trying fallback method")
            
            # Look for isolated A, B, C, D answers
            for line in lines:
                line = line.strip()
                if re.match(r'^[A-D]$', line):
                    answers.append(line)
            
            # Limit to reasonable number (avoid parsing MCQ options as answers)
            if len(answers) > 200:  # If we get too many, it's likely wrong
                answers = answers[:100]  # Take first 100
    
        self.logger.info(f"Final extracted answers count: {len(answers)}")
        return answers
    
    def save_extracted_data(self, questions: List[Dict], answers: List[str], output_dir: str = "temp_test_data"):
        """Save extracted questions and answers for debugging - improved version"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save questions
        with open(output_path / "extracted_questions.json", 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
    
        # Save answers
        with open(output_path / "extracted_answers.json", 'w', encoding='utf-8') as f:
            json.dump(answers, f, indent=2, ensure_ascii=False)
    
        # Save formatted view with validation
        with open(output_path / "formatted_test.md", 'w', encoding='utf-8') as f:
            f.write("# Extracted Test Data\n\n")
            f.write("## Questions\n\n")
            
            for i, question in enumerate(questions):
                f.write(f"### Question {i+1}\n")
                f.write(f"{question['question']}\n\n")
                
                if question.get('options'):
                    for option in question['options']:
                        f.write(f"- {option}\n")
                
                # Add correct answer if available
                if i < len(answers):
                    f.write(f"\n**Correct Answer**: {answers[i]}\n\n")
                else:
                    f.write(f"\n**Correct Answer**: NOT FOUND\n\n")
    
        # Add validation summary
        with open(output_path / "validation_summary.txt", 'w', encoding='utf-8') as f:
            f.write(f"Extraction Validation Summary\n")
            f.write(f"============================\n\n")
            f.write(f"Questions extracted: {len(questions)}\n")
            f.write(f"Answers extracted: {len(answers)}\n")
            f.write(f"Match status: {'MATCH' if len(questions) == len(answers) else 'MISMATCH'}\n\n")
            
            if len(questions) != len(answers):
                f.write("WARNING: Question and answer counts don't match!\n")
                f.write("This will cause issues during test execution.\n\n")
            
            # Show answer distribution
            if answers:
                from collections import Counter
                answer_counts = Counter(answers)
                f.write("Answer Distribution:\n")
                for letter in ['A', 'B', 'C', 'D']:
                    count = answer_counts.get(letter, 0)
                    percentage = (count / len(answers)) * 100 if answers else 0
                    f.write(f"  {letter}: {count} ({percentage:.1f}%)\n")

        self.logger.info(f"Extracted data saved to {output_path}")
        
        # Log validation results
        if len(questions) == len(answers):
            self.logger.info("Questions and answers count match")
        else:
            self.logger.error(f"Mismatch: {len(questions)} questions vs {len(answers)} answers")