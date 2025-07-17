import pdfplumber
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

class TestProcessor:
    """
    A clean, maintainable processor for extracting questions and answers from PDF files.
    
    Usage:
        processor = TestProcessor()
        questions = processor.extract_questions_from_pdf("path/to/questions.pdf")
        answers = processor.extract_answers_from_pdf("path/to/answers.pdf")
        processor.save_to_json(questions, answers, "output_dir")
        
        # Later load the data
        questions, answers = processor.load_from_json("output_dir")
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def extract_questions_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract questions from practice test PDF."""
        self.logger.info(f"Extracting questions from {pdf_path}")
        
        try:
            text = self._extract_text_from_pdf(pdf_path)
            if not text:
                self.logger.error("No text extracted from PDF")
                return []
            
            questions = self._extract_questions_from_text(text)
            questions = self._validate_and_clean_questions(questions)
            
            self.logger.info(f"Successfully extracted {len(questions)} questions")
            return questions
        
        except Exception as e:
            self.logger.error(f"Error extracting questions: {e}")
            raise  # Re-raise for easier debugging
    
    def extract_answers_from_pdf(self, pdf_path: str) -> List[str]:
        """Extract correct answers from answer key PDF."""
        self.logger.info(f"Extracting answers from {pdf_path}")
        
        try:
            text = self._extract_text_from_pdf(pdf_path)
            if not text:
                self.logger.error("No text extracted from PDF")
                return []
            
            answers = self._extract_answers_from_text(text)
            self.logger.info(f"Successfully extracted {len(answers)} answers")
            return answers
            
        except Exception as e:
            self.logger.error(f"Error extracting answers: {e}")
            raise  # Re-raise for easier debugging
    
    def save_to_json(self, questions: List[Dict], answers: List[str], output_dir: str = "Outputs/extraction"):
        """Save extracted questions and answers to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save questions
        questions_file = output_path / "questions.json"
        with open(questions_file, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        
        # Save answers
        answers_file = output_path / "answers.json"
        with open(answers_file, 'w', encoding='utf-8') as f:
            json.dump(answers, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Data saved to {output_path}")
        
        # Log validation results
        if len(questions) == len(answers):
            self.logger.info(f"✓ Questions and answers count match: {len(questions)}")
        else:
            self.logger.warning(f"⚠ Mismatch: {len(questions)} questions vs {len(answers)} answers")
    
    # Add backward compatibility method
    def save_extracted_data(self, questions: List[Dict], answers: List[str], output_dir: str = "Outputs/extraction"):
        """Backward compatibility wrapper for save_to_json."""
        self.logger.warning("save_extracted_data is deprecated, use save_to_json instead")
        return self.save_to_json(questions, answers, output_dir)
    
    def load_from_json(self, input_dir: str = "Outputs/extraction") -> Tuple[List[Dict], List[str]]:
        """Load questions and answers from JSON files."""
        input_path = Path(input_dir)
        
        # Load questions
        questions_file = input_path / "questions.json"
        if not questions_file.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_file}")
        
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        # Load answers
        answers_file = input_path / "answers.json"
        if not answers_file.exists():
            raise FileNotFoundError(f"Answers file not found: {answers_file}")
        
        with open(answers_file, 'r', encoding='utf-8') as f:
            answers = json.load(f)
        
        self.logger.info(f"Loaded {len(questions)} questions and {len(answers)} answers from {input_path}")
        return questions, answers
    
    # --- Private Helper Methods ---
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract plain text from PDF file."""
        try:
            text_output = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_output += page_text + "\n\n"
            return text_output
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def _extract_questions_from_text(self, text: str) -> List[Dict]:
        """Extract questions with options from text."""
        questions = []
        lines = text.split('\n')
        
        current_question = None
        current_options = []
        question_buffer = ""
        
        # Question patterns to look for
        question_patterns = [
            r'^(\d+)\.\s*(.+)',                    # 1. Question text
            r'^Question\s*(\d+):?\s*(.+)',         # Question 1: text
            r'^Q(\d+):?\s*(.+)',                   # Q1: text
        ]
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            # Check if this line starts a new question
            question_match = self._match_question_pattern(line, question_patterns)
            
            if question_match:
                # Save previous question if it exists and has options
                if current_question and current_options:
                    questions.append({
                        'question': current_question.strip(),
                        'options': current_options.copy()
                    })
                
                # Start new question
                question_buffer = question_match.group(2).strip()
                current_question = None
                current_options = []
                
                # Look ahead to collect complete question text
                i = self._collect_complete_question_text(lines, i, question_buffer, question_patterns)
                current_question = question_buffer.strip()
            
            # Check for answer options (A, B, C, D)
            elif current_question:
                option_match = re.match(r'^([A-D])\.?\s*(.+)', line)
                if option_match:
                    option_letter = option_match.group(1)
                    option_text = option_match.group(2).strip()
                    current_options.append(f"{option_letter}. {option_text}")
            
            i += 1
        
        # Don't forget the last question
        if current_question and current_options:
            questions.append({
                'question': current_question.strip(),
                'options': current_options.copy()
            })
        
        return questions
    
    def _extract_answers_from_text(self, text: str) -> List[str]:
        """Extract answers using multiple methods and choose the best result."""
        # Try different extraction methods
        extraction_methods = [
            self._extract_answers_by_pattern,
            self._extract_answers_by_simple_key,
            self._extract_answers_by_position,
        ]
        
        results = []
        for method in extraction_methods:
            try:
                answers = method(text)
                if answers:
                    score = self._score_answer_extraction(answers)
                    results.append((answers, score, method.__name__))
                    self.logger.info(f"{method.__name__}: {len(answers)} answers, score: {score:.1f}")
            except Exception as e:
                self.logger.warning(f"Method {method.__name__} failed: {e}")
        
        if not results:
            self.logger.error("No extraction method succeeded")
            return []
        
        # Choose best result
        best_answers, best_score, best_method = max(results, key=lambda x: x[1])
        self.logger.info(f"Best method: {best_method} (score: {best_score:.1f})")
        
        return self._validate_answers(best_answers)
    
    def _extract_answers_by_pattern(self, text: str) -> List[str]:
        """Extract answers by looking for 'Correct Answer' patterns."""
        answer_dict = {}
        lines = text.split('\n')
        
        # Enhanced answer patterns with more variations
        answer_patterns = [
            r'\*\*Correct Answer\*\*:\s*([A-D])',
            r'Correct Answer:\s*([A-D])',
            r'Answer:\s*([A-D])',
            r'Solution:\s*([A-D])',
            r'Key:\s*([A-D])',
            r'Ans:\s*([A-D])',
            r'Response:\s*([A-D])',
            r'\*\*Answer:\*\*\s*([A-D])',
            r'\*\*([A-D])\*\*\s*(?:is|correct)',
            r'The correct answer is\s*([A-D])',
            r'Correct:\s*([A-D])',
        ]
        
        for i, line in enumerate(lines):
            for pattern in answer_patterns:
                match = re.search(pattern, line.strip(), re.IGNORECASE)
                if match:
                    answer_letter = match.group(1).upper()
                    question_num = self._find_question_number_backwards(lines, i)
                    if question_num:
                        answer_dict[question_num] = answer_letter
                        break
        
        return self._convert_answer_dict_to_list(answer_dict)
    
    def _extract_answers_by_simple_key(self, text: str) -> List[str]:
        """Extract answers using simple answer key patterns."""
        answer_dict = {}
        lines = text.split('\n')
        
        key_patterns = [
            r'^(\d+)\.\s*([A-D])',                # 1. A
            r'^(\d+)\)\s*([A-D])',                # 1) A
            r'^Question\s*(\d+):\s*([A-D])',      # Question 1: A
            r'^(\d+)\s*-\s*([A-D])',              # 1 - A
            r'^(\d+)\s+([A-D])',                  # 1 A
            r'^(\d+):\s*([A-D])',                 # 1: A
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            for pattern in key_patterns:
                match = re.search(pattern, line)
                if match:
                    question_num = int(match.group(1))
                    answer_letter = match.group(2).upper()
                    answer_dict[question_num] = answer_letter
                    break
        
        return self._convert_answer_dict_to_list(answer_dict)
    
    def _extract_answers_by_position(self, text: str) -> List[str]:
        """Extract answers by looking for answer patterns near question markers."""
        answer_dict = {}
        
        # Split into question sections
        question_sections = re.split(r'(?=Question\s*\d+)', text, flags=re.IGNORECASE)
        
        for section in question_sections:
            if not section.strip():
                continue
            
            # Find question number
            question_match = re.search(r'Question\s*(\d+)', section, re.IGNORECASE)
            if not question_match:
                continue
            
            question_num = int(question_match.group(1))
            
            # Look for answer in this section with expanded patterns
            answer_patterns = [
                r'correct answer:\s*([A-D])',
                r'answer:\s*([A-D])',
                r'solution:\s*([A-D])',
                r'key:\s*([A-D])',
                r'ans:\s*([A-D])',
                r'the answer is\s*([A-D])',
                r'correct:\s*([A-D])',
            ]
            
            for pattern in answer_patterns:
                matches = re.findall(pattern, section, re.IGNORECASE)
                if matches:
                    answer_dict[question_num] = matches[-1].upper()
                    break
        
        return self._convert_answer_dict_to_list(answer_dict)
    
    # --- Utility Helper Methods ---
    
    def _match_question_pattern(self, line: str, patterns: List[str]) -> Optional[re.Match]:
        """Check if line matches any question pattern."""
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match
        return None
    
    def _collect_complete_question_text(self, lines: List[str], start_idx: int, 
                                      question_buffer: str, question_patterns: List[str]) -> int:
        """Collect complete question text by looking ahead."""
        j = start_idx + 1
        while j < len(lines) and j < start_idx + 10:  # Look ahead max 10 lines
            next_line = lines[j].strip()
            
            # Stop conditions
            if (not next_line or 
                re.match(r'^[A-D]\.', next_line) or  # Hit options
                any(re.search(pattern, next_line, re.IGNORECASE) for pattern in question_patterns) or  # Hit next question
                re.search(r'correct answer|answer:', next_line, re.IGNORECASE)):  # Hit answer
                break
            
            # Continue building question text
            if next_line and not next_line.startswith('**'):
                question_buffer += " " + next_line
            
            j += 1
        
        return j - 1
    
    def _find_question_number_backwards(self, lines: List[str], current_index: int) -> Optional[int]:
        """Find question number by looking backwards from current position."""
        for j in range(current_index - 1, max(0, current_index - 20), -1):
            prev_line = lines[j].strip()
            
            patterns = [
                r'### Question (\d+)',
                r'Question (\d+)',
                r'(\d+)\.',
                r'Q(\d+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, prev_line, re.IGNORECASE)
                if match:
                    return int(match.group(1))
        
        return None
    
    def _convert_answer_dict_to_list(self, answer_dict: Dict[int, str]) -> List[str]:
        """Convert answer dictionary to ordered list, filling gaps."""
        if not answer_dict:
            return []
        
        max_question = max(answer_dict.keys())
        answers = []
        
        for i in range(1, max_question + 1):
            if i in answer_dict:
                answers.append(answer_dict[i])
            else:
                self.logger.warning(f"Missing answer for question {i}, using 'A'")
                answers.append("A")  # Default fallback
        
        return answers
    
    def _score_answer_extraction(self, answers: List[str]) -> float:
        """Score extraction method based on quality indicators."""
        if not answers:
            return 0.0
        
        score = 0.0
        
        # Score based on reasonable count (20-200 questions)
        if 20 <= len(answers) <= 200:
            score += 50.0
        elif 10 <= len(answers) <= 20:
            score += 30.0
        elif len(answers) < 10:
            score += 10.0
        else:
            score -= 20.0
        
        # Score based on answer distribution
        answer_counts = Counter(answers)
        total = len(answers)
        
        for letter, count in answer_counts.items():
            if letter in ['A', 'B', 'C', 'D']:
                percentage = count / total
                if 0.15 <= percentage <= 0.45:  # Reasonable range
                    score += 10.0
                elif percentage > 0.6:  # Too dominant
                    score -= 15.0
        
        # Score based on valid answers only
        valid_answers = [a for a in answers if a in ['A', 'B', 'C', 'D']]
        if len(valid_answers) == len(answers):
            score += 20.0
        else:
            score -= 10.0
        
        return score
    
    def _validate_answers(self, answers: List[str]) -> List[str]:
        """Validate and clean answer list."""
        validated = []
        
        for i, answer in enumerate(answers):
            if answer in ['A', 'B', 'C', 'D']:
                validated.append(answer)
            else:
                self.logger.warning(f"Invalid answer '{answer}' for question {i+1}, using 'A'")
                validated.append('A')
        
        return validated
    
    def _validate_and_clean_questions(self, questions: List[Dict]) -> List[Dict]:
        """Final validation and cleanup of extracted questions."""
        cleaned_questions = []
        
        for i, q_data in enumerate(questions):
            question = q_data['question'].strip()
            options = q_data.get('options', [])
            
            # Skip if no options
            if not options:
                self.logger.warning(f"Question {i+1} has no options, skipping")
                continue
            
            # Clean up question text
            question = re.sub(r'\s+', ' ', question)  # Normalize whitespace
            question = question.replace('\x00', ' ')  # Remove null characters
            
            # Ensure question has proper punctuation
            if not question.endswith(('?', ':', '.')):
                question += "?"
            
            cleaned_questions.append({
                'question_number': i + 1,
                'question': question,
                'options': options
            })
        
        return cleaned_questions