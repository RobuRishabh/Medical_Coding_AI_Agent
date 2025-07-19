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
        """Extract questions from practice test PDF using comprehensive extraction"""
        self.logger.info(f"Extracting questions from {pdf_path}")
        
        try:
            text = self._extract_text_from_pdf(pdf_path)
            if not text:
                self.logger.error("No text extracted from PDF")
                return []
            
            # Step 1: Parse questions with improved method
            questions = self._parse_questions_from_text_improved(text)
            self.logger.info(f"Initial extraction: {len(questions)} questions")
            
            # Step 2: Enhance questions with context
            questions = self._enhance_questions_with_context(questions, text)
            self.logger.info("Enhanced questions with context")
            
            # Step 3: Reconstruct incomplete questions
            questions = self._reconstruct_incomplete_questions(questions)
            self.logger.info("Reconstructed incomplete questions")
            
            # Step 4: Final validation and cleanup
            questions = self._validate_and_clean_questions(questions)
            
            self.logger.info(f"Successfully extracted {len(questions)} complete questions")
            return questions
        
        except Exception as e:
            self.logger.error(f"Error extracting questions: {e}")
            return []
    
    def extract_answers_from_pdf(self, pdf_path: str) -> List[str]:
        """Extract correct answers from answer key PDF with correction pipeline"""
        self.logger.info(f"Extracting answers from {pdf_path}")
        
        try:
            text = self._extract_text_from_pdf(pdf_path)
            if not text:
                self.logger.error("No text extracted from PDF")
                return []
            
            # Use the correction pipeline
            answers = self._extract_answers_with_correction_pipeline(text)
            self.logger.info(f"Successfully extracted {len(answers)} answers")
            return answers
            
        except Exception as e:
            self.logger.error(f"Error extracting answers: {e}")
            return []
    
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

    def save_extracted_data(self, questions: List[Dict], answers: List[str], output_dir: str = "Outputs/extraction"):
        """Save extracted questions and answers with comprehensive validation (backward compatibility method)"""
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
        
        # Add comprehensive validation summary
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
                answer_counts = Counter(answers)
                f.write("Answer Distribution:\n")
                for letter in ['A', 'B', 'C', 'D']:
                    count = answer_counts.get(letter, 0)
                    percentage = (count / len(answers)) * 100 if answers else 0
                    f.write(f"  {letter}: {count} ({percentage:.1f}%)\n")
                
                f.write("\nAnswer Quality Check:\n")
                # Check for reasonable distribution
                max_percentage = max(answer_counts.values()) / len(answers) * 100 if answer_counts else 0
                if max_percentage > 60:
                    f.write("WARNING: One answer dominates (>60%), may indicate extraction error\n")
                else:
                    f.write("Answer distribution appears reasonable\n")
                
                # Check for invalid answers
                invalid_answers = [a for a in answers if a not in ['A', 'B', 'C', 'D']]
                if invalid_answers:
                    f.write(f"WARNING: {len(invalid_answers)} invalid answers found: {set(invalid_answers)}\n")
                else:
                    f.write("All answers are valid (A, B, C, D)\n")

        self.logger.info(f"Extracted data saved to {output_path}")
        
        # Log validation results
        if len(questions) == len(answers):
            self.logger.info("✓ Questions and answers count match")
        else:
            self.logger.error(f"⚠ Mismatch: {len(questions)} questions vs {len(answers)} answers")

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
    
    def _parse_questions_from_text_improved(self, text: str) -> List[Dict]:
        """
        Improved question parsing that handles incomplete sentences and context
        """
        questions = []
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Question patterns
        question_patterns = [
            r'^(\d+)\.\s*(.+)',                    # 1. Question text
            r'^Question\s*(\d+):?\s*(.+)',         # Question 1: text
            r'^Q(\d+):?\s*(.+)',                   # Q1: text
            r'^(\d+)\)\s*(.+)',                    # 1) Question text
        ]
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if this line starts a new question
            question_match = self._match_question_pattern(line, question_patterns)
            
            if question_match:
                question_num = int(question_match.group(1))
                initial_question_text = question_match.group(2).strip()
                
                # Collect complete question and options
                question_data = self._collect_complete_question_and_options_v2(
                    lines, i, initial_question_text, question_patterns)
                
                if question_data:
                    question_data['question_number'] = question_num
                    questions.append(question_data)
                    i = question_data.get('next_index', i + 1)
                else:
                    i += 1
            else:
                i += 1
        
        return questions
    
    def _collect_complete_question_and_options_v2(self, lines: List[str], start_idx: int, 
                                                 initial_question_text: str, question_patterns: List[str]) -> Optional[Dict]:
        """
        Improved method to collect complete question text and all options
        """
        # Phase 1: Collect all content lines until next question
        all_content = []
        i = start_idx + 1
        
        while i < len(lines) and i < start_idx + 25:  # Look ahead max 25 lines
            line = lines[i].strip()
            
            # Stop if we hit next question
            if self._match_question_pattern(line, question_patterns):
                break
                
            # Stop if we hit answer indicators
            if re.search(r'(correct\s+answer|answer\s*:|solution\s*:)', line, re.IGNORECASE):
                break
                
            # Skip obvious headers/footers
            if (line and 
                not line.startswith('**') and 
                not line.startswith('---') and
                not re.match(r'^(page|chapter|\d+\s*$)', line, re.IGNORECASE) and
                'Medical Coding Ace' not in line and
                len(line) > 2):  # Skip very short lines
                all_content.append(line)
            
            i += 1
        
        # Phase 2: Intelligently separate question text from options
        question_text = initial_question_text
        options = []
        
        # Find where options start
        option_start_idx = None
        for idx, content_line in enumerate(all_content):
            if re.match(r'^[A-D]\.?\s+', content_line, re.IGNORECASE):
                option_start_idx = idx
                break
        
        if option_start_idx is not None:
            # Build complete question from lines before options
            question_lines = all_content[:option_start_idx]
            for q_line in question_lines:
                if len(q_line) > 10:  # Reasonable length
                    question_text += " " + q_line
            
            # Parse options
            options = self._parse_options_from_lines(all_content[option_start_idx:])
        else:
            # No clear options found - this might be a question that spans multiple lines
            # Try to find options in a different way
            question_continuation = []
            potential_options = []
            
            for content_line in all_content:
                if re.match(r'^[A-D]\.?\s+', content_line, re.IGNORECASE):
                    potential_options.append(content_line)
                else:
                    question_continuation.append(content_line)
            
            # Add continuation to question
            for q_line in question_continuation:
                if len(q_line) > 10:
                    question_text += " " + q_line
            
            # Parse found options
            if potential_options:
                options = self._parse_options_from_lines(potential_options)
        
        # Handle special case: question text in option A
        if options and len(options) > 0:
            first_option = options[0]
            if self._option_contains_question_text(first_option):
                question_text, options = self._extract_question_from_option_a(question_text, options)
        
        # Clean up
        question_text = self._clean_question_text(question_text)
        options = self._clean_options(options)
        
        # Validate
        if len(question_text.strip()) < 15 or len(options) < 2:
            return None
        
        return {
            'question': question_text,
            'options': options,
            'next_index': start_idx + len(all_content) + 1
        }
    
    def _parse_options_from_lines(self, option_lines: List[str]) -> List[str]:
        """Parse options from a list of lines"""
        options = []
        current_option = None
        current_text = ""
        
        for line in option_lines:
            # Check if this starts a new option
            option_match = re.match(r'^([A-D])\.?\s*(.+)', line, re.IGNORECASE)
            
            if option_match:
                # Save previous option
                if current_option and current_text.strip():
                    options.append(f"{current_option}. {current_text.strip()}")
                
                # Start new option
                current_option = option_match.group(1).upper()
                current_text = option_match.group(2).strip()
            else:
                # Continue current option
                if current_option and line:
                    current_text += " " + line
        
        # Don't forget the last option
        if current_option and current_text.strip():
            options.append(f"{current_option}. {current_text.strip()}")
        
        return options
    
    def _option_contains_question_text(self, option: str) -> bool:
        """Check if option A contains question text"""
        option_text = option.lower()
        indicators = [
            'which cpt code', 'what cpt code', 'which icd', 'what icd', 
            'which hcpcs', 'what hcpcs', 'which code', 'what code',
            'procedure?', 'diagnosis?', 'condition?'
        ]
        return any(indicator in option_text for indicator in indicators)
    
    def _extract_question_from_option_a(self, question_text: str, options: List[str]) -> Tuple[str, List[str]]:
        """Extract question text from option A and reconstruct proper options"""
        if not options:
            return question_text, options
        
        first_option = options[0]
        
        # Try to split option A intelligently
        option_text = first_option[2:].strip()  # Remove "A. "
        
        # Look for question indicators
        question_indicators = ['Which CPT code', 'What CPT code', 'Which ICD', 'What ICD', 'Which HCPCS']
        
        split_point = None
        for indicator in question_indicators:
            if indicator in option_text:
                split_point = option_text.find(indicator)
                break
        
        if split_point and split_point > 10:
            # Split found - add first part to question
            question_addition = option_text[:split_point].strip()
            remaining_text = option_text[split_point:].strip()
            
            new_question = f"{question_text} {question_addition} {remaining_text}"
            
            # Remove the problematic first option and shift others up
            new_options = []
            for i, opt in enumerate(options[1:], 1):
                if opt.startswith(('B.', 'C.', 'D.')):
                    # Convert B->A, C->B, D->C
                    new_letter = chr(ord('A') + i - 1)
                    new_options.append(f"{new_letter}. {opt[2:].strip()}")
            
            return new_question, new_options
        
        return question_text, options
    
    def _clean_question_text(self, text: str) -> str:
        """Clean and normalize question text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove problematic characters
        text = text.replace('\x00', ' ').replace('\r', ' ')
        
        # Remove "Medical Coding Ace" artifacts
        text = re.sub(r'\s*Medical Coding Ace\s*', '', text)
        
        # Remove trailing/leading whitespace
        text = text.strip()
        
        # Ensure question ends with appropriate punctuation
        if text and not text.endswith(('?', ':', '.', '!')):
            if any(word in text.lower() for word in ['what', 'which', 'how', 'when', 'where', 'who', 'why']):
                text += '?'
            else:
                text += ':'
        
        return text
    
    def _clean_options(self, options: List[str]) -> List[str]:
        """Clean up options list"""
        cleaned = []
        for option in options:
            # Remove "Medical Coding Ace" artifacts
            cleaned_option = re.sub(r'\s*Medical Coding Ace\s*', '', option).strip()
            # Remove trailing question marks from options
            cleaned_option = re.sub(r'\?\s*$', '', cleaned_option).strip()
            if cleaned_option:
                cleaned.append(cleaned_option)
        return cleaned
    
    def _extract_answers_with_correction_pipeline(self, text: str) -> List[str]:
        """Complete correction pipeline for answer extraction"""
        self.logger.info("Starting answer extraction correction pipeline")
        
        # Try multiple extraction methods
        methods_results = []
        
        methods = [
            self._extract_answers_method1,
            self._extract_answers_method2,
            self._extract_answers_method3,
            self._extract_answers_method4,
        ]
        
        for i, method in enumerate(methods, 1):
            try:
                results = method(text)
                if results:
                    score = self._score_extraction_method(results)
                    methods_results.append((f'method{i}', results, score))
                    self.logger.info(f"Method {i} extracted {len(results)} answers, score: {score:.1f}")
            except Exception as e:
                self.logger.warning(f"Method {i} failed: {e}")
        
        if not methods_results:
            self.logger.error("No extraction method succeeded")
            return []
        
        # Choose best method
        best_method = max(methods_results, key=lambda x: x[2])
        self.logger.info(f"Best method: {best_method[0]} with score {best_method[2]}")
        
        # Apply final validation
        return self._validate_answers(best_method[1])
    
    def _extract_answers_method1(self, text: str) -> List[str]:
        """Method 1: Look for 'Correct Answer' patterns"""
        answer_dict = {}
        lines = text.split('\n')
        
        answer_patterns = [
            r'\*\*Correct Answer\*\*:\s*([A-D])',
            r'Correct Answer:\s*([A-D])',
            r'Answer:\s*([A-D])',
            r'Solution:\s*([A-D])',
            r'Key:\s*([A-D])',
            r'Ans:\s*([A-D])',
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
    
    def _extract_answers_method2(self, text: str) -> List[str]:
        """Method 2: Simple answer key patterns"""
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
    
    def _extract_answers_method3(self, text: str) -> List[str]:
        """Method 3: Position-based extraction"""
        answer_dict = {}
        question_sections = re.split(r'(?=Question\s*\d+)', text, flags=re.IGNORECASE)
        
        for section in question_sections:
            if not section.strip():
                continue
            
            question_match = re.search(r'Question\s*(\d+)', section, re.IGNORECASE)
            if not question_match:
                continue
            
            question_num = int(question_match.group(1))
            
            answer_patterns = [
                r'correct answer:\s*([A-D])',
                r'answer:\s*([A-D])',
                r'solution:\s*([A-D])',
            ]
            
            for pattern in answer_patterns:
                matches = re.findall(pattern, section, re.IGNORECASE)
                if matches:
                    answer_dict[question_num] = matches[-1].upper()
                    break
        
        return self._convert_answer_dict_to_list(answer_dict)
    
    def _extract_answers_method4(self, text: str) -> List[str]:
        """Method 4: Smart detection with context"""
        answers = []
        question_sections = re.split(r'(?=Question\s*\d+)', text, flags=re.IGNORECASE)
        
        for section in question_sections:
            if not section.strip():
                continue
            
            question_match = re.search(r'Question\s*(\d+)', section, re.IGNORECASE)
            if not question_match:
                continue
            
            question_num = int(question_match.group(1))
            
            # Look for standalone letters that might be answers
            standalone_letters = re.findall(r'\b([A-D])\b', section)
            if standalone_letters:
                found_answer = standalone_letters[-1].upper()
                
                # Ensure we have the right number of answers
                while len(answers) < question_num - 1:
                    answers.append("A")
                
                if len(answers) == question_num - 1:
                    answers.append(found_answer)
        
        return answers
    
    # Utility methods
    def _match_question_pattern(self, line: str, patterns: List[str]) -> Optional[re.Match]:
        """Check if line matches any question pattern."""
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match
        return None
    
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
                answers.append("A")
        
        return answers
    
    def _score_extraction_method(self, answers: List[str]) -> float:
        """Score extraction method based on quality indicators."""
        if not answers:
            return 0.0
        
        score = 0.0
        
        # Score based on reasonable count
        if 20 <= len(answers) <= 200:
            score += 50.0
        elif 10 <= len(answers) <= 20:
            score += 30.0
        else:
            score += 10.0
        
        # Score based on answer distribution
        answer_counts = Counter(answers)
        total = len(answers)
        
        for letter, count in answer_counts.items():
            if letter in ['A', 'B', 'C', 'D']:
                percentage = count / total
                if 0.15 <= percentage <= 0.45:
                    score += 10.0
                elif percentage > 0.6:
                    score -= 15.0
        
        # Score based on valid answers
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
    
    def _enhance_questions_with_context(self, questions: List[Dict], full_text: str) -> List[Dict]:
        """Enhance incomplete questions by finding more context"""
        enhanced_questions = []
        
        for i, q_data in enumerate(questions):
            question = q_data['question']
            options = q_data['options']
            
            # If question seems incomplete, try to find complete version
            if len(question) < 50 or question.endswith(('a', 'an', 'the', 'for', 'to', 'with', 'and')):
                enhanced_question = self._find_complete_question(question, full_text, i + 1)
                if enhanced_question and len(enhanced_question) > len(question):
                    question = enhanced_question
            
            enhanced_questions.append({
                'question': question,
                'options': options
            })
        
        return enhanced_questions
    
    def _find_complete_question(self, partial_question: str, full_text: str, question_num: int) -> str:
        """Find complete question text from full document"""
        # Look for complete question around the question number
        pattern = rf"(?:Question\s*{question_num}|{question_num}\.)([^?]*\?)"
        match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
        if match:
            complete_question = match.group(1).strip()
            if len(complete_question) > len(partial_question):
                return complete_question
        
        return partial_question
    
    def _reconstruct_incomplete_questions(self, questions: List[Dict]) -> List[Dict]:
        """Reconstruct incomplete questions based on common patterns"""
        reconstructed = []
        
        for q_data in questions:
            question = q_data['question']
            options = q_data['options']
            
            if self._is_question_incomplete(question):
                completed_question = self._complete_question_from_context(question, options)
                question = completed_question
            
            reconstructed.append({
                'question': question,
                'options': options
            })
        
        return reconstructed
    
    def _is_question_incomplete(self, question: str) -> bool:
        """Check if a question appears to be incomplete"""
        incomplete_indicators = [
            len(question) < 40,
            question.endswith(('a', 'an', 'the', 'for', 'to', 'with', 'and', 'or', 'but')),
            question.endswith(('Dr.', 'patient', 'procedure', 'code')),
            not question.endswith(('?', ':', '.', 'code?', 'procedure?')),
        ]
        
        return any(incomplete_indicators)
    
    def _complete_question_from_context(self, partial_question: str, options: List[str]) -> str:
        """Complete questions based on options and common patterns"""
        # Analyze options to understand question type
        if all(self._looks_like_cpt_code(opt) for opt in options):
            if 'discovered a' in partial_question:
                return partial_question + " suspicious lesion. Which CPT code covers the excision?"
            elif 'diagnosed with' in partial_question:
                return partial_question + " a condition. Which CPT code should be used?"
            elif partial_question.endswith('for'):
                return partial_question + " a procedure. Which CPT code applies?"
        
        elif all(self._looks_like_icd_code(opt) for opt in options):
            if 'diagnosed with' in partial_question:
                return partial_question + " a condition. Which ICD-10-CM code should be assigned?"
        
        # Add generic completion
        if not partial_question.endswith('?'):
            return partial_question + "?"
        
        return partial_question
    
    def _looks_like_cpt_code(self, option: str) -> bool:
        """Check if option looks like a CPT code"""
        return bool(re.search(r'\b\d{5}\b', option))
    
    def _looks_like_icd_code(self, option: str) -> bool:
        """Check if option looks like an ICD code"""
        return bool(re.search(r'\b[A-Z]\d+(?:\.\d+)?\b', option))
    
    def _validate_and_clean_questions(self, questions: List[Dict]) -> List[Dict]:
        """Final validation and cleanup of extracted questions"""
        cleaned_questions = []
        
        for i, q_data in enumerate(questions):
            question = q_data['question'].strip()
            options = q_data.get('options', [])
            
            # Skip if no options
            if not options:
                self.logger.warning(f"Question {i+1} has no options, skipping")
                continue
            
            # Clean up question text
            question = re.sub(r'\s+', ' ', question)
            question = question.replace('\x00', ' ')
            
            # Ensure question has proper punctuation
            if not question.endswith(('?', ':', '.')):
                question += "?"
            
            cleaned_questions.append({
                'question_number': q_data.get('question_number', i + 1),
                'question': question,
                'options': options
            })
        
        return cleaned_questions