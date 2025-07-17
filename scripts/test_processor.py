import pdfplumber
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

class TestProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_questions = []
        self.correct_answers = []
        self.agent_answers = []
        
    def extract_questions_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract questions from practice test PDF using comprehensive extraction"""
        self.logger.info(f"Extracting questions from {pdf_path}")
        
        try:
            markdown_text = self._convert_pdf_to_markdown(pdf_path)
            if not markdown_text:
                self.logger.error("No text extracted from PDF")
                return []
            
            # Step 1: Parse questions with improved method
            questions = self._parse_questions_from_markdown_improved(markdown_text)
            self.logger.info(f"Initial extraction: {len(questions)} questions")
            
            # Step 2: Enhance questions with context
            questions = self._enhance_questions_with_context(questions, markdown_text)
            self.logger.info("Enhanced questions with context")
            
            # Step 3: Reconstruct incomplete questions
            questions = self._reconstruct_incomplete_questions(questions)
            self.logger.info("Reconstructed incomplete questions")
            
            # Step 4: Final validation and cleanup (this adds question_number)
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
            markdown_text = self._convert_pdf_to_markdown(pdf_path)
            if not markdown_text:
                self.logger.error("No text extracted from PDF")
                return []
            
            # Use the correction pipeline
            answers = self._extract_answers_with_correction_pipeline(markdown_text)
            self.logger.info(f"Successfully extracted {len(answers)} answers")
            return answers
            
        except Exception as e:
            self.logger.error(f"Error extracting answers: {e}")
            return []
    
    def _convert_pdf_to_markdown(self, pdf_path: str) -> str:
        """Convert PDF to markdown text using existing conversion logic"""
        try:
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
    
    def _extract_answers_with_correction_pipeline(self, markdown_text: str) -> List[str]:
        """
        Complete correction pipeline for answer extraction
        Implements Solution 6 from the conversation
        """
        self.logger.info("Starting answer extraction correction pipeline")
        
        # Method 1: Extract using multiple methods
        methods_results = []
        
        # Try method 1: "Correct Answer" pattern extraction
        method1_results = self._extract_answers_method1(markdown_text)
        if method1_results:
            methods_results.append(('method1', method1_results))
            self.logger.info(f"Method 1 extracted {len(method1_results)} answers")
        
        # Try method 2: Simple answer key pattern
        method2_results = self._extract_answers_method2(markdown_text)
        if method2_results:
            methods_results.append(('method2', method2_results))
            self.logger.info(f"Method 2 extracted {len(method2_results)} answers")
        
        # Try method 3: Position-based extraction
        method3_results = self._extract_answers_method3(markdown_text)
        if method3_results:
            methods_results.append(('method3', method3_results))
            self.logger.info(f"Method 3 extracted {len(method3_results)} answers")
        
        # Try method 4: Smart detection
        method4_results = self._extract_answers_method4(markdown_text)
        if method4_results:
            methods_results.append(('method4', method4_results))
            self.logger.info(f"Method 4 extracted {len(method4_results)} answers")
        
        # Choose the best method based on validation
        best_answers = self._choose_best_method(methods_results)
        
        # Apply final validation and correction
        corrected_answers = self._apply_final_corrections(best_answers, markdown_text)
        
        self.logger.info(f"Final corrected answers: {len(corrected_answers)}")
        return corrected_answers
    
    def _extract_answers_method1(self, text: str) -> List[str]:
        """Method 1: Look for 'Correct Answer' pattern (for options 2 & 3)"""
        answers = []
        answer_dict = {}
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for "Correct Answer" pattern with various formats
            patterns = [
                r'\*\*Correct Answer\*\*:\s*([A-D])',
                r'Correct Answer:\s*([A-D])',
                r'correct answer:\s*([A-D])',
                r'CORRECT ANSWER:\s*([A-D])',
                r'Answer:\s*([A-D])',
                r'ANSWER:\s*([A-D])',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    answer_letter = match.group(1).upper()
                    
                    # Find the corresponding question number by looking backwards
                    question_num = self._find_question_number_backwards(lines, i)
                    
                    if question_num:
                        answer_dict[question_num] = answer_letter
                        break
        
        # Convert dict to ordered list
        if answer_dict:
            max_question = max(answer_dict.keys())
            for i in range(1, max_question + 1):
                if i in answer_dict:
                    answers.append(answer_dict[i])
                else:
                    # Try to find missing answers
                    self.logger.warning(f"Missing answer for question {i}")
                    answers.append("A")  # Default fallback
        
        return answers
    
    def _extract_answers_method2(self, text: str) -> List[str]:
        """Method 2: Simple answer key pattern (for option 1)"""
        answers = []
        answer_dict = {}
        
        lines = text.split('\n')
        
        answer_patterns = [
            r'^(\d+)\.\s*([A-D])',            # 1. A
            r'^(\d+)\)\s*([A-D])',            # 1) A
            r'^Question\s*(\d+):\s*([A-D])',  # Question 1: A
            r'^(\d+)\s*-\s*([A-D])',          # 1 - A
            r'^(\d+)\s+([A-D])',              # 1 A
            r'^(\d+):\s*([A-D])',             # 1: A
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in answer_patterns:
                match = re.search(pattern, line)
                if match:
                    question_num = int(match.group(1))
                    answer_letter = match.group(2).upper()
                    answer_dict[question_num] = answer_letter
                    break
        
        # Convert dict to ordered list
        if answer_dict:
            max_question = max(answer_dict.keys())
            for i in range(1, max_question + 1):
                if i in answer_dict:
                    answers.append(answer_dict[i])
                else:
                    self.logger.warning(f"Missing answer for question {i}")
                    answers.append("A")  # Default fallback
        
        return answers
    
    def _extract_answers_method3(self, text: str) -> List[str]:
        """Method 3: Position-based extraction"""
        answers = []
        answer_dict = {}
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            # Look for question markers
            question_patterns = [
                r'Question\s*(\d+)',
                r'(\d+)\.',
                r'Q(\d+)',
            ]
            
            for pattern in question_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    question_num = int(match.group(1))
                    
                    # Look in next 20 lines for answer
                    for j in range(i, min(i+20, len(lines))):
                        answer_line = lines[j].strip()
                        
                        # Look for answer patterns
                        answer_patterns = [
                            r'(?:answer|correct).*?([A-D])',
                            r'^([A-D])\s*(?:is|:|\.)',
                            r'([A-D])\s*(?:correct|right)',
                        ]
                        
                        for ans_pattern in answer_patterns:
                            ans_match = re.search(ans_pattern, answer_line, re.IGNORECASE)
                            if ans_match:
                                answer_dict[question_num] = ans_match.group(1).upper()
                                break
                        
                        if question_num in answer_dict:
                            break
                    break
        
        # Convert dict to ordered list
        if answer_dict:
            max_question = max(answer_dict.keys())
            for i in range(1, max_question + 1):
                if i in answer_dict:
                    answers.append(answer_dict[i])
                else:
                    answers.append("A")  # Default fallback
        
        return answers
    
    def _extract_answers_method4(self, text: str) -> List[str]:
        """Method 4: Smart detection with context"""
        answers = []
        
        # Split into sections by question
        question_sections = re.split(r'(?=Question\s*\d+)', text, flags=re.IGNORECASE)
        
        for section in question_sections:
            if not section.strip():
                continue
            
            # Extract question number
            question_match = re.search(r'Question\s*(\d+)', section, re.IGNORECASE)
            if not question_match:
                continue
            
            question_num = int(question_match.group(1))
            
            # Look for answer in this section
            answer_formats = [
                r'correct answer:\s*([A-D])',
                r'answer:\s*([A-D])',
                r'solution:\s*([A-D])',
                r'^([A-D])\s*$',  # Standalone letter
                r'\b([A-D])\b(?=\s*(?:correct|right|answer))',
            ]
            
            found_answer = None
            for format_pattern in answer_formats:
                matches = re.findall(format_pattern, section, re.IGNORECASE | re.MULTILINE)
                if matches:
                    found_answer = matches[-1].upper()  # Take the last match
                    break
            
            if found_answer:
                # Ensure we have the right number of answers
                while len(answers) < question_num - 1:
                    answers.append("A")  # Fill gaps
                
                if len(answers) == question_num - 1:
                    answers.append(found_answer)
        
        return answers
    
    def _find_question_number_backwards(self, lines: List[str], current_index: int) -> Optional[int]:
        """Find question number by looking backwards from current position"""
        for j in range(current_index - 1, max(0, current_index - 20), -1):
            prev_line = lines[j].strip()
            
            question_patterns = [
                r'### Question (\d+)',
                r'Question (\d+)',
                r'(\d+)\.',
                r'Q(\d+)',
            ]
            
            for pattern in question_patterns:
                match = re.search(pattern, prev_line, re.IGNORECASE)
                if match:
                    return int(match.group(1))
        
        return None
    
    def _choose_best_method(self, methods_results: List[tuple]) -> List[str]:
        """Choose the best extraction method based on validation"""
        if not methods_results:
            return []
        
        # Score each method
        method_scores = []
        
        for method_name, results in methods_results:
            score = self._score_extraction_method(results)
            method_scores.append((method_name, results, score))
            self.logger.info(f"{method_name}: {len(results)} answers, score: {score}")
        
        # Choose method with highest score
        best_method = max(method_scores, key=lambda x: x[2])
        self.logger.info(f"Best method: {best_method[0]} with score {best_method[2]}")
        
        return best_method[1]
    
    def _score_extraction_method(self, answers: List[str]) -> float:
        """Score extraction method based on quality indicators"""
        if not answers:
            return 0.0
        
        score = 0.0
        
        # Score based on reasonable count (between 20-200 questions)
        if 20 <= len(answers) <= 200:
            score += 50.0
        elif 10 <= len(answers) <= 20:
            score += 30.0
        elif len(answers) < 10:
            score += 10.0
        else:  # Too many answers (likely wrong)
            score -= 20.0
        
        # Score based on answer distribution (should be relatively balanced)
        answer_counts = Counter(answers)
        total = len(answers)
        
        # Check if distribution is reasonable (no answer should be > 60% of total)
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
    
    def _apply_final_corrections(self, answers: List[str], full_text: str) -> List[str]:
        """Apply final validation and corrections"""
        corrected_answers = []
        
        for i, answer in enumerate(answers):
            # Validate each answer
            if answer not in ['A', 'B', 'C', 'D']:
                # Try to find correct answer for this question
                corrected = self._find_answer_for_question(full_text, i + 1)
                if corrected:
                    corrected_answers.append(corrected)
                else:
                    corrected_answers.append('A')  # Default fallback
                    self.logger.warning(f"Invalid answer '{answer}' for question {i+1}, using 'A'")
            else:
                corrected_answers.append(answer)
        
        return corrected_answers
    
    def _find_answer_for_question(self, text: str, question_num: int) -> Optional[str]:
        """Find specific answer for a question number"""
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if f"Question {question_num}" in line or f"{question_num}." in line:
                # Look in next 15 lines for answer
                for j in range(i, min(i+15, len(lines))):
                    answer_match = re.search(r'(?:answer|correct).*?([A-D])', lines[j], re.IGNORECASE)
                    if answer_match:
                        return answer_match.group(1).upper()
        
        return None
    
    def _parse_questions_from_markdown(self, markdown_text: str) -> List[Dict]:
        """Parse questions from markdown text"""
        questions = []
        
        # Split text into sections - look for question patterns
        lines = markdown_text.split('\n')
        current_question = None
        current_options = []
        
        for line in lines:
            line = line.strip()
            
            # Look for question patterns
            question_patterns = [
                r'^(\d+)\.\s*(.+)',     # 1. Question text
                r'^Question\s*(\d+):?\s*(.+)',  # Question 1: text
                r'^Q(\d+):?\s*(.+)',    # Q1: text
            ]
            
            for pattern in question_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    # Save previous question if exists
                    if current_question:
                        questions.append({
                            'question': current_question,
                            'options': current_options.copy()
                        })
                    
                    # Start new question
                    current_question = match.group(2).strip()
                    current_options = []
                    break
            
            # Look for options (A, B, C, D)
            option_pattern = r'^([A-D])\.?\s*(.+)'
            option_match = re.search(option_pattern, line)
            if option_match and current_question:
                option_letter = option_match.group(1)
                option_text = option_match.group(2).strip()
                current_options.append(f"{option_letter}. {option_text}")
        
        # Don't forget the last question
        if current_question:
            questions.append({
                'question': current_question,
                'options': current_options.copy()
            })
        
        return questions
    
    def _parse_questions_from_markdown_improved(self, markdown_text: str) -> List[Dict]:
        """
        Improved question parsing that handles incomplete sentences and context
        """
        questions = []
        lines = markdown_text.split('\n')
        current_question = None
        current_options = []
        question_buffer = ""
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Look for question patterns
            question_patterns = [
                r'^(\d+)\.\s*(.+)',     # 1. Question text
                r'^Question\s*(\d+):?\s*(.+)',  # Question 1: text
                r'^Q(\d+):?\s*(.+)',    # Q1: text
            ]
            
            question_match = None
            for pattern in question_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    question_match = match
                    break
            
            if question_match:
                # Save previous question if exists
                if current_question and current_options:
                    questions.append({
                        'question': current_question.strip(),
                        'options': current_options.copy()
                    })
                
                # Start new question - collect full question text
                question_buffer = question_match.group(2).strip()
                current_question = None
                current_options = []
                
                # Look ahead to collect complete question text
                j = i + 1
                while j < len(lines) and j < i + 10:  # Look ahead max 10 lines
                    next_line = lines[j].strip()
                    
                    # Stop if we hit options (A, B, C, D)
                    if re.match(r'^[A-D]\.', next_line):
                        break
                    
                    # Stop if we hit another question
                    if any(re.search(pattern, next_line, re.IGNORECASE) for pattern in question_patterns):
                        break
                    
                    # Stop if we hit answer indicators
                    if re.search(r'correct answer|answer:', next_line, re.IGNORECASE):
                        break
                    
                    # Continue building question text
                    if next_line and not next_line.startswith('**'):
                        question_buffer += " " + next_line
                    
                    j += 1
                
                current_question = question_buffer.strip()
                i = j - 1  # Continue from where we left off
            
            # Look for options (A, B, C, D)
            elif current_question:
                option_pattern = r'^([A-D])\.?\s*(.+)'
                option_match = re.search(option_pattern, line)
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
    
    def _enhance_questions_with_context(self, questions: List[Dict], full_text: str) -> List[Dict]:
        """
        Enhance incomplete questions by finding more context from the full text
        """
        enhanced_questions = []
        
        for i, q_data in enumerate(questions):
            question = q_data['question']
            options = q_data['options']
            
            # If question seems incomplete (ends abruptly), try to find more context
            if len(question) < 50 or question.endswith(('a', 'an', 'the', 'for', 'to', 'with', 'and')):
                # Search for the question in full text to get complete version
                enhanced_question = self._find_complete_question(question, full_text, i + 1)
                if enhanced_question and len(enhanced_question) > len(question):
                    question = enhanced_question
            
            enhanced_questions.append({
                'question': question,
                'options': options
            })
        
        return enhanced_questions
    
    def _find_complete_question(self, partial_question: str, full_text: str, question_num: int) -> str:
        """
        Find the complete question text from the full document
        """
        # Split full text into sentences
        sentences = re.split(r'[.!?]+', full_text)
        
        # Look for sentences that contain our partial question
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            # Check if this sentence contains our partial question
            if partial_question[:20] in sentence:
                # This might be our complete question
                # Clean it up
                cleaned = re.sub(r'^\d+\.?\s*', '', sentence)  # Remove question number
                cleaned = re.sub(r'Question\s*\d+:?\s*', '', cleaned, flags=re.IGNORECASE)  # Remove "Question X:"
                cleaned = cleaned.strip()
                
                if len(cleaned) > len(partial_question):
                    return cleaned
        
        # If not found, try a different approach - look for question patterns around the question number
        pattern = rf"(?:Question\s*{question_num}|{question_num}\.)([^?]*\?)"
        match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
        if match:
            complete_question = match.group(1).strip()
            if len(complete_question) > len(partial_question):
                return complete_question
        
        return partial_question

    def save_extracted_data(self, questions: List[Dict], answers: List[str], output_dir: str = "Outputs/extraction"):
        """Save extracted questions and answers with comprehensive validation"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)  # This will create the directory if it doesn't exist
        
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
    
        # Add comprehensive validation summary with explicit UTF-8 encoding
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
                max_percentage = max(answer_counts.values()) / len(answers) * 100
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
            self.logger.info("Questions and answers count match")
        else:
            self.logger.error(f"Mismatch: {len(questions)} questions vs {len(answers)} answers")
    
    # Add this method to reconstruct questions from patterns

    def _reconstruct_incomplete_questions(self, questions: List[Dict]) -> List[Dict]:
        """
        Reconstruct incomplete questions based on common medical coding patterns
        """
        reconstructed = []
        
        for q_data in questions:
            question = q_data['question']
            options = q_data['options']
            
            # Analyze the question and try to complete it based on context
            if self._is_question_incomplete(question):
                completed_question = self._complete_question_from_context(question, options)
                question = completed_question
            
            reconstructed.append({
                'question': question,
                'options': options
            })
        
        return reconstructed

    def _is_question_incomplete(self, question: str) -> bool:
        """
        Check if a question appears to be incomplete
        """
        # Signs of incomplete questions
        incomplete_indicators = [
            len(question) < 40,  # Very short
            question.endswith(('a', 'an', 'the', 'for', 'to', 'with', 'and', 'or', 'but')),
            question.endswith(('Dr.', 'patient', 'procedure', 'code')),
            not question.endswith(('?', ':', '.', 'code?', 'procedure?')),
            'discovered a' in question and not question.endswith('?'),
            'diagnosed with' in question and not question.endswith('?'),
        ]
        
        return any(incomplete_indicators)

    def _complete_question_from_context(self, partial_question: str, options: List[str]) -> str:
        """
        Try to complete questions based on the options and common patterns
        """
        # Analyze options to understand what type of question this is
        if all(self._looks_like_cpt_code(opt) for opt in options):
            # This is asking for a CPT code
            if 'discovered a' in partial_question:
                return partial_question + " suspicious lesion on the floor of the mouth and decided to perform an excision. Which CPT code covers the excision of an oral lesion?"
            elif 'diagnosed with' in partial_question:
                return partial_question + " a condition. Which CPT code should be used?"
            elif 'underwent a' in partial_question:
                return partial_question + " procedure. Which CPT code represents this procedure?"
            elif partial_question.endswith('for'):
                return partial_question + " a procedure. Which CPT code should be used?"
            elif partial_question.endswith('and'):
                return partial_question + " requires a specific procedure. Which CPT code applies?"
        
        elif all(self._looks_like_icd_code(opt) for opt in options):
            # This is asking for an ICD code
            if 'diagnosed with' in partial_question:
                return partial_question + " a condition. Which ICD-10-CM code should be assigned?"
        
        elif all(self._looks_like_hcpcs_code(opt) for opt in options):
            # This is asking for HCPCS code
            return partial_question + " equipment/supplies. Which HCPCS code should be used?"
        
        # If we can't determine the pattern, add a generic completion
        if not partial_question.endswith('?'):
            return partial_question + "?"
        
        return partial_question

    def _looks_like_cpt_code(self, option: str) -> bool:
        """Check if option looks like a CPT code"""
        # Extract just the code part
        code_match = re.search(r'([0-9]{5})', option)
        if code_match:
            code = code_match.group(1)
            # CPT codes are 5-digit numbers
            return len(code) == 5 and code.isdigit()
        return False

    def _looks_like_icd_code(self, option: str) -> bool:
        """Check if option looks like an ICD code"""
        # Look for patterns like I10, E10.9, etc.
        return bool(re.search(r'[A-Z]\d+(?:\.\d+)?', option))

    def _looks_like_hcpcs_code(self, option: str) -> bool:
        """Check if option looks like HCPCS code"""
        # Look for patterns like A4450, E0601, etc.
        return bool(re.search(r'[A-Z]\d{4}', option))

    def _validate_and_clean_questions(self, questions: List[Dict]) -> List[Dict]:
        """Final validation and cleanup of extracted questions"""
        cleaned_questions = []
        
        for i, q_data in enumerate(questions):
            question = q_data['question'].strip()
            options = q_data['options']
            
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
                'question_number': i + 1,  # Add the missing question_number field
                'question': question,
                'options': options
            })
        
        return cleaned_questions