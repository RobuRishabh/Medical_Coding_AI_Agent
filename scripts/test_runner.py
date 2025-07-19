import json
import time
import logging
from datetime import datetime
from pathlib import Path
import sys
from scripts.test_processor import TestProcessor
from smolagents import CodeAgent
from typing import Dict, List
import re
import os
import concurrent.futures
from functools import partial
import threading

class AutomatedTestRunner:
    def __init__(self, agent_config: Dict):
        self.agent_config = agent_config
        self.processor = TestProcessor()
        self.logger = logging.getLogger(__name__)
        self.results = {
            'agent_config': agent_config,
            'test_start_time': None,
            'test_end_time': None,
            'questions_answered': 0,
            'correct_answers': 0,
            'score_percentage': 0,
            'detailed_results': [],
            'performance_metrics': {}
        }
        
        # Load the practice test prompt
        self.practice_test_prompt = self._load_practice_test_prompt()
        
        # Single agent - no pool needed
        self.agent = None
        
    def _create_single_agent(self):
        """Create a single optimized agent for test taking"""
        if self.agent is None:
            # Import from parent directory
            sys.path.append(str(Path(__file__).parent.parent))
            from app import create_test_optimized_agent
            
            try:
                self.agent = create_test_optimized_agent()
                self.logger.info("Single agent created successfully")
            except Exception as e:
                self.logger.error(f"Failed to create agent: {e}")
                self.agent = None
        
        return self.agent

    def run_test_simplified(self, questions: List[Dict], answers: List[str], 
                          progress_callback=None) -> Dict:
        """Run test with single agent - simplified approach"""
        self.results['test_start_time'] = datetime.now()
        
        # Validate input
        if not questions or not answers:
            raise ValueError("Questions and answers cannot be empty")
        
        # Create single agent
        agent = self._create_single_agent()
        if not agent:
            raise Exception("Failed to create agent")
        
        # Process questions sequentially
        total_questions = len(questions)
        self.logger.info(f"Starting test with {total_questions} questions using single agent")
        
        for i, (question_data, correct_answer) in enumerate(zip(questions, answers)):
            question_num = i + 1
            
            if progress_callback:
                progress_callback(i, total_questions, f"Processing question {question_num}/{total_questions}")
            
            try:
                # Process single question
                result = self._process_single_question(agent, question_data, correct_answer, question_num)
                
                # Update results
                if result['is_correct']:
                    self.results['correct_answers'] += 1
                self.results['questions_answered'] += 1
                
                self.results['detailed_results'].append(result)
                
                # Small delay between questions to respect rate limits
                if i < total_questions - 1:  # Don't delay after last question
                    time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error processing question {question_num}: {e}")
                # Add error result
                self.results['detailed_results'].append({
                    'question_number': question_num,
                    'question': question_data.get('question', 'Error'),
                    'options': question_data.get('options', []),
                    'agent_answer': 'A',
                    'correct_answer': correct_answer,
                    'is_correct': False,
                    'reasoning': f'Error occurred: {str(e)}'
                })
                self.results['questions_answered'] += 1
        
        # Calculate final score
        if self.results['questions_answered'] > 0:
            self.results['score_percentage'] = (
                self.results['correct_answers'] / self.results['questions_answered'] * 100
            )
        
        self.results['test_end_time'] = datetime.now()
        return self.results

    def run_test_with_extracted_data(self, questions: List[Dict], answers: List[str], progress_callback=None):
        """Run test using pre-extracted questions and answers"""
        return self.run_test_simplified(questions, answers, progress_callback)

    def run_test_with_cached_data(self, progress_callback=None):
        """Run test using cached data"""
        self.results['test_start_time'] = datetime.now()
        
        # Look for existing cached data files
        cached_data_path = Path("Outputs/extraction")
        
        if not cached_data_path.exists():
            self.logger.error("No cached data found")
            raise FileNotFoundError("No cached test data found. Please run extraction first.")
        
        try:
            # Load cached questions and answers
            with open(cached_data_path / "questions.json", 'r') as f:
                questions = json.load(f)
                
            with open(cached_data_path / "answers.json", 'r') as f:
                answers = json.load(f)
                
            # Use the simplified method
            return self.run_test_simplified(questions, answers, progress_callback)
            
        except Exception as e:
            self.logger.error(f"Error loading cached data: {e}")
            raise Exception(f"Failed to load cached test data: {e}")

    def _load_practice_test_prompt(self) -> str:
        """Load practice test prompt from prompts.json"""
        try:
            # Get the parent directory (where prompts.json is located)
            parent_dir = Path(__file__).parent.parent
            prompts_path = parent_dir / "prompts.json"
            
            with open(prompts_path, "r", encoding="utf-8") as file:
                prompts = json.load(file)
            
            prompt = prompts.get("PRACTICE_TEST_PROMPT", "")
            if not prompt:
                self.logger.warning("PRACTICE_TEST_PROMPT not found in prompts.json, using default")
                return self._get_default_prompt()
            
            self.logger.info("Practice test prompt loaded successfully")
            return prompt
            
        except Exception as e:
            self.logger.error(f"Failed to load practice test prompt: {e}")
            return self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """Fallback prompt if loading from file fails"""
        return """You are a Certified Professional Coder (CPC) taking a practice medical coding examination. Answer each multiple-choice question by selecting the most appropriate option and providing detailed reasoning.

Please provide your answer as a single letter (A, B, C, or D) at the beginning of your response, followed by your reasoning.
Format: A. [your reasoning here]"""
        
    def _calculate_delay(self, question_index: int) -> int:
        """Calculate delay between questions to respect rate limits"""
        base_delay = 2
        
        # Progressive delay
        if question_index < 5:
            return base_delay
        elif question_index < 10:
            return base_delay + 1
        elif question_index < 15:
            return base_delay + 2
        else:
            return base_delay + 3
    
    def _process_single_question(self, agent: CodeAgent, question_data: Dict, 
                               correct_answer: str, question_num: int) -> Dict:
        """Process a single question synchronously"""
        try:
            formatted_question = self._format_question_for_agent(question_data)
            
            # Get agent response with retry logic
            agent_answer, reasoning = self._get_agent_answer_with_retry(agent, question_data, question_num)
            
            # Score the answer
            is_correct = self._score_answer(agent_answer, correct_answer)
            
            return {
                'question_number': question_num,
                'question': question_data['question'],
                'options': question_data.get('options', []),
                'agent_answer': agent_answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'reasoning': reasoning
            }
            
        except Exception as e:
            self.logger.error(f"Error processing question {question_num}: {e}")
            return {
                'question_number': question_num,
                'question': question_data['question'],
                'options': question_data.get('options', []),
                'agent_answer': 'A',
                'correct_answer': correct_answer,
                'is_correct': False,
                'reasoning': f'Error occurred: {str(e)}'
            }
    
    def _warm_up_agents(self, questions: List[Dict]) -> None:
        """Warm up agents with common medical coding terms"""
        self.logger.info("Warming up agents...")
        
        # Extract common terms from questions
        common_terms = set()
        for question in questions[:5]:  # Use first 5 questions
            question_text = question['question'].lower()
            # Extract medical terms
            medical_terms = re.findall(r'\b(?:cpt|icd|procedure|diagnosis|code|medical)\b', question_text)
            common_terms.update(medical_terms)
        
        # Pre-warm agents with common searches
        if common_terms and self.agent:
            for i, agent in enumerate([self.agent]):  # Only warm the single agent
                try:
                    term = list(common_terms)[i % len(common_terms)]
                    agent.run(f"What is {term}?")
                except:
                    pass  # Ignore errors during warm-up
        
        self.logger.info("Agent warm-up completed")
    
    def run_test_optimized(self, questions: List[Dict], answers: List[str], 
                          progress_callback=None) -> Dict:
        """Run test with optimized batch processing"""
        self.results['test_start_time'] = datetime.now()
        
        # Validate input
        if not questions or not answers:
            raise ValueError("Questions and answers cannot be empty")
        
        # Create single agent
        agent = self._create_single_agent()
        if not agent:
            raise Exception("Failed to create agent")
        
        # Warm up agent
        self._warm_up_agents(questions)
        
        # Ensure questions have question_number field
        for i, question in enumerate(questions):
            if 'question_number' not in question:
                question['question_number'] = i + 1
        
        # Process questions in batches
        total_questions = len(questions)
        processed_results = []
        
        self.logger.info(f"Starting optimized test with {total_questions} questions")
        
        for i, (question_data, correct_answer) in enumerate(zip(questions, answers)):
            question_num = i + 1
            
            if progress_callback:
                progress_callback(i, total_questions, f"Processing question {question_num}/{total_questions}")
            
            # Process single question
            result = self._process_single_question(agent, question_data, correct_answer, question_num)
            
            # Update results
            if result['is_correct']:
                self.results['correct_answers'] += 1
            self.results['questions_answered'] += 1
            
            processed_results.append(result)
            
            # Add delay between questions to respect rate limits
            if i < total_questions - 1:  # Don't delay after last question
                self.logger.info(f"Waiting 1 second before next question...")
                time.sleep(1)
        
        self.results['detailed_results'] = processed_results
        
        # Calculate final score
        if self.results['questions_answered'] > 0:
            self.results['score_percentage'] = (
                self.results['correct_answers'] / self.results['questions_answered'] * 100
            )
        
        self.results['test_end_time'] = datetime.now()
        return self.results
    
    def _get_agent_answer_with_retry(self, agent: CodeAgent, question_data: Dict, question_num: int) -> tuple:
        """Get agent's answer with retry logic for rate limiting"""
        max_retries = 3
        base_delay = 5  # Start with 5 seconds
        
        for attempt in range(max_retries):
            try:
                # Format question for agent
                formatted_question = self._format_question_for_agent(question_data)
                
                # Get agent response
                response = agent.run(formatted_question)
                full_response = str(response)
                
                # Extract the chosen answer (A, B, C, D)
                answer = self._extract_answer_choice(full_response)
                
                # Extract reasoning from the response
                reasoning = self._extract_reasoning(full_response, answer)
                
                return answer, reasoning
                 
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a rate limit error
                if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                    if attempt < max_retries - 1:
                        # Exponential backoff: 5, 10, 20 seconds
                        delay = base_delay * (2 ** attempt)
                        self.logger.warning(f"Rate limit hit on question {question_num}, attempt {attempt + 1}. Retrying in {delay} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        # Max retries exceeded
                        self.logger.error(f"Max retries exceeded for question {question_num}")
                        return 'A', 'Unable to generate reasoning due to rate limiting'
                else:
                    # Non-rate-limit error
                    self.logger.error(f"Non-rate-limit error on question {question_num}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Short delay for other errors
                        continue
                    else:
                        return 'A', f'Error occurred while generating response: {str(e)}'
        
        return 'A', 'Unable to generate reasoning due to multiple failures'
    
    def _extract_reasoning(self, response: str, answer: str) -> str:
        """Extract reasoning from agent response"""
        # Remove any leading/trailing whitespace
        response = response.strip()
        
        # Look for reasoning after the answer choice
        patterns = [
            rf'^{answer}\.\s*(.+)',  # "A. [reasoning]"
            rf'Answer:\s*{answer}\s*[.\-:]\s*(.+)',  # "Answer: A. [reasoning]"
            rf'{answer}\s*[.\-:]\s*(.+)',  # "A - [reasoning]" or "A: [reasoning]"
            rf'answer\s*is\s*{answer}\s*[.\-:]\s*(.+)',  # "answer is A. [reasoning]"
            rf'correct\s*answer\s*is\s*{answer}\s*[.\-:]\s*(.+)',  # "correct answer is A. [reasoning]"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                reasoning = match.group(1).strip()
                # Clean up the reasoning
                reasoning = self._clean_reasoning(reasoning)
                return reasoning
        
        # If no specific pattern found, look for reasoning after first occurrence of answer
        lines = response.split('\n')
        found_answer = False
        reasoning_lines = []
        
        for line in lines:
            line = line.strip()
            if not found_answer and answer in line:
                found_answer = True
                # Check if there's reasoning on the same line after the answer
                after_answer = line.split(answer, 1)
                if len(after_answer) > 1:
                    remainder = after_answer[1].strip()
                    if remainder.startswith(('.', ':', '-')):
                        remainder = remainder[1:].strip()
                    if remainder:
                        reasoning_lines.append(remainder)
                continue
            
            if found_answer and line:
                reasoning_lines.append(line)
        
        if reasoning_lines:
            reasoning = ' '.join(reasoning_lines)
            return self._clean_reasoning(reasoning)
        
        # Last resort: return the entire response if we can't parse it
        return self._clean_reasoning(response)
    
    def _clean_reasoning(self, reasoning: str) -> str:
        """Clean up the reasoning text"""
        # Remove excessive whitespace
        reasoning = re.sub(r'\s+', ' ', reasoning)
        
        # Remove common prefixes that might be included
        prefixes_to_remove = [
            r'^(reasoning|explanation|because|since|the reason is|this is because):\s*',
            r'^[.\-:]\s*',
        ]
        
        for prefix in prefixes_to_remove:
            reasoning = re.sub(prefix, '', reasoning, flags=re.IGNORECASE)
        
        # Capitalize first letter
        if reasoning:
            reasoning = reasoning[0].upper() + reasoning[1:]
        
        return reasoning.strip()
    
    def run_complete_test(self, test_pdf_path: str, answers_pdf_path: str) -> Dict:
        """Run the complete automated test"""
        self.results['test_start_time'] = datetime.now()
        
        # Extract questions and answers using markdown conversion
        questions = self.processor.extract_questions_from_pdf(test_pdf_path)
        correct_answers = self.processor.extract_answers_from_pdf(answers_pdf_path)
        
        # Save extracted data for debugging
        self.processor.save_extracted_data(questions, correct_answers)
        
        # Check if extraction was successful
        if not questions:
            self.logger.error("No questions extracted from PDF")
            raise Exception("No questions could be extracted from the test PDF. Please check the file format.")
        
        if not correct_answers:
            self.logger.error("No answers extracted from answer key")
            raise Exception("No answers could be extracted from the answer key PDF. Please check the file format.")
        
        # Log extraction results instead of using Streamlit
        self.logger.info(f"Successfully extracted {len(questions)} questions and {len(correct_answers)} answers")
        
        if len(questions) != len(correct_answers):
            self.logger.warning(f"Mismatch: {len(questions)} questions vs {len(correct_answers)} answers")
            # Use the minimum to avoid index errors
            min_count = min(len(questions), len(correct_answers))
            questions = questions[:min_count]
            correct_answers = correct_answers[:min_count]
        
        # Use optimized processing
        return self.run_test_optimized(questions, correct_answers)
    
    def _format_question_for_agent(self, question_data: Dict) -> str:
        """Format question in optimal way for agent"""
        # Get question number safely
        question_number = question_data.get('question_number', 1)
        
        formatted = f"Medical Coding Question {question_number}:\n\n"
        formatted += f"{question_data['question']}\n\n"
        
        if question_data.get('options'):
            formatted += "Options:\n"
            for option in question_data['options']:
                formatted += f"{option}\n"
        
        # Add the loaded prompt instead of hardcoded instruction
        formatted += f"\n{self.practice_test_prompt}"
        
        return formatted
    
    def _extract_answer_choice(self, response: str) -> str:
        """Extract A, B, C, or D from agent response"""
        # Look for patterns like "A.", "Answer: A", "The answer is A", etc.
        patterns = [
            r'^([A-D])\.',  # Starts with "A."
            r'Answer:\s*([A-D])',  # "Answer: A"
            r'answer\s*is\s*([A-D])',  # "answer is A"
            r'correct\s*answer\s*is\s*([A-D])',  # "correct answer is A"
            r'\b([A-D])\b',  # Any single letter A-D
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # If no pattern found, look for the first A, B, C, or D
        for char in response:
            if char.upper() in ['A', 'B', 'C', 'D']:
                return char.upper()
        
        # Default to A if nothing found
        self.logger.warning(f"Could not extract answer from response: {response[:100]}...")
        return 'A'
    
    def _score_answer(self, agent_answer: str, correct_answer: str) -> bool:
        """Score individual answer"""
        return agent_answer.strip().upper() == correct_answer.strip().upper()
    
    def run_test_batch_processing(self, questions: List[Dict], answers: List[str], 
                                progress_callback=None) -> Dict:
        """Run test with batch processing - all questions at once"""
        self.results['test_start_time'] = datetime.now()
        
        # Validate input
        if not questions or not answers:
            raise ValueError("Questions and answers cannot be empty")
        
        # Create single agent
        agent = self._create_single_agent()
        if not agent:
            raise Exception("Failed to create agent")
        
        total_questions = len(questions)
        self.logger.info(f"Starting batch test with {total_questions} questions")
        
        if progress_callback:
            progress_callback(0, total_questions, "Preparing batch request...")
        
        try:
            # Format all questions for batch processing
            batch_question = self._format_all_questions_for_batch(questions)
            
            if progress_callback:
                progress_callback(total_questions//4, total_questions, "Sending batch request to agent...")
            
            # Get agent response for all questions at once
            response = agent.run(batch_question)
            full_response = str(response)
            
            if progress_callback:
                progress_callback(total_questions//2, total_questions, "Processing agent response...")
            
            # Extract all answers from the batch response
            agent_answers = self._extract_batch_answers(full_response, total_questions)
            
            if progress_callback:
                progress_callback(3*total_questions//4, total_questions, "Scoring answers...")
            
            # Score all answers
            for i, (question_data, correct_answer, agent_answer) in enumerate(zip(questions, answers, agent_answers)):
                question_num = i + 1
                is_correct = self._score_answer(agent_answer, correct_answer)
                
                if is_correct:
                    self.results['correct_answers'] += 1
                self.results['questions_answered'] += 1
                
                self.results['detailed_results'].append({
                    'question_number': question_num,
                    'question': question_data['question'],
                    'options': question_data.get('options', []),
                    'agent_answer': agent_answer,
                    'correct_answer': correct_answer,
                    'is_correct': is_correct,
                    'reasoning': 'Batch processing - no individual reasoning'
                })
            
            if progress_callback:
                progress_callback(total_questions, total_questions, "Batch processing completed!")
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
            # Fallback to individual processing if batch fails
            self.logger.info("Falling back to individual question processing...")
            return self.run_test_simplified(questions, answers, progress_callback)
        
        # Calculate final score
        if self.results['questions_answered'] > 0:
            self.results['score_percentage'] = (
                self.results['correct_answers'] / self.results['questions_answered'] * 100
            )
        
        self.results['test_end_time'] = datetime.now()
        return self.results

    def _format_all_questions_for_batch(self, questions: List[Dict]) -> str:
        """Format all questions for batch processing"""
        formatted = "Medical Coding Practice Test - Answer ALL questions:\n\n"
        
        for i, question_data in enumerate(questions, 1):
            formatted += f"Question {i}:\n"
            formatted += f"{question_data['question']}\n"
            
            if question_data.get('options'):
                for option in question_data['options']:
                    formatted += f"{option}\n"
            formatted += "\n"
        
        # Add the batch processing prompt
        formatted += f"\n{self.practice_test_prompt}"
        
        return formatted

    def _extract_batch_answers(self, response: str, expected_count: int) -> List[str]:
        """Extract answers from batch response"""
        self.logger.info("Extracting batch answers from response")
        
        # Look for the ANSWERS section
        answers_section = ""
        if "**ANSWERS:**" in response:
            answers_section = response.split("**ANSWERS:**")[1]
        elif "ANSWERS:" in response:
            answers_section = response.split("ANSWERS:")[1]
        else:
            # Fallback: look for numbered answers in the entire response
            answers_section = response
        
        # Extract answers using multiple patterns
        patterns = [
            r'(\d+)\.\s*([A-D])',  # "1. A"
            r'(\d+)\)\s*([A-D])',  # "1) A"
            r'(\d+):\s*([A-D])',   # "1: A"
            r'(\d+)\s+([A-D])',    # "1 A"
        ]
        
        answer_dict = {}
        for pattern in patterns:
            matches = re.findall(pattern, answers_section, re.IGNORECASE)
            for match in matches:
                question_num = int(match[0])
                answer_letter = match[1].upper()
                if question_num <= expected_count:
                    answer_dict[question_num] = answer_letter
        
        # Convert to ordered list
        answers = []
        for i in range(1, expected_count + 1):
            if i in answer_dict:
                answers.append(answer_dict[i])
            else:
                self.logger.warning(f"Missing answer for question {i}, using 'A'")
                answers.append('A')  # Default fallback
        
        self.logger.info(f"Extracted {len(answers)} answers from batch response")
        return answers

    # Update the main methods to use batch processing
    def run_test_with_extracted_data(self, questions: List[Dict], answers: List[str], progress_callback=None):
        """Run test using pre-extracted questions and answers with batch processing"""
        return self.run_test_batch_processing(questions, answers, progress_callback)

    def run_test_with_cached_data(self, progress_callback=None):
        """Run test using cached data with batch processing"""
        self.results['test_start_time'] = datetime.now()
        
        # Look for existing cached data files
        cached_data_path = Path("Outputs/extraction")
        
        if not cached_data_path.exists():
            self.logger.error("No cached data found")
            raise FileNotFoundError("No cached test data found. Please run extraction first.")
        
        try:
            # Load cached questions and answers
            with open(cached_data_path / "questions.json", 'r') as f:
                questions = json.load(f)
                
            with open(cached_data_path / "answers.json", 'r') as f:
                answers = json.load(f)
                
            # Use the batch processing method
            return self.run_test_batch_processing(questions, answers, progress_callback)
            
        except Exception as e:
            self.logger.error(f"Error loading cached data: {e}")
            raise Exception(f"Failed to load cached test data: {e}")
