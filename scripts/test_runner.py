import json
import time
import logging
from datetime import datetime
from pathlib import Path
import sys
from scripts.test_processor import TestProcessor
from smolagents import CodeAgent
import streamlit as st
from typing import Dict
import re
import os

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
        # Gemini free tier: 15 requests per minute
        # Base delay: 4 seconds (60/15 = 4 seconds per request)
        base_delay = 4
        
        # Progressive delay - increase delay as we go to be safe
        if question_index < 5:
            return base_delay  # 4 seconds for first 5 questions
        elif question_index < 10:
            return base_delay + 1  # 5 seconds for next 5 questions
        elif question_index < 15:
            return base_delay + 2  # 6 seconds for next 5 questions
        else:
            return base_delay + 3  # 7 seconds for remaining questions
    
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
        
        # Show extraction results
        st.success(f"✅ Successfully extracted {len(questions)} questions and {len(correct_answers)} answers")
        
        if len(questions) != len(correct_answers):
            self.logger.warning(f"Mismatch: {len(questions)} questions vs {len(correct_answers)} answers")
            # Use the minimum to avoid index errors
            min_count = min(len(questions), len(correct_answers))
            questions = questions[:min_count]
            correct_answers = correct_answers[:min_count]
        
        # Create agent
        agent = self._create_optimized_agent()
        if not agent:
            self.logger.error("Failed to create agent")
            raise Exception("Failed to create AI agent")
        
        # Process each question with rate limiting
        for i, question_data in enumerate(questions):
            try:
                self.logger.info(f"Processing question {i+1}/{len(questions)}")
                
                # Generate answer and reasoning using agent with retry logic
                agent_answer, reasoning = self._get_agent_answer_with_retry(agent, question_data, i+1)
                correct_answer = correct_answers[i]
                
                # Score the answer
                is_correct = self._score_answer(agent_answer, correct_answer)
                
                # Store detailed results
                self.results['detailed_results'].append({
                    'question_number': i + 1,
                    'question': question_data['question'],
                    'options': question_data.get('options', []),
                    'agent_answer': agent_answer,
                    'correct_answer': correct_answer,
                    'is_correct': is_correct,
                    'reasoning': reasoning  # Now contains actual agent reasoning
                })
                
                if is_correct:
                    self.results['correct_answers'] += 1
                    
                self.results['questions_answered'] += 1
                
                # Show progress
                progress = (i + 1) / len(questions)
                st.progress(progress, f"Processing question {i+1}/{len(questions)}")
                
                # Add delay between questions to respect rate limits
                if i < len(questions) - 1:  # Don't delay after last question
                    delay = self._calculate_delay(i)
                    self.logger.info(f"Waiting {delay} seconds before next question...")
                    time.sleep(delay)
                
            except Exception as e:
                self.logger.error(f"Error processing question {i+1}: {e}")
                continue
        
        # Calculate final score
        if self.results['questions_answered'] > 0:
            self.results['score_percentage'] = (
                self.results['correct_answers'] / self.results['questions_answered'] * 100
            )
        
        self.results['test_end_time'] = datetime.now()
        
        return self.results
    
    def _create_optimized_agent(self) -> CodeAgent:
        """Create agent optimized for test taking"""
        # Import from parent directory
        sys.path.append(str(Path(__file__).parent.parent))
        from app import create_dynamic_agent
        
        try:
            # Create agent with both tools for maximum accuracy
            agent = create_dynamic_agent(use_knowledge_base=True, use_web_search=True)
            return agent
        except Exception as e:
            self.logger.error(f"Failed to create agent: {e}")
            return None
    
    def _format_question_for_agent(self, question_data: Dict) -> str:
        """Format question in optimal way for agent"""
        formatted = f"Medical Coding Question {question_data['question_number']}:\n\n"
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
    
    def run_test_with_cached_data(self):
        """Run test using already extracted and cached data"""
        self.results['test_start_time'] = datetime.now()
        
        # Look for existing cached data files
        cached_data_path = "temp_test_data"
        
        if not os.path.exists(cached_data_path):
            self.logger.error("No cached test data found")
            raise FileNotFoundError("No cached test data found. Please run the test with PDFs first to generate cached data.")
        
        try:
            # Load cached questions and answers
            questions_file = os.path.join(cached_data_path, "extracted_questions.json")
            answers_file = os.path.join(cached_data_path, "extracted_answers.json")

            if not os.path.exists(questions_file) or not os.path.exists(answers_file):
                self.logger.error("Cached data files not found")
                raise FileNotFoundError("Cached data files (extracted_questions.json or extracted_answers.json) not found in temp_test_data directory.")
            
            # Load the cached data
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            
            with open(answers_file, 'r', encoding='utf-8') as f:
                correct_answers = json.load(f)
            
            # Validate loaded data
            if not questions:
                self.logger.error("No questions found in cached data")
                raise ValueError("No questions found in cached data")
            
            if not correct_answers:
                self.logger.error("No answers found in cached data")
                raise ValueError("No answers found in cached data")
            
            # Show cached data info
            st.success(f"✅ Loaded cached data: {len(questions)} questions and {len(correct_answers)} answers")
            
            # Ensure questions and answers match in count
            if len(questions) != len(correct_answers):
                self.logger.warning(f"Mismatch in cached data: {len(questions)} questions vs {len(correct_answers)} answers")
                # Use the minimum to avoid index errors
                min_count = min(len(questions), len(correct_answers))
                questions = questions[:min_count]
                correct_answers = correct_answers[:min_count]
            
            # Create agent
            agent = self._create_optimized_agent()
            if not agent:
                self.logger.error("Failed to create agent")
                # Initialize default results before raising exception
                self.results['test_end_time'] = datetime.now()
                self.results['score_percentage'] = 0
                raise Exception("Failed to create AI agent")
            
            # Process each question with rate limiting
            for i, question_data in enumerate(questions):
                try:
                    self.logger.info(f"Processing question {i+1}/{len(questions)}")
                    
                    # Generate answer and reasoning using agent with retry logic
                    agent_answer, reasoning = self._get_agent_answer_with_retry(agent, question_data, i+1)
                    correct_answer = correct_answers[i]
                    
                    # Score the answer
                    is_correct = self._score_answer(agent_answer, correct_answer)
                    
                    # Store detailed results
                    self.results['detailed_results'].append({
                        'question_number': i + 1,
                        'question': question_data['question'],
                        'options': question_data.get('options', []),
                        'agent_answer': agent_answer,
                        'correct_answer': correct_answer,
                        'is_correct': is_correct,
                        'reasoning': reasoning
                    })
                    
                    if is_correct:
                        self.results['correct_answers'] += 1
                        
                    self.results['questions_answered'] += 1
                    
                    # Show progress
                    progress = (i + 1) / len(questions)
                    st.progress(progress, f"Processing question {i+1}/{len(questions)}")
                    
                    # Add delay between questions to respect rate limits
                    if i < len(questions) - 1:  # Don't delay after last question
                        delay = self._calculate_delay(i)
                        self.logger.info(f"Waiting {delay} seconds before next question...")
                        time.sleep(delay)
                
                except Exception as e:
                    self.logger.error(f"Error processing question {i+1}: {e}")
                    # Continue processing other questions
                    continue
        
        except Exception as e:
            self.logger.error(f"Error in run_test_with_cached_data: {e}")
            # Ensure we always return a results dictionary, even on error
            self.results['test_end_time'] = datetime.now()
            self.results['score_percentage'] = 0
            # Re-raise the exception after setting default values
            raise Exception(f"Failed to load cached data: {str(e)}")