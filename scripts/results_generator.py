import json
from typing import Dict

class ResultsGenerator:
    def __init__(self, results: Dict):
        self.results = results
        
    def generate_comprehensive_report(self) -> str:
        """Generate detailed results report"""
        report = f"""
            # Medical Coding AI Agent - Practice Test Results

            ## Test Summary
            - **Date**: {self.results['test_start_time'].strftime('%Y-%m-%d %H:%M:%S')}
            - **Total Questions**: {self.results['questions_answered']}
            - **Correct Answers**: {self.results['correct_answers']}
            - **Score**: {self.results['score_percentage']:.1f}%

            ## Agent Configuration
            - **Model**: {self.results['agent_config']['model']}
            - **Tools Used**: {', '.join(self.results['agent_config']['tools'])}
            - **Temperature**: {self.results['agent_config']['temperature']}

            ## Infrastructure Details
            - **Framework**: SmolaGents CodeAgent
            - **Vector Database**: ChromaDB with HuggingFace Embeddings
            - **Web Search**: DuckDuckGo
            - **Knowledge Base**: Embedded PDF documents

            ## Detailed Question Analysis
        """
        
        # Add detailed question-by-question results
        for result in self.results['detailed_results']:
            report += f"""
                ### Question {result['question_number']}
                **Question**: {result['question']}

                **Agent Answer**: {result['agent_answer']}
                **Correct Answer**: {result['correct_answer']}
                **Result**: {'✅ Correct' if result['is_correct'] else '❌ Incorrect'}
            """
        
        return report
    
    def save_results(self, output_path: str):
        """Save results to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_comprehensive_report())
        
        # Also save raw JSON data
        json_path = output_path.replace('.md', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str, ensure_ascii=False)