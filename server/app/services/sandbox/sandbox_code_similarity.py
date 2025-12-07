#Service file for intelligent code review validation
#Uses OpenAI API to perform semantic code review against design specifications
import logging
import json
import os
from typing import Dict, List, Optional
from pathlib import Path

from openai import OpenAI


class CodeReviewValidator:
    #Intelligent code validator that uses LLM to perform semantic code review
    #Checks if generated code correctly implements design specifications
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        #Initialize code review validator with OpenAI API
        #api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
        #model: OpenAI model to use for reviews (gpt-4o-mini for speed, gpt-4o for depth)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        logging.info(f"Initialized CodeReviewValidator with model {model}")
    
    def review_code(
        self,
        code_content: str,
        design_spec: str,
        file_path: Optional[str] = None,
        execution_results: Optional[Dict] = None
    ) -> Dict:
        #Performs semantic code review to validate if code implements design correctly
        #code_content: The generated code to review
        #design_spec: The original design specification/requirements
        #file_path: Optional file path for context
        #execution_results: Optional test execution results from sandbox.test_file()
        #Returns dict with: approved, score, issues, suggestions, summary
        logging.info(f"Starting code review for {file_path or 'code snippet'}")
        
        #Build the review prompt
        prompt = self._build_review_prompt(code_content, design_spec, file_path, execution_results)
        
        try:
            #Call OpenAI API for code review
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,  #Lower temperature for more consistent reviews
                max_tokens=2000
            )
            
            #Parse the JSON response
            review_result = json.loads(response.choices[0].message.content)
            
            #Validate and normalize the response
            normalized = self._normalize_review_result(review_result)
            
            logging.info(
                f"Code review complete: approved={normalized['approved']}, "
                f"score={normalized['score']:.2f}, issues={len(normalized['issues'])}"
            )
            
            return normalized
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse review response as JSON: {e}")
            return self._create_error_result("Invalid JSON response from model")
        except Exception as e:
            logging.error(f"Code review failed: {e}")
            return self._create_error_result(str(e))
    
    def _get_system_prompt(self) -> str:
        #Returns the system prompt for code review
        return """You are an expert code reviewer. Your job is to validate if generated code correctly implements the given design specification.

Analyze the code for:
1. **Correctness**: Does it implement the specified functionality?
2. **Completeness**: Are all requirements addressed?
3. **Quality**: Is it well-structured, readable, and maintainable?
4. **Best Practices**: Does it follow language conventions and patterns?
5. **Edge Cases**: Does it handle errors and edge cases appropriately?

You must respond ONLY with valid JSON in this exact format:
{
  "approved": true/false,
  "score": 0.85,
  "issues": ["Critical issue 1", "Problem 2"],
  "suggestions": ["Suggestion 1", "Improvement 2"],
  "summary": "Brief summary of the review"
}

- approved: true if code meets requirements and is production-ready
- score: 0.0 to 1.0 rating (0.8+ is good, 0.9+ is excellent)
- issues: Critical problems that must be fixed (empty array if none)
- suggestions: Optional improvements (can be empty)
- summary: 1-2 sentence overview of the review"""
    
    def _build_review_prompt(
        self,
        code_content: str,
        design_spec: str,
        file_path: Optional[str],
        execution_results: Optional[Dict]
    ) -> str:
        #Builds the review prompt with all context
        prompt_parts = [
            "# DESIGN SPECIFICATION:",
            design_spec,
            "\n# GENERATED CODE:"
        ]
        
        #Add file path if provided
        if file_path:
            prompt_parts.append(f"File: {file_path}")
        
        #Add code content
        prompt_parts.extend([f"```\n{code_content}\n```"])
        
        #Add test execution results if available
        if execution_results:
            prompt_parts.append("\n# TEST EXECUTION RESULTS:")
            prompt_parts.append(json.dumps(execution_results, indent=2))
        
        prompt_parts.append("\nReview this code and return your analysis as JSON.")
        
        return "\n".join(prompt_parts)
    
    def _normalize_review_result(self, result: Dict) -> Dict:
        #Ensures review result has all required fields with correct types
        normalized = {
            'approved': bool(result.get('approved', False)),
            'score': float(result.get('score', 0.0)),
            'issues': result.get('issues', []),
            'suggestions': result.get('suggestions', []),
            'summary': result.get('summary', 'No summary provided'),
            'needs_revision': False
        }
        
        #Clamp score to 0-1 range
        normalized['score'] = max(0.0, min(1.0, normalized['score']))
        
        #Set needs_revision based on approval and issues
        normalized['needs_revision'] = not normalized['approved'] or len(normalized['issues']) > 0
        
        #Ensure lists are actually lists
        if not isinstance(normalized['issues'], list):
            normalized['issues'] = []
        if not isinstance(normalized['suggestions'], list):
            normalized['suggestions'] = []
        
        return normalized
    
    def _create_error_result(self, error_message: str) -> Dict:
        #Creates a standardized error result when review fails
        return {
            'approved': False,
            'score': 0.0,
            'issues': [f"Review failed: {error_message}"],
            'suggestions': [],
            'summary': f"Code review encountered an error: {error_message}",
            'needs_revision': True
        }
    
    def batch_review(self, files: List[Dict[str, str]], design_spec: str) -> Dict[str, Dict]:
        #Reviews multiple files against the same design specification
        #files: List of dicts with 'path' and 'content' keys
        #design_spec: The design specification for all files
        #Returns dict mapping file paths to review results
        results = {}
        
        for file_info in files:
            file_path = file_info.get('path', 'unknown')
            content = file_info.get('content', '')
            
            try:
                review = self.review_code(
                    code_content=content,
                    design_spec=design_spec,
                    file_path=file_path
                )
                results[file_path] = review
            except Exception as e:
                logging.error(f"Failed to review {file_path}: {e}")
                results[file_path] = self._create_error_result(str(e))
        
        return results
    
    def quick_check(self, code_content: str, design_spec: str) -> bool:
        #Quick approval check - returns True if code is approved, False otherwise
        #Useful for simple pass/fail validation without detailed feedback
        result = self.review_code(code_content, design_spec)
        return result['approved']