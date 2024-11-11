import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfFolder, login
import uuid
import random
from typing import Dict, List, Tuple
import json
from datetime import datetime
import time
import subprocess
import sys
import gc

def setup_huggingface():
    """Setup Hugging Face authentication and requirements."""
    token = "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"
    
    try:
        os.environ["HUGGINGFACE_TOKEN"] = token
        login(token=token)
        HfFolder.save_token(token)
        
        required_packages = ["transformers", "torch", "accelerate"]
        for package in required_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

class HaystackGenerator:
    """Generate haystacks with token limit control."""
    
    def __init__(self, size: int, tokenizer):
        self.size = min(size, 100)  # UUID 쌍의 수를 100개로 제한
        self.tokenizer = tokenizer
        self.max_tokens = 1024  # 토큰 수 제한 (프롬프트 템플릿 공간 고려)
    
    def generate(self) -> Dict[str, str]:
        """Generate a haystack with UUID pairs while monitoring token count."""
        haystack = {}
        base_prompt = '<s>[INST] Find the value for key "KEY" in:\n'  # 기본 프롬프트 템플릿
        
        for _ in range(self.size):
            # 현재 haystack으로 프롬프트 생성
            current_json = json.dumps(haystack)
            test_prompt = base_prompt + current_json + "\n[/INST]"
            
            # 토큰 수 확인
            tokens = self.tokenizer(test_prompt, return_tensors="pt")
            current_tokens = len(tokens['input_ids'][0])
            
            if current_tokens > self.max_tokens:
                print(f"Token limit reached at {len(haystack)} pairs")
                break
                
            # 새로운 UUID 쌍 추가
            key = str(uuid.uuid4())
            value = str(uuid.uuid4())
            haystack[key] = value
        
        if not haystack:
            raise ValueError("Could not generate haystack within token limits")
            
        return haystack

    def create_thread(self, haystack: Dict[str, str], thread_length: int, 
                     direction: str = "forward") -> Tuple[str, List[str]]:
        """Create a thread with UUIDs."""
        keys = list(haystack.keys())
        thread_length = min(thread_length, len(keys), 5)  # 스레드 길이 제한
        
        if direction == "forward":
            start_idx = random.randint(0, len(keys) - thread_length)
            indices = range(start_idx, start_idx + thread_length)
        else:
            start_idx = random.randint(thread_length - 1, len(keys) - 1)
            indices = range(start_idx, start_idx - thread_length, -1)
            
        thread_keys = [keys[i] for i in indices]
        thread_values = [haystack[thread_keys[i]] for i in range(thread_length)]
        
        return thread_keys[0], thread_values

class NeedleThreadingTest:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B"):
        print(f"Loading model {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = 2048
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        self.model.config.pad_token_id = self.model.config.eos_token_id
        
        # HaystackGenerator 초기화 시 tokenizer 전달
        self.haystack_generator = HaystackGenerator(
            size=50, 
            tokenizer=self.tokenizer
        )
    def __resize_generator(self, size: int):
        """새로운 크기로 generator 재설정"""
        self.haystack_generator = HaystackGenerator(
            size=size,
            tokenizer=self.tokenizer
        )
    def cleanup(self):
        """Clean up GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    def format_prompt(self, task_type: str, haystack: Dict[str, str], 
                    query_params: Dict) -> str:
        """개선된 프롬프트 형식."""
        json_str = json.dumps(haystack, indent=2)  # 가독성을 위해 들여쓰기 추가
        
        if task_type == "single_needle":
            prompt = f"""<s>[INST]
    Given this key-value store:
    {json_str}

    Return the exact value that corresponds to the key: "{query_params['key']}"
    Return only the UUID value, nothing else.
    [/INST]"""

        elif task_type == "threading":
            prompt = f"""<s>[INST]
    Given this key-value store:
    {json_str}

    Starting with key "{query_params['start_key']}", perform these steps:
    1. Find the value for this key
    2. Check if this value exists as a key
    3. If it exists as a key, get its value and go to step 2
    4. If it doesn't exist as a key, return that final value

    Return only the final UUID value, nothing else.
    [/INST]"""
        
        tokens = self.tokenizer(prompt, return_tensors="pt")
        if len(tokens['input_ids'][0]) > self.tokenizer.model_max_length:
            raise ValueError(f"Prompt too long: {len(tokens['input_ids'][0])} tokens")
            
        return prompt
    
    def generate_response(self, prompt: str) -> str:
        """개선된 응답 생성 및 처리."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=0.0,
                    num_beams=1,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 응답 디코딩
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 프롬프트 제거
            response = response[len(prompt):].strip()
            
            # UUID 패턴 매칭
            import re
            
            # JSON 형식의 응답에서 value만 추출 시도
            json_pattern = r'"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"'
            json_matches = re.findall(json_pattern, response.lower())
            if json_matches:
                # 마지막 UUID 반환 (JSON 출력의 경우)
                return json_matches[-1]
                
            # 일반 UUID 패턴 매칭
            uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
            matches = re.findall(uuid_pattern, response.lower())
            if matches:
                # 첫 번째 UUID 반환
                return matches[0]
                
            # 다른 형식의 매칭도 시도하지만 UUID 형식만 반환
            quote_pattern = r'"([^"]+)"'
            matches = re.findall(quote_pattern, response)
            if matches:
                for match in matches:
                    if re.match(uuid_pattern, match.lower()):
                        return match
            
            return ""
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return ""
    
    def validate_response(self, response: str, expected: str) -> bool:
        """응답 검증."""
        response = response.strip().lower()
        expected = expected.strip().lower()
        return response == expected
    
    def run_tests(self, num_trials: int = 3):
        """더 작은 크기로 시작하여 점진적으로 증가."""
        results = {
            "model": "Llama-3.2-1B",
            "timestamp": datetime.now().isoformat(),
            "tests": []
        }
        
        # 더 작은 크기부터 시작
        initial_sizes = [5, 10, 15, 20, 25]
        
        for size in initial_sizes:
            print(f"\n{'='*50}")
            print(f"Testing with {size} UUID pairs")
            print(f"{'='*50}")
            
            # generator 크기 조정
            self.__resize_generator(size)
            
            for trial in range(num_trials):
                print(f"\n{'-'*30}")
                print(f"Trial {trial + 1}/{num_trials}")
                print(f"{'-'*30}")
                
                torch.cuda.empty_cache()
                
                try:
                    # Single needle test
                    haystack = self.haystack_generator.generate()
                    key = random.choice(list(haystack.keys()))
                    expected_value = haystack[key]
                    
                    print("\nSingle Needle Test:")
                    print(f"Key: {key}")
                    print(f"Expected: {expected_value}")
                    
                    prompt = self.format_prompt("single_needle", haystack, {"key": key})
                    start_time = time.time()
                    response = self.generate_response(prompt)
                    duration = time.time() - start_time
                    
                    is_correct = self.validate_response(response, expected_value)
                    print(f"Got: {response}")
                    print(f"Correct: {is_correct}")
                    
                    results["tests"].append({
                        "task": "single_needle",
                        "size": size,
                        "trial": trial,
                        "duration": duration,
                        "is_correct": is_correct,
                        "expected": expected_value,
                        "received": response
                    })
                    
                    # Threading test
                    print("\nThreading Test:")
                    start_key, thread_values = self.haystack_generator.create_thread(
                        haystack,
                        thread_length=2,
                        direction="forward"
                    )
                    
                    print(f"Thread start: {start_key}")
                    print(f"Expected final: {thread_values[-1]}")
                    
                    prompt = self.format_prompt("threading", haystack, {"start_key": start_key})
                    start_time = time.time()
                    response = self.generate_response(prompt)
                    duration = time.time() - start_time
                    
                    is_correct = self.validate_response(response, thread_values[-1])
                    print(f"Got: {response}")
                    print(f"Correct: {is_correct}")
                    
                    results["tests"].append({
                        "task": "threading",
                        "size": size,
                        "trial": trial,
                        "duration": duration,
                        "is_correct": is_correct,
                        "expected": thread_values[-1],
                        "received": response
                    })
                    
                except Exception as e:
                    print(f"Error in trial: {str(e)}")
                    print(f"Error type: {type(e)}")
                    import traceback
                    print(traceback.format_exc())
                    continue
                
                torch.cuda.empty_cache()
                gc.collect()
        
        return results

def main():
    if not setup_huggingface():
        return
    
    tester = NeedleThreadingTest()
    
    try:
        results = tester.run_tests()
        
        output_file = f"results_llama_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\nTest Summary:")
        for task in ["single_needle", "threading"]:
            task_results = [t for t in results["tests"] if t["task"] == task]
            correct = sum(1 for t in task_results if t["is_correct"])
            total = len(task_results)
            if total > 0:
                print(f"{task} accuracy: {correct/total:.2%}")
    
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()