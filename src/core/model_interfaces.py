# src/core/model_interfaces.py
import requests
import json
import time
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from pydantic import BaseModel
import httpx
import asyncio
import os


class ModelConfig(BaseModel):
    """Configuration for model interfaces"""
    name: str
    provider: str  # ollama, openai, anthropic, huggingface, local
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: str
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 60


class GenerationResult(BaseModel):
    """Result from code generation"""
    code: str
    model_name: str
    provider: str
    generation_time: float
    token_count: Optional[int] = None
    error: Optional[str] = None


class ModelInterface(ABC):
    """Abstract base class for all model interfaces"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.name = config.name
        self.model_name = config.model_name
        self.provider = config.provider
    
    @abstractmethod
    def generate_code(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Generate code from a prompt"""
        pass
    
    @abstractmethod
    async def generate_code_async(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Async version of code generation"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the model is accessible"""
        pass


class VLLMInterface(ModelInterface):
    """Interface for vLLM servers (OpenAI-compatible)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:8000"
        
    def test_connection(self) -> bool:
        """Test vLLM server connection"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def generate_code(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Generate code using vLLM server (OpenAI-compatible API)"""
        start_time = time.time()
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": False
            }
            
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            generation_time = time.time() - start_time
            
            return GenerationResult(
                code=result["choices"][0]["message"]["content"],
                model_name=self.model_name,
                provider=self.provider,
                generation_time=generation_time,
                token_count=result.get("usage", {}).get("total_tokens")
            )
            
        except Exception as e:
            return GenerationResult(
                code="",
                model_name=self.model_name,
                provider=self.provider,
                generation_time=time.time() - start_time,
                error=str(e)
            )
    
    async def generate_code_async(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Async version of vLLM generation"""
        start_time = time.time()
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": False
            }
            
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                result = response.json()
            
            generation_time = time.time() - start_time
            
            return GenerationResult(
                code=result["choices"][0]["message"]["content"],
                model_name=self.model_name,
                provider=self.provider,
                generation_time=generation_time,
                token_count=result.get("usage", {}).get("total_tokens")
            )
            
        except Exception as e:
            return GenerationResult(
                code="",
                model_name=self.model_name,
                provider=self.provider,
                generation_time=time.time() - start_time,
                error=str(e)
            )


class CustomServerInterface(ModelInterface):
    """Interface for custom LLM servers with flexible API"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:8000"
        self.api_format = getattr(config, 'api_format', 'openai')  # openai, custom
        
    def test_connection(self) -> bool:
        """Test custom server connection"""
        try:
            # Try different endpoints based on API format
            test_endpoints = [
                f"{self.base_url}/v1/models",
                f"{self.base_url}/models",
                f"{self.base_url}/health",
                f"{self.base_url}/"
            ]
            
            for endpoint in test_endpoints:
                try:
                    response = requests.get(endpoint, timeout=5)
                    if response.status_code == 200:
                        return True
                except:
                    continue
            return False
        except Exception:
            return False
    
    def generate_code(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Generate code using custom server"""
        start_time = time.time()
        
        try:
            if self.api_format == 'openai':
                return self._generate_openai_format(prompt, system_prompt, start_time)
            else:
                return self._generate_custom_format(prompt, system_prompt, start_time)
                
        except Exception as e:
            return GenerationResult(
                code="",
                model_name=self.model_name,
                provider=self.provider,
                generation_time=time.time() - start_time,
                error=str(e)
            )
    
    def _generate_openai_format(self, prompt: str, system_prompt: Optional[str], start_time: float) -> GenerationResult:
        """Generate using OpenAI-compatible API"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.config.timeout
        )
        
        response.raise_for_status()
        result = response.json()
        
        return GenerationResult(
            code=result["choices"][0]["message"]["content"],
            model_name=self.model_name,
            provider=self.provider,
            generation_time=time.time() - start_time,
            token_count=result.get("usage", {}).get("total_tokens")
        )
    
    def _generate_custom_format(self, prompt: str, system_prompt: Optional[str], start_time: float) -> GenerationResult:
        """Generate using custom API format"""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        payload = {
            "prompt": full_prompt,
            "model": self.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        response = requests.post(
            f"{self.base_url}/generate",
            json=payload,
            headers=headers,
            timeout=self.config.timeout
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Handle different response formats
        code = result.get("text", result.get("response", result.get("output", "")))
        
        return GenerationResult(
            code=code,
            model_name=self.model_name,
            provider=self.provider,
            generation_time=time.time() - start_time
        )
    
    async def generate_code_async(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Async version - just calls sync version for simplicity"""
        return self.generate_code(prompt, system_prompt)


class OllamaInterface(ModelInterface):
    """Interface for Ollama local models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
        
    def test_connection(self) -> bool:
        """Test Ollama connection"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def generate_code(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Generate code using Ollama"""
        start_time = time.time()
        
        try:
            # Combine system and user prompts
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            generation_time = time.time() - start_time
            
            return GenerationResult(
                code=result.get("response", ""),
                model_name=self.model_name,
                provider=self.provider,
                generation_time=generation_time,
                token_count=result.get("eval_count")
            )
            
        except Exception as e:
            return GenerationResult(
                code="",
                model_name=self.model_name,
                provider=self.provider,
                generation_time=time.time() - start_time,
                error=str(e)
            )
    
    async def generate_code_async(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Async version of Ollama generation"""
        start_time = time.time()
        
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            }
            
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
            
            generation_time = time.time() - start_time
            
            return GenerationResult(
                code=result.get("response", ""),
                model_name=self.model_name,
                provider=self.provider,
                generation_time=generation_time,
                token_count=result.get("eval_count")
            )
            
        except Exception as e:
            return GenerationResult(
                code="",
                model_name=self.model_name,
                provider=self.provider,
                generation_time=time.time() - start_time,
                error=str(e)
            )


class OpenAIInterface(ModelInterface):
    """Interface for OpenAI models - Fixed for compatibility"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Initialize OpenAI client with proper error handling
        try:
            import openai
            # Use the newer OpenAI client initialization
            self.client = openai.OpenAI(
                api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
                timeout=config.timeout
            )
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        except Exception as e:
            # For testing purposes, allow creation even if API key is invalid
            import openai
            self.client = openai.OpenAI(
                api_key=config.api_key or "test-key",
                timeout=config.timeout
            )
        
    def test_connection(self) -> bool:
        """Test OpenAI connection"""
        try:
            # Try to list models to test connection
            self.client.models.list()
            return True
        except Exception:
            return False
    
    def generate_code(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Generate code using OpenAI"""
        start_time = time.time()
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            generation_time = time.time() - start_time
            
            return GenerationResult(
                code=response.choices[0].message.content,
                model_name=self.model_name,
                provider=self.provider,
                generation_time=generation_time,
                token_count=response.usage.total_tokens if response.usage else None
            )
            
        except Exception as e:
            return GenerationResult(
                code="",
                model_name=self.model_name,
                provider=self.provider,
                generation_time=time.time() - start_time,
                error=str(e)
            )
    
    async def generate_code_async(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Async version of OpenAI generation"""
        start_time = time.time()
        
        try:
            import openai
            async_client = openai.AsyncOpenAI(
                api_key=self.config.api_key or os.getenv("OPENAI_API_KEY")
            )
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            generation_time = time.time() - start_time
            
            return GenerationResult(
                code=response.choices[0].message.content,
                model_name=self.model_name,
                provider=self.provider,
                generation_time=generation_time,
                token_count=response.usage.total_tokens if response.usage else None
            )
            
        except Exception as e:
            return GenerationResult(
                code="",
                model_name=self.model_name,
                provider=self.provider,
                generation_time=time.time() - start_time,
                error=str(e)
            )


class AnthropicInterface(ModelInterface):
    """Interface for Anthropic models - Fixed for compatibility"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY")
            )
        except ImportError:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")
        except Exception:
            # For testing, allow creation with test key
            import anthropic
            self.client = anthropic.Anthropic(api_key=config.api_key or "test-key")
        
    def test_connection(self) -> bool:
        """Test Anthropic connection"""
        try:
            # Try a simple message to test connection
            self.client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except Exception:
            return False
    
    def generate_code(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Generate code using Anthropic"""
        start_time = time.time()
        
        try:
            kwargs = {
                "model": self.model_name,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            if system_prompt:
                kwargs["system"] = system_prompt
            
            response = self.client.messages.create(**kwargs)
            
            generation_time = time.time() - start_time
            
            return GenerationResult(
                code=response.content[0].text,
                model_name=self.model_name,
                provider=self.provider,
                generation_time=generation_time,
                token_count=response.usage.input_tokens + response.usage.output_tokens
            )
            
        except Exception as e:
            return GenerationResult(
                code="",
                model_name=self.model_name,
                provider=self.provider,
                generation_time=time.time() - start_time,
                error=str(e)
            )
    
    async def generate_code_async(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Async version of Anthropic generation"""
        start_time = time.time()
        
        try:
            import anthropic
            async_client = anthropic.AsyncAnthropic(
                api_key=self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
            )
            
            kwargs = {
                "model": self.model_name,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            if system_prompt:
                kwargs["system"] = system_prompt
            
            response = await async_client.messages.create(**kwargs)
            
            generation_time = time.time() - start_time
            
            return GenerationResult(
                code=response.content[0].text,
                model_name=self.model_name,
                provider=self.provider,
                generation_time=generation_time,
                token_count=response.usage.input_tokens + response.usage.output_tokens
            )
            
        except Exception as e:
            return GenerationResult(
                code="",
                model_name=self.model_name,
                provider=self.provider,
                generation_time=time.time() - start_time,
                error=str(e)
            )


class HuggingFaceInterface(ModelInterface):
    """Interface for HuggingFace models via API"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.base_url = config.base_url or "https://api-inference.huggingface.co/models"
        
    def test_connection(self) -> bool:
        """Test HuggingFace connection"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.post(
                f"{self.base_url}/{self.model_name}",
                headers=headers,
                json={"inputs": "test", "parameters": {"max_length": 10}},
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def generate_code(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Generate code using HuggingFace"""
        start_time = time.time()
        
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "inputs": full_prompt,
                "parameters": {
                    "max_length": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "return_full_text": False
                }
            }
            
            response = requests.post(
                f"{self.base_url}/{self.model_name}",
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            generation_time = time.time() - start_time
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
            else:
                generated_text = str(result)
            
            return GenerationResult(
                code=generated_text,
                model_name=self.model_name,
                provider=self.provider,
                generation_time=generation_time
            )
            
        except Exception as e:
            return GenerationResult(
                code="",
                model_name=self.model_name,
                provider=self.provider,
                generation_time=time.time() - start_time,
                error=str(e)
            )
    
    async def generate_code_async(self, prompt: str, system_prompt: Optional[str] = None) -> GenerationResult:
        """Async version of HuggingFace generation"""
        start_time = time.time()
        
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "inputs": full_prompt,
                "parameters": {
                    "max_length": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "return_full_text": False
                }
            }
            
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/{self.model_name}",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
            
            generation_time = time.time() - start_time
            
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
            else:
                generated_text = str(result)
            
            return GenerationResult(
                code=generated_text,
                model_name=self.model_name,
                provider=self.provider,
                generation_time=generation_time
            )
            
        except Exception as e:
            return GenerationResult(
                code="",
                model_name=self.model_name,
                provider=self.provider,
                generation_time=time.time() - start_time,
                error=str(e)
            )


class ModelFactory:
    """Factory for creating model interfaces"""
    
    @staticmethod
    def create_interface(config: ModelConfig) -> ModelInterface:
        """Create appropriate model interface based on provider"""
        provider = config.provider.lower()
        
        if provider == "ollama":
            return OllamaInterface(config)
        elif provider == "openai":
            return OpenAIInterface(config)
        elif provider == "anthropic":
            return AnthropicInterface(config)
        elif provider == "huggingface":
            return HuggingFaceInterface(config)
        elif provider == "vllm":
            return VLLMInterface(config)
        elif provider == "custom":
            return CustomServerInterface(config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def get_default_configs() -> List[ModelConfig]:
        """Get default model configurations"""
        return [
            # Ollama local models
            ModelConfig(
                name="CodeLlama 7B",
                provider="ollama",
                model_name="codellama:7b",
                base_url="http://localhost:11434"
            ),
            ModelConfig(
                name="DeepSeek Coder 6.7B",
                provider="ollama",
                model_name="deepseek-coder:6.7b",
                base_url="http://localhost:11434"
            ),
            ModelConfig(
                name="Qwen2.5 Coder 7B",
                provider="ollama",
                model_name="qwen2.5-coder:7b",
                base_url="http://localhost:11434"
            ),
            # vLLM servers (OpenAI-compatible)
            ModelConfig(
                name="vLLM Server (Local)",
                provider="vllm",
                model_name="your-model-name",
                base_url="http://localhost:8000"
            ),
            # Custom servers
            ModelConfig(
                name="Custom Server (OpenAI API)",
                provider="custom",
                model_name="your-model",
                base_url="http://localhost:8000"
            ),
            # API models (require API keys)
            ModelConfig(
                name="GPT-4 Turbo",
                provider="openai",
                model_name="gpt-4-turbo-preview",
                api_key="${OPENAI_API_KEY}"
            ),
            ModelConfig(
                name="Claude 3 Sonnet",
                provider="anthropic",
                model_name="claude-3-sonnet-20240229",
                api_key="${ANTHROPIC_API_KEY}"
            )
        ]


# Example usage and testing
if __name__ == "__main__":
    # Test Ollama connection
    config = ModelConfig(
        name="CodeLlama Test",
        provider="ollama",
        model_name="codellama:7b",
        base_url="http://localhost:11434"
    )
    
    interface = ModelFactory.create_interface(config)
    
    if interface.test_connection():
        print("✅ Model connection successful!")
        
        # Test code generation
        result = interface.generate_code(
            prompt="Write a Python function to calculate fibonacci numbers",
            system_prompt="You are an expert Python developer. Write clean, efficient code."
        )
        
        print(f"Generated code in {result.generation_time:.2f}s:")
        print(result.code)
    else:
        print("❌ Model connection failed!")