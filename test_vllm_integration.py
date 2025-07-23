#!/usr/bin/env python3
"""
Test script for vLLM and Custom Server integration
Verifies the new model interfaces work correctly
"""

import sys
import os
sys.path.append('.')

from src.core.model_interfaces import ModelConfig, ModelFactory, VLLMInterface, CustomServerInterface

def test_vllm_interface():
    """Test vLLM interface creation and configuration"""
    print("🧪 Testing vLLM Interface...")
    
    # Test vLLM configuration
    vllm_config = ModelConfig(
        name="Test vLLM Server",
        provider="vllm",
        model_name="test-model",
        base_url="http://localhost:8000",
        temperature=0.1,
        max_tokens=1000
    )
    
    # Create interface via factory
    try:
        vllm_interface = ModelFactory.create_interface(vllm_config)
        assert isinstance(vllm_interface, VLLMInterface), "Factory should create VLLMInterface"
        assert vllm_interface.base_url == "http://localhost:8000"
        assert vllm_interface.model_name == "test-model"
        print("✅ vLLM interface creation successful")
    except Exception as e:
        print(f"❌ vLLM interface creation failed: {e}")
        return False
    
    # Test connection (will fail without server, but should not crash)
    try:
        connection_result = vllm_interface.test_connection()
        print(f"🔗 vLLM connection test: {'✅ Connected' if connection_result else '⚠️  No server (expected)'}")
    except Exception as e:
        print(f"❌ vLLM connection test failed: {e}")
        return False
    
    return True

def test_custom_server_interface():
    """Test custom server interface creation and configuration"""
    print("\n🧪 Testing Custom Server Interface...")
    
    # Test OpenAI-compatible custom server
    custom_config = ModelConfig(
        name="Test Custom Server",
        provider="custom",
        model_name="custom-model",
        base_url="http://localhost:8001",
        api_key="test-key",
        temperature=0.2,
        max_tokens=2000
    )
    
    # Add api_format attribute for custom format
    custom_config.api_format = "openai"
    
    try:
        custom_interface = ModelFactory.create_interface(custom_config)
        assert isinstance(custom_interface, CustomServerInterface), "Factory should create CustomServerInterface"
        assert custom_interface.base_url == "http://localhost:8001"
        assert custom_interface.model_name == "custom-model"
        assert custom_interface.api_format == "openai"
        print("✅ Custom server interface creation successful")
    except Exception as e:
        print(f"❌ Custom server interface creation failed: {e}")
        return False
    
    # Test connection
    try:
        connection_result = custom_interface.test_connection()
        print(f"🔗 Custom server connection test: {'✅ Connected' if connection_result else '⚠️  No server (expected)'}")
    except Exception as e:
        print(f"❌ Custom server connection test failed: {e}")
        return False
    
    return True

def test_model_factory_integration():
    """Test ModelFactory integration with new providers"""
    print("\n🧪 Testing ModelFactory Integration...")
    
    # Test all supported providers
    test_configs = [
        ("ollama", "ollama"),
        ("openai", "openai"), 
        ("anthropic", "anthropic"),
        ("huggingface", "huggingface"),
        ("vllm", "vllm"),
        ("custom", "custom")
    ]
    
    for provider_name, provider_type in test_configs:
        try:
            config = ModelConfig(
                name=f"Test {provider_name}",
                provider=provider_type,
                model_name="test-model",
                base_url="http://localhost:8000"
            )
            
            interface = ModelFactory.create_interface(config)
            print(f"✅ {provider_name.capitalize()} provider: {type(interface).__name__}")
        except Exception as e:
            print(f"❌ {provider_name.capitalize()} provider failed: {e}")
            return False
    
    # Test unsupported provider
    try:
        bad_config = ModelConfig(
            name="Bad Provider",
            provider="unsupported",
            model_name="test"
        )
        ModelFactory.create_interface(bad_config)
        print("❌ Should have failed for unsupported provider")
        return False
    except ValueError:
        print("✅ Correctly rejected unsupported provider")
    
    return True

def test_default_configs():
    """Test default model configurations include new providers"""
    print("\n🧪 Testing Default Configurations...")
    
    try:
        default_configs = ModelFactory.get_default_configs()
        
        # Check for vLLM and custom servers in defaults
        vllm_found = any(config.provider == "vllm" for config in default_configs)
        custom_found = any(config.provider == "custom" for config in default_configs)
        
        print(f"📋 Total default configs: {len(default_configs)}")
        print(f"🔧 vLLM configs found: {'✅' if vllm_found else '❌'}")
        print(f"🔧 Custom configs found: {'✅' if custom_found else '❌'}")
        
        # Show all provider types
        providers = set(config.provider for config in default_configs)
        print(f"🎯 Available providers: {', '.join(sorted(providers))}")
        
        return vllm_found and custom_found
        
    except Exception as e:
        print(f"❌ Default configs test failed: {e}")
        return False

def test_code_generation_mock():
    """Test code generation with mock data (no actual server)"""
    print("\n🧪 Testing Code Generation (Mock)...")
    
    # Test vLLM generation (will fail gracefully without server)
    vllm_config = ModelConfig(
        name="Mock vLLM",
        provider="vllm", 
        model_name="mock-model",
        base_url="http://nonexistent:8000"
    )
    
    try:
        vllm_interface = ModelFactory.create_interface(vllm_config)
        result = vllm_interface.generate_code(
            prompt="Write a hello world function",
            system_prompt="You are a helpful coding assistant"
        )
        
        # Should have error but not crash
        assert result.error is not None, "Should have connection error"
        assert result.model_name == "mock-model"
        assert result.provider == "vllm"
        print("✅ vLLM graceful error handling")
        
    except Exception as e:
        print(f"❌ vLLM generation test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Starting vLLM and Custom Server Integration Tests\n")
    
    tests = [
        test_vllm_interface,
        test_custom_server_interface, 
        test_model_factory_integration,
        test_default_configs,
        test_code_generation_mock
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print("⚠️  Test failed!")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! vLLM and Custom Server integration is working correctly.")
        print("\n✨ New Features Available:")
        print("   • vLLM server support (OpenAI-compatible)")
        print("   • Custom server configuration")
        print("   • Dynamic server detection")
        print("   • Web UI for adding custom servers")
        print("   • Extensible model selection")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
