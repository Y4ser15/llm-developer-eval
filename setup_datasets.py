#!/usr/bin/env python3
"""
Setup Custom Datasets for Production
Creates domain-specific datasets that work immediately without external dependencies.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_custom_datasets():
    """Create custom dataset files for immediate use"""
    
    datasets_dir = Path("datasets")
    
    # Frontend dataset
    frontend_tasks = {
        "name": "Frontend Development Tasks",
        "description": "React components, UI logic, and frontend development",
        "tasks": [
            {
                "task_id": "frontend_001",
                "prompt": "Create a React component that displays a user profile card with name, email, avatar, and a follow button. Include hover effects and responsive design.",
                "solution": "// React UserProfile component with modern styling",
                "difficulty": "medium",
                "tags": ["react", "component", "ui", "responsive"],
                "entry_point": "UserProfileCard",
                "test": "// Test for UserProfileCard component"
            },
            {
                "task_id": "frontend_002", 
                "prompt": "Implement a shopping cart component that allows adding/removing items, calculates total price, and persists data in localStorage.",
                "solution": "// Shopping cart with state management",
                "difficulty": "hard",
                "tags": ["react", "state", "localStorage", "ecommerce"],
                "entry_point": "ShoppingCart",
                "test": "// Test for ShoppingCart component"
            },
            {
                "task_id": "frontend_003",
                "prompt": "Create a form validation component with real-time validation for email, password strength, and confirm password fields.",
                "solution": "// Form validation component",
                "difficulty": "medium", 
                "tags": ["forms", "validation", "regex", "ux"],
                "entry_point": "FormValidator",
                "test": "// Test for FormValidator component"
            },
            {
                "task_id": "frontend_004",
                "prompt": "Build a responsive navigation menu that works on both desktop and mobile, with dropdown menus and hamburger menu for mobile.",
                "solution": "// Responsive navigation component",
                "difficulty": "medium",
                "tags": ["navigation", "responsive", "mobile", "css"],
                "entry_point": "Navigation",
                "test": "// Test for Navigation component"
            },
            {
                "task_id": "frontend_005",
                "prompt": "Create a data table component with sorting, filtering, pagination, and row selection capabilities.",
                "solution": "// Data table with advanced features",
                "difficulty": "hard",
                "tags": ["table", "sorting", "filtering", "pagination"],
                "entry_point": "DataTable",
                "test": "// Test for DataTable component"
            }
        ]
    }
    
    # Backend dataset
    backend_tasks = {
        "name": "Backend Development Tasks", 
        "description": "API endpoints, database operations, and server logic",
        "tasks": [
            {
                "task_id": "backend_001",
                "prompt": "Create a FastAPI endpoint for user authentication that validates credentials, generates JWT tokens, and handles login/logout.",
                "solution": "# FastAPI authentication endpoint",
                "difficulty": "hard",
                "tags": ["fastapi", "auth", "jwt", "security"],
                "entry_point": "authenticate_user",
                "test": "# Test for authentication endpoint"
            },
            {
                "task_id": "backend_002",
                "prompt": "Implement a REST API for a blog system with CRUD operations for posts, including pagination and search functionality.",
                "solution": "# Blog API with CRUD operations",
                "difficulty": "medium",
                "tags": ["rest", "crud", "pagination", "search"],
                "entry_point": "blog_api",
                "test": "# Test for blog API"
            },
            {
                "task_id": "backend_003",
                "prompt": "Create a database model and API endpoint for a many-to-many relationship between users and roles with proper permissions.",
                "solution": "# User-Role relationship model",
                "difficulty": "hard",
                "tags": ["database", "relationships", "permissions", "orm"],
                "entry_point": "user_role_model",
                "test": "# Test for user-role model"
            },
            {
                "task_id": "backend_004",
                "prompt": "Design a caching system using Redis that stores frequently accessed data with automatic expiration and cache invalidation.",
                "solution": "# Redis caching system",
                "difficulty": "hard",
                "tags": ["redis", "caching", "performance", "optimization"],
                "entry_point": "cache_manager",
                "test": "# Test for cache manager"
            },
            {
                "task_id": "backend_005",
                "prompt": "Implement a file upload service that handles multiple file types, validates file sizes, and stores files securely.",
                "solution": "# File upload service",
                "difficulty": "medium",
                "tags": ["upload", "files", "validation", "security"],
                "entry_point": "file_upload_service",
                "test": "# Test for file upload service"
            }
        ]
    }
    
    # Testing dataset
    testing_tasks = {
        "name": "Testing and QA Tasks",
        "description": "Unit tests, integration tests, and test automation",
        "tasks": [
            {
                "task_id": "testing_001",
                "prompt": "Write comprehensive unit tests for a function that calculates compound interest with various edge cases and error conditions.",
                "solution": "# Comprehensive unit tests for compound interest",
                "difficulty": "medium",
                "tags": ["unittest", "edge-cases", "finance", "pytest"],
                "entry_point": "test_compound_interest",
                "test": "# Compound interest test cases"
            },
            {
                "task_id": "testing_002",
                "prompt": "Create integration tests for a REST API that test user registration, login, and protected endpoint access.",
                "solution": "# API integration tests",
                "difficulty": "hard",
                "tags": ["integration", "api", "auth", "testing"],
                "entry_point": "test_api_integration",
                "test": "# API integration test cases"
            },
            {
                "task_id": "testing_003",
                "prompt": "Implement mock objects and test a service that depends on external APIs, database, and file system operations.",
                "solution": "# Service testing with mocks",
                "difficulty": "hard",
                "tags": ["mocking", "dependencies", "isolation", "pytest"],
                "entry_point": "test_service_with_mocks",
                "test": "# Service mock test cases"
            },
            {
                "task_id": "testing_004",
                "prompt": "Design and implement end-to-end tests for a web application using Selenium, covering user workflows and error scenarios.",
                "solution": "# E2E tests with Selenium",
                "difficulty": "hard",
                "tags": ["e2e", "selenium", "automation", "workflows"],
                "entry_point": "test_e2e_workflows",
                "test": "# E2E test scenarios"
            },
            {
                "task_id": "testing_005",
                "prompt": "Create performance tests that measure API response times, database query performance, and identify bottlenecks.",
                "solution": "# Performance testing suite",
                "difficulty": "medium",
                "tags": ["performance", "benchmarking", "optimization", "profiling"],
                "entry_point": "test_performance",
                "test": "# Performance test cases"
            }
        ]
    }
    
    # Create dataset files
    for domain, data in [
        ("frontend", frontend_tasks),
        ("backend", backend_tasks), 
        ("testing", testing_tasks)
    ]:
        domain_dir = datasets_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_file = domain_dir / f"{domain}_dataset.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Created {domain} dataset: {dataset_file} ({len(data['tasks'])} tasks)")
    
    # Create README
    readme_content = """# Custom Domain Datasets

This directory contains custom datasets for domain-specific LLM evaluation.

## Datasets

- **frontend/**: React components, UI development, responsive design
- **backend/**: API development, database operations, server logic  
- **testing/**: Unit tests, integration tests, test automation

## Format

Each dataset file follows this structure:

```json
{
    "name": "Dataset Name",
    "description": "Dataset description", 
    "tasks": [
        {
            "task_id": "domain_001",
            "prompt": "Task description...",
            "solution": "Expected solution...",
            "difficulty": "easy|medium|hard",
            "tags": ["tag1", "tag2"],
            "entry_point": "function_name",
            "test": "Test code..."
        }
    ]
}
```

## Usage

These datasets are automatically loaded by the evaluation platform and provide
immediate evaluation capabilities without requiring external authentication.
"""
    
    readme_file = datasets_dir / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    logger.info(f"✅ Created README: {readme_file}")
    logger.info(f"🎉 Custom datasets setup complete! Total datasets: 3")


if __name__ == "__main__":
    setup_custom_datasets()
