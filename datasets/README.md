# Custom Datasets Documentation

This folder contains custom evaluation datasets for the LLM Coding Evaluation Platform.

## Purpose
- Store custom coding tasks specific to your evaluation needs
- Extend the platform beyond BigCodeBench and HumanEval
- Add domain-specific tasks for specialized evaluation

## Structure
```
datasets/
├── frontend/        # Frontend development tasks
├── backend/         # Backend development tasks  
├── testing/         # Testing and QA tasks
└── README.md        # This file
```

## Adding Custom Tasks

### Frontend Tasks (React, Vue, Angular)
Create JSON files in `datasets/frontend/` with this format:
```json
{
  "task_id": "custom_frontend_001",
  "title": "Interactive Todo List Component",
  "description": "Create a React component for managing todos",
  "prompt": "Build a TodoList component with add, delete, and toggle functionality...",
  "difficulty": "medium",
  "tags": ["react", "hooks", "state-management"],
  "expected_technologies": ["React", "JavaScript", "CSS"]
}
```

### Backend Tasks (APIs, Databases)
Create JSON files in `datasets/backend/` with this format:
```json
{
  "task_id": "custom_backend_001", 
  "title": "User Authentication API",
  "description": "Build secure user authentication endpoints",
  "prompt": "Create FastAPI endpoints for user registration, login, and JWT validation...",
  "difficulty": "medium",
  "tags": ["api", "authentication", "security"],
  "expected_technologies": ["Python", "FastAPI", "JWT"]
}
```

### Testing Tasks (Unit, Integration, E2E)
Create JSON files in `datasets/testing/` with this format:
```json
{
  "task_id": "custom_testing_001",
  "title": "API Integration Test Suite", 
  "description": "Write comprehensive API tests",
  "prompt": "Create integration tests for a REST API with authentication...",
  "difficulty": "medium",
  "tags": ["testing", "api", "integration"],
  "expected_technologies": ["pytest", "requests", "fixtures"]
}
```

## Current Status
The platform uses 9 built-in tasks from `src/core/custom_datasets.py`:
- 3 Frontend tasks (React components, UI interactions)
- 3 Backend tasks (APIs, databases, microservices)  
- 3 Testing tasks (Unit, integration, E2E)

## Loading Custom Tasks
Custom tasks from this folder will be automatically loaded and integrated with the built-in tasks for comprehensive evaluation.

## Examples
See `examples/` folder for sample custom task definitions you can use as templates.
