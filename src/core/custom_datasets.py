# src/core/custom_datasets.py
import json
import yaml
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from enum import Enum
import os


class TaskType(str, Enum):
    """Task types for evaluation"""
    FRONTEND = "frontend"
    BACKEND = "backend"
    TESTING = "testing"


class DifficultyLevel(str, Enum):
    """Difficulty levels for tasks"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestCase(BaseModel):
    """Individual test case for a task"""
    name: str
    description: str
    input_data: Optional[Dict[str, Any]] = None
    expected_output: Optional[Any] = None
    test_code: Optional[str] = None
    timeout: int = 30


class EvaluationCriteria(BaseModel):
    """Evaluation criteria for tasks"""
    functionality: float = 0.3  # Weight for basic functionality
    code_quality: float = 0.2   # Weight for code quality/style
    security: float = 0.15      # Weight for security considerations
    performance: float = 0.15   # Weight for performance
    accessibility: float = 0.1  # Weight for accessibility (frontend)
    error_handling: float = 0.1 # Weight for error handling


class Task(BaseModel):
    """Individual coding task"""
    task_id: str
    title: str
    description: str
    prompt: str
    task_type: TaskType
    difficulty: DifficultyLevel
    tags: List[str] = []
    system_prompt: Optional[str] = None
    test_cases: List[TestCase] = []
    evaluation_criteria: EvaluationCriteria = EvaluationCriteria()
    expected_technologies: List[str] = []  # e.g., ["React", "TypeScript"]
    time_limit: int = 300  # seconds
    context_files: List[str] = []  # Additional context files if needed


class Dataset(BaseModel):
    """Collection of tasks"""
    name: str
    description: str
    version: str
    task_type: TaskType
    tasks: List[Task] = []
    
    def add_task(self, task: Task):
        """Add a task to the dataset"""
        self.tasks.append(task)
    
    def get_tasks_by_difficulty(self, difficulty: DifficultyLevel) -> List[Task]:
        """Get tasks filtered by difficulty"""
        return [task for task in self.tasks if task.difficulty == difficulty]
    
    def get_tasks_by_tags(self, tags: List[str]) -> List[Task]:
        """Get tasks that contain any of the specified tags"""
        return [task for task in self.tasks if any(tag in task.tags for tag in tags)]


class FrontendDataset(Dataset):
    """Frontend-specific dataset"""
    
    def __init__(self):
        super().__init__(
            name="Frontend Development Tasks",
            description="Comprehensive frontend development evaluation tasks",
            version="1.0.0",
            task_type=TaskType.FRONTEND
        )
        self._initialize_tasks()
    
    def _initialize_tasks(self):
        """Initialize frontend tasks"""
        
        # React Component Tasks
        react_component_easy = Task(
            task_id="frontend_react_01",
            title="User Profile Card Component",
            description="Create a responsive React component for displaying user profile information",
            prompt="""Create a React component called 'UserProfileCard' that displays user information.

Requirements:
- Accept props: user (object with name, email, avatar, role)
- Display user avatar, name, email, and role
- Make it responsive (works on mobile and desktop)
- Include hover effects
- Use modern CSS (flexbox/grid)
- Handle missing avatar gracefully

Example usage:
```jsx
<UserProfileCard 
  user={{
    name: "John Doe",
    email: "john@example.com", 
    avatar: "https://example.com/avatar.jpg",
    role: "Developer"
  }}
/>
```""",
            task_type=TaskType.FRONTEND,
            difficulty=DifficultyLevel.EASY,
            tags=["react", "component", "css", "responsive"],
            system_prompt="You are an expert React developer. Write clean, modern React components with proper styling.",
            expected_technologies=["React", "CSS", "JSX"],
            test_cases=[
                TestCase(
                    name="renders_with_valid_props",
                    description="Component renders correctly with valid user props",
                    test_code="""
import { render, screen } from '@testing-library/react';
import UserProfileCard from './UserProfileCard';

test('renders user profile card with valid props', () => {
  const user = {
    name: "John Doe",
    email: "john@example.com",
    avatar: "https://example.com/avatar.jpg",
    role: "Developer"
  };
  
  render(<UserProfileCard user={user} />);
  
  expect(screen.getByText('John Doe')).toBeInTheDocument();
  expect(screen.getByText('john@example.com')).toBeInTheDocument();
  expect(screen.getByText('Developer')).toBeInTheDocument();
});
"""
                ),
                TestCase(
                    name="handles_missing_avatar",
                    description="Component handles missing avatar gracefully",
                    test_code="""
test('handles missing avatar gracefully', () => {
  const user = {
    name: "Jane Doe",
    email: "jane@example.com",
    role: "Designer"
  };
  
  render(<UserProfileCard user={user} />);
  expect(screen.getByText('Jane Doe')).toBeInTheDocument();
});
"""
                )
            ]
        )
        
        # Interactive Component Task
        react_interactive_medium = Task(
            task_id="frontend_react_02",
            title="Searchable Dropdown Component",
            description="Create an accessible searchable dropdown with keyboard navigation",
            prompt="""Create a React component called 'SearchableDropdown' with the following features:

Requirements:
- Search functionality (filter options as user types)
- Keyboard navigation (arrow keys, enter, escape)
- Click outside to close
- Accessibility (ARIA labels, roles, keyboard support)
- Custom styling
- Support for custom option rendering
- Loading state support

Props interface:
```typescript
interface Option {
  value: string;
  label: string;
  disabled?: boolean;
}

interface SearchableDropdownProps {
  options: Option[];
  value?: string;
  placeholder?: string;
  onChange: (value: string) => void;
  loading?: boolean;
  disabled?: boolean;
}
```

The component should be fully accessible and work well with screen readers.""",
            task_type=TaskType.FRONTEND,
            difficulty=DifficultyLevel.MEDIUM,
            tags=["react", "accessibility", "keyboard-navigation", "typescript"],
            system_prompt="You are an expert React developer focused on accessibility and user experience.",
            expected_technologies=["React", "TypeScript", "CSS", "ARIA"],
            test_cases=[
                TestCase(
                    name="filters_options_on_search",
                    description="Options are filtered when user types in search",
                    test_code="""
import { render, screen, fireEvent } from '@testing-library/react';
import SearchableDropdown from './SearchableDropdown';

test('filters options when user types', () => {
  const options = [
    { value: 'apple', label: 'Apple' },
    { value: 'banana', label: 'Banana' },
    { value: 'orange', label: 'Orange' }
  ];
  
  render(<SearchableDropdown options={options} onChange={() => {}} />);
  
  const input = screen.getByRole('textbox');
  fireEvent.change(input, { target: { value: 'app' } });
  
  expect(screen.getByText('Apple')).toBeInTheDocument();
  expect(screen.queryByText('Banana')).not.toBeInTheDocument();
});
"""
                )
            ]
        )
        
        # Complex Frontend Task
        frontend_complex_hard = Task(
            task_id="frontend_complex_01",
            title="Dashboard with Real-time Data",
            description="Create a responsive dashboard with real-time data visualization",
            prompt="""Build a React dashboard component with the following features:

Requirements:
- Multiple chart types (line, bar, pie)
- Real-time data updates (WebSocket simulation)
- Responsive grid layout
- Dark/light theme toggle
- Data filtering and date range selection
- Export functionality (CSV/PDF)
- Loading states and error handling
- Accessibility compliance

Components needed:
1. DashboardLayout - Main layout component
2. ChartWidget - Reusable chart component
3. FilterPanel - Data filtering controls
4. ThemeToggle - Theme switching
5. ExportButton - Data export functionality

Use modern React patterns (hooks, context) and ensure excellent UX.""",
            task_type=TaskType.FRONTEND,
            difficulty=DifficultyLevel.HARD,
            tags=["react", "dashboard", "charts", "websocket", "responsive"],
            expected_technologies=["React", "TypeScript", "Chart.js", "CSS Grid"],
            time_limit=600
        )
        
        self.add_task(react_component_easy)
        self.add_task(react_interactive_medium)
        self.add_task(frontend_complex_hard)


class BackendDataset(Dataset):
    """Backend-specific dataset"""
    
    def __init__(self):
        super().__init__(
            name="Backend Development Tasks",
            description="Comprehensive backend development evaluation tasks",
            version="1.0.0",
            task_type=TaskType.BACKEND
        )
        self._initialize_tasks()
    
    def _initialize_tasks(self):
        """Initialize backend tasks"""
        
        # REST API Task
        api_easy = Task(
            task_id="backend_api_01",
            title="User Management REST API",
            description="Create a RESTful API for user management",
            prompt="""Create a REST API for user management using Python (Flask/FastAPI).

Requirements:
- CRUD operations for users (Create, Read, Update, Delete)
- Input validation and error handling
- JSON responses with proper HTTP status codes
- Basic authentication/authorization
- Database integration (SQLite/PostgreSQL)
- API documentation (OpenAPI/Swagger)

Endpoints needed:
- POST /users - Create new user
- GET /users - List all users (with pagination)
- GET /users/{id} - Get specific user
- PUT /users/{id} - Update user
- DELETE /users/{id} - Delete user

User model should include: id, username, email, created_at, updated_at""",
            task_type=TaskType.BACKEND,
            difficulty=DifficultyLevel.EASY,
            tags=["api", "rest", "crud", "database"],
            system_prompt="You are an expert backend developer. Write clean, secure, and well-documented APIs.",
            expected_technologies=["Python", "FastAPI", "SQLAlchemy", "Pydantic"],
            test_cases=[
                TestCase(
                    name="create_user_success",
                    description="Successfully create a new user",
                    test_code="""
def test_create_user():
    response = client.post("/users", json={
        "username": "testuser",
        "email": "test@example.com"
    })
    assert response.status_code == 201
    assert response.json()["username"] == "testuser"
"""
                )
            ]
        )
        
        # Database Operations Task
        database_medium = Task(
            task_id="backend_db_01",
            title="Advanced Database Operations",
            description="Implement complex database operations with transactions",
            prompt="""Implement a database service class for an e-commerce system.

Requirements:
- Transaction handling for order processing
- Complex queries with joins and aggregations
- Database connection pooling
- Error handling and rollback mechanisms
- Performance optimization (indexing, query optimization)
- Data integrity constraints

Implement these operations:
1. Process order (deduct inventory, create order, payment processing)
2. Generate sales reports with aggregations
3. Handle concurrent inventory updates
4. Implement efficient product search with filters
5. Manage user preferences and recommendations

Use proper database design principles and handle edge cases.""",
            task_type=TaskType.BACKEND,
            difficulty=DifficultyLevel.MEDIUM,
            tags=["database", "transactions", "sql", "performance"],
            expected_technologies=["Python", "SQLAlchemy", "PostgreSQL"],
            time_limit=450
        )
        
        # Microservices Task
        microservices_hard = Task(
            task_id="backend_micro_01",
            title="Microservices Architecture",
            description="Design and implement a microservices system",
            prompt="""Design a microservices architecture for an e-commerce platform.

Services to implement:
1. User Service - Authentication and user management
2. Product Service - Product catalog and inventory
3. Order Service - Order processing and management
4. Payment Service - Payment processing
5. Notification Service - Email/SMS notifications

Requirements:
- Service-to-service communication (REST/gRPC)
- Message queues for async processing
- Centralized logging and monitoring
- Circuit breaker pattern
- API Gateway for routing
- Database per service pattern
- Health checks and service discovery
- Docker containerization

Implement at least 2 services with proper communication patterns.""",
            task_type=TaskType.BACKEND,
            difficulty=DifficultyLevel.HARD,
            tags=["microservices", "architecture", "docker", "messaging"],
            expected_technologies=["Python", "Docker", "Redis", "RabbitMQ"],
            time_limit=900
        )
        
        self.add_task(api_easy)
        self.add_task(database_medium)
        self.add_task(microservices_hard)


class TestingDataset(Dataset):
    """Testing-specific dataset"""
    
    def __init__(self):
        super().__init__(
            name="Test Generation Tasks",
            description="Comprehensive test generation and testing strategy tasks",
            version="1.0.0",
            task_type=TaskType.TESTING
        )
        self._initialize_tasks()
    
    def _initialize_tasks(self):
        """Initialize testing tasks"""
        
        # Unit Testing Task
        unit_testing_easy = Task(
            task_id="testing_unit_01",
            title="Comprehensive Unit Tests",
            description="Generate comprehensive unit tests for a given function",
            prompt="""Generate comprehensive unit tests for the following Python function:

```python
def calculate_discount(price, discount_percent, customer_type='regular'):
    \"\"\"
    Calculate discounted price based on discount percentage and customer type.
    
    Args:
        price: Original price (must be positive)
        discount_percent: Discount percentage (0-100)
        customer_type: 'regular', 'premium', or 'vip'
    
    Returns:
        Final price after applying discount and customer-specific bonuses
    \"\"\"
    if price <= 0:
        raise ValueError("Price must be positive")
    
    if not 0 <= discount_percent <= 100:
        raise ValueError("Discount must be between 0 and 100")
    
    # Apply base discount
    discounted_price = price * (1 - discount_percent / 100)
    
    # Apply customer-specific bonuses
    if customer_type == 'premium':
        discounted_price *= 0.95  # Additional 5% off
    elif customer_type == 'vip':
        discounted_price *= 0.9   # Additional 10% off
    
    return round(discounted_price, 2)
```

Create tests that cover:
- All valid input combinations
- Edge cases and boundary conditions
- Error cases and exception handling
- Different customer types
- Proper assertions and test structure""",
            task_type=TaskType.TESTING,
            difficulty=DifficultyLevel.EASY,
            tags=["unit-testing", "pytest", "edge-cases"],
            system_prompt="You are an expert in software testing. Write thorough, well-structured tests.",
            expected_technologies=["Python", "pytest"],
            test_cases=[
                TestCase(
                    name="test_coverage",
                    description="Tests should achieve high code coverage",
                    test_code="# Coverage will be measured automatically"
                )
            ]
        )
        
        # Integration Testing Task
        integration_medium = Task(
            task_id="testing_integration_01",
            title="API Integration Tests",
            description="Create integration tests for a REST API",
            prompt="""Create comprehensive integration tests for a user management API.

API Endpoints:
- POST /users - Create user
- GET /users - List users
- GET /users/{id} - Get user by ID
- PUT /users/{id} - Update user
- DELETE /users/{id} - Delete user

Requirements:
- Test complete user workflows
- Database integration testing
- Authentication and authorization tests
- Input validation testing
- Error handling verification
- Performance and load testing
- Test data setup and teardown
- Mock external services

Use pytest and appropriate testing libraries. Include:
1. Happy path scenarios
2. Error scenarios
3. Security testing
4. Performance benchmarks
5. Data consistency checks""",
            task_type=TaskType.TESTING,
            difficulty=DifficultyLevel.MEDIUM,
            tags=["integration-testing", "api-testing", "database"],
            expected_technologies=["Python", "pytest", "requests"],
            time_limit=400
        )
        
        # E2E Testing Task
        e2e_hard = Task(
            task_id="testing_e2e_01",
            title="End-to-End Test Automation",
            description="Create comprehensive E2E test suite",
            prompt="""Design and implement an end-to-end test automation framework.

Application: E-commerce web application

Test scenarios to automate:
1. User registration and login
2. Product browsing and search
3. Shopping cart operations
4. Checkout process
5. Order management
6. User profile management

Requirements:
- Use Selenium WebDriver or Playwright
- Page Object Model pattern
- Cross-browser testing (Chrome, Firefox)
- Mobile responsive testing
- Test data management
- Screenshot capture on failures
- Parallel test execution
- CI/CD integration ready
- Comprehensive reporting

Include:
- Setup and teardown procedures
- Reusable components and utilities
- Configuration management
- Error handling and retry mechanisms""",
            task_type=TaskType.TESTING,
            difficulty=DifficultyLevel.HARD,
            tags=["e2e-testing", "selenium", "automation", "ci-cd"],
            expected_technologies=["Python", "Selenium", "Playwright"],
            time_limit=600
        )
        
        self.add_task(unit_testing_easy)
        self.add_task(integration_medium)
        self.add_task(e2e_hard)


class DatasetManager:
    """Manager for loading and managing datasets"""
    
    def __init__(self, datasets_dir: str = "datasets"):
        self.datasets_dir = datasets_dir
        self.datasets: Dict[TaskType, Dataset] = {}
        self._load_datasets()
    
    def _load_datasets(self):
        """Load all available datasets"""
        self.datasets[TaskType.FRONTEND] = FrontendDataset()
        self.datasets[TaskType.BACKEND] = BackendDataset()
        self.datasets[TaskType.TESTING] = TestingDataset()
    
    def get_dataset(self, task_type: TaskType) -> Dataset:
        """Get dataset by task type"""
        return self.datasets.get(task_type)
    
    def get_all_tasks(self) -> List[Task]:
        """Get all tasks from all datasets"""
        all_tasks = []
        for dataset in self.datasets.values():
            all_tasks.extend(dataset.tasks)
        return all_tasks
    
    def get_tasks_by_type(self, task_type: TaskType) -> List[Task]:
        """Get tasks filtered by type"""
        dataset = self.get_dataset(task_type)
        return dataset.tasks if dataset else []
    
    def get_tasks_by_difficulty(self, difficulty: DifficultyLevel) -> List[Task]:
        """Get tasks filtered by difficulty across all datasets"""
        all_tasks = self.get_all_tasks()
        return [task for task in all_tasks if task.difficulty == difficulty]
    
    def save_dataset(self, dataset: Dataset, filename: str):
        """Save dataset to file"""
        filepath = os.path.join(self.datasets_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(dataset.dict(), f, indent=2)
    
    def load_dataset_from_file(self, filepath: str) -> Dataset:
        """Load dataset from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            return Dataset(**data)
    
    def export_tasks_summary(self) -> Dict[str, Any]:
        """Export summary of all tasks"""
        summary = {
            "total_tasks": len(self.get_all_tasks()),
            "by_type": {},
            "by_difficulty": {},
            "datasets": []
        }
        
        # Count by type
        for task_type in TaskType:
            tasks = self.get_tasks_by_type(task_type)
            summary["by_type"][task_type.value] = len(tasks)
        
        # Count by difficulty
        for difficulty in DifficultyLevel:
            tasks = self.get_tasks_by_difficulty(difficulty)
            summary["by_difficulty"][difficulty.value] = len(tasks)
        
        # Dataset info
        for task_type, dataset in self.datasets.items():
            summary["datasets"].append({
                "name": dataset.name,
                "type": task_type.value,
                "version": dataset.version,
                "task_count": len(dataset.tasks)
            })
        
        return summary


# Example usage
if __name__ == "__main__":
    # Initialize dataset manager
    manager = DatasetManager()
    
    # Print summary
    summary = manager.export_tasks_summary()
    print("Dataset Summary:")
    print(f"Total tasks: {summary['total_tasks']}")
    print(f"By type: {summary['by_type']}")
    print(f"By difficulty: {summary['by_difficulty']}")
    
    # Get frontend tasks
    frontend_tasks = manager.get_tasks_by_type(TaskType.FRONTEND)
    print(f"\nFrontend tasks: {len(frontend_tasks)}")
    for task in frontend_tasks:
        print(f"- {task.title} ({task.difficulty.value})")
    
    # Get easy tasks across all types
    easy_tasks = manager.get_tasks_by_difficulty(DifficultyLevel.EASY)
    print(f"\nEasy tasks across all types: {len(easy_tasks)}")
    for task in easy_tasks:
        print(f"- {task.title} ({task.task_type.value})")