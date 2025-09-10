# Design Document

## Overview

This design document outlines a comprehensive refactoring approach to improve the code quality of the Chicken Farm Harvest Plan Optimization System. The current codebase suffers from several architectural and code quality issues that need systematic resolution while maintaining all existing functionality.

The refactoring will follow a phased approach, breaking down large monolithic functions, implementing proper error handling, adding comprehensive testing, and establishing clean architecture patterns. The design emphasizes maintainability, testability, and extensibility while preserving the complex optimization logic that is core to the system's value.

## Architecture

### Current Architecture Issues

1. **Monolithic Functions**: The main.py file contains 1826+ lines with massive functions handling multiple responsibilities
2. **Tight Coupling**: Direct dependencies between high-level orchestration and low-level implementation details
3. **Inconsistent Error Handling**: Ad-hoc error handling without consistent patterns or logging
4. **Missing Abstractions**: Concrete implementations mixed with business logic
5. **Poor Separation of Concerns**: Data processing, optimization, and export logic intermingled

### Target Architecture

The refactored system will implement a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  (Service Classes, Configuration Management)                 │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Domain Layer                              │
│  (Business Logic, Domain Models, Interfaces)                │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                 Infrastructure Layer                         │
│  (Data Access, File I/O, External Services)                 │
└─────────────────────────────────────────────────────────────┘
```

### Key Architectural Principles

1. **Dependency Inversion**: High-level modules depend on abstractions, not concretions
2. **Single Responsibility**: Each class and function has one clear purpose
3. **Open/Closed**: System is open for extension, closed for modification
4. **Interface Segregation**: Clients depend only on interfaces they use
5. **Liskov Substitution**: Derived classes are substitutable for base classes

## Components and Interfaces

### Core Interfaces

```python
# Data Processing Interfaces
class DataLoader(Protocol):
    def load_growth_data(self, file_path: str) -> pd.DataFrame: ...
    def load_market_data(self, file_path: str) -> pd.DataFrame: ...
    def validate_data_schema(self, df: pd.DataFrame) -> ValidationResult: ...

class DataProcessor(Protocol):
    def preprocess_data(self, df: pd.DataFrame, config: ProcessingConfig) -> ProcessedData: ...
    def calculate_mortality(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def apply_weight_filters(self, df: pd.DataFrame, weight_range: WeightRange) -> pd.DataFrame: ...

# Optimization Interfaces
class OptimizationModelBuilder(Protocol):
    def build_model(self, data: ProcessedData, constraints: Constraints) -> OptimizationModel: ...
    def add_constraints(self, model: OptimizationModel, constraints: List[Constraint]) -> None: ...
    def set_objective(self, model: OptimizationModel, objective: ObjectiveFunction) -> None: ...

class OptimizationSolver(Protocol):
    def solve(self, model: OptimizationModel) -> SolutionResult: ...
    def solve_with_fallback(self, model: OptimizationModel, fallback_strategies: List[Strategy]) -> SolutionResult: ...

class SolutionProcessor(Protocol):
    def process_solution(self, solution: SolutionResult, data: ProcessedData) -> HarvestPlan: ...
    def validate_solution(self, solution: SolutionResult) -> ValidationResult: ...

# Export Interfaces
class DataExporter(Protocol):
    def export_harvest_plan(self, plan: HarvestPlan, output_config: ExportConfig) -> ExportResult: ...
    def export_summaries(self, summaries: List[Summary], output_config: ExportConfig) -> ExportResult: ...
```

### Refactored Component Structure

#### 1. Data Management Layer

```python
# harvest/data/
├── loaders/
│   ├── csv_loader.py          # CSV file loading with validation
│   ├── excel_loader.py        # Excel file loading (future extension)
│   └── base_loader.py         # Abstract base loader
├── processors/
│   ├── growth_processor.py    # Growth data processing
│   ├── market_processor.py    # Market data processing
│   ├── mortality_processor.py # Mortality calculations
│   └── weight_processor.py    # Weight filtering and validation
├── validators/
│   ├── schema_validator.py    # Data schema validation
│   ├── business_validator.py  # Business rule validation
│   └── integrity_validator.py # Data integrity checks
└── models/
    ├── raw_data.py           # Raw data models
    ├── processed_data.py     # Processed data models
    └── validation_result.py  # Validation result models
```

#### 2. Optimization Engine Layer

```python
# harvest/optimization/
├── models/
│   ├── slaughterhouse_model.py    # Slaughterhouse optimization model
│   ├── market_model.py            # Market optimization model
│   └── base_model.py              # Abstract optimization model
├── solvers/
│   ├── pulp_solver.py             # PuLP solver implementation
│   ├── ortools_solver.py          # OR-Tools solver (future)
│   └── base_solver.py             # Abstract solver interface
├── processors/
│   ├── solution_processor.py      # Solution processing logic
│   ├── constraint_processor.py    # Constraint handling
│   └── objective_processor.py     # Objective function handling
├── strategies/
│   ├── fallback_strategy.py       # Fallback optimization strategies
│   ├── relaxation_strategy.py     # Constraint relaxation strategies
│   └── base_strategy.py           # Abstract strategy interface
└── constraints/
    ├── capacity_constraints.py    # Capacity constraint definitions
    ├── weight_constraints.py      # Weight constraint definitions
    ├── timing_constraints.py      # Timing constraint definitions
    └── base_constraints.py        # Abstract constraint interface
```

#### 3. Export and Reporting Layer

```python
# harvest/export/
├── exporters/
│   ├── csv_exporter.py           # CSV export functionality
│   ├── excel_exporter.py         # Excel export with formatting
│   └── base_exporter.py          # Abstract exporter interface
├── formatters/
│   ├── harvest_formatter.py      # Harvest plan formatting
│   ├── summary_formatter.py      # Summary data formatting
│   └── report_formatter.py       # Report formatting utilities
├── templates/
│   ├── excel_templates.py        # Excel formatting templates
│   └── report_templates.py       # Report structure templates
└── validators/
    ├── export_validator.py       # Export data validation
    └── file_validator.py         # File output validation
```

#### 4. Service and Application Layer

```python
# harvest/services/
├── harvest_service.py            # Main harvest orchestration service
├── optimization_service.py       # Optimization workflow service
├── export_service.py             # Export workflow service
├── validation_service.py         # Validation workflow service
└── configuration_service.py      # Configuration management service
```

## Data Models

### Enhanced Data Models with Validation

```python
@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationWarning]
    
    def add_error(self, error: ValidationError) -> None: ...
    def add_warning(self, warning: ValidationWarning) -> None: ...
    def has_errors(self) -> bool: ...

@dataclass
class ProcessedData:
    growth_data: pd.DataFrame
    market_data: pd.DataFrame
    metadata: DataMetadata
    validation_result: ValidationResult
    
    def validate(self) -> ValidationResult: ...
    def get_date_range(self) -> Tuple[datetime, datetime]: ...
    def get_farm_house_combinations(self) -> List[Tuple[str, str]]: ...

@dataclass
class OptimizationModel:
    problem: LpProblem
    variables: Dict[str, LpVariable]
    constraints: Dict[str, Any]
    objective: Any
    metadata: ModelMetadata
    
    def add_constraint(self, name: str, constraint: Any) -> None: ...
    def remove_constraint(self, name: str) -> None: ...
    def get_constraint_summary(self) -> Dict[str, Any]: ...

@dataclass
class SolutionResult:
    status: SolutionStatus
    objective_value: Optional[float]
    variables: Dict[str, float]
    solve_time: float
    solver_info: Dict[str, Any]
    
    def is_optimal(self) -> bool: ...
    def is_feasible(self) -> bool: ...
    def get_solution_summary(self) -> Dict[str, Any]: ...

@dataclass
class HarvestPlan:
    harvest_entries: List[HarvestEntry]
    summary_data: SummaryData
    metadata: PlanMetadata
    validation_result: ValidationResult
    
    def get_total_harvest(self) -> int: ...
    def get_total_meat(self) -> float: ...
    def get_harvest_by_type(self, harvest_type: HarvestType) -> List[HarvestEntry]: ...
    def validate_plan(self) -> ValidationResult: ...
```

## Error Handling

### Comprehensive Error Handling Strategy

#### 1. Exception Hierarchy

```python
class HarvestOptimizationError(Exception):
    """Base exception for harvest optimization system."""
    pass

class DataValidationError(HarvestOptimizationError):
    """Raised when data validation fails."""
    pass

class OptimizationError(HarvestOptimizationError):
    """Raised when optimization fails."""
    pass

class ExportError(HarvestOptimizationError):
    """Raised when export operations fail."""
    pass

class ConfigurationError(HarvestOptimizationError):
    """Raised when configuration is invalid."""
    pass
```

#### 2. Error Context and Recovery

```python
@dataclass
class ErrorContext:
    operation: str
    component: str
    input_data: Dict[str, Any]
    timestamp: datetime
    stack_trace: str
    
class ErrorHandler:
    def handle_data_error(self, error: DataValidationError, context: ErrorContext) -> RecoveryAction: ...
    def handle_optimization_error(self, error: OptimizationError, context: ErrorContext) -> RecoveryAction: ...
    def handle_export_error(self, error: ExportError, context: ErrorContext) -> RecoveryAction: ...
```

#### 3. Logging Strategy

```python
class StructuredLogger:
    def log_operation_start(self, operation: str, context: Dict[str, Any]) -> None: ...
    def log_operation_success(self, operation: str, result: Dict[str, Any]) -> None: ...
    def log_operation_failure(self, operation: str, error: Exception, context: Dict[str, Any]) -> None: ...
    def log_performance_metrics(self, operation: str, metrics: Dict[str, Any]) -> None: ...
```

## Testing Strategy

### Multi-Level Testing Approach

#### 1. Unit Testing Structure

```python
# tests/unit/
├── data/
│   ├── test_loaders.py
│   ├── test_processors.py
│   └── test_validators.py
├── optimization/
│   ├── test_model_builders.py
│   ├── test_solvers.py
│   └── test_solution_processors.py
├── export/
│   ├── test_exporters.py
│   └── test_formatters.py
└── services/
    ├── test_harvest_service.py
    └── test_optimization_service.py
```

#### 2. Integration Testing Structure

```python
# tests/integration/
├── test_data_pipeline.py        # End-to-end data processing
├── test_optimization_pipeline.py # End-to-end optimization
├── test_export_pipeline.py      # End-to-end export
└── test_full_workflow.py        # Complete system workflow
```

#### 3. Test Data Management

```python
# tests/fixtures/
├── sample_data/
│   ├── growth_data_small.csv
│   ├── growth_data_large.csv
│   ├── market_data_sample.csv
│   └── invalid_data_samples/
├── expected_outputs/
│   ├── harvest_plans/
│   ├── summaries/
│   └── exports/
└── test_configurations/
    ├── valid_configs/
    └── invalid_configs/
```

#### 4. Testing Utilities

```python
class TestDataBuilder:
    def build_growth_data(self, farms: int, houses: int, days: int) -> pd.DataFrame: ...
    def build_market_data(self, date_range: Tuple[datetime, datetime]) -> pd.DataFrame: ...
    def build_invalid_data(self, error_type: str) -> pd.DataFrame: ...

class OptimizationTestHelper:
    def create_simple_model(self) -> OptimizationModel: ...
    def create_infeasible_model(self) -> OptimizationModel: ...
    def verify_solution_validity(self, solution: SolutionResult, expected: Dict[str, Any]) -> bool: ...

class ExportTestHelper:
    def verify_csv_structure(self, file_path: str, expected_columns: List[str]) -> bool: ...
    def verify_excel_formatting(self, file_path: str, expected_format: Dict[str, Any]) -> bool: ...
    def compare_export_outputs(self, actual: str, expected: str) -> ComparisonResult: ...
```

### Performance Testing

```python
class PerformanceTestSuite:
    def test_large_dataset_processing(self) -> None: ...
    def test_optimization_scalability(self) -> None: ...
    def test_memory_usage_patterns(self) -> None: ...
    def test_concurrent_operations(self) -> None: ...
```

## Implementation Phases

### Phase 1: Foundation and Infrastructure (Requirements 1, 2, 5)
- Establish project structure and interfaces
- Implement comprehensive error handling and logging
- Set up code formatting and documentation standards
- Create base classes and abstract interfaces

### Phase 2: Data Layer Refactoring (Requirements 4, 7)
- Extract and modularize data loading functionality
- Implement robust input validation and data integrity checks
- Create reusable data processing components
- Add comprehensive unit tests for data layer

### Phase 3: Optimization Layer Refactoring (Requirements 1, 6)
- Break down monolithic optimization functions
- Implement clean separation between model building, solving, and processing
- Add strategy pattern for fallback optimization approaches
- Create focused, testable optimization components

### Phase 4: Export Layer Enhancement (Requirements 4, 8)
- Modularize CSV and Excel export functionality
- Implement pluggable export formats
- Add export data validation
- Clean up dependency management

### Phase 5: Service Layer and Integration (Requirements 3, 6)
- Create high-level service orchestration
- Implement comprehensive integration testing
- Add end-to-end workflow testing
- Performance optimization and monitoring

### Phase 6: Documentation and Finalization (Requirements 5)
- Complete API documentation
- Add architectural decision records
- Create developer onboarding guides
- Final code review and quality assurance