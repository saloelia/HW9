# Architecture Documentation
# DeepFake Video Detection System

## 1. System Overview

The DeepFake Detector is designed using a modular, layered architecture that separates concerns and enables easy extension and testing.

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│                   (Python API / CLI)                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DeepFakeDetector                            │
│                    (High-Level API)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  DeepFakeDetectorAgent                          │
│              (Orchestration & LLM Reasoning)                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Prompts    │  │    State     │  │  LLM Client  │          │
│  │  Templates   │  │  Management  │  │  (Anthropic/ │          │
│  │              │  │              │  │   OpenAI)    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Tools Layer                              │
├──────────┬──────────┬──────────┬──────────┬─────────────────────┤
│  Video   │   Face   │ Temporal │Frequency │   Optical Flow      │
│Extractor │ Analyzer │ Analyzer │ Analyzer │     Analyzer        │
└──────────┴──────────┴──────────┴──────────┴─────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Core Components                             │
│          (Models, Configuration, Utilities)                     │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Component Design

### 2.1 Core Module

#### 2.1.1 Models (`core/models.py`)

Data classes following the Building Blocks pattern:

```python
class VideoMetadata:
    """
    Input Data: file_path, duration, fps, resolution
    Output Data: total_frames (computed), aspect_ratio (computed)
    """

class AnalysisResult:
    """
    Input Data: video_metadata, tool results
    Output Data: verdict, confidence, reasoning
    Setup Data: analysis_id, timestamp
    """
```

#### 2.1.2 Detector (`core/detector.py`)

High-level API wrapper:
- Validates inputs
- Creates and manages agent
- Handles errors gracefully

### 2.2 Agent Module

#### 2.2.1 DetectorAgent (`agent/detector_agent.py`)

Central orchestration component:

```
┌───────────────────────────────────────────────────────┐
│                DeepFakeDetectorAgent                  │
├───────────────────────────────────────────────────────┤
│ Properties:                                          │
│   - config: AgentConfig                              │
│   - tools: Dict[str, BaseTool]                       │
│   - state: AgentState                                │
├───────────────────────────────────────────────────────┤
│ Methods:                                             │
│   + analyze(video_path) -> AnalysisResult           │
│   - _extract_frames()                                │
│   - _run_sequential_analysis()                       │
│   - _run_autonomous_analysis()                       │
│   - _synthesize_results()                            │
│   - _call_llm()                                      │
└───────────────────────────────────────────────────────┘
```

#### 2.2.2 Prompts (`agent/prompts.py`)

Prompt engineering templates:
- **SYSTEM_PROMPT**: Agent persona and capabilities
- **ANALYSIS_PROMPT_TEMPLATE**: Analysis initiation
- **REASONING_PROMPT_TEMPLATE**: Result synthesis
- **TOOL_SELECTION_PROMPT**: Autonomous mode

### 2.3 Tools Module

All tools inherit from `BaseTool`:

```
┌─────────────────────────────────────────────────────────┐
│                       BaseTool                          │
├─────────────────────────────────────────────────────────┤
│ Properties:                                             │
│   + name: str                                           │
│   + description: str                                    │
│   + config: ToolConfig                                  │
├─────────────────────────────────────────────────────────┤
│ Abstract Methods:                                       │
│   + execute(**kwargs) -> ToolResult                    │
│   + get_schema() -> Dict                               │
├─────────────────────────────────────────────────────────┤
│ Concrete Methods:                                       │
│   + get_tool_definition() -> Dict                      │
│   + validate_input() -> List[str]                      │
└─────────────────────────────────────────────────────────┘
            △
            │ inherits
  ┌─────────┼─────────┬──────────┬──────────┐
  │         │         │          │          │
┌─┴──┐  ┌───┴──┐  ┌───┴──┐  ┌────┴──┐  ┌────┴───┐
│Video│  │Face │  │Temp. │  │Freq. │  │Optical │
│Extr.│  │Anlz.│  │Anlz. │  │Anlz. │  │ Flow   │
└─────┘  └─────┘  └──────┘  └──────┘  └────────┘
```

#### Tool Descriptions:

| Tool | Purpose | Key Outputs |
|------|---------|-------------|
| VideoExtractor | Extract frames | frames[], metadata |
| FaceAnalyzer | Detect/analyze faces | face_data, consistency_score |
| TemporalAnalyzer | Temporal patterns | blink_rate, movement_scores |
| FrequencyAnalyzer | FFT analysis | gan_fingerprint, noise_score |
| OpticalFlowAnalyzer | Motion analysis | flow_consistency, boundary_score |

## 3. Data Flow

### 3.1 Analysis Pipeline

```
Video File
    │
    ▼
┌────────────────┐
│ Video Extractor│──────────────────────────────┐
└───────┬────────┘                              │
        │ frames[]                             │
        ▼                                      │
┌────────────────┐                             │
│ Face Analyzer  │                             │
└───────┬────────┘                             │
        │ frame_analyses[]                     │
        │                                      │
        ├──────────────────┐                   │
        ▼                  ▼                   ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│   Temporal     │  │   Frequency    │  │ Optical Flow   │
│   Analyzer     │  │   Analyzer     │  │   Analyzer     │
└───────┬────────┘  └───────┬────────┘  └───────┬────────┘
        │                  │                   │
        └──────────┬───────┴───────────────────┘
                   │
                   ▼
        ┌──────────────────┐
        │ Evidence & Scores│
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  LLM Synthesis   │
        │  (or Rule-based) │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ AnalysisResult   │
        │ verdict, conf,   │
        │ evidence, reason │
        └──────────────────┘
```

### 3.2 Tool Result Structure

```python
@dataclass
class ToolResult(Generic[T]):
    """
    Standardized tool output container.

    Building Block Output Data pattern.
    """
    success: bool
    data: Optional[T]
    error: Optional[str]
    metadata: Dict[str, Any]
```

## 4. Design Patterns

### 4.1 Building Blocks Pattern

Each component clearly defines:
- **Input Data**: Parameters for operation
- **Output Data**: Results of operation
- **Setup Data**: Configuration and defaults

### 4.2 Strategy Pattern (Tools)

Tools implement a common interface but have different algorithms:
- `execute()` method performs the analysis
- `get_schema()` returns input specification

### 4.3 Template Method (Agent)

Agent defines analysis skeleton:
```python
def analyze(self, video_path):
    frames = self._extract_frames(video_path)
    results = self._run_analysis(frames)
    verdict = self._synthesize_results(results)
    return verdict
```

### 4.4 Factory Pattern (LLM Client)

Client creation based on provider:
```python
if provider == "anthropic":
    client = anthropic.Anthropic()
elif provider == "openai":
    client = openai.OpenAI()
```

## 5. Configuration Architecture

### 5.1 Configuration Hierarchy

```
Environment Variables (.env)
         │
         ▼ (override)
Configuration File (config.yaml)
         │
         ▼ (override)
Default Values (code)
```

### 5.2 Configuration Scope

| Level | Scope | Example |
|-------|-------|---------|
| Environment | Secrets | API_KEY |
| Config File | Tunables | max_frames |
| Defaults | Safe values | temperature=0.3 |

## 6. Error Handling

### 6.1 Error Categories

| Category | Handling | Example |
|----------|----------|---------|
| Input Validation | Return immediately | Invalid video path |
| Tool Errors | Log and continue | Frame extraction failed |
| LLM Errors | Fallback to rules | API timeout |
| Critical | Return error result | All tools failed |

### 6.2 Fallback Strategy

```
LLM Available?
    │
    ├── Yes ──► LLM Synthesis
    │
    └── No ──► Rule-based Synthesis
```

## 7. Testing Architecture

### 7.1 Test Layers

```
┌─────────────────────────────────────┐
│        Integration Tests           │
│    (Full pipeline tests)           │
├─────────────────────────────────────┤
│         Unit Tests                 │
│    (Individual components)         │
├─────────────────────────────────────┤
│         Mock Tests                 │
│    (Isolated with mocks)           │
└─────────────────────────────────────┘
```

### 7.2 Test Coverage Target

| Component | Target Coverage |
|-----------|----------------|
| Core | 90% |
| Tools | 85% |
| Agent | 80% |
| Utils | 75% |

## 8. Extensibility

### 8.1 Adding New Tools

1. Create class inheriting from `BaseTool`
2. Implement `execute()` and `get_schema()`
3. Register in agent's tools dictionary
4. Add prompt guidance in `prompts.py`

### 8.2 Adding LLM Providers

1. Add provider handling in `_initialize_client()`
2. Add response parsing in `_call_llm()`
3. Update configuration options

## 9. Architectural Decision Records (ADR)

### ADR-001: Use Building Blocks Pattern

**Decision**: Implement Building Blocks pattern for all components

**Rationale**:
- Clear separation of input/output/setup
- Self-documenting code
- Easier testing

### ADR-002: LLM Fallback Strategy

**Decision**: Implement rule-based fallback when LLM unavailable

**Rationale**:
- System remains functional offline
- Reduces API costs for simple cases
- Faster processing when confident

### ADR-003: Sequential vs Autonomous Mode

**Decision**: Default to sequential analysis, autonomous as option

**Rationale**:
- Sequential is deterministic and testable
- Autonomous requires more LLM calls
- Sequential provides baseline performance

## 10. Security Considerations

### 10.1 API Key Management
- Keys stored in environment variables
- Never hardcoded or logged
- `.env` file in `.gitignore`

### 10.2 Input Validation
- Video path validated before processing
- File type checked against whitelist
- Size limits enforced
