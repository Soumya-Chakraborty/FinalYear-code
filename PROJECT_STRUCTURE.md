# Raga AI Project Structure

## Overview

This document describes the organized structure of the Raga AI project for analyzing Indian classical music performances.

## Directory Structure

```
FinalYear/
├── main.py                 # Main application entry point
├── module_1_ear.py         # Audio processing and pitch extraction module
├── module_2_translator.py  # Pitch to swara mapping module
├── module_3_judge.py       # Raga analysis and error detection module
├── __init__.py             # Package initialization
├── README.md               # Main project documentation
├── LICENSE                 # License information
├── requirements.txt        # Python dependencies
├── test_imports.py         # Dependency testing script
├── assets/                 # Project assets and diagrams
│   ├── system_architecture.png
│   ├── workflow_diagram.png
│   ├── analysis_steps.png
│   └── [generated visualizations]
├── docs/                   # Documentation
│   ├── requirements.md     # System requirements
│   └── research.md         # Research papers and references
├── venv/                   # Virtual environment (if used)
└── __pycache__/            # Python cache files
```

## File Descriptions

### Core Modules
- **main.py**: The main application that orchestrates the entire analysis process
- **module_1_ear.py**: Handles audio input, recording, and pitch extraction using CREPE
- **module_2_translator.py**: Converts pitch frequencies to swara notation and performs tonic calibration
- **module_3_judge.py**: Applies raga rules and detects violations in the performance

### Documentation Files
- **README.md**: Comprehensive project overview, architecture, and usage instructions
- **LICENSE**: Legal licensing information
- **requirements.txt**: List of required Python packages
- **docs/**: Additional documentation including requirements and research references

### Asset Files
- **assets/**: Contains system diagrams, workflow charts, and generated visualizations
- **Generated visualization files**: Step-by-step analysis outputs created during processing

### Configuration Files
- **setup.py**: Package configuration for distribution
- **__init__.py**: Package initialization file

## Generated Files During Operation

When the system runs, it creates the following visualization files:
- `performance_analysis.png` - Combined analysis view
- `step1_raw_pitch_contour.png` - Raw pitch tracking results
- `step2_segmented_notes.png` - Identified note events
- `step3_rule_violations.png` - Major rule violations
- `step4_complete_analysis.png` - Complete analysis with all feedback

## Development Guidelines

### Adding New Features
1. Follow the modular architecture pattern
2. Maintain backward compatibility
3. Update documentation accordingly
4. Add appropriate error handling

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

### Testing
- Run `test_imports.py` to verify dependencies
- Test each module independently
- Validate visualization outputs
- Check error handling paths

## Maintenance

### Updating Dependencies
- Modify `requirements.txt` as needed
- Test with updated dependencies
- Update documentation if necessary

### Documentation Updates
- Keep README current with new features
- Update research references as needed
- Document breaking changes

## Deployment

The system can be deployed in various environments:
- Local development machines
- Cloud computing platforms
- Educational institutions
- Research facilities

For deployment, ensure all dependencies in `requirements.txt` are installed and the system meets the hardware requirements documented in `docs/requirements.md`.