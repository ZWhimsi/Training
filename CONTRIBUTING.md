# Contributing to ML Training Program

Thank you for your interest in contributing! This document provides guidelines for contributing to the training program.

## How to Contribute

### Reporting Issues

- Use GitHub Issues to report bugs or suggest improvements
- Include the day/track where you found the issue
- Provide clear reproduction steps if reporting a bug

### Improving Exercises

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b improve-triton-day05`
3. **Make your changes**
4. **Test your changes**: Ensure exercises work and tests pass
5. **Submit a pull request**

### Types of Contributions Welcome

- **Bug fixes**: Typos, incorrect hints, failing tests
- **Clarity improvements**: Better explanations, additional hints
- **New exercises**: Additional practice problems for a day
- **Documentation**: README improvements, resource links
- **Translations**: Translating exercises to other languages

## Exercise Guidelines

When creating or modifying exercises:

### Structure

```python
"""
Day XX: Topic Name
==================
Estimated time: 1-2 hours
Prerequisites: Day XX-1

Learning objectives:
- Clear, measurable objectives
- What the learner will be able to do

Hints:
- Specific function names to use
- Common pitfalls to avoid

Resources:
- Links to documentation
- Relevant papers or tutorials
"""

import required_modules

# ============================================================================
# Exercise 1: Descriptive Title
# ============================================================================
# API to look up: function_name(arg1, arg2), other_fn(); list names/args only, no full solution
def exercise_1():
    """
    Brief docstring explaining the function.
    
    Returns:
        Expected return type and description
    """
    # Your implementation here
    pass


# ============================================================================
# Tests
# ============================================================================

def test_exercise_1():
    """Test exercise 1 implementation."""
    result = exercise_1()
    assert result is not None, "Function should return a value"
    # More specific assertions


if __name__ == "__main__":
    print("Day XX: Topic Name")
    print("=" * 50)
    
    print("\nExercise 1:")
    exercise_1()
    
    print("\nRunning tests...")
    test_exercise_1()
    print("All tests passed!")
```

### Quality Checklist

- [ ] Estimated time is realistic (1-2 hours for all exercises combined)
- [ ] Prerequisites are correctly listed
- [ ] Learning objectives are clear and achievable
- [ ] Hints guide without spoiling the solution
- [ ] Tests verify correct implementation
- [ ] Code follows PEP 8 style guidelines
- [ ] Comments explain the "why", not just the "what"

### Difficulty Progression

Each day should:
- Build on concepts from previous days
- Introduce 1-2 new concepts maximum
- Include both guided exercises (more hints) and challenge exercises (fewer hints)
- Be completable in 1-2 hours for someone following the curriculum

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Include type hints where helpful
- Keep functions focused and well-documented

## Testing

Before submitting:

```bash
# Run all tests in a track
cd triton  # or pytorch, autodiff
pytest . -v

# Run specific day
pytest day05.py -v

# Check code style
black --check .
isort --check .
```

## Pull Request Process

1. Update documentation if needed
2. Ensure all tests pass
3. Keep commits focused and well-described
4. Reference related issues in the PR description

## Questions?

Open an issue with the "question" label if you need clarification on contributing.

Thank you for helping improve this training program!
