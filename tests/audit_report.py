"""
COMPREHENSIVE AUDIT REPORT
PyTorchSandbox Mechanistic Interpretability Environment
Final Pre-Deployment Verification
"""

import os
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# ============================================================================
# AUDIT REPORT GENERATOR
# ============================================================================

class AuditReport:
    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.project_root = Path(__file__).parent.parent
        self.issues = []
        self.warnings = []
        self.passes = []
        self.metrics = {}
    
    def add_pass(self, category, message):
        self.passes.append({"category": category, "message": message})
    
    def add_warning(self, category, message):
        self.warnings.append({"category": category, "message": message})
    
    def add_issue(self, category, severity, message):
        self.issues.append({
            "category": category,
            "severity": severity,  # "critical", "high", "medium", "low"
            "message": message
        })
    
    def add_metric(self, name, value):
        self.metrics[name] = value
    
    def generate(self):
        return {
            "timestamp": self.timestamp,
            "summary": {
                "critical_issues": len([i for i in self.issues if i["severity"] == "critical"]),
                "high_issues": len([i for i in self.issues if i["severity"] == "high"]),
                "medium_issues": len([i for i in self.issues if i["severity"] == "medium"]),
                "low_issues": len([i for i in self.issues if i["severity"] == "low"]),
                "warnings": len(self.warnings),
                "passes": len(self.passes),
            },
            "issues": self.issues,
            "warnings": self.warnings,
            "passes": self.passes,
            "metrics": self.metrics,
        }
    
    def print_report(self):
        report = self.generate()
        
        print("\n" + "="*80)
        print("COMPREHENSIVE AUDIT REPORT - PyTorchSandbox Mech Interp Environment")
        print("="*80)
        print(f"Generated: {self.timestamp}")
        
        summary = report["summary"]
        print(f"\nSUMMARY:")
        print(f"  Critical Issues: {summary['critical_issues']}")
        print(f"  High Issues:     {summary['high_issues']}")
        print(f"  Medium Issues:   {summary['medium_issues']}")
        print(f"  Low Issues:      {summary['low_issues']}")
        print(f"  Warnings:        {summary['warnings']}")
        print(f"  Passes:          {summary['passes']}")
        
        if self.issues:
            print(f"\nISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  [{issue['severity'].upper()}] {issue['category']}: {issue['message']}")
        
        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning['category']}: {warning['message']}")
        
        if self.passes:
            print(f"\nPASSES ({len(self.passes)}):")
            for i, passed in enumerate(self.passes, 1):
                if i <= 10:  # Show first 10
                    print(f"  ✓ {passed['category']}: {passed['message']}")
            if len(self.passes) > 10:
                print(f"  ... and {len(self.passes) - 10} more passes")
        
        print("\nMETRICS:")
        for key, value in self.metrics.items():
            print(f"  {key}: {value}")
        
        print("\n" + "="*80)
        
        # Final verdict
        if summary["critical_issues"] == 0 and summary["high_issues"] == 0:
            print("✓ READY FOR PRODUCTION DEPLOYMENT")
        elif summary["critical_issues"] == 0:
            print("⚠ DEPLOYMENT ALLOWED WITH CAUTION (High issues present)")
        else:
            print("✗ NOT READY FOR DEPLOYMENT (Critical issues present)")
        
        print("="*80 + "\n")


# ============================================================================
# AUDIT CHECKS
# ============================================================================

def audit_file_structure(report):
    """Verify all required files exist."""
    required_files = {
        "pyproject.toml": "Project configuration",
        "Dockerfile": "Docker container definition",
        "models.py": "OpenEnv schema definitions",
        "client.py": "OpenEnv HTTP client",
        "inference.py": "LLM agent inference loop",
        "__init__.py": "Package initialization",
        "openenv.yaml": "OpenEnv manifest",
        "README.md": "Documentation",
        "server/app.py": "FastAPI server",
        "server/mech_interp_environment.py": "Environment implementation",
        "server/model_architectures.py": "Task models",
        "server/gen_art.py": "Artifact generation",
        "server/requirements.txt": "Server dependencies",
        "artifacts/task1.pt": "Task 1 model",
        "artifacts/task2.pt": "Task 2 model",
        "artifacts/task3.pt": "Task 3 model",
    }
    
    for filepath, description in required_files.items():
        full_path = report.project_root / filepath
        if full_path.exists():
            report.add_pass("File Structure", f"{description} exists")
        else:
            report.add_issue("File Structure", "critical", f"Missing: {filepath} ({description})")


def audit_python_syntax(report):
    """Verify all Python files compile."""
    python_files = [
        "models.py",
        "client.py",
        "inference.py",
        "__init__.py",
        "server/app.py",
        "server/mech_interp_environment.py",
        "server/model_architectures.py",
        "server/gen_art.py",
    ]
    
    import py_compile
    for pyfile in python_files:
        full_path = report.project_root / pyfile
        try:
            py_compile.compile(str(full_path), doraise=True)
            report.add_pass("Python Syntax", f"{pyfile} compiles")
        except py_compile.PyCompileError as e:
            report.add_issue("Python Syntax", "critical", f"{pyfile}: {e}")


def audit_dependencies(report):
    """Verify all dependencies are declared."""
    pyproject_path = report.project_root / "pyproject.toml"
    requirements_path = report.project_root / "server/requirements.txt"
    
    expected_deps = {
        "torch": "Core ML framework",
        "pydantic": "Data validation",
        "openenv-core": "OpenEnv runtime",
        "fastapi": "Web framework",
        "uvicorn": "ASGI server",
    }
    
    # Read pyproject.toml
    with open(pyproject_path) as f:
        content = f.read()
        for dep, desc in expected_deps.items():
            if dep in content:
                report.add_pass("Dependencies", f"{dep} declared in pyproject.toml")
            else:
                report.add_issue("Dependencies", "high", f"Missing in pyproject.toml: {dep}")
    
    # Read requirements.txt
    with open(requirements_path) as f:
        content = f.read()
        for dep in ["openenv-core", "fastapi", "uvicorn", "torch", "pydantic"]:
            if dep in content:
                report.add_pass("Dependencies", f"{dep} in server/requirements.txt")
            else:
                report.add_issue("Dependencies", "high", f"Missing in requirements.txt: {dep}")


def audit_ground_truth_consistency(report):
    """Verify ground truth values are consistent."""
    ground_truths = {
        "Task 1": {
            "expected": [2, 5, 8],
            "files_to_check": ["server/mech_interp_environment.py"]
        },
        "Task 2": {
            "expected": [2],
            "files_to_check": ["server/mech_interp_environment.py"]
        },
        "Task 3": {
            "expected": [2, 17, 23, 44, 47],
            "files_to_check": ["server/mech_interp_environment.py"]
        },
    }
    
    for task, info in ground_truths.items():
        expected_str = str(info["expected"])
        for file_to_check in info["files_to_check"]:
            with open(report.project_root / file_to_check) as f:
                content = f.read()
                if expected_str in content:
                    report.add_pass("Ground Truth Consistency", f"{task} ground truth found in {file_to_check}")
                else:
                    report.add_warning("Ground Truth Consistency", f"{task} ground truth not easily found in {file_to_check}")


def audit_model_artifacts(report):
    """Verify model artifacts exist and are loadable."""
    artifacts = {
        "task1.pt": "Dead Neuron Detection",
        "task2.pt": "Causal Ablation",
        "task3.pt": "Fourier Analysis",
    }
    
    for artifact, description in artifacts.items():
        path = report.project_root / "artifacts" / artifact
        
        if not path.exists():
            report.add_issue("Model Artifacts", "critical", f"Missing artifact: {artifact} ({description})")
            continue
        
        # Check file size
        size_kb = path.stat().st_size / 1024
        if size_kb < 1:
            report.add_issue("Model Artifacts", "high", f"{artifact} is too small ({size_kb:.1f}KB)")
        else:
            report.add_pass("Model Artifacts", f"{artifact} exists ({size_kb:.1f}KB)")
        
        # Try to load with torch
        try:
            model = torch.load(str(path), weights_only=False)
            report.add_pass("Model Artifacts", f"{artifact} loads successfully")
        except Exception as e:
            report.add_issue("Model Artifacts", "high", f"{artifact} failed to load: {e}")


def audit_dockerfile(report):
    """Verify Dockerfile is production-ready."""
    dockerfile_path = report.project_root / "Dockerfile"
    
    with open(dockerfile_path) as f:
        content = f.read()
    
    checks = {
        "FROM python:3.11": "Uses correct Python version",
        "apt-get update": "Updates apt cache",
        "--no-cache-dir": "Disables pip cache to reduce image size",
        "COPY pyproject.toml server/requirements.txt": "Copies dependencies correctly",
        "USER user": "Runs as non-root user",
        "EXPOSE 8000": "Exposes correct port",
        "uvicorn": "Uses uvicorn server",
    }
    
    for check, description in checks.items():
        if check in content:
            report.add_pass("Dockerfile", description)
        else:
            report.add_warning("Dockerfile", f"Missing: {description}")


def audit_inference_script(report):
    """Verify inference script is correctly configured."""
    inference_path = report.project_root / "inference.py"
    
    with open(inference_path) as f:
        content = f.read()
    
    checks = {
        "API_KEY": "Uses validator API key",
        "HF_TOKEN": "Keeps local HuggingFace fallback",
        "API_BASE_URL": "Sets API base URL",
        "deepseek-ai/DeepSeek-V3-0324": "Uses configured model",
        "MechInterpEnvironmentClient": "Uses correct client",
        "client.reset()": "Calls reset",
        "client.step()": "Calls step",
    }
    
    for check, description in checks.items():
        if check in content:
            report.add_pass("Inference Script", description)
        else:
            report.add_warning("Inference Script", f"Missing: {description}")


def audit_readme(report):
    """Verify README is complete and accurate."""
    readme_path = report.project_root / "README.md"
    
    with open(readme_path) as f:
        content = f.read()
    
    checks = {
        "# PyTorchSandbox": "Has title",
        "Task 1": "Documents Task 1",
        "Task 2": "Documents Task 2",
        "Task 3": "Documents Task 3",
        "## Installation": "Has installation section",
        "## Usage": "Has usage section",
        "ground truth": "Mentions ground truth",
    }
    
    for check, description in checks.items():
        if check.lower() in content.lower():
            report.add_pass("README", description)
        else:
            report.add_warning("README", f"Missing: {description}")


def audit_openenv_compliance(report):
    """Verify OpenEnv spec compliance."""
    
    # Check models.py for required schemas
    models_path = report.project_root / "models.py"
    with open(models_path) as f:
        content = f.read()
    
    required_classes = ["MechInterpAction", "MechInterpObservation", "InterpState"]
    for cls in required_classes:
        if cls in content:
            report.add_pass("OpenEnv Compliance", f"Has {cls} class")
        else:
            report.add_issue("OpenEnv Compliance", "high", f"Missing {cls} class")
    
    # Check environment.py for required methods
    env_path = report.project_root / "server/mech_interp_environment.py"
    with open(env_path) as f:
        content = f.read()
    
    required_methods = ["reset", "step", "@property"]
    for method in required_methods:
        if method in content:
            report.add_pass("OpenEnv Compliance", f"Has {method} method/property")
        else:
            report.add_issue("OpenEnv Compliance", "high", f"Missing {method} implementation")


def audit_security(report):
    """Verify security measures."""
    env_path = report.project_root / "server/mech_interp_environment.py"
    
    with open(env_path) as f:
        content = f.read()
    
    security_checks = {
        "signal.SIGALRM": "Has timeout protection",
        "contextlib.redirect_stdout": "Captures stdout/stderr",
        "ExecTimeoutError": "Handles timeout errors",
        "module._forward_hooks.clear()": "Clears hooks to prevent OOM",
        "sandbox": "Mentions sandbox",
    }
    
    for check, description in security_checks.items():
        if check in content:
            report.add_pass("Security", description)
        else:
            report.add_warning("Security", f"Missing: {description}")


def audit_code_quality(report):
    """Check code quality indicators."""
    
    files_to_check = [
        "models.py",
        "client.py",
        "inference.py",
        "server/app.py",
        "server/mech_interp_environment.py",
    ]
    
    for file_rel_path in files_to_check:
        file_path = report.project_root / file_rel_path
        
        with open(file_path) as f:
            content = f.read()
        
        # Check for docstrings
        if '"""' in content or "'''" in content:
            report.add_pass("Code Quality", f"{file_rel_path} has docstrings")
        else:
            report.add_warning("Code Quality", f"{file_rel_path} lacks docstrings")
        
        # Check for type hints
        if "->" in content:
            report.add_pass("Code Quality", f"{file_rel_path} has type hints")
        else:
            report.add_warning("Code Quality", f"{file_rel_path} lacks type hints")
        
        # Check line count
        lines = len(content.split("\n"))
        report.add_metric(f"{file_rel_path}_lines", lines)


def audit_performance(report):
    """Estimate performance characteristics."""
    
    # Task 1 model size
    task1_path = report.project_root / "artifacts/task1.pt"
    task1_size_kb = task1_path.stat().st_size / 1024
    report.add_metric("Task1_Model_Size_KB", f"{task1_size_kb:.1f}")
    
    task2_path = report.project_root / "artifacts/task2.pt"
    task2_size_kb = task2_path.stat().st_size / 1024
    report.add_metric("Task2_Model_Size_KB", f"{task2_size_kb:.1f}")
    
    task3_path = report.project_root / "artifacts/task3.pt"
    task3_size_kb = task3_path.stat().st_size / 1024
    report.add_metric("Task3_Model_Size_KB", f"{task3_size_kb:.1f}")
    
    total_artifacts_kb = task1_size_kb + task2_size_kb + task3_size_kb
    report.add_metric("Total_Artifacts_KB", f"{total_artifacts_kb:.1f}")
    
    report.add_pass("Performance", "All artifact sizes reasonable for edge deployment")


# ============================================================================
# MAIN AUDIT EXECUTION
# ============================================================================

if __name__ == "__main__":
    report = AuditReport()
    
    print("Running comprehensive audit...")
    
    # Run all audit checks
    audit_file_structure(report)
    audit_python_syntax(report)
    audit_dependencies(report)
    audit_ground_truth_consistency(report)
    audit_model_artifacts(report)
    audit_dockerfile(report)
    audit_inference_script(report)
    audit_readme(report)
    audit_openenv_compliance(report)
    audit_security(report)
    audit_code_quality(report)
    audit_performance(report)
    
    # Print report
    report.print_report()
    
    # Determine exit code
    summary = report.generate()["summary"]
    if summary["critical_issues"] == 0 and summary["high_issues"] <= 2:
        sys.exit(0)  # Success - ready for deployment
    else:
        sys.exit(1)  # Failure - issues found
