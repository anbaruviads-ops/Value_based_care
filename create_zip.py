"""
Packaging utility script.
Creates a clean, ready-to-distribute ZIP archive of the complete AI Assistant module.
Excludes virtual environments, pycache, and temporary files.
"""

import os
import sys
import zipfile
from pathlib import Path

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def create_project_zip(output_zip_name="aco_ai_assistant_module.zip"):
    project_root = Path(__file__).resolve().parent
    output_path = project_root / output_zip_name

    exclude_dirs = {".git", ".pytest_cache", "__pycache__", "venv", ".venv", "env"}
    exclude_extensions = {".pyc", ".pyo", ".pyd", ".zip"}

    print(f"[PACKAGING] Packaging project into: {output_path.name} ...")

    file_count = 0
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(project_root):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                file_path = Path(root) / file
                if file == output_zip_name or file_path.suffix in exclude_extensions:
                    continue

                arcname = file_path.relative_to(project_root)
                zipf.write(file_path, arcname)
                file_count += 1

    print(f"[SUCCESS] Created {output_zip_name} containing {file_count} files ({output_path.stat().st_size / 1024:.1f} KB).")
    return output_path

if __name__ == "__main__":
    create_project_zip()
