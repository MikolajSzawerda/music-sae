from pathlib import Path
import os
def _is_project_root(path: Path) -> bool:
	return all((path / dir_name).is_dir() for dir_name in ['data', 'docs', 'scripts', 'musicsae'])

def _find_project_root(starting_dir: Path = None) -> Path:
	if starting_dir is None:
		starting_dir = Path.cwd()
	current_dir = starting_dir

	while current_dir != current_dir.parent:
		if _is_project_root(current_dir):
			return current_dir
		current_dir = current_dir.parent

	raise FileNotFoundError("Project root not found. Could not find directories: data, docs, scripts, src")

class CallableString(str):
	def __init__(self, value):
		super().__init__()
		self._value = value
		self._func = lambda *args: os.path.join(self._value, *args)

	def __str__(self):
		return self._value

	def __call__(self, *args, **kwargs):
		return self._func(*args)

PROJECT_ROOT = _find_project_root()
OUTPUT_PATH = CallableString(os.path.join(PROJECT_ROOT, 'data', 'output'))
INPUT_PATH = CallableString(os.path.join(PROJECT_ROOT, 'data', 'input'))
RAW_PATH = CallableString(os.path.join(PROJECT_ROOT, 'data', 'raw'))
LOGS_PATH = CallableString(os.path.join(PROJECT_ROOT, 'logs'))
MODELS_PATH = CallableString(os.path.join(PROJECT_ROOT, 'models'))
SCRIPTS_PATH = CallableString(os.path.join(PROJECT_ROOT, 'scripts'))

