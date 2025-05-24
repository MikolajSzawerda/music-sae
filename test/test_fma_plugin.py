from musicsae.ds_plugins.fma_plugin import FmaPlugin
from src.project_config import RAW_DATA_DIR

if __name__ == "__main__":
    ds = FmaPlugin(32_000, 100, 100, 42)
    ds.prepare(base_dir=RAW_DATA_DIR / "fma_medium")
