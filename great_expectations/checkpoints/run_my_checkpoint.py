
import sys

from great_expectations.checkpoint.types.checkpoint_result import (
    CheckpointResult,  # noqa: TCH001
)
from great_expectations.data_context import FileDataContext  # noqa: TCH001
from great_expectations.util import get_context

data_context: FileDataContext = get_context(
    context_root_dir=None
)

result: CheckpointResult = data_context.run_checkpoint(
    checkpoint_name="my_checkpoint",
    batch_request=None,
    run_name=None,
)

if not result["success"]:
    print("Validation failed!")
    sys.exit(1)

print("Validation succeeded!")
sys.exit(0)