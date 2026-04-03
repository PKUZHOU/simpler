# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""HostTask: orchestration unit for HostWorker."""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class HostTask:
    """A unit of work for HostWorker.execute().

    orch is called as orch(hw, args) where hw is the HostWorker instance.
    Dependencies between tasks are inferred automatically from tensor base pointers
    via the distributed runtime's TensorMap.
    """

    orch: Callable
    args: Any = field(default=None)
