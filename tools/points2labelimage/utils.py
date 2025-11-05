from typing import Any


def get_list_depth(nested_list: Any) -> int:
    if isinstance(nested_list, list):
        if len(nested_list) > 0:
            return 1 + max(map(get_list_depth, nested_list))
        else:
            return 1
    else:
        return 0
