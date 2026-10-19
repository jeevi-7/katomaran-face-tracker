
from database import (
    save_visitor,
    load_visitors,
    get_unique_visitor_count,
)
import numpy as np

# Temporary test embedding — not a real face
test_embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

save_visitor("test_visitor_001", test_embedding)

visitors = load_visitors()

print("Saved visitor IDs:", list(visitors.keys()))
print("Unique visitor count:", get_unique_visitor_count())
print("Test embedding:", visitors["test_visitor_001"])
