import numpy as np

class InMemoryVectorStore:
    def __init__(self):
        self.items = []


    def _validate_item(self, item):
        if not isinstance(item, dict):
            raise TypeError(f"item must be dict, got {type(item).__name__}")

        required = {"id", "text", "embedding", "metadata"}
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"Missing required keys: {sorted(missing)}")

        


    def add(self, item):
        self._validate_item(item)
        self.items.append(item)

            
    def add_many(self, items):
        if not isinstance(items, list):
            raise TypeError(f"items must be list, got {type(items).__name__}")

        for item in items:
            self._validate_item(item)

        self.items.extend(items)


    def search(self, query_vector, top_k):
        if top_k <= 0:
            raise ValueError("top_k must be a positive number")

        if not self.items:
            return []

        score_item_pair = []

        for item in self.items:
            score_item_pair.append((float(np.dot(query_vector, item["embedding"])), item)) 

        
        score_item_pair.sort(key=lambda x: x[0], reverse=True)

        scored_items = []

        for pair in score_item_pair[:top_k]:
            scored_items.append(
                {
                    "score": pair[0],
                    "id": pair[1]["id"],
                    "text": pair[1]["text"],
                    "metadata":  pair[1]["metadata"]

                }
            )
        return scored_items

    
    def count(self):
        return len(self.items)

    
    def clear(self):
        self.items.clear()





# m = InMemoryVectorStore()

# m._validate_item({"id": "dasdsad", "text": "fsdff"})
