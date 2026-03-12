from geosight.rag.retriever import retrieve

result = retrieve("flood risk development planning restrictions")
print(f"Query: {result.query}")
print(f"Found {len(result.chunks)} chunks\n")
for chunk in result.chunks:
    print(f"[{chunk.score:.3f}] {chunk.source}, p.{chunk.page}")
    print(f"  {chunk.text[:150]}...")
    print()