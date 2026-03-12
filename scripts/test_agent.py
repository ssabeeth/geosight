from geosight.agent import run_agent

print("Running GeoSight agent for TQ13 8HH (Dartmoor)...")
print("This will take 30-60 seconds for the LLM to generate the report.\n")

result = run_agent("TQ13 8HH")

if result["errors"]:
    print("Warnings:", result["errors"])

print(result["report"])