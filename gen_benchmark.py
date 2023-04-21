from sysidexpr.constants import benchmarks
import json

# write the benchmarks to a json file
with open("benchmarks.json", "w") as f:
    json.dump(benchmarks, f)