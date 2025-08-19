import glob
import os
from collections import defaultdict

# Time dictionaries
time_per_function_all_ranks = defaultdict(list)
total_times = []

# Dictionary for each function for each file
# time_per_function_per_file: {filename: {function: total_time}}
time_per_function_per_file = defaultdict(lambda: defaultdict(float))

gh_comm_times = 0.0
cg_comm_times = 0.0

max_index = 0

for filename in glob.glob("profiling_rank_*.txt"):
    with open(filename) as f:
        for line in f:
            try:
                index = int(filename.split("_")[-1].split(".")[0])
                func_name, time_str = line.strip().split(':')
                func_name = func_name.strip().split('(')[0]
                exec_time = float(time_str.strip().split()[0])
                if func_name == "TOTAL_TIME" and exec_time > max_index:
                    max_index = index
            except ValueError:
                print(f"Errore nel parsing della riga: '{line.strip()}'")

# Read profiling for each rank
filename = f"profiling_rank_{max_index}.txt"
with open(filename) as f:
    for line in f:
        if ':' not in line:
            continue
        try:
            func_name, time_str = line.strip().split(':')
            func_name = func_name.strip().split('(')[0]
            exec_time = float(time_str.strip().split()[0])
            if "(GH)" in line:
                gh_comm_times += exec_time
            elif "(CG)" in line:
                cg_comm_times += exec_time
            elif func_name == "TOTAL_TIME":
                total_times.append(exec_time)
            else:
                time_per_function_per_file[filename][func_name] += exec_time
        except ValueError:
            print(f"Errore nel parsing della riga: '{line.strip()}'")

# Maximum time of TOTAL_TIME (wall-clock parallelo)
wall_clock_total_time = total_times[0] if total_times else 0.0

# Reorganizing datas for each file: totals for each file
# function_totals: {function: [tempo_file1, tempo_file2, ...]}
function_totals = defaultdict(list)
for file_times in time_per_function_per_file.values():
    for func, total_time in file_times.items():
        function_totals[func].append(total_time)

# Output
output_lines = ["Profiling globale (funzioni parallele sommate per file e max tra i processi):\n"]
output_lines.append(f"TEMPO TOTALE PROGRAMMA (wall-clock): {wall_clock_total_time:.4f} s\n")

# Times and percentages of the slowest rank (not alwaays rank 0, ex. parallel3BatchGHT.c)
for func, times in function_totals.items():
    max_time = times[0]
    pct_total = 100 * max_time / wall_clock_total_time if wall_clock_total_time > 0 else 0.0
    if (func == "generalized_hough "):
        comm_times = gh_comm_times
    elif (func == "compute_gradient "):
        comm_times = cg_comm_times
    else: comm_times = 0
    output_lines.append(f"{func:<30}: {max_time:.4f} s ({max_index}) -> {pct_total:.2f}% del wall-clock [whose {comm_times:.4f} s of communication]")

print("\n".join(output_lines))

# Save resulting file
with open("profiling_summary.txt", "w") as out_f:
    out_f.write("\n".join(output_lines) + "\n")

# Delete profiling files for next execution
for filename in glob.glob("profiling_rank_*.txt"):
    try:
        os.remove(filename)
        print(f"Rimosso: {filename}")
    except OSError as e:
        print(f"Errore durante la rimozione di {filename}: {e}")
