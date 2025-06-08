import glob

# Dizionari per raccogliere tutti i tempi per ogni funzione
from collections import defaultdict

# Lista dei tempi per ciascuna funzione in ogni processo
time_per_function_all_ranks = defaultdict(list)
total_times = []

# Cerca tutti i file generati dai vari processi MPI
for filename in glob.glob("profiling_rank_*.txt"):
    with open(filename) as f:
        for line in f:
            if ':' not in line:
                continue
            try:
                func_name, time_str = line.strip().split(':')
                func_name = func_name.strip().split('(')[0]
                exec_time = float(time_str.strip().split()[0])

                if func_name == "TOTAL_TIME":
                    total_times.append(exec_time)
                else:
                    time_per_function_all_ranks[func_name].append(exec_time)
            except ValueError:
                print(f"Errore nel parsing della riga: '{line.strip()}'")

# Calcola il tempo totale come il massimo dei TOTAL_TIME (tempo wall-clock parallelo)
wall_clock_total_time = max(total_times) if total_times else 0.0

# Costruisce l'output
output_lines = ["Profiling globale (funzioni parallele misurate con max tempo tra i processi):\n"]
output_lines.append(f"TEMPO TOTALE PROGRAMMA (wall-clock): {wall_clock_total_time:.4f} s\n")

for func, times in sorted(time_per_function_all_ranks.items(), key=lambda x: max(x[1]), reverse=True):
    max_time = max(times)
    pct_total = 100 * max_time / wall_clock_total_time if wall_clock_total_time > 0 else 0.0
    output_lines.append(f"{func:<30}: {max_time:.4f} s (max) -> {pct_total:.2f}% del wall-clock")

# Stampa a console
print("\n".join(output_lines))

# Salva su file
with open("profiling_summary.txt", "w") as out_f:
    out_f.write("\n".join(output_lines) + "\n")
