import glob
from collections import defaultdict

# Dizionari per raccogliere i tempi
time_per_function_all_ranks = defaultdict(list)
total_times = []

# Ricostruisci i tempi per funzione per ciascun file
# time_per_function_per_file: {filename: {function: total_time}}
time_per_function_per_file = defaultdict(lambda: defaultdict(float))

# Leggi tutti i file di profiling
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
                    # Aggiungi al dizionario aggregato per file
                    time_per_function_all_ranks[func_name].append(exec_time)
                    time_per_function_per_file[filename][func_name] += exec_time
            except ValueError:
                print(f"Errore nel parsing della riga: '{line.strip()}'")

# Calcola il tempo totale come il massimo dei TOTAL_TIME (wall-clock parallelo)
wall_clock_total_time = max(total_times) if total_times else 0.0

# Riorganizza i dati per funzione: raccoglie tutti i totali per file
# function_totals: {function: [tempo_file1, tempo_file2, ...]}
function_totals = defaultdict(list)
for file_times in time_per_function_per_file.values():
    for func, total_time in file_times.items():
        function_totals[func].append(total_time)

# Costruisce l'output
output_lines = ["Profiling globale (funzioni parallele sommate per file e max tra i processi):\n"]
output_lines.append(f"TEMPO TOTALE PROGRAMMA (wall-clock): {wall_clock_total_time:.4f} s\n")

# Ordina per funzione con massimo tempo totale decrescente
for func, times in sorted(function_totals.items(), key=lambda x: max(x[1]), reverse=True):
    max_time = max(times)
    pct_total = 100 * max_time / wall_clock_total_time if wall_clock_total_time > 0 else 0.0
    output_lines.append(f"{func:<30}: {max_time:.4f} s (max totale per processo) -> {pct_total:.2f}% del wall-clock")

# Stampa a console
print("\n".join(output_lines))

# Salva su file
with open("profiling_summary.txt", "w") as out_f:
    out_f.write("\n".join(output_lines) + "\n")
