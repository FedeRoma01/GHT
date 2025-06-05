# Nome degli eseguibili
PARALLEL = parall1
SERIAL = serial

# File sorgente
PARALLEL_SRC = parall_mult_template_rot_scale_master.c
SERIAL_SRC = serial_mult_template_rot_scale.c

# Compilatori
MPICC = mpicc
GCC = gcc

# Flags di compilazione
CFLAGS_PARALLEL = -g -O2
CFLAGS_SERIAL = -pg -O0

# Librerie
LIBS = -lm

# Obiettivo principale: costruisce entrambi
all: $(PARALLEL) $(SERIAL)

# Compilazione parallela
$(PARALLEL): $(PARALLEL_SRC)
	$(MPICC) $(CFLAGS_PARALLEL) -o $(PARALLEL) $(PARALLEL_SRC) $(LIBS)

# Compilazione seriale
$(SERIAL): $(SERIAL_SRC)
	$(GCC) $(CFLAGS_SERIAL) -o $(SERIAL) $(SERIAL_SRC) $(LIBS)

# Pulizia dei file compilati
clean:
	rm -f $(PARALLEL) $(SERIAL)
