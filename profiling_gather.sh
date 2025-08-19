#!/bin/bash

# Lista dei nodi da cui copiare i file
NODES=("node2" "node3" "node4" "node5" "node6" "node7" "node8")  # aggiungi tutti i nodi necessari

# Cartella dove salvare i file sul nodo 1
DEST_DIR="/home/roma/GHT"

# Cartella sui nodi remoti dove sono salvati i file
REMOTE_DIR="/home/roma/GHT"

echo "Inizio copia file da tutti i nodi..."

for NODE in "${NODES[@]}"; do
    echo "Copio file da $NODE..."
    scp "$NODE:$REMOTE_DIR/*.txt" "$DEST_DIR/"
    if [ $? -ne 0 ]; then
        echo "Attenzione: errore nella copia da $NODE"
    fi
done

echo "Copia completata. Tutti i file sono in $DEST_DIR"
