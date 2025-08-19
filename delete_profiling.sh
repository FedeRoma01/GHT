#!/bin/bash

# Lista dei nodi su cui cancellare i file
NODES=("node1" "node2" "node3" "node4" "node5" "node6" "node7" "node8")  # aggiungi tutti i nodi necessari

# Cartella sui nodi remoti dove sono salvati i file
REMOTE_DIR="/home/roma/GHT"

echo "Inizio cancellazione file .txt da tutti i nodi..."

for NODE in "${NODES[@]}"; do
    echo "Cancello file su $NODE..."
    ssh "$NODE" "rm -f $REMOTE_DIR/*.txt"
    if [ $? -ne 0 ]; then
        echo "Attenzione: errore nella cancellazione su $NODE"
    fi
done

echo "Cancellazione completata su tutti i nodi."
