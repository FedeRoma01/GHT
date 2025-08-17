#!/bin/bash

# Lista degli IP interni (o hostname) delle tue 8 VM
VMS=("10.128.0.4" "10.128.0.5" "10.128.0.6" "10.128.0.7" "10.128.0.8" "10.128.0.9" "10.128.0.10" "10.128.0.11")

# Percorso dove installare Open MPI
MPI_PREFIX="/opt/openmpi"

# Versione di Open MPI da compilare
MPI_VERSION="5.1.0"
MPI_TAR="openmpi-$MPI_VERSION.tar.gz"
MPI_URL="https://download.open-mpi.org/release/open-mpi/v5.1/$MPI_TAR"

# Comandi da eseguire su ciascun nodo
for vm in "${VMS[@]}"; do
  echo ">>> Configuro e compilo Open MPI su $vm"

  ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no roma@$vm bash -c "'
    set -e

    # Installazione dipendenze
    sudo dnf groupinstall -y \"Development Tools\"
    sudo dnf install -y wget tar gcc gcc-c++ make zlib zlib-devel

    # Scarico e scompatto Open MPI
    cd /tmp
    if [ ! -f $MPI_TAR ]; then
      wget $MPI_URL
    fi
    rm -rf openmpi-$MPI_VERSION
    tar -xzf $MPI_TAR
    cd openmpi-$MPI_VERSION

    # Configuro, compilo e installo
    ./configure --prefix=$MPI_PREFIX --with-zlib
    make -j\$(nproc)
    sudo make install

    # Aggiorno PATH e LD_LIBRARY_PATH
    if ! grep -q \"$MPI_PREFIX/bin\" ~/.bashrc; then
      echo 'export PATH=$MPI_PREFIX/bin:\$PATH' >> ~/.bashrc
      echo 'export LD_LIBRARY_PATH=$MPI_PREFIX/lib:\$LD_LIBRARY_PATH' >> ~/.bashrc
    fi

    # Verifica installazione
    $MPI_PREFIX/bin/ompi_info | grep -i compress
  '"
done
