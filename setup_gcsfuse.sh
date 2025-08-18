#!/bin/bash

# Lista degli IP interni o hostname delle tue 8 VM
VMS=("10.128.0.4" "10.128.0.5" "10.128.0.6" "10.128.0.7" "10.128.0.8" "10.128.0.9" "10.128.0.10" "10.128.0.11")

# Nome del bucket GCS
BUCKET="profiling-bucket-mpi"

# Percorso di mount
MOUNTPOINT="/shared"

# Comandi da eseguire su ogni VM
for vm in "${VMS[@]}"; do
  echo ">>> Configurazione gcsfuse su $vm"

  ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no roma@$vm bash -c "'
    set -e

    # Installazione gcsfuse (Ubuntu/Debian)
    if [ -f /etc/debian_version ]; then
      sudo apt-get update
      sudo apt-get install -y gcsfuse
    fi

    # Installazione gcsfuse (CentOS/RHEL)
    if [ -f /etc/redhat-release ]; then
      if ! rpm -q gcsfuse >/dev/null 2>&1; then
        sudo tee /etc/yum.repos.d/gcsfuse.repo <<EOF
[gcsfuse]
name=gcsfuse (packages.cloud.google.com)
baseurl=https://packages.cloud.google.com/yum/repos/gcsfuse-el7-x86_64
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://packages.cloud.google.com/yum/doc/yum-key.gpg https://packages.cloud.google.com/yum/doc/rpm-package-key.gpg
EOF
        sudo yum install -y gcsfuse
      fi
    fi

    # Crea la cartella di mount se non esiste
    sudo mkdir -p $MOUNTPOINT

    # Monta il bucket
    sudo fusermount -u $MOUNTPOINT >/dev/null 2>&1 || true
    sudo gcsfuse --implicit-dirs \
        --uid=$(id -u roma) --gid=$(id -g roma) \
        --file-mode=644 --dir-mode=755 \
        $BUCKET $MOUNTPOINT
  '"
done
