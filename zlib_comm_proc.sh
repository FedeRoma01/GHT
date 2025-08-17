#!/bin/bash

# Lista degli IP interni (o esterni) delle tue 8 VM
VMS=("10.128.0.4" "10.128.0.5" "10.128.0.6" "10.128.0.7" "10.128.0.8" "10.128.0.9" "10.128.0.10" "10.128.0.11")

for vm in "${VMS[@]}"; do
  echo ">>> Installo zlib su $vm"
  ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no roma@$vm "sudo dnf install -y zlib zlib-devel"
done
