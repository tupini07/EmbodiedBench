
# create a qcow2 disk image of 20GB
qemu-img create -f qcow2 ubuntu18.qcow2 20G

# install Ubuntu 18.04.6 from ISO
sudo qemu-system-x86_64 -enable-kvm  -m 4096  -cdrom "./ubuntu-18.04.6-desktop-amd64.iso"  -boot d  -drive file=ubuntu18.qcow2,format=qcow2  -netdev user,id=net0,hostfwd=tcp::43289-:43289  -device e1000,netdev=net0

# after installation, run the VM with the following command
sudo qemu-system-x86_64 \
  -enable-kvm \
  -m 4096 \
  -cpu host \
  -drive file=ubuntu18.qcow2,format=qcow2 \
  -netdev user,id=net0,hostfwd=tcp::43289-:43289 \
  -device e1000,netdev=net0
