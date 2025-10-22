
qemu-img.exe create -f qcow2 ubuntu18.qcow2 20G

qemu-system-x86_64.exe  -m 4096  -accel whpx  -cdrom "./ubuntu-18.04.6-desktop-amd64.iso"  -boot d  -drive file=ubuntu18.qcow2,format=qcow2  -netdev user,id=net0,hostfwd=tcp::43289-:43289  -device e1000,netdev=net0


qemu-system-x86_64.exe  -m 4096  -accel whpx -drive file=ubuntu18.qcow2,format=qcow2  -netdev user,id=net0,hostfwd=tcp::43289-:43289  -device e1000,netdev=net0