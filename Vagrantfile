# -*- mode: ruby -*-
# vi: set ft=ruby :

# Check if running on Windows (not WSL)
unless Gem.win_platform?
  puts ""
  puts "❌ ERROR: This Vagrantfile must be run from Windows (not WSL)"
  puts ""
  exit 1
end

Vagrant.configure("2") do |config|
  # Base box - Ubuntu 18.04 LTS
  config.vm.box = "ubuntu/bionic64"

  # Install vagrant-disksize plugin if not present
  unless Vagrant.has_plugin?("vagrant-disksize")
    puts "Installing vagrant-disksize plugin..."
    system("vagrant plugin install vagrant-disksize")
    puts "Please run 'vagrant up' again."
    exit
  end

  # Set disk size to 100GB for EmbodiedBench
  config.disksize.size = "100GB"

  # VirtualBox provider configuration
  config.vm.provider "virtualbox" do |vb|
    # Display the VirtualBox GUI when booting the machine
    vb.gui = true

    # Increased memory and CPU for EmbodiedBench (8GB recommended)
    vb.memory = 8192
    vb.cpus = 4

    # VM name
    vb.name = "embodiedbench-ubuntu-18.04"

    # Graphics configuration - use VBoxSVGA for better compatibility
    vb.customize ["modifyvm", :id, "--graphicscontroller", "vmsvga"]
    vb.customize ["modifyvm", :id, "--vram", "128"]
    vb.customize ["modifyvm", :id, "--accelerate3d", "on"]
  end

  # Network configuration - port forwarding for vLLM server
  config.vm.network "forwarded_port", guest: 43289, host: 43289, protocol: "tcp"

  # Sync folder - WSL compatibility fix
  # Disable default synced folder and use rsync instead
  config.vm.synced_folder ".", "/vagrant", disabled: true
  config.vm.synced_folder ".", "/vagrant", type: "rsync",
                                           rsync__exclude: [".git/", "*.qcow2", "*.iso", ".vagrant/",
                                                            "*.pt", "CoppeliaSim_Pro_V4_1_0_Ubuntu20_04/",
                                                            "habitat-lab/", "embodiedbench/envs/eb_habitat/data/",
                                                            "embodiedbench/envs/eb_alfred/data/json_2.1.0/",
                                                            "embodiedbench/envs/eb_manipulation/EB-Manipulation/",
                                                            "embodiedbench/envs/eb_manipulation/PyRep/",
                                                            "*.tar.xz", "install.log"]

  # Initial provisioning - install system dependencies
  config.vm.provision "shell", inline: <<-SHELL
    echo "=== Installing system dependencies ==="
    
    # Update package list
    apt-get update
    
    # Install Ubuntu desktop (minimal) - needed for GUI applications
    DEBIAN_FRONTEND=noninteractive apt-get install -y ubuntu-desktop-minimal
    
    # Install essential build tools and libraries
    apt-get install -y build-essential git wget curl vim
    apt-get install -y dkms linux-headers-$(uname -r)
    
    # Install additional dependencies for EmbodiedBench
    apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
    apt-get install -y git-lfs
    
    # Enable graphical login
    systemctl set-default graphical.target
    
    echo "=== System dependencies installed ==="
    echo "=== Disk space available: $(df -h / | tail -1 | awk '{print $4}') ==="
  SHELL

  # Install Miniconda for the vagrant user
  config.vm.provision "shell", privileged: false, inline: <<-SHELL
    echo "=== Installing Miniconda ==="
    
    if [ ! -d "$HOME/miniconda3" ]; then
      wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
      bash /tmp/miniconda.sh -b -p $HOME/miniconda3
      rm /tmp/miniconda.sh
      
      # Initialize conda for bash
      $HOME/miniconda3/bin/conda init bash
      
      echo "Miniconda installed successfully"
    else
      echo "Miniconda already installed"
    fi
  SHELL

  # Run the EmbodiedBench installation script
  config.vm.provision "shell", privileged: false, inline: <<-SHELL
    echo "=== Running EmbodiedBench installation ==="
    
    cd /vagrant
    
    # Check if already installed (marker file approach)
    if [ -f /home/vagrant/.embodiedbench_installed ]; then
      echo "EmbodiedBench already installed (found marker file)"
      echo "To reinstall, remove /home/vagrant/.embodiedbench_installed and run 'vagrant provision'"
    else
      # Source conda
      source "$HOME/miniconda3/etc/profile.d/conda.sh"
      
      # Make install script executable
      chmod +x install.sh
      
      # Run the installation script
      bash install.sh 2>&1 | tee /vagrant/install.log
      
      # Create marker file on successful installation
      touch /home/vagrant/.embodiedbench_installed
      
      echo "=== EmbodiedBench installation complete ==="
      echo "Installation log saved to /vagrant/install.log"
    fi
  SHELL

  # Prepare the run script
  config.vm.provision "shell", privileged: false, inline: <<-SHELL
    echo "=== Setting up EmbodiedBench run script ==="
    
    cd /vagrant
    
    # Make scripts executable
    chmod +x install.sh
    chmod +x scripts/run_embodiedbench.sh
    
    # Create a desktop shortcut that points to the actual script
    mkdir -p $HOME/Desktop
    cp scripts/run_embodiedbench.sh $HOME/Desktop/
    
    echo "=== Setup complete! ==="
    echo ""
    echo "To run EmbodiedBench evaluations:"
    echo "  1. SSH into VM: vagrant ssh"
    echo "  2. Navigate to project: cd /vagrant"
    echo "  3. Run evaluations: ./scripts/run_embodiedbench.sh <job_name> <exp_name> [port]"
    echo ""
    echo "Or use the desktop shortcut when in GUI mode"
  SHELL

  # Auto-start GUI after provisioning
  config.vm.provision "shell", run: "always", inline: <<-SHELL
    # Ensure the desktop service is running
    systemctl start gdm3 || true
  SHELL
end
