# Camera-Radar Fusion Attack Validation Framework

This project is an advanced development framework built on top of [commaai/openpilot v0.9.6](https://github.com/commaai/openpilot/tree/v0.9.6). It is specifically designed to validate the effectiveness of our proposed **Camera-Radar Fusion Attack** against autonomous driving perception systems.

## ✨ Key Features & Improvements

1. **Radar Spoofing Simulation**: Modified `simulated_car.py` to simulate the generation, parsing, and transmission of radar signals based on specific vehicle models, routing the spoofed data directly into the OpenPilot pipeline for downstream processing.
2. **Real-time Camera Adversarial Patch Injection**: Modified `modeld.py` to dynamically apply pre-computed adversarial patches onto the rear of the lead vehicle in real-time, effectively executing a camera-side attack.
3. **On-the-fly Attack Toggle**: Implemented a dynamic switch mechanism in `modeld.py` and `simulated_car.py`. The attack can be toggled ON or OFF in real-time during simulation without needing to restart the system.

---

## 🛠️ Deployment & Installation

Follow these steps to set up the environment and deploy the attack framework.

### 1. Clone & Setup OpenPilot v0.9.6

First, set up the official OpenPilot v0.9.6 environment:

```bash
# Clone the specific v0.9.6 branch with its submodules
git clone -b v0.9.6 --recurse-submodules https://github.com/commaai/openpilot.git
cd openpilot

# Pull large files
git lfs pull

# Run the setup script (grant execution permissions if necessary)
chmod +x tools/ubuntu_setup.sh
tools/ubuntu_setup.sh

# Enter the virtual environment and compile
poetry shell
scons -u -j$(nproc)

```

> **⚠️ Troubleshooting Compilation (PYAV Error):**
> If you encounter compilation errors related to `av` or `pyav`, replace the `pyav` library in your virtual environment (`<path_to_openpilot>/.venv/lib/python3.11/site-packages`). After replacing it, re-run `tools/ubuntu_setup.sh`, open a **new terminal**, and recompile with `scons`.

### 2. Apply Fusion Attack Modifications

Next, clone this repository and apply the modifications to the OpenPilot directory:

```bash
# Navigate to a workspace directory (outside of the openpilot folder)
git clone -b openpilot0.9.6-upload-20260602 https://github.com/pengjk016/Fusion-Attack.git Fusion-Attack
cd Fusion-Attack

# Grant execution permissions to the patch script
chmod +x apply_fusion_attack_changes.sh

# Apply changes (replace the path below with your actual openpilot path)
./apply_fusion_attack_changes.sh /path/to/your/openpilot

```

---

## 🚀 Usage & Attack Execution

To run the simulation and execute the attack, open **four separate terminals** in the `openpilot` root directory.

### Terminal 1: Attack Controller (Toggle Switch)

Use this terminal to control the attack state in real-time. Make sure you are in the `poetry shell`.

```bash
poetry shell

# Enable the attack
touch /tmp/adversarial_patch_enabled

# Disable the attack
rm -f /tmp/adversarial_patch_enabled

# Check current status
ls -l /tmp/adversarial_patch_enabled
# (If the file exists, the attack is ON. If "No such file", the attack is OFF.)

```

### Terminal 2: Data Logging

Run the script to dump and record the model's inputs/outputs.

```bash
poetry shell
cd selfdrive/controls
python dump_model_input.py

```

### Terminal 3: Launch OpenPilot

Start the main OpenPilot processes.

```bash
poetry shell
./tools/sim/launch_openpilot.sh

```

### Terminal 4: Launch MetaDrive Bridge

Start the simulation bridge to connect MetaDrive with OpenPilot.

```bash
poetry shell
./tools/sim/run_bridge.py

```

---

## 📂 Modified Files Overview

Compared to the official OpenPilot v0.9.6 release, the following files have been modified or introduced in this repository:

1. **Adversarial Patch Carriers (`.npy` files)**: Placed in the same directory as `modeld.py`. You can replace these files to test different adversarial patches.
2. `openpilot/tools/sim/bridge/metadrive/metadrive_bridge.py`
3. `openpilot/tools/sim/bridge/metadrive/metadrive_process.py`
4. `openpilot/tools/sim/bridge/metadrive/metadrive_world.py`
5. `openpilot/tools/sim/lib/simulated_car.py`
6. `openpilot/tools/sim/launch_openpilot.sh`
7. `openpilot/selfdrive/modeld/modeld.py`
8. `openpilot/selfdrive/controls/radard.py`
9. `openpilot/selfdrive/controls/dump_model_input.py`
10. `visionipc_pyx.so` (Compiled library update)
11. **External Dependency Modification**:
* `<path_to_openpilot>/.venv/lib/python3.11/site-packages/metadrive/component/sensors/lidar.py`
* *(Note: This file is located inside OpenPilot's poetry virtual environment, not the main source tree).*
