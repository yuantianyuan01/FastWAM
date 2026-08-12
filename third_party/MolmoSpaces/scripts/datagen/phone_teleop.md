## Teleoperate in MolmoSpaces using your phone

1. Install Mujoco AR from the App Store
   https://apps.apple.com/us/app/mujoco-ar/id6612039501

2. Run the datagen pipeline with the teleop policy
   ```bash
   python scripts/datagen/run_pipeline.py \
     --eval ~/datagen/pick_v1/20251202_134009_benchmark \
     --policy teleop \
     --robot rum   # or: droid
   ```
3. Enter the printed IP address and port into the app. Example terminal output:
   ```bash
   MujocoARConnector Starting on port 8888...
   MujocoARConnector Started. Details:
   IP Address: 192.168.50.126
   Port: 8888
   Waiting for a device to connect...
   ```
4. Start teleoperating!

- Click the Toggle to Grasp
- Click the Button to go to the next episode
