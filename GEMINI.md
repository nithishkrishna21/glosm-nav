# GLOSM-Nav: Global-to-Local Object-Centric Semantic Mapping

## Project Context
GLOSM-Nav is an Object Goal Navigation framework designed to mitigate "spatial amnesia" inherent in reactive 2D mapping policies. It achieves zero-shot exploration by maintaining a persistent 3D Semantic Dictionary, storing voxelized point clouds and feature embeddings for discrete object instances across massive 3D environments. 

### Core Architecture & Capabilities
- **Object-Centric Navigation Policy:** Features an Acyclic Enforcer and Global CLIP Fallback mechanism to actively penalize frontier cyclic-looping and enable exploration until specific targets are verified.
- **ValueMap Fusion:** Utilizes FOV cone confidence projections for weighted-average spatial scoring, optimizing trajectory efficiency (SPL of 30.3%).
- **Hierarchical Vision Pipeline:** Combines YOLOv7, MobileSAM, and OpenCLIP/MetaCLIP to localize raw geometry into robust, view-invariant descriptors.
- **Dual-Similarity Geometric Tracking:** Uses 3D-IoU volumetric overlap and cosine-similarity hashing to consolidate duplicate detections into persistent world-frame instances.
- **RL Integration:** Incorporates a low-level PointNav motor policy trained via the VER distributed RL algorithm for collision-free routing.

## Current Objective: Multi-Floor Navigation (Branch: v6/staircase-navigation)
- **Goal:** Enable dynamic multi-floor map swapping by implementing 3D position tracking.
- **Challenge:** Overcoming "altitude-blindness" by extracting the true Z-coordinate (height) from the `gps_sensor` to allow the policy to create and switch between isolated 2D navigation maps per floor, while maintaining the global 3D Semantic Map.

## System Instructions & Constraints
- **GPU Server Constraint:** DO NOT attempt to run `python` scripts or deep learning models directly via `run_command` tools on this machine. The codebase must be executed on the user's remote GPU server.
- If you need to test or verify a script, provide the code block directly to the user so they can run it on the GPU server and return the results.
- Never try to import `torch`, `habitat`, or heavy libraries using the local terminal.
- DO NOT add changes to the .py files without the user's permission or his agreement to the proposed code changes. The user will provide an explicit green signal on when to add changes. 
- DO NOT add or change the code even if you think it will fix the issue, always ask for permission first.
- When the user says refer to other methods, he means to check the cloned git repos in the codebase folder, so only check there unless the user explicitly asks you to use an internet search. 
- DO NOT RUN python even in the terminal, it wont work for this project.