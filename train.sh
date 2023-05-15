MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --real_robot=True \
                --utd_ratio=20 \
                --start_training=2000 \
                --max_steps=100000 \
                --config=configs/droq_config.py \
                --task=hover

# MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false py-spy record -o profile.svg --format speedscope --   python train_online.py --real_robot=True \
# MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --real_robot=True \