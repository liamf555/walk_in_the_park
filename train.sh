echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --real_robot=True \
                --utd_ratio=20 \
                --start_training=1300 \
                --max_steps=100000 \
                --config=configs/droq_config.py \
                --task=knife \
                --save_state=True \
                --acro_init=False \
                --control_frequency=20 \
                --wind=True

# MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false py-spy record -o profile.svg --format speedscope --   python train_online.py --real_robot=True \
# MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --real_robot=True \