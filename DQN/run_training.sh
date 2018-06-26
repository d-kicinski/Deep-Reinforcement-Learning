
# Experiment description:

# -render : Flaga mówiąca o tym czy renderować animacje
# -n_experiments : liczba uruchomień eksperymentów z różnymi seed
# -lr : learning rate
# --step_lr : co ile epizodów lr scheduler  ma zmniejszac lr
# -ln : liczba warstw ukrytych
# -ls : liczba unitów
# --target_update_freq :  co ile iteracji ma byc update target network

# --use_double_dqn
# więcej opcji zostało opisanych w pliku dqn.py


python src/dqn.py --env_name MountainCar-v0 \
	--render \
       --n_experiments 1 \
       --seed 1 \
       --episode_num 1001 --episode_len 200  \
       -lr 0.0005 --step_lr 300 --batch_size 64 -ln 1 -ls 64 \
       --target_update_freq 1000 \
       --replay_buffer_size 100000 \
       --exp_name best_params


