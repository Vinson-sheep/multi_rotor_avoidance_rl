# Multi-Rotor Obstacle Avoidance : Reinfocement Learning 1.1.3

## Latest Update
1. move some API to training_node, samplify training process.
2. remove server.sh
3. remove model/falco_01
4. add model/falco
5. remove model/EXP_iris

6. fix world train_env_3m_light's bug
7. fix bug of training_node.py
8. add cur_trap_num in testing_node
...


## log:
train empty_3m 50

fix train_env_3m_light 30 
-> critic loss 5

train train_env_3m_light 80
-> actor loss -23
-> critic loss 7

fix train_env_3m 30
-> critic loss 6

train train_env_3m 80
-> actor loss -9
-> critic loss 4.3 

fix empty_7m 10 (directly arrived, path curve)
-> critic loss 0.1

train empty_7m 20 (actor_lr = 3e-6)
-> actor loss -44
-> critic loss 0.2

fix train_env_7m 30
-> critic loss 6

train train_7m 80
-> actor loss -23
-> critic loss 4

fix empty_10m 10 ()
-> critic loss 0.12

train empty_10m 20 (actor_lr = 3e-6)
-> actor loss -45
-> critic loss 0.11

fix train_env_10m 30
-> critic loss 4.85

train train_env_10m 150
-> actor loss -28
-> critic loss 3




