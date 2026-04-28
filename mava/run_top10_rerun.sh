#!/bin/bash

# Reads WANDB_API_KEY from env. Set via: export WANDB_API_KEY=...
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your environment before running}"

source .venv/bin/activate

# Top 10 runs from ppo_crl_continuous-sweep-04-24-2026, sorted by evaluator/success/mean.
# Each is rerun with seeds 0-4 for statistical robustness.
# All runs share: energy_fn=dot, rollout_length=128, num_updates=1500,
#   use_achieved_goal=true, use_gae=false, use_reinforce=false, reward_advantage_coeff=0

COMMON_ARGS=(
    env=jaxnav
    arch.num_envs=256
    arch.num_evaluation=25
    arch.num_eval_episodes=512
    arch.evaluation_greedy=false
    system.rollout_length=128
    system.num_updates=1500
    system.rep_size=64
    system.vf_coef=0.5
    system.gae_lambda=0.95
    system.ent_coef_end=0.01
    system.log_std_min=-5
    system.log_std_max=2
    system.max_grad_norm=1
    system.energy_fn=dot
    system.use_achieved_goal=true
    system.use_gae=false
    system.reward_advantage_coeff=0.0
    system.use_reinforce=false
    env.goal_type=distance
    env.eval_metric=success
    logger.loggers.wandb.enabled=true
    logger.loggers.wandb.project=jaxnav
    logger.loggers.wandb.entity=asim_osman-aimst-university
    'logger.loggers.wandb.tags=[jaxnav,ppo_crl_continuous,no-reward,top10-rerun]'
    logger.loggers.wandb.group=ppo_crl_continuous-top10-rerun-04-25-2026
)

# ── Rank 1 | success=0.5723 | id=2fc1q1y3
# contrastive_loss_fn=fwd_infonce, gamma=0.9999, num_epochs=8, use_adaptive_entropy=false
uv run mava/systems/icrl/anakin/ppo_crl_continuous.py -m \
    system.seed=0,1,2,3,4 \
    "${COMMON_ARGS[@]}" \
    system.actor_lr=1.5119633084265482e-06 \
    system.q_lr=3.791367239991691e-06 \
    system.gamma=0.9999 \
    system.clip_eps=0.12749272261772338 \
    system.ent_coef=0.25910466133652155 \
    system.num_epochs=8 \
    system.batch_size=256 \
    system.num_mc_samples=16 \
    system.contrastive_loss_fn=fwd_infonce \
    system.contrastive_temperature=2.3940798909533996 \
    system.logsumexp_penalty_coeff=0.11647839049071546 \
    system.num_critic_warmup_epochs=1 \
    system.use_adaptive_entropy=false \
    system.target_entropy=3.995752377443284 \
    system.lr_linear_decay=false \
    logger.loggers.wandb.run_name=top10-rank1-jaxnav-rerun

# # ── Rank 2 | success=0.5469 | id=b7xq1kuy
# # contrastive_loss_fn=bwd_infonce, gamma=0.98, num_epochs=2, use_adaptive_entropy=true
# uv run mava/systems/icrl/anakin/ppo_crl_continuous.py -m \
#     system.seed=0,1,2,3,4 \
#     "${COMMON_ARGS[@]}" \
#     system.actor_lr=1.1400863701127326e-05 \
#     system.q_lr=2.3423849847112885e-05 \
#     system.gamma=0.98 \
#     system.clip_eps=0.11375540844608736 \
#     system.ent_coef=0.2737029166028468 \
#     system.num_epochs=2 \
#     system.batch_size=256 \
#     system.num_mc_samples=16 \
#     system.contrastive_loss_fn=bwd_infonce \
#     system.contrastive_temperature=3.726982248607548 \
#     system.logsumexp_penalty_coeff=0.03037864935284442 \
#     system.num_critic_warmup_epochs=1 \
#     system.use_adaptive_entropy=true \
#     system.target_entropy=4.819459112971966 \
#     system.lr_linear_decay=false \
#     logger.loggers.wandb.run_name=top10-rank2-jaxnav-rerun

# # ── Rank 3 | success=0.5195 | id=c7im9khf
# # contrastive_loss_fn=binary_nce, gamma=0.9999, num_epochs=8, use_adaptive_entropy=true
# uv run mava/systems/icrl/anakin/ppo_crl_continuous.py -m \
#     system.seed=0,1,2,3,4 \
#     "${COMMON_ARGS[@]}" \
#     system.actor_lr=5.602265006952132e-06 \
#     system.q_lr=3.3622288498144896e-06 \
#     system.gamma=0.9999 \
#     system.clip_eps=0.48235634835209173 \
#     system.ent_coef=0.27741331957092596 \
#     system.num_epochs=8 \
#     system.batch_size=256 \
#     system.num_mc_samples=32 \
#     system.contrastive_loss_fn=binary_nce \
#     system.contrastive_temperature=3.461270671298143 \
#     system.logsumexp_penalty_coeff=0.09426917282238752 \
#     system.num_critic_warmup_epochs=2 \
#     system.use_adaptive_entropy=true \
#     system.target_entropy=1.4332991068892862 \
#     system.lr_linear_decay=false \
#     logger.loggers.wandb.run_name=top10-rank3-jaxnav-rerun

# # ── Rank 4 | success=0.5156 | id=wtdqk38g
# # contrastive_loss_fn=bwd_infonce, gamma=0.9999, num_epochs=8, use_adaptive_entropy=true
# uv run mava/systems/icrl/anakin/ppo_crl_continuous.py -m \
#     system.seed=0,1,2,3,4 \
#     "${COMMON_ARGS[@]}" \
#     system.actor_lr=3.948047484912802e-06 \
#     system.q_lr=7.811997185358953e-06 \
#     system.gamma=0.9999 \
#     system.clip_eps=0.14435366352643647 \
#     system.ent_coef=0.21239140739264223 \
#     system.num_epochs=8 \
#     system.batch_size=256 \
#     system.num_mc_samples=32 \
#     system.contrastive_loss_fn=bwd_infonce \
#     system.contrastive_temperature=2.926704846133571 \
#     system.logsumexp_penalty_coeff=0.01912690140646517 \
#     system.num_critic_warmup_epochs=2 \
#     system.use_adaptive_entropy=true \
#     system.target_entropy=4.093906020627942 \
#     system.lr_linear_decay=false \
#     logger.loggers.wandb.run_name=top10-rank4-jaxnav-rerun

# # ── Rank 5 | success=0.5059 | id=iv52hob7
# # contrastive_loss_fn=fwd_infonce, gamma=0.9999, num_epochs=8, use_adaptive_entropy=false
# uv run mava/systems/icrl/anakin/ppo_crl_continuous.py -m \
#     system.seed=0,1,2,3,4 \
#     "${COMMON_ARGS[@]}" \
#     system.actor_lr=3.45454898401252e-06 \
#     system.q_lr=4.989300807107994e-06 \
#     system.gamma=0.9999 \
#     system.clip_eps=0.14517198922716126 \
#     system.ent_coef=0.2229910192838783 \
#     system.num_epochs=8 \
#     system.batch_size=256 \
#     system.num_mc_samples=16 \
#     system.contrastive_loss_fn=fwd_infonce \
#     system.contrastive_temperature=2.726708582147127 \
#     system.logsumexp_penalty_coeff=0.0140776820366976 \
#     system.num_critic_warmup_epochs=1 \
#     system.use_adaptive_entropy=false \
#     system.target_entropy=4.175905364020111 \
#     system.lr_linear_decay=false \
#     logger.loggers.wandb.run_name=top10-rank5-jaxnav-rerun

# # ── Rank 6 | success=0.4980 | id=u5yzb56b
# # contrastive_loss_fn=binary_nce, gamma=0.9999, num_epochs=8, use_adaptive_entropy=true
# uv run mava/systems/icrl/anakin/ppo_crl_continuous.py -m \
#     system.seed=0,1,2,3,4 \
#     "${COMMON_ARGS[@]}" \
#     system.actor_lr=6.564268628739694e-06 \
#     system.q_lr=9.159262317348824e-06 \
#     system.gamma=0.9999 \
#     system.clip_eps=0.12789041968732087 \
#     system.ent_coef=0.24060321974264048 \
#     system.num_epochs=8 \
#     system.batch_size=256 \
#     system.num_mc_samples=32 \
#     system.contrastive_loss_fn=binary_nce \
#     system.contrastive_temperature=2.9741819461111043 \
#     system.logsumexp_penalty_coeff=0.07800728302545767 \
#     system.num_critic_warmup_epochs=2 \
#     system.use_adaptive_entropy=true \
#     system.target_entropy=4.577796334157728 \
#     system.lr_linear_decay=false \
#     logger.loggers.wandb.run_name=top10-rank6-jaxnav-rerun

# # ── Rank 7 | success=0.4941 | id=qcvh19ae
# # contrastive_loss_fn=fwd_infonce, gamma=0.9999, num_epochs=8, use_adaptive_entropy=true
# uv run mava/systems/icrl/anakin/ppo_crl_continuous.py -m \
#     system.seed=0,1,2,3,4 \
#     "${COMMON_ARGS[@]}" \
#     system.actor_lr=1.9162248786913294e-06 \
#     system.q_lr=1.2792085022271152e-05 \
#     system.gamma=0.9999 \
#     system.clip_eps=0.33022572647545106 \
#     system.ent_coef=0.27392514533295986 \
#     system.num_epochs=8 \
#     system.batch_size=256 \
#     system.num_mc_samples=32 \
#     system.contrastive_loss_fn=fwd_infonce \
#     system.contrastive_temperature=2.20432193206924 \
#     system.logsumexp_penalty_coeff=0.13894203953072087 \
#     system.num_critic_warmup_epochs=2 \
#     system.use_adaptive_entropy=true \
#     system.target_entropy=4.56589078362585 \
#     system.lr_linear_decay=false \
#     logger.loggers.wandb.run_name=top10-rank7-jaxnav-rerun

# # ── Rank 8 | success=0.4863 | id=4ujs5llo
# # contrastive_loss_fn=fwd_infonce, gamma=0.9999, num_epochs=8, use_adaptive_entropy=false
# uv run mava/systems/icrl/anakin/ppo_crl_continuous.py -m \
#     system.seed=0,1,2,3,4 \
#     "${COMMON_ARGS[@]}" \
#     system.actor_lr=8.910893094634089e-06 \
#     system.q_lr=1.4230507654113668e-05 \
#     system.gamma=0.9999 \
#     system.clip_eps=0.13427058225777008 \
#     system.ent_coef=0.28966633594891067 \
#     system.num_epochs=8 \
#     system.batch_size=256 \
#     system.num_mc_samples=16 \
#     system.contrastive_loss_fn=fwd_infonce \
#     system.contrastive_temperature=4.406680612285592 \
#     system.logsumexp_penalty_coeff=0.21117290662396593 \
#     system.num_critic_warmup_epochs=1 \
#     system.use_adaptive_entropy=false \
#     system.target_entropy=1.2287918299604432 \
#     system.lr_linear_decay=false \
#     logger.loggers.wandb.run_name=top10-rank8-jaxnav-rerun

# # ── Rank 9 | success=0.4219 | id=0jn16czs
# # contrastive_loss_fn=bwd_infonce, gamma=0.99, num_epochs=6, batch_size=512, max_grad_norm=10
# uv run mava/systems/icrl/anakin/ppo_crl_continuous.py -m \
#     system.seed=0,1,2,3,4 \
#     "${COMMON_ARGS[@]}" \
#     system.max_grad_norm=10 \
#     system.actor_lr=3.0458275267904657e-05 \
#     system.q_lr=4.578335757382371e-05 \
#     system.gamma=0.99 \
#     system.clip_eps=0.10205945836370356 \
#     system.ent_coef=0.1284435297213816 \
#     system.num_epochs=6 \
#     system.batch_size=512 \
#     system.num_mc_samples=32 \
#     system.contrastive_loss_fn=bwd_infonce \
#     system.contrastive_temperature=4.034675573876615 \
#     system.logsumexp_penalty_coeff=0.7540972905676309 \
#     system.num_critic_warmup_epochs=2 \
#     system.use_adaptive_entropy=true \
#     system.target_entropy=3.2199850133763896 \
#     system.lr_linear_decay=true \
#     logger.loggers.wandb.run_name=top10-rank9-jaxnav-rerun

# # ── Rank 10 | success=0.4062 | id=ha0vfcc1
# # contrastive_loss_fn=bwd_infonce, gamma=0.99, num_epochs=6, max_grad_norm=10
# uv run mava/systems/icrl/anakin/ppo_crl_continuous.py -m \
#     system.seed=0,1,2,3,4 \
#     "${COMMON_ARGS[@]}" \
#     system.max_grad_norm=10 \
#     system.actor_lr=1.4467053264901472e-05 \
#     system.q_lr=3.233547957550202e-05 \
#     system.gamma=0.99 \
#     system.clip_eps=0.1733221423256303 \
#     system.ent_coef=0.11344415121534054 \
#     system.num_epochs=6 \
#     system.batch_size=256 \
#     system.num_mc_samples=16 \
#     system.contrastive_loss_fn=bwd_infonce \
#     system.contrastive_temperature=4.4960895471572195 \
#     system.logsumexp_penalty_coeff=0.724726025755096 \
#     system.num_critic_warmup_epochs=2 \
#     system.use_adaptive_entropy=true \
#     system.target_entropy=3.3617885107282683 \
#     system.lr_linear_decay=false \
#     logger.loggers.wandb.run_name=top10-rank10-jaxnav-rerun
