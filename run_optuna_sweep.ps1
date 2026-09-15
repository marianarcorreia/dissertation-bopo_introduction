# Full Optuna sweep: all representations (oo, om, ojm) x both valdata strategies (our, gen).
# 500 trials/representation, 250 episodes/trial, validation every 10 episodes.
# Each command tunes all three representations sequentially (see param.py: run_tuning loops
# tune_representation() over --representations). --storage makes each sweep resumable if
# interrupted (rerun the same command and Optuna picks up where it left off).

python main.py --mode optuna --representation all `
    --max-episodes 250 --trials 500 --validation-freq 10 `
    --valdata our --storage optuna_our.db --run-name optuna_our

python main.py --mode optuna --representation all `
    --max-episodes 250 --trials 500 --validation-freq 10 `
    --valdata gen --storage optuna_gen.db --run-name optuna_gen
