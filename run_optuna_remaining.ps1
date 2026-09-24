Set-Location "D:\Dissertação\dissertation-bopo_introduction"
"=== resuming optuna sweep: oo (3 more trials) ===" | Out-File -Append -Encoding utf8 sweep_logs/optuna_full_sweep.log
python -u main.py --mode optuna --representation oo --trials 3 --max-episodes 100 --validation-freq 15 --validation-size 20 --storage sqlite:///optuna.db --no-dashboard *>> sweep_logs/optuna_full_sweep.log
"=== optuna sweep: om (10 trials) ===" | Out-File -Append -Encoding utf8 sweep_logs/optuna_full_sweep.log
python -u main.py --mode optuna --representation om --trials 10 --max-episodes 100 --validation-freq 15 --validation-size 20 --storage sqlite:///optuna.db --no-dashboard *>> sweep_logs/optuna_full_sweep.log
"=== ALL OPTUNA STUDIES COMPLETE (ojm skipped per instruction) ===" | Out-File -Append -Encoding utf8 sweep_logs/optuna_full_sweep.log
