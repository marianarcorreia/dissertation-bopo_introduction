Set-Location "D:\Dissertação\dissertation-bopo_introduction"
$common = "--trials", "10", "--max-episodes", "100", "--validation-freq", "15", "--validation-size", "20", "--storage", "sqlite:///optuna.db", "--no-dashboard"

"=== optuna: om + gat (resume) ===" | Out-File -Append -Encoding utf8 sweep_logs/optuna_backbones.log
python -u main.py --mode optuna --representation om --gnn-type gat @common *>> sweep_logs/optuna_backbones.log

"=== optuna: om + gin ===" | Out-File -Append -Encoding utf8 sweep_logs/optuna_backbones.log
python -u main.py --mode optuna --representation om --gnn-type gin @common *>> sweep_logs/optuna_backbones.log

"=== optuna: om + transformer ===" | Out-File -Append -Encoding utf8 sweep_logs/optuna_backbones.log
python -u main.py --mode optuna --representation om --gnn-type transformer @common *>> sweep_logs/optuna_backbones.log

"=== optuna: oo + gin ===" | Out-File -Append -Encoding utf8 sweep_logs/optuna_backbones.log
python -u main.py --mode optuna --representation oo --gnn-type gin @common *>> sweep_logs/optuna_backbones.log

"=== optuna: oo + transformer ===" | Out-File -Append -Encoding utf8 sweep_logs/optuna_backbones.log
python -u main.py --mode optuna --representation oo --gnn-type transformer @common *>> sweep_logs/optuna_backbones.log

"=== ALL BACKBONE OPTUNA STUDIES COMPLETE ===" | Out-File -Append -Encoding utf8 sweep_logs/optuna_backbones.log
