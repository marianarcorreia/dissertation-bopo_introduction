import math
import os
import shutil
import sys
import importlib
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.generator import generate_instance_list #gerador de instancia sinteticas
from src.parsedata import get_data, parse #parce de ficheiros de benchmark
import json #para ler ficheiros de configuração e resultados
# Load PyTorch dynamically so static analysis does not require the optional
# dependency to be installed in the interpreter used to inspect this module.
torch = importlib.import_module("torch")
from datetime import datetime #timestamp  para logs
import random
import numpy as np
import time #tempo de execução
from src.utils import build_validation_dataset, run_validation, OutputManager, get_test_dataset
from src.bopo_utils import run_behavior_cloning

_DBG = int(os.environ.get("FJSP_DEBUG", "0"))

def _dbg(level, *args, **kwargs):
    if _DBG >= level:
        print("[TRAIN]", *args, **kwargs)


def _resolve_representation_modules(representation: str):
    rep = representation.lower().strip()
    rep_map = {
        "oo": ("src.envo_o", "FJSPEnvOO", "src.bopo_oo", "BOPO"),
        "om": ("src.envheterogeneosmo", "FJSPEnvMO", "src.bopomo", "BOPO"),
        "ojm": ("src.env", "FJSSPEnv", "src.bopo", "BOPO"),
    }
    if rep not in rep_map:
        raise ValueError(f"Unsupported representation '{representation}'. Use one of: oo, om, ojm")

    env_module_name, env_class_name, bopo_module_name, bopo_class_name = rep_map[rep]
    env_module = importlib.import_module(env_module_name)
    bopo_module = importlib.import_module(bopo_module_name)
    env_class = getattr(env_module, env_class_name)
    bopo_class = getattr(bopo_module, bopo_class_name)
    return rep, env_class, bopo_class

#cria pasta para ficheiros se não existirem
os.makedirs('candidate_models', exist_ok=True)
os.makedirs('models', exist_ok=True)

#cria lista de instancias sinteticas com base na configuração
def generate_train_instances(train_config):
    _dbg(1, f"generate_train_instances | config={train_config}")
    list_instances = generate_instance_list(**train_config)
    instances = []
    for instance in list_instances:
        jobs, operations, info, maximum = get_data(parse(instance))
        instances.append({ "jobs": jobs, "operations": operations, "maximum": maximum, "num_machines": info["machinesNb"]})
    _dbg(1, f"  generated {len(instances)} training instance(s)")
    return instances


def train(max_episodes = 100,new_freq=500, n_cases = 100, mask_option=0, sel_k=100, B=64, K=16, use_greedy=True,lr=0.0005, hidden_channels=128,num_layers = 3, heads = 3,j_max = 10, j_min = 8, m_max = 10, m_min = 5, op_max = 6, max_processing = 100,
         checkpoint_smooth_window=3, warm_start_steps=200, lr_min_ratio=0.2, seed=None, logp_norm="mean", exclude_greedy_from_loss=True, jm_design="baseline", representation="oo", gnn_type="gat", validation_freq=20, validation_size=20, dbg_fn=None, run_name="train_run"):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    #inicialização
    #NOTA: "max_episodes" passou a contar passos de treino do BOPO, não episódios PPO.
    #Cada passo amostra B trajetórias paralelas da MESMA instância e faz UMA atualização
    #(sem critic, sem buffer multi-episódio) - ver src/ppomo.py, src/ppo.py, src/ppoo_o.py.
    print("=" * 60)
    print("[TRAIN] Training started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"[TRAIN] Hyperparams | max_steps={max_episodes} | new_freq={new_freq}")
    print(f"[TRAIN]             | n_cases={n_cases} | B={B} | K={K} | use_greedy={use_greedy} | lr={lr} (decaying to {lr*lr_min_ratio:.2e})")
    print(f"[TRAIN]             | warm_start_steps={warm_start_steps} | checkpoint_smooth_window={checkpoint_smooth_window} | seed={seed}")
    print(f"[TRAIN]             | hidden_channels={hidden_channels} | num_layers={num_layers} | heads={heads}")
    print(f"[TRAIN]             | logp_norm={logp_norm} | exclude_greedy_from_loss={exclude_greedy_from_loss}")
    print(f"[TRAIN] Representation | {representation} | GNN | {gnn_type} | jm_design | {jm_design}")
    print(f"[TRAIN] Problem size | jobs=[{j_min},{j_max}] | machines=[{m_min},{m_max}] | ops_per_job=[5,{op_max}] | max_proc={max_processing}")
    print("=" * 60)
    rep_name, EnvClass, BOPOClass = _resolve_representation_modules(representation)
    # only passed when set, so om/oo envs (which don't take it) are built exactly as before
    jm_kwargs = {}
    if jm_design != "baseline":
        if rep_name != "ojm":
            raise ValueError(f"jm_design={jm_design!r} only applies to the ojm representation")
        jm_kwargs = {"jm_design": jm_design}
    output_manager = OutputManager(output_dir="results", run_name=run_name)
    print(f"[TRAIN] Output run folder: {output_manager.run_dir}")
    run_start_time = time.time()
    validation_set = build_validation_dataset(sample_size=validation_size, dbg_fn=_dbg)
    print(f"[TRAIN] Validation config | every={validation_freq} steps | instances={len(validation_set)} (representative subset)")
    _dbg(1, f"Loaded validation set from val/instances + val/solutions: {len(validation_set)} instance(s)")

    #ambiente de validação e extrai o metadata (informação sobre o ambiente, como número de jobs, máquinas, etc.)
    val_env = EnvClass(validation_set, mask_option, sel_k, **jm_kwargs)
    s = val_env.reset()
    metadata = s.metadata()

    #define e gera instancias de treino
    train_config = {
        "n_cases": n_cases,
        "range_jobs": (j_min, j_max),
        "range_machines": (m_min, m_max),
        "range_op_per_job": (5, op_max),
        "max_processing": max_processing
    }

    instances = generate_train_instances(train_config)

    #cria ambiente de treino
    env = EnvClass(instances, mask_option, sel_k, **jm_kwargs)
    #cria agente BOPO (só ator, sem critic - ver src/bopo_utils.py para a SROLoss)
    bopo_agent = BOPOClass(lr, env, metadata, hidden_channels, num_layers, heads, B, K, use_greedy, gnn_type=gnn_type,
                           logp_norm=logp_norm, exclude_greedy_from_loss=exclude_greedy_from_loss, **jm_kwargs)

    if warm_start_steps > 0:
        print(f"[TRAIN] Warm-start (behavior cloning vs. dispatch heuristic) | {warm_start_steps} step(s)...")
        for ws in range(1, warm_start_steps + 1):
            ws_instance = random.randrange(len(env.instances))
            bc_loss, bc_steps = run_behavior_cloning(env, bopo_agent.policy, bopo_agent.optimizer, ws_instance)
            if ws % max(1, warm_start_steps // 20) == 0 or ws == warm_start_steps:
                print(f"[TRAIN][WARMSTART] step {ws}/{warm_start_steps} | instance={ws_instance} | bc_loss={bc_loss:.4f} | decisions={bc_steps}")
        print("[TRAIN] Warm-start complete, switching to BOPO preference optimization.")
        print("-" * 60)

    print(f"[TRAIN] BOPO agent ready. Starting training loop for {max_episodes} step(s)...")
    print("-" * 60)
    validation_history = []
    episode_metrics = []
    update_metrics = []
    best_avg_gap = float('inf')  # tracks the SMOOTHED avg_gap (see checkpoint_smooth_window)
    best_q80_gap = float('inf')
    best_difference = 0.0
    best_model_path = None
    validation_gap_window: list = []
    step_number = 1
    #ordem (baralhada) das instâncias de treino dentro de cada "epoch" sobre o pool atual
    instance_order = []
    # training loop
    while True:
        step_start_time = time.time()
        if step_number > max_episodes:
            break
        print(f"[TRAIN] Step {step_number}/{max_episodes} | {datetime.now().strftime('%H:%M:%S')}")

        # Cosine-decay the optimizer's lr over the BOPO phase (warm-start already ran at
        # the full, undecayed lr).
        progress = 0.0 if max_episodes <= 1 else (step_number - 1) / (max_episodes - 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        current_lr = lr * (lr_min_ratio + (1 - lr_min_ratio) * cosine)
        for param_group in bopo_agent.optimizer.param_groups:
            param_group['lr'] = current_lr

        if not instance_order:
            instance_order = list(range(len(env.instances)))
            random.shuffle(instance_order)
        instance_index = instance_order.pop()

        #amostra B trajetórias da mesma instância, auto-rotula pares e faz UMA atualização
        loss, best_ms, rounds, entropy_stats = bopo_agent.update(instance_index)
        update_duration_sec = time.time() - step_start_time
        _dbg(1, f"  Step {step_number} finished | instance={instance_index} | decision_rounds={rounds} | best_makespan={best_ms:.2f} | loss={loss:.6f}")
        print(f"[TRAIN] Step {step_number} | Loss={loss:.6f} | Best makespan in group={best_ms:.2f}")

        update_entry = {
            "episode": step_number,
            "actor_loss": float(loss),
            "lr": float(current_lr),
            "update_duration_sec": float(update_duration_sec),
            **entropy_stats,
        }
        update_metrics = output_manager.append_update_metrics(update_metrics, update_entry)

        #gera nova instancia de treino se estiver na altura de renovar o pool
        if step_number % new_freq == 0:
            print(f"[TRAIN] Refreshing training instances (step {step_number})...")
            instances = generate_train_instances(train_config)
            env = EnvClass(instances, mask_option, sel_k, **jm_kwargs)
            bopo_agent.env = env
            instance_order = []

        validation_avg_gap = None
        validation_std_gap = None
        validation_q80_gap = None
        if step_number % validation_freq == 0 or step_number == max_episodes:
            val_metrics = run_validation(bopo_agent, val_env, validation_set, step_number, dbg_fn=_dbg)
            validation_avg_gap = float(val_metrics["avg_gap"])
            validation_std_gap = float(val_metrics["std_gap"])
            validation_q80_gap = float(val_metrics["q80_gap"])
            best_q80_gap = min(best_q80_gap, validation_q80_gap)

            # Moving average over the last checkpoint_smooth_window validation checkpoints
            # (including this one) - the actual criterion for "improved"/saved below, so a
            # single lucky checkpoint on only validation_size instances can't get crowned
            # "best" on its own; it has to be part of a sustained good streak.
            validation_gap_window.append(validation_avg_gap)
            if len(validation_gap_window) > checkpoint_smooth_window:
                validation_gap_window.pop(0)
            smoothed_avg_gap = sum(validation_gap_window) / len(validation_gap_window)

            history_entry = {
                "episode": step_number,
                "avg_gap": validation_avg_gap,
                "smoothed_avg_gap": smoothed_avg_gap,
                "std_gap": validation_std_gap,
                "q80_gap": validation_q80_gap,
                "all_gaps": val_metrics["all_gaps"],
                "instances": [v["name"] for v in validation_set],
            }
            validation_history = output_manager.append_validation_history(validation_history, history_entry)
            output_manager.plot_validation_gap(validation_history)

            if smoothed_avg_gap < best_avg_gap:
                improvement = 0.0 if np.isinf(best_avg_gap) else best_avg_gap - smoothed_avg_gap
                best_difference = max(best_difference, float(improvement))
                best_avg_gap = smoothed_avg_gap

                name = str(int(random.uniform(10**10, 10**15)))
                print(f"[TRAIN] Validation improved | smoothed_avg_gap={best_avg_gap:.4f} (raw={validation_avg_gap:.4f}) | saving candidate: {name}.pth")
                with open('candidate_models/model_params.json', 'r') as infile:
                    model_params = json.load(infile)

                model_params.append({
                    "name": name + ".pth",
                    "representation": rep_name,
                    "sel_k": sel_k,
                    "mask_option": mask_option,
                    "num_layers": num_layers,
                    "hidden_channels": hidden_channels,
                    "heads": heads,
                    "gnn_type": gnn_type,
                    "jm_design": jm_design,
                    "all_val_results": val_metrics["all_gaps"],
                    "avg_gap": val_metrics["avg_gap"],
                    "smoothed_avg_gap": smoothed_avg_gap,
                    "std_gap": val_metrics["std_gap"],
                    "q80_gap": val_metrics["q80_gap"],
                    "validation_instances": [v["name"] for v in validation_set],
                    "episode": step_number,
                })

                with open('candidate_models/model_params.json', 'w') as outfile:
                    json.dump(model_params, outfile)

                best_model_path = "candidate_models/" + name + ".pth"
                bopo_agent.save(best_model_path)
                shutil.copy(best_model_path, "models/" + name + ".pth")

        episode_entry = {
            "episode": step_number,
            "steps": int(rounds),
            "makespan": float(best_ms),
            "actor_loss": float(loss),
            "update_duration_sec": float(update_duration_sec),
            "validation_avg_gap": validation_avg_gap,
            "validation_std_gap": validation_std_gap,
            "validation_q80_gap": validation_q80_gap,
            "best_validation_avg_gap": float(best_avg_gap) if not np.isinf(best_avg_gap) else None,
            "best_validation_q80_gap": float(best_q80_gap) if not np.isinf(best_q80_gap) else None,
            "episode_duration_sec": float(time.time() - step_start_time),
            **entropy_stats,
        }
        episode_metrics = output_manager.append_episode_metrics(episode_metrics, episode_entry)

        step_number += 1
    env.close()
    val_env.close()
    actor_param_count = int(sum(p.numel() for p in bopo_agent.policy.actor.parameters()))
    plot_episode_outputs = output_manager.plot_episode_metrics(episode_metrics)
    plot_update_outputs = output_manager.plot_update_metrics(update_metrics)

    # One-time, final report on the held-out TEST split (see src/utils/validation_utils.py:
    # get_test_dataset). Disjoint from the validation split used above for checkpoint
    # selection, and never looked at until now - best_avg_gap/best_q80_gap above answer
    # "which checkpoint did we pick", this answers "how good is that checkpoint", without
    # the two questions being asked of the same data.
    test_avg_gap = None
    test_std_gap = None
    test_q80_gap = None
    test_all_gaps = None
    if best_model_path is not None:
        print(f"[TRAIN] Evaluating best checkpoint ({best_model_path}) on the held-out test split...")
        test_set = get_test_dataset(sample_size=validation_size, dbg_fn=_dbg)
        test_env = EnvClass(test_set, mask_option, sel_k, **jm_kwargs)
        bopo_agent.load(best_model_path)
        test_metrics = run_validation(bopo_agent, test_env, test_set, dbg_fn=_dbg,
                                       print_fn=lambda msg: print(msg.replace("[TRAIN][VAL]", "[TRAIN][TEST]")))
        test_env.close()
        test_avg_gap = float(test_metrics["avg_gap"])
        test_std_gap = float(test_metrics["std_gap"])
        test_q80_gap = float(test_metrics["q80_gap"])
        test_all_gaps = test_metrics["all_gaps"]
        output_manager.save_test_metrics({
            "best_model_path": best_model_path,
            "avg_gap": test_avg_gap,
            "std_gap": test_std_gap,
            "q80_gap": test_q80_gap,
            "all_gaps": test_all_gaps,
            "instances": [t["name"] for t in test_set],
        })
    else:
        print("[TRAIN] No checkpoint ever improved validation avg_gap - skipping held-out test evaluation.")

    summary = {
        "run_name": run_name,
        "representation": rep_name,
        "gnn_type": gnn_type,
        "num_layers": int(num_layers),
        "mask_option": int(mask_option),
        "sel_k": int(sel_k),
        "seed": seed,
        "logp_norm": logp_norm,
        "exclude_greedy_from_loss": bool(exclude_greedy_from_loss),
        "jm_design": jm_design,
        "run_dir": output_manager.run_dir,
        "max_episodes": int(max_episodes),
        "episodes_completed": int(len(episode_metrics)),
        "updates_completed": int(len(update_metrics)),
        "best_validation_avg_gap": float(best_avg_gap) if not np.isinf(best_avg_gap) else None,
        "best_validation_q80_gap": float(best_q80_gap) if not np.isinf(best_q80_gap) else None,
        "best_model_path": best_model_path,
        "test_avg_gap": test_avg_gap,
        "test_std_gap": test_std_gap,
        "test_q80_gap": test_q80_gap,
        "actor_param_count": actor_param_count,
        "best_difference": float(best_difference),
        "total_runtime_sec": float(time.time() - run_start_time),
        "validation_metrics_file": output_manager.history_path(),
        "episode_metrics_file": output_manager.episode_metrics_path(),
        "update_metrics_file": output_manager.update_metrics_path(),
        "plot_episode_outputs": plot_episode_outputs,
        "plot_update_outputs": plot_update_outputs,
    }
    output_manager.save_run_summary(summary)
    output_manager.log("Training ended. Validation history and plots updated.", tag="TRAIN")

    return best_difference

def test_model(model_name, folder, filename, models_file="models/model_params.json", representation="oo"):

    start_time = [time.time()]
    folder_path = folder
    test_instances = []
    
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'r') as file:
        contents = file.read()
        jobs, operations, info, maximum = get_data(parse(contents))
        test_instances.append({"jobs": jobs, "operations": operations, "maximum": maximum, "num_machines": info["machinesNb"]})

    models = [model_name]

    best_results = [float('inf')]*len(test_instances)

    all_results = []
    with open(models_file, 'r') as infile:
        model_params = json.load(infile)
    models_dir = os.path.dirname(models_file) or "."

    with torch.no_grad():
        for m in models:
            model_results = []
            param = [p for p in model_params if p["name"]==m][0]
            model_rep = param.get("representation", representation)
            _, ModelEnvClass, ModelBOPOClass = _resolve_representation_modules(model_rep)
            # metadata must come from an env built with this model's own mask_option/sel_k —
            # those change the graph's feature dimensions, so reusing metadata from a
            # differently-configured env produces mismatched GNN layer shapes at load time.
            jm_design = param.get("jm_design", "baseline")
            jm_kwargs = {} if jm_design == "baseline" else {"jm_design": jm_design}
            test_env = ModelEnvClass(test_instances, param["mask_option"], param["sel_k"], **jm_kwargs)
            metadata = test_env.reset().metadata()
            t_ppo_agent = ModelBOPOClass(0.001, test_env, metadata, param["hidden_channels"], param["num_layers"], param["heads"],
                                          gnn_type=param.get("gnn_type", "gat"), **jm_kwargs)
            # preTrained weights directory
            t_ppo_agent.load(os.path.join(models_dir, m))

            start_time = time.time()

            for i in range(len(test_instances)):
                v_state = test_env.reset()
                for q in range(1, 10**10):
                    v_action = t_ppo_agent.select_action(v_state, 2, q)
                    v_state, v_reward, v_done, _ = test_env.step(v_action)
                    if v_done:
                        model_results.append(test_env.mk)
                        if test_env.mk< best_results[i]:
                            best_results[i] = test_env.mk
                        break


            all_results.append(model_results)
    

    return best_results, start_time, time.time()
