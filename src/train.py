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
from src.utils import build_validation_dataset, run_validation, OutputManager

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

    env_module_name, env_class_name, ppo_module_name, ppo_class_name = rep_map[rep]
    env_module = importlib.import_module(env_module_name)
    ppo_module = importlib.import_module(ppo_module_name)
    env_class = getattr(env_module, env_class_name)
    ppo_class = getattr(ppo_module, ppo_class_name)
    return rep, env_class, ppo_class

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

#o código abaixo é o código original do gerador de instancias, mantido para referência e possível reutilização futura
def train(max_episodes = 10,
             new_freq=1, n_cases = 100, mask_option=1, sel_k=1, B=64, K=16, use_greedy=True, lr=0.0001, hidden_channels=128, num_layers = 1, heads = 3
         ,j_max = 15, j_min = 5, m_max = 13, m_min = 4, op_max = 9, max_processing = 25,
         validation_freq=10, validation_size=20, run_name="train_run", representation="oo", gnn_type="gat"):
    #inicialização
    #NOTA: "max_episodes" passou a contar passos de treino do BOPO, não episódios PPO.
    #Cada passo amostra B trajetórias paralelas da MESMA instância e faz UMA atualização
    #(sem critic, sem buffer multi-episódio) - ver src/ppomo.py, src/ppo.py, src/ppoo_o.py.
    print("=" * 60)
    print("[TRAIN] Training started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"[TRAIN] Hyperparams | max_steps={max_episodes} | new_freq={new_freq}")
    print(f"[TRAIN]             | n_cases={n_cases} | B={B} | K={K} | use_greedy={use_greedy} | lr={lr}")
    print(f"[TRAIN]             | hidden_channels={hidden_channels} | num_layers={num_layers} | heads={heads}")
    print(f"[TRAIN] Representation | {representation} | GNN | {gnn_type}")
    print(f"[TRAIN] Problem size | jobs=[{j_min},{j_max}] | machines=[{m_min},{m_max}] | ops_per_job=[4,{op_max}] | max_proc={max_processing}")
    print("=" * 60)
    rep_name, EnvClass, BOPOClass = _resolve_representation_modules(representation)
    output_manager = OutputManager(output_dir="results", run_name=run_name)
    print(f"[TRAIN] Output run folder: {output_manager.run_dir}")
    run_start_time = time.time()
    validation_set = build_validation_dataset(sample_size=validation_size, dbg_fn=_dbg)
    print(f"[TRAIN] Validation config | every={validation_freq} steps | instances={len(validation_set)} (representative subset)")
    _dbg(1, f"Loaded validation set from val/instances + val/solutions: {len(validation_set)} instance(s)")

    #ambiente de validação e extrai o metadata (informação sobre o ambiente, como número de jobs, máquinas, etc.)
    val_env = EnvClass(validation_set, mask_option, sel_k)
    s = val_env.reset()
    metadata = s.metadata()

    max_episodes = max_episodes
    new_freq = new_freq

    #define e gera instancias de treino
    train_config = {
        "n_cases": n_cases,
        "range_jobs": (j_min, j_max),
        "range_machines": (m_min, m_max),
        "range_op_per_job": (4, op_max),
        "max_processing": max_processing
    }

    instances = generate_train_instances(train_config)

    #cria ambiente de treino
    env = EnvClass(instances, mask_option, sel_k)
    #cria agente BOPO (só ator, sem critic - ver src/bopo_utils.py para a SROLoss)
    bopo_agent = BOPOClass(lr, env, metadata, hidden_channels, num_layers, heads, B, K, use_greedy, gnn_type=gnn_type)
    print(f"[TRAIN] BOPO agent ready. Starting training loop for {max_episodes} step(s)...")
    print("-" * 60)
    validation_history = []
    episode_metrics = []
    update_metrics = []
    best_avg_gap = float('inf')
    best_q80_gap = float('inf')
    best_difference = 0.0
    step_number = 1
    #ordem (baralhada) das instâncias de treino dentro de cada "epoch" sobre o pool atual
    instance_order = []
    # training loop
    while True:
        step_start_time = time.time()
        if step_number > max_episodes:
            break
        print(f"[TRAIN] Step {step_number}/{max_episodes} | {datetime.now().strftime('%H:%M:%S')}")

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
            "policy_loss": float(loss),  # BOPO não tem critic separado - a actor loss É a policy loss
            "critic_loss": 0.0,  # BOPO não tem critic
            "update_duration_sec": float(update_duration_sec),
            **entropy_stats,
        }
        update_metrics = output_manager.append_update_metrics(update_metrics, update_entry)

        #gera nova instancia de treino se estiver na altura de renovar o pool
        if step_number % new_freq == 0:
            print(f"[TRAIN] Refreshing training instances (step {step_number})...")
            instances = generate_train_instances(train_config)
            env = EnvClass(instances, mask_option, sel_k)
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
            history_entry = {
                "episode": step_number,
                "avg_gap": validation_avg_gap,
                "std_gap": validation_std_gap,
                "q80_gap": validation_q80_gap,
                "all_gaps": val_metrics["all_gaps"],
                "instances": [v["name"] for v in validation_set],
            }
            validation_history = output_manager.append_validation_history(validation_history, history_entry)
            output_manager.plot_validation_gap(validation_history)

            if val_metrics["avg_gap"] < best_avg_gap:
                improvement = 0.0 if np.isinf(best_avg_gap) else best_avg_gap - val_metrics["avg_gap"]
                best_difference = max(best_difference, float(improvement))
                best_avg_gap = val_metrics["avg_gap"]

                name = str(int(random.uniform(10**10, 10**15)))
                print(f"[TRAIN] Validation improved | avg_gap={best_avg_gap:.4f} | saving candidate: {name}.pth")
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
                    "all_val_results": val_metrics["all_gaps"],
                    "avg_gap": val_metrics["avg_gap"],
                    "std_gap": val_metrics["std_gap"],
                    "q80_gap": val_metrics["q80_gap"],
                    "validation_instances": [v["name"] for v in validation_set],
                    "episode": step_number,
                })

                with open('candidate_models/model_params.json', 'w') as outfile:
                    json.dump(model_params, outfile)

                bopo_agent.save("candidate_models/" + name + ".pth")
                shutil.copy("candidate_models/" + name + ".pth", "models/" + name + ".pth")

        episode_entry = {
            "episode": step_number,
            "steps": int(rounds),
            "episode_reward": float(-best_ms),
            "makespan": float(best_ms),
            "actor_loss": float(loss),
            "policy_loss": float(loss),  # BOPO não tem critic separado - a actor loss É a policy loss
            "critic_loss": 0.0,
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
    summary = {
        "run_name": run_name,
        "representation": rep_name,
        "run_dir": output_manager.run_dir,
        "max_episodes": int(max_episodes),
        "episodes_completed": int(len(episode_metrics)),
        "updates_completed": int(len(update_metrics)),
        "best_validation_avg_gap": float(best_avg_gap) if not np.isinf(best_avg_gap) else None,
        "best_validation_q80_gap": float(best_q80_gap) if not np.isinf(best_q80_gap) else None,
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
            test_env = ModelEnvClass(test_instances, param["mask_option"], param["sel_k"])
            metadata = test_env.reset().metadata()
            t_ppo_agent = ModelBOPOClass(0.001, test_env, metadata, param["hidden_channels"], param["num_layers"], param["heads"],
                                          gnn_type=param.get("gnn_type", "gat"))
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
