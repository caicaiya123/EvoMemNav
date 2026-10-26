"""Evaluate EvoMemNav on the validation unseen split of GOAT-Bench.

All episodes (10 per scene):
    python run_goatbench.py --cfg cfg/evomemnav_goatbench.yaml
Selected episodes, e.g. episode 1 of each scene:
    python run_goatbench.py --cfg cfg/evomemnav_goatbench.yaml --episodes 1
Shard the scenes over several processes / GPUs:
    python run_goatbench.py --cfg cfg/evomemnav_goatbench.yaml --start_ratio 0.0 --end_ratio 0.5
Override any config value:
    python run_goatbench.py --cfg cfg/evomemnav_goatbench.yaml coarse.anchor_budget=3 coarse.frontier_budget=2

The episodes of a scene run in order, so that Scene-LTM accumulates over them.
"""

import argparse
import json
import logging
import os
import random

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("MAGNUM_LOG", "quiet")

import numpy as np
from omegaconf import OmegaConf

from evomemnav.config import load_config
from evomemnav.env.goatbench import list_scene_files, load_scene_file, prepare_goatbench_navigation_goals, resolve_scene_id
from evomemnav.eval.recorder import ResultRecorder, load_records, summarize
from evomemnav.experience.memory import ExperienceMemory


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", default="cfg/evomemnav_goatbench.yaml")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="1-based episode indices (default: all)")
    parser.add_argument("--start_ratio", type=float, default=0.0)
    parser.add_argument("--end_ratio", type=float, default=1.0)
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. fine.verify_threshold=0.62")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.cfg, args.overrides)
    output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    tag = f"{args.start_ratio:.2f}_{args.end_ratio:.2f}"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(os.path.join(output_dir, f"log_{tag}.log")), logging.StreamHandler()],
    )
    OmegaConf.save(cfg, os.path.join(output_dir, f"cfg_{tag}.yaml"))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # heavy imports after the config is validated
    from evomemnav.agent import EvoMemNav
    from evomemnav.env.scene import Scene
    from evomemnav.vlm.client import VLMClient

    cg_cfg = OmegaConf.load(cfg.perception.concept_graph_config)
    agent = EvoMemNav(cfg, cg_cfg, VLMClient(cfg.vlm))
    recorder = ResultRecorder(output_dir, tag)
    finished = recorder.finished_episodes()

    scene_files = list_scene_files(cfg.data.goatbench_dir)
    n = len(scene_files)
    scene_files = scene_files[int(args.start_ratio * n) : int(args.end_ratio * n)]
    all_scene_ids = sorted(
        os.listdir(os.path.join(cfg.data.scene_data_path, "train")) + os.listdir(os.path.join(cfg.data.scene_data_path, "val"))
    )
    logging.info(f"***** {cfg.exp_name}: {len(scene_files)} scenes, episodes={args.episodes or 'all'} *****")

    for scene_file in scene_files:
        scene_name = scene_file.split(".")[0]
        scene_id = resolve_scene_id(scene_name, all_scene_ids)
        data = load_scene_file(cfg.data.goatbench_dir, scene_file)
        indices = args.episodes or list(range(1, len(data["episodes"]) + 1))
        experience = ExperienceMemory(cfg.experience, os.path.join(output_dir, "scene_ltm", f"{scene_id}.json"))

        for ep_idx in indices:
            episode = data["episodes"][ep_idx - 1]
            goal_types, goals = prepare_goatbench_navigation_goals(scene_name, episode, data["goals"])
            episode_key = f"{scene_id}_ep{episode['episode_id']}"
            if finished.get(episode_key, 0) >= len(goals):
                logging.info(f"{episode_key} already finished")
                continue
            scene = Scene(scene_id, cfg, cg_cfg, agent.detection_model, agent.sam_predictor, agent.clip_model, agent.clip_preprocess, agent.clip_tokenizer)
            try:
                for record in agent.run_episode(scene, episode, goal_types, goals, experience, episode_key):
                    record.update(
                        scene_id=scene_id,
                        episode_id=episode["episode_id"],
                        episode_key=episode_key,
                        subtask_id=f"{episode_key}_{record['subtask_idx']}",
                    )
                    recorder.log(record)
            finally:
                scene.close()

    summary = summarize(load_records(output_dir))
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
