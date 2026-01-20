import csv
import json
import os
import shutil
import warnings

import hydra
import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from metrics import get_all_evals, get_dataloader, get_eval_results
from utils import get_model_identifiers_from_yaml

warnings.filterwarnings("ignore")


def setup_distributed():
    """Initialize torch.distributed for eval."""
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def model_eval(
    cfg,
    task_id,
    unlearn_times,
    model,
    tokenizer,
    save_dir,
    curr_forget_path,
    eval_unlearn_step=None,
):
    eval_unlearn_step = "last" if eval_unlearn_step is None else eval_unlearn_step
    aggregated_eval_logs = {}

    for (
        folder,
        split,
        question_key,
        answer_key,
        eval_task,
        base_answer_key,
        perturbed_answer_key,
    ) in zip(
        cfg.eval.data_path,
        cfg.eval.split_list,
        cfg.eval.question_key,
        cfg.eval.answer_key,
        cfg.eval.eval_task,
        cfg.eval.base_answer_key,
        cfg.eval.perturbed_answer_key,
    ):
        if eval_task == "eval_log_forget":
            folder = curr_forget_path
            split = "forget_perturbed"

        os.makedirs(save_dir, exist_ok=True)
        save_filename = os.path.join(save_dir, f"{eval_task}.json")

        if os.path.exists(save_filename):
            eval_logs = json.load(open(save_filename, "r"))
        else:
            (
                eval_dataloader,
                base_eval_dataloader,
                perturb_dataloader,
                idk_dataloader,
            ) = get_dataloader(
                cfg.eval,
                eval_task,
                tokenizer,
                folder,
                split,
                question_key,
                answer_key,
                base_answer_key,
                perturbed_answer_key,
            )

            with torch.no_grad():
                eval_logs = get_all_evals(
                    cfg.eval,
                    model,
                    tokenizer,
                    folder,
                    split,
                    eval_task,
                    eval_dataloader,
                    base_eval_dataloader,
                    perturb_dataloader,
                    idk_dataloader,
                    True,
                )

            with open(save_filename, "w") as f:
                json.dump(eval_logs, f, indent=4)

        aggregated_eval_logs[f"{eval_task}.json"] = eval_logs

    aggregated_eval_log_filename = os.path.join(save_dir, "eval_log_aggregated.json")
    with open(aggregated_eval_log_filename, "w") as f:
        json.dump(aggregated_eval_logs, f, indent=4)

    eval_results = get_eval_results(aggregated_eval_logs)
    return eval_results


@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):
    local_rank = setup_distributed()
    is_rank0 = local_rank == 0

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    task_list = [int(i) for i in os.getenv("TASK_LIST").split(",")]
    cfg.save_dir = os.path.join(cfg.save_dir, os.getenv("TASK_LIST").replace(",", "-"))

    unlearn_times = task_list.index(cfg.task_id) + 1
    curr_save_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times}")
    curr_data_path = os.path.join(curr_save_dir, "task_data")

    curr_checkpoint_dir = (
        cfg.model_path
        if cfg.eval_unlearn_step == 0
        else os.path.join(curr_save_dir, f"checkpoint-{cfg.eval_unlearn_step}")
    )

    if not os.path.exists(curr_checkpoint_dir):
        if is_rank0:
            print(f"{curr_checkpoint_dir} does not exist.")
        cleanup_distributed()
        return

    curr_eval_dir = os.path.join(curr_save_dir, f"eval_results-{cfg.eval_unlearn_step}")
    if os.path.exists(os.path.join(curr_eval_dir, "aggregate_stat.csv")):
        if is_rank0:
            print(f"{curr_eval_dir} already evaluated.")
        cleanup_distributed()
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_id)

    device_map = {"": local_rank}

    if cfg.use_LoRA:
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device_map,
        )
        model = PeftModel.from_pretrained(base_model, curr_checkpoint_dir)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            curr_checkpoint_dir,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device_map,
        )

    model.eval()
    model.config.use_cache = False

    eval_results = model_eval(
        cfg,
        cfg.task_id,
        unlearn_times,
        model,
        tokenizer,
        curr_eval_dir,
        curr_data_path,
        cfg.eval_unlearn_step,
    )

    if is_rank0:
        print(
            "After Unlearn Task %d, Step %s | Untargeted %.6f | Targeted %.6f | Utility %.6f"
            % (
                cfg.task_id,
                cfg.eval_unlearn_step,
                eval_results["Untargeted Forget Efficacy"],
                eval_results["Targeted Forget Efficacy"],
                eval_results["Model Utility"],
            )
        )

    cleanup_distributed()


if __name__ == "__main__":
    main()

