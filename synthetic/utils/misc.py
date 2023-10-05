import os
import torch
import wandb
from absl import logging
from datetime import datetime

def makedir(path):
    if not os.path.exists(path):
        print('creating dir: {}'.format(path))
        os.makedirs(path)
    else:
        print(path, "already exist!")

class Logger:
    def __init__(
        self,
        exp_name,
        exp_suffix="",
        save_dir=None,
        print_every=100,
        save_every=100,
        total_step=0,
        print_to_stdout=True,
        wandb_project_name=None,
        wandb_tags=[],
        wandb_config=None,
    ):
        if save_dir is not None:
            self.save_dir = save_dir
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = None

        self.print_every = print_every
        self.save_every = save_every
        self.step_count = 0
        self.total_step = total_step
        self.print_to_stdout = print_to_stdout

        self.writer = None
        self.start_time = None
        self.groups = dict()
        self.models_to_save = dict()
        self.objects_to_save = dict()
        if "/" in exp_suffix:
            exp_suffix = "_".join(exp_suffix.split("/")[:-1])
        wandb.init(entity="anonymous", project=wandb_project_name, name=exp_name + "_" + exp_suffix, tags=wandb_tags, reinit=True)
        wandb.config.update(wandb_config)

    def register_model_to_save(self, model, name):
        assert name not in self.models_to_save.keys(), "Name is already registered."

        self.models_to_save[name] = model

    def register_object_to_save(self, object, name):
        assert name not in self.objects_to_save.keys(), "Name is already registered."

        self.objects_to_save[name] = object

    def step(self):
        if self.step_count % self.print_every == 0:
            if self.print_to_stdout:
                self.print_log(self.step_count, self.total_step, elapsed_time=datetime.now() - self.start_time)
            self.write_log(self.step_count)

        if self.step_count % self.save_every == 0:
            self.save_models()
            self.save_objects()
        self.step_count += 1

    def meter(self, group_name, log_name, value):
        if group_name not in self.groups.keys():
            self.groups[group_name] = dict()

        if log_name not in self.groups[group_name].keys():
            self.groups[group_name][log_name] = Accumulator()

        self.groups[group_name][log_name].update_state(value)

    def reset_state(self):
        for _, group in self.groups.items():
            for _, log in group.items():
                log.reset_state()

    def print_log(self, step, total_step, elapsed_time=None):
        print(f"[Step {step:5d}/{total_step}]", end="  ")

        for name, group in self.groups.items():
            print(f"({name})", end="  ")
            for log_name, log in group.items():
                res = log.result()
                if res is None:
                    continue

                if "acc" in log_name.lower():
                    print(f"{log_name} {res:.2f}", end=" | ")
                else:
                    print(f"{log_name} {res:.4f}", end=" | ")

        if elapsed_time is not None:
            print(f"(Elapsed time) {elapsed_time}")
        else:
            print()

    def write_log(self, step):
        log_dict = {}
        for group_name, group in self.groups.items():
            for log_name, log in group.items():
                res = log.result()
                if res is None:
                    continue
                log_dict["{}/{}".format(group_name, log_name)] = res
        wandb.log(log_dict, step=step)

        self.reset_state()

    def write_log_individually(self, name, value, step):
        if self.use_wandb:
            wandb.log({name: value}, step=step)
        else:
            self.writer.add_scalar(name, value, step=step)

    def save_models(self, suffix=None):
        if self.save_dir is None:
            return

        for name, model in self.models_to_save.items():
            _name = name
            if suffix:
                _name += f"_{suffix}"
            torch.save(model.state_dict(), os.path.join(self.save_dir, f"{_name}.pth"))

            if self.print_to_stdout:
                logging.info(f"{name} is saved to {self.save_dir}")

    def save_objects(self, suffix=None):
        if self.save_dir is None:
            return

        for name, obj in self.objects_to_save.items():
            _name = name
            if suffix:
                _name += f"_{suffix}"
            torch.save(obj, os.path.join(self.save_dir, f"{_name}.pth"))

            if self.print_to_stdout:
                logging.info(f"{name} is saved to {self.save_dir}")

    def start(self):
        if self.print_to_stdout:
            logging.info("Training starts!")
        self.start_time = datetime.now()

    def finish(self):
        if self.step_count % self.save_every != 0:
            self.save_models(self.step_count)
            self.save_objects(self.step_count)

        if self.print_to_stdout:
            logging.info("Training is finished!")
        wandb.join()

class Accumulator:
    def __init__(self):
        self.data = 0
        self.num_data = 0

    def reset_state(self):
        self.data = 0
        self.num_data = 0

    def update_state(self, tensor):
        with torch.no_grad():
            self.data += tensor
            self.num_data += 1

    def result(self):
        if self.num_data == 0:
            return None
        data = self.data.item() if hasattr(self.data, 'item') else self.data
        return float(data) / self.num_data
