"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


import logging
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch_geometric
from tqdm import tqdm

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.models.equiformer_v2.trainers.forces_trainer import (
    EquiformerV2ForcesTrainer,
)
from ocpmodels.modules.evaluator import mae
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.modules.scaling.util import ensure_fitted


@dataclass
class DenoisingPosParams:
    prob: float = 0.0
    fixed_noise_std: bool = False
    std: float = None
    num_steps: int = None
    std_low: float = None
    std_high: float = None
    corrupt_ratio: float = None
    all_atoms: bool = False
    denoising_pos_coefficient: float = None


def add_gaussian_noise_to_position(
    batch, std, corrupt_ratio=None, all_atoms=False
):
    """
    1.  Update `pos` in `batch`.
    2.  Add `noise_vec` to `batch`, which will serve as the target for denoising positions.
    3.  Add `denoising_pos_forward` to switch to denoising mode during training.
    4.  Add `noise_mask` for partially corrupted structures when `corrupt_ratio` is not None.
    5.  If `all_atoms` == True, we add noise to all atoms including fixed ones.
    6.  Check whether `batch` has `md`. We do not add noise to structures from MD split.
    """
    noise_vec = torch.zeros_like(batch.pos)
    noise_vec = noise_vec.normal_(mean=0.0, std=std)

    if corrupt_ratio is not None:
        noise_mask = torch.rand(
            (batch.pos.shape[0]),
            dtype=batch.pos.dtype,
            device=batch.pos.device,
        )
        noise_mask = noise_mask < corrupt_ratio
        noise_vec[(~noise_mask)] *= 0
        batch.noise_mask = noise_mask
    
    # Not add noise to structures from MD split
    if hasattr(batch, 'md'):
        batch_index = batch.batch
        md_index = batch.md.bool()
        md_index = md_index[batch_index]
        noise_mask = (~md_index)
        noise_vec[(~noise_mask)] *= 0
        if hasattr(batch, 'noise_mask'):
            batch.noise_mask = batch.noise_mask * noise_mask
        else:
            batch.noise_mask = noise_mask

    pos = batch.pos
    new_pos = pos + noise_vec
    if all_atoms:
        batch.pos = new_pos
    else:
        free_mask = batch.fixed == 0.0
        batch.pos[free_mask] = new_pos[free_mask]

    batch.noise_vec = noise_vec
    batch.denoising_pos_forward = True

    return batch


def add_gaussian_noise_schedule_to_position(
    batch, std_low, std_high, num_steps, corrupt_ratio=None, all_atoms=False
):
    """
    1.  Similar to above, update positions in batch with gaussian noise, but
        additionally, also save the sigmas the noise vectors are sampled from.
    2.  Add `noise_mask` for partially corrupted structures when `corrupt_ratio`
        is not None.
    3.  If `all_atoms` == True, we add noise to all atoms including fixed ones.
    4.  Check whether `batch` has `md`. We do not add noise to structures from MD split.
    """
    sigmas = torch.tensor(
        np.exp(np.linspace(np.log(std_low), np.log(std_high), num_steps)),
        dtype=torch.float32,
    )
    # select a sigma for each structure, and project it all atoms in the structure.
    ts = torch.randint(0, num_steps, size=(batch.natoms.size(0),))
    batch.sigmas = sigmas[ts][batch.batch][:, None]  # (natoms, 1)
    noise_vec = torch.zeros_like(batch.pos)
    noise_vec = noise_vec.normal_() * batch.sigmas

    if corrupt_ratio is not None:
        noise_mask = torch.rand(
            (batch.pos.shape[0]),
            dtype=batch.pos.dtype,
            device=batch.pos.device,
        )
        noise_mask = noise_mask < corrupt_ratio
        #noise_vec[(~noise_mask)] *= 0
        batch.noise_mask = noise_mask

    # Not add noise to structures from MD split
    if hasattr(batch, 'md'):
        batch_index = batch.batch
        md_index = batch.md.bool()
        md_index = md_index[batch_index]
        noise_mask = (~md_index)
        #noise_vec[(~noise_mask)] *= 0
        if hasattr(batch, 'noise_mask'):
            batch.noise_mask = batch.noise_mask * noise_mask
        else:
            batch.noise_mask = noise_mask

    if hasattr(batch, 'noise_mask'):
        noise_vec[(~batch.noise_mask)] *= 0

    # only add noise to free atoms
    pos = batch.pos
    new_pos = pos + noise_vec
    if all_atoms:
        batch.pos = new_pos
    else:
        free_mask = batch.fixed == 0.0
        batch.pos[free_mask] = new_pos[free_mask]

    batch.noise_vec = noise_vec
    batch.denoising_pos_forward = True

    return batch


def denoising_pos_eval(
    evaluator, prediction, target, prev_metrics={}, denoising_pos_forward=False
):
    """
    1.  Overwrite the original Evaluator.eval() here: https://github.com/Open-Catalyst-Project/ocp/blob/5a7738f9aa80b1a9a7e0ca15e33938b4d2557edd/ocpmodels/modules/evaluator.py#L69-L81
    2.  This is to make sure we separate forces MAE and denoising positions MAE.
    """

    if not denoising_pos_forward:
        return evaluator.eval(prediction, target, prev_metrics)

    # for attr in evaluator.task_attributes[evaluator.task]:
    #     assert attr in prediction
    #     assert attr in target
    #     assert prediction[attr].shape == target[attr].shape

    metrics = prev_metrics

    if target.get("noise_mask", None) is None:
        # Only update `denoising_energy_mae` and `denoising_pos_mae` during denoising positions if not using partially corrupted structures
        res = eval("mae")(prediction, target, "energy")
        metrics = evaluator.update("denoising_energy_mae", res, metrics)
        res = eval("mae")(prediction, target, "forces")
        metrics = evaluator.update("denoising_pos_mae", res, metrics)
    else:
        # Update `denoising_energy_mae`, `denoising_pos_mae` and `denoising_force_mae` if using partially corrupted structures
        res = eval("mae")(prediction, target, "energy")
        metrics = evaluator.update("denoising_energy_mae", res, metrics)
        # separate S2EF and denoising positions results based on `noise_mask`
        target_tensor = target["forces"]
        prediction_tensor = prediction["forces"]
        noise_mask = target["noise_mask"]
        s2ef_index = torch.where(noise_mask == 0)
        s2ef_prediction = {"forces": prediction_tensor[s2ef_index]}
        s2ef_target = {"forces": target_tensor[s2ef_index]}
        res = eval("mae")(s2ef_prediction, s2ef_target, "forces")
        if res["numel"] != 0:
            metrics = evaluator.update("denoising_force_mae", res, metrics)
        denoising_pos_index = torch.where(noise_mask == 1)
        denoising_pos_prediction = {
            "forces": prediction_tensor[denoising_pos_index]
        }
        denoising_pos_target = {"forces": target_tensor[denoising_pos_index]}
        res = eval("mae")(
            denoising_pos_prediction, denoising_pos_target, "forces"
        )
        if res["numel"] != 0:
            metrics = evaluator.update("denoising_pos_mae", res, metrics)
    return metrics


def compute_atomwise_denoising_pos_and_force_hybrid_loss(
    pred, target, noise_mask, force_mult, denoising_pos_mult, mask=None
):
    loss = torch.norm(pred - target, p=2, dim=-1, keepdim=True)
    force_index = torch.where(noise_mask == 0)
    denoising_pos_index = torch.where(noise_mask == 1)
    mult_tensor = torch.ones_like(loss)
    mult_tensor[force_index] *= force_mult
    mult_tensor[denoising_pos_index] *= denoising_pos_mult
    loss = loss * mult_tensor
    if mask is not None:
        loss = loss[mask]
    loss = torch.mean(loss)
    return loss


@registry.register_trainer("denoising_forces")
class DenoisingForcesTrainer(EquiformerV2ForcesTrainer):
    """
    1.  We add a denoising objective to the original S2EF task.
    2.  The denoising objective is that we take as input
        atom types, node-wise forces and 3D coordinates perturbed with Gaussian noises and then
        output energy of the original structure (3D coordinates without any perturbation) and
        the node-wise noises added to the original structure.
    3.  This should make models leverage more from training data and enable data augmentation for
        the S2EF task.
    4.  We should only modify the training part.
    5.  For normalizing the outputs of noise prediction, if we use `fixed_noise_std = True`, we use 
        `std` for the normalization factor. Otherwise, we use `std_high` when `fixed_noise_std = False`.

    Args:
        task (dict): Task configuration.
        model (dict): Model configuration.
        dataset (dict): Dataset configuration. The dataset needs to be a SinglePointLMDB dataset.
        optimizer (dict): Optimizer configuration.
        identifier (str): Experiment identifier that is appended to log directory.
        run_dir (str, optional): Path to the run directory where logs are to be saved.
            (default: :obj:`None`)
        is_debug (bool, optional): Run in debug mode.
            (default: :obj:`False`)
        is_hpo (bool, optional): Run hyperparameter optimization with Ray Tune.
            (default: :obj:`False`)
        print_every (int, optional): Frequency of printing logs.
            (default: :obj:`100`)
        seed (int, optional): Random number seed.
            (default: :obj:`None`)
        logger (str, optional): Type of logger to be used.
            (default: :obj:`tensorboard`)
        local_rank (int, optional): Local rank of the process, only applicable for distributed training.
            (default: :obj:`0`)
        amp (bool, optional): Run using automatic mixed precision.
            (default: :obj:`False`)
        slurm (dict): Slurm configuration. Currently just for keeping track.
            (default: :obj:`{}`)
    """

    def __init__(
        self,
        task,
        model,
        dataset,
        optimizer,
        identifier,
        timestamp_id=None,
        run_dir=None,
        is_debug=False,
        print_every=100,
        seed=None,
        logger="tensorboard",
        local_rank=0,
        amp=False,
        cpu=False,
        slurm={},
        noddp=False,
        outputs={},
        loss_fns={},
        eval_metrics={},
        name="ocp",
    ):
        super().__init__(
            task=task,
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            identifier=identifier,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            is_debug=is_debug,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            slurm=slurm,
            noddp=noddp,
            outputs=outputs,
            loss_fns=loss_fns,
            eval_metrics=eval_metrics,
            name=name,
        )

        # for denoising positions
        self.use_denoising_pos = self.config["optim"]["use_denoising_pos"]
        self.denoising_pos_params = DenoisingPosParams(
            **self.config["optim"]["denoising_pos_params"]
        )
        self.denoising_pos_params.denoising_pos_coefficient = self.config[
            "optim"
        ]["denoising_pos_coefficient"]
        self.normalizers["denoising_pos_target"] = Normalizer(
            mean=0.0,
            std=(
                self.denoising_pos_params.std if self.denoising_pos_params.fixed_noise_std
                else self.denoising_pos_params.std_high
            ),
            device=self.device,
        )

    def train(self, disable_eval_tqdm=False):
        ensure_fitted(self._unwrapped_model, warn=True)

        eval_every = self.config["optim"].get(
            "eval_every", len(self.train_loader)
        )
        checkpoint_every = self.config["optim"].get(
            "checkpoint_every", eval_every
        )
        primary_metric = self.evaluation_metrics.get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )
        if (
            not hasattr(self, "primary_metric")
            or self.primary_metric != primary_metric
        ):
            self.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
        else:
            primary_metric = self.primary_metric
        self.metrics = {}

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)

        for epoch_int in range(
            start_epoch, self.config["optim"]["max_epochs"]
        ):
            skip_steps = self.step % len(self.train_loader)
            self.train_sampler.set_epoch_and_start_iteration(
                epoch_int, skip_steps
            )
            train_loader_iter = iter(self.train_loader)

            self.metrics = {}

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch.
                batch = next(train_loader_iter)

                # for denoising positions
                if self.use_denoising_pos:
                    if np.random.rand() < self.denoising_pos_params.prob:
                        if self.denoising_pos_params.fixed_noise_std:
                            batch = add_gaussian_noise_to_position(
                                batch,
                                std=self.denoising_pos_params.std,
                                corrupt_ratio=self.denoising_pos_params.corrupt_ratio,
                                all_atoms=self.denoising_pos_params.all_atoms,
                            )
                        else:
                            batch = add_gaussian_noise_schedule_to_position(
                                batch,
                                std_low=self.denoising_pos_params.std_low,
                                std_high=self.denoising_pos_params.std_high,
                                num_steps=self.denoising_pos_params.num_steps,
                                corrupt_ratio=self.denoising_pos_params.corrupt_ratio,
                                all_atoms=self.denoising_pos_params.all_atoms,
                            )

                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch)
                loss = self.scaler.scale(loss) if self.scaler else loss
                self._backward(loss)
                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    self.metrics,
                )
                self.metrics = self.evaluator.update(
                    "loss", loss.item() / scale, self.metrics
                )

                # Log metrics.
                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                log_dict.update(
                    {
                        "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                    }
                )
                if (
                    self.step % self.config["cmd"]["print_every"] == 0
                    or i == 0
                    or i == (len(self.train_loader) - 1)
                ) and distutils.is_master():
                    log_str = [
                        "{}: {:.2e}".format(k, v) for k, v in log_dict.items()
                    ]
                    logging.info(", ".join(log_str))
                    self.metrics = {}

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split="train",
                    )

                if (
                    checkpoint_every != -1
                    and self.step % checkpoint_every == 0
                ):
                    self.save(
                        checkpoint_file="checkpoint.pt", training_state=True
                    )

                # Evaluate on val set every `eval_every` iterations.
                if self.step % eval_every == 0 or i == (
                    len(self.train_loader) - 1
                ):
                    if self.val_loader is not None:
                        if i == (len(self.train_loader) - 1):
                            self.save(
                                checkpoint_file="checkpoint.pt",
                                training_state=True,
                            )

                        val_metrics = self.validate(
                            split="val", disable_tqdm=disable_eval_tqdm
                        )
                        self.update_best(
                            primary_metric,
                            val_metrics,
                            disable_eval_tqdm=disable_eval_tqdm,
                        )

                    if self.config["task"].get("eval_relaxations", False):
                        if "relax_dataset" not in self.config["task"]:
                            logging.warning(
                                "Cannot evaluate relaxations, relax_dataset not specified"
                            )
                        else:
                            self.run_relaxations()

                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    self.scheduler.step()

            # torch.cuda.empty_cache()

            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)

        self.train_dataset.close_db()
        if self.config.get("val_dataset", False):
            self.val_dataset.close_db()
        if self.config.get("test_dataset", False):
            self.test_dataset.close_db()

    def _compute_loss(self, out, batch):
        batch_size = batch.natoms.numel()
        fixed = batch.fixed
        mask = fixed == 0

        loss = []
        for loss_fn in self.loss_fns:
            target_name, loss_info = loss_fn

            if target_name == "forces" and batch.get(
                "denoising_pos_forward", False
            ):
                denoising_pos_target = batch.noise_vec
                if self.normalizers.get("denoising_pos_target", False):
                    denoising_pos_target = self.normalizers[
                        "denoising_pos_target"
                    ].norm(denoising_pos_target)

                if hasattr(batch, "noise_mask"):
                    # for partially corrupted structures
                    target = batch.forces
                    if self.normalizers.get("forces", False):
                        target = self.normalizers["forces"].norm(target)
                    noise_mask = batch.noise_mask.view(-1, 1)
                    target = denoising_pos_target * noise_mask + target * (~noise_mask)
                else:
                    target = denoising_pos_target

                pred = out["forces"]
                natoms = batch.natoms
                natoms = torch.repeat_interleave(natoms, natoms)

                force_mult = loss_info["coefficient"]
                denoising_pos_mult = self.denoising_pos_params.denoising_pos_coefficient

                if (
                    self.output_targets[target_name]["level"] == "atom"
                    and self.output_targets[target_name]["train_on_free_atoms"]
                ):
                    # If `all_atoms` == True when training on only free atoms,
                    # we also add noise to and denoise fixed atoms.
                    if self.denoising_pos_params.all_atoms:
                        if hasattr(batch, "noise_mask"):
                            mask = mask.view(-1, 1) | noise_mask
                        else:
                            mask = torch.ones_like(
                                mask, dtype=torch.bool, device=mask.device
                            ).view(-1, 1)

                    if hasattr(batch, "noise_mask"):
                        # for partially corrupted structures
                        loss.append(
                            compute_atomwise_denoising_pos_and_force_hybrid_loss(
                                pred=pred,
                                target=target,
                                noise_mask=noise_mask,
                                force_mult=force_mult,
                                denoising_pos_mult=denoising_pos_mult,
                                mask=mask,
                            )
                        )
                    else:
                        target = target[mask]
                        pred = pred[mask]
                        natoms = natoms[mask]

                        loss.append(
                            denoising_pos_mult
                            * loss_info["fn"](
                                pred,
                                target,
                                natoms=natoms,
                                batch_size=batch_size,
                            )
                        )
                else:
                    if hasattr(batch, "noise_mask"):
                        # for partially corrupted structures
                        loss.append(
                            compute_atomwise_denoising_pos_and_force_hybrid_loss(
                                pred=pred,
                                target=target,
                                noise_mask=noise_mask,
                                force_mult=force_mult,
                                denoising_pos_mult=denoising_pos_mult,
                                mask=None,
                            )
                        )
                    else:
                        loss.append(
                            denoising_pos_mult
                            * loss_info["fn"](
                                pred,
                                target,
                                natoms=natoms,
                                batch_size=batch_size,
                            )
                        )

            else:
                target = batch[target_name]
                pred = out[target_name]
                natoms = batch.natoms
                natoms = torch.repeat_interleave(natoms, natoms)

                if (
                    self.output_targets[target_name]["level"] == "atom"
                    and self.output_targets[target_name]["train_on_free_atoms"]
                ):
                    target = target[mask]
                    pred = pred[mask]
                    natoms = natoms[mask]

                num_atoms_in_batch = natoms.numel()
                if self.normalizers.get(target_name, False):
                    target = self.normalizers[target_name].norm(target)

                ### reshape accordingly: num_atoms_in_batch, -1 or num_systems_in_batch, -1
                if self.output_targets[target_name]["level"] == "atom":
                    target = target.view(num_atoms_in_batch, -1)
                else:
                    target = target.view(batch_size, -1)

                mult = loss_info["coefficient"]
                loss.append(
                    mult
                    * loss_info["fn"](
                        pred,
                        target,
                        natoms=natoms,
                        batch_size=batch_size,
                    )
                )

        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        loss = sum(loss)
        return loss

    def _compute_metrics(self, out, batch, evaluator, metrics={}):
        # This assumes batch.fixed is specified correctly for each dataset.

        natoms = batch.natoms
        batch_size = natoms.numel()

        ### Retrieve free atoms
        fixed = batch.fixed
        mask = fixed == 0

        s_idx = 0
        natoms_free = []
        for _natoms in natoms:
            natoms_free.append(torch.sum(mask[s_idx : s_idx + _natoms]).item())
            s_idx += _natoms
        natoms = torch.LongTensor(natoms_free).to(self.device)

        denoising_pos_forward = False
        if batch.get("denoising_pos_forward", False):
            denoising_pos_forward = True

        targets = {}
        for target_name in self.output_targets:
            num_atoms_in_batch = batch.natoms.sum()

            if denoising_pos_forward and target_name == "forces":
                if hasattr(batch, "noise_mask"):
                    force_target = batch.forces
                    denoising_pos_target = batch.noise_vec
                    noise_mask = batch.noise_mask
                    s2ef_index = torch.where(noise_mask == 0)
                    denoising_pos_index = torch.where(noise_mask == 1)
                    noise_mask_tensor = noise_mask.view(-1, 1)
                    targets["forces"] = (
                        denoising_pos_target * noise_mask_tensor
                        + force_target * (~noise_mask_tensor)
                    )
                    targets["noise_mask"] = noise_mask
                else:
                    targets["forces"] = batch.noise_vec

                if self.normalizers.get("denoising_pos_target", False):
                    if hasattr(batch, "noise_mask"):
                        out["forces"][denoising_pos_index] = self.normalizers[
                            "denoising_pos_target"
                        ].denorm(out["forces"][denoising_pos_index])
                    else:
                        out["forces"] = self.normalizers[
                            "denoising_pos_target"
                        ].denorm(out["forces"])

                if hasattr(batch, "noise_mask"):
                    out["forces"][s2ef_index] = self.normalizers[
                        "forces"
                    ].denorm(out["forces"][s2ef_index])

                if (
                    self.output_targets[target_name]["level"] == "atom"
                    and self.output_targets[target_name]["eval_on_free_atoms"]
                ):
                    if self.denoising_pos_params.all_atoms:
                        if hasattr(batch, "noise_mask"):
                            mask = mask | noise_mask
                        else:
                            mask = torch.ones_like(
                                mask, dtype=torch.bool, device=mask.device
                            )

                    targets["forces"] = targets["forces"][mask]
                    out["forces"] = out["forces"][mask]
                    num_atoms_in_batch = natoms.sum()
                    if "noise_mask" in targets:
                        targets["noise_mask"] = targets["noise_mask"][mask]
            else:
                target = batch[target_name]

                if (
                    self.output_targets[target_name]["level"] == "atom"
                    and self.output_targets[target_name]["eval_on_free_atoms"]
                ):
                    target = target[mask]
                    out[target_name] = out[target_name][mask]
                    num_atoms_in_batch = natoms.sum()

                ### reshape accordingly: num_atoms_in_batch, -1 or num_systems_in_batch, -1
                if self.output_targets[target_name]["level"] == "atom":
                    target = target.view(num_atoms_in_batch, -1)
                else:
                    target = target.view(batch_size, -1)

                targets[target_name] = target
                if self.normalizers.get(target_name, False):
                    out[target_name] = self.normalizers[target_name].denorm(
                        out[target_name]
                    )

        targets["natoms"] = natoms
        out["natoms"] = natoms

        metrics = denoising_pos_eval(
            evaluator,
            out,
            targets,
            prev_metrics=metrics,
            denoising_pos_forward=denoising_pos_forward,
        )

        return metrics


    @torch.no_grad()
    def predict(
        self,
        data_loader,
        per_image: bool = True,
        results_file: Optional[str] = None,
        disable_tqdm: bool = False,
    ):
        if self.is_debug and per_image:
            raise FileNotFoundError(
                "Predictions require debug mode to be turned off."
            )

        ensure_fitted(self._unwrapped_model, warn=True)

        if distutils.is_master() and not disable_tqdm:
            logging.info("Predicting on test.")
        assert isinstance(
            data_loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(data_loader, torch_geometric.data.Batch):
            data_loader = [data_loader]

        self.model.eval()
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        predictions = defaultdict(list)

        for key in self.normalizers.keys():
            self.normalizers[key].to(self.device)

        for i, batch in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                batch = batch.to(self.device)
                out = self._forward(batch)

            for key in out.keys():
                out[key] = out[key].float()

            for target_key in self.config["outputs"]:
                pred = out[target_key]
                if self.normalizers.get(target_key, False):
                    pred = self.normalizers[target_key].denorm(pred)

                if per_image:
                    ### Save outputs in desired precision, default float16
                    if (
                        self.config["outputs"][target_key].get(
                            "prediction_dtype", "float16"
                        )
                        == "float32"
                        or self.config["task"].get(
                            "prediction_dtype", "float16"
                        )
                        == "float32"
                        or self.config["task"].get("dataset", "lmdb")
                        == "oc22_lmdb"
                    ):
                        dtype = torch.float32
                    else:
                        dtype = torch.float16

                    #pred = pred.cpu().detach().to(dtype)
                    pred = pred.detach().cpu().to(dtype)
                    
                    ### Split predictions into per-image predictions
                    if self.config["outputs"][target_key]["level"] == "atom":
                        batch_natoms = batch.natoms
                        batch_fixed = batch.fixed
                        per_image_pred = torch.split(
                            pred, batch_natoms.tolist()
                        )

                        ### Save out only free atom, EvalAI does not need fixed atoms
                        _per_image_fixed = torch.split(
                            batch_fixed, batch_natoms.tolist()
                        )
                        _per_image_free_preds = [
                            _pred[(fixed == 0).tolist()].numpy()
                            for _pred, fixed in zip(
                                per_image_pred, _per_image_fixed
                            )
                        ]
                        _chunk_idx = np.array(
                            [
                                free_pred.shape[0]
                                for free_pred in _per_image_free_preds
                            ]
                        )
                        per_image_pred = _per_image_free_preds
                    ### Assumes system level properties are of the same dimension
                    else:
                        per_image_pred = pred.numpy()
                        _chunk_idx = None

                    predictions[f"{target_key}"].extend(per_image_pred)
                    ### Backwards compatibility, retain 'chunk_idx' for forces.
                    if _chunk_idx is not None:
                        if target_key == "forces":
                            predictions["chunk_idx"].extend(_chunk_idx)
                        else:
                            predictions[f"{target_key}_chunk_idx"].extend(
                                _chunk_idx
                            )
                else:
                    predictions[f"{target_key}"] = pred.detach()

            if not per_image:
                return predictions

            ### Get unique system identifiers
            sids = (
                batch.sid.tolist()
                if isinstance(batch.sid, torch.Tensor)
                else batch.sid
            )
            ## Support naming structure for OC20 S2EF
            if "fid" in batch:
                fids = (
                    batch.fid.tolist()
                    if isinstance(batch.fid, torch.Tensor)
                    else batch.fid
                )
                systemids = [f"{sid}_{fid}" for sid, fid in zip(sids, fids)]
            else:
                systemids = [f"{sid}" for sid in sids]

            predictions["ids"].extend(systemids)

        # for key in predictions:
            # if isinstance(predictions[key][0], np.ndarray):
                # predictions[key] = np.concatenate(predictions[key], axis=0)
            # else:
                # predictions[key] = np.array(predictions[key])
        self.save_results(predictions, results_file)

        if self.ema:
            self.ema.restore()

        return predictions
