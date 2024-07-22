"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import time

import numpy as np
import math
import torch
import torch.nn as nn
from pyexpat.model import XML_CQUANT_OPT

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel
from ocpmodels.models.escn.so3 import (
    CoefficientMapping,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
)
from ocpmodels.models.scn.sampling import CalcSpherePoints
from ocpmodels.models.scn.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)

from ocpmodels.models.escn.escn import (
    eSCN, 
    LayerBlock,
    MessageBlock,
    SO2Block,
    SO2Conv,
    EdgeBlock,
    EnergyBlock,
    ForceBlock
)
from ..equiformer_v2.so3 import SO3_LinearV2

try:
    from e3nn import o3
except ImportError:
    pass


@registry.register_model("escn_dens")
class eSCN_DenoisingPos(eSCN):
    """Equivariant Spherical Channel Network
    Paper: Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs

    1.  We use the default architectural parameters for denoising positions.
        a.  We encode all degrees for forces.
        b.  We share the energy head for predicting denoising energy.
    2.  Following that eSCN uses small initialization (multiplying with 0.001) for 
        energy predictions and atom-edge embeddings, we use small initializations 
        for encoding forces.
    
    Args:
        use_pbc (bool):         Use periodic boundary conditions
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_neighbors (int):    Maximum number of neighbors per atom
        cutoff (float):         Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_layers (int):             Number of layers in the GNN
        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        hidden_channels (int):        Number of hidden units in message passing
        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks
        edge_channels (int):          Number of channels for the edge invariant features
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances
        basis_width_scalar (float):   Width of distance basis function
        distance_resolution (float):  Distance between distance basis functions in Angstroms
        
        use_force_encoding (bool):    Whether to use force encoding when denoising positions

        show_timing_info (bool):      Show timing and memory info
    """

    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,  # not used
        use_pbc=True,
        regress_forces=True,
        otf_graph=False,
        max_neighbors=40,
        cutoff=8.0,
        max_num_elements=90,
        num_layers=8,
        lmax_list=[6],
        mmax_list=[2],
        sphere_channels=128,
        hidden_channels=256,
        edge_channels=128,
        use_grid=True,
        num_sphere_samples=128,
        distance_function="gaussian",
        basis_width_scalar=1.0,
        distance_resolution=0.02,
        use_force_encoding=True,
        show_timing_info=True,
    ):
        super().__init__(
            num_atoms,      # not used
            bond_feat_dim,  # not used
            num_targets,    # not used
            use_pbc,
            regress_forces,
            otf_graph,
            max_neighbors,
            cutoff,
            max_num_elements,
            num_layers,
            lmax_list,
            mmax_list,
            sphere_channels,
            hidden_channels,
            edge_channels,
            use_grid,
            num_sphere_samples,
            distance_function,
            basis_width_scalar,
            distance_resolution,
            show_timing_info,
        )

        # for denoising positions
        self.use_force_encoding = use_force_encoding
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=max(self.lmax_list), p=1)
        self.force_embedding = SO3_LinearV2(
            in_features=1, 
            out_features=self.sphere_channels, 
            lmax=max(self.lmax_list)
        )
        torch.nn.init.uniform_(self.force_embedding.weight, -0.001, 0.001)

        if self.regress_forces:
            self.denoising_pos_block = ForceBlock(
                self.sphere_channels_all, self.num_sphere_samples, self.act
            )


    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device

        atomic_numbers = data.atomic_numbers.long()
        num_atoms = len(atomic_numbers)
        pos = data.pos

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, edge_index, edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        self.SO3_edge_rot = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_edge_rot.append(
                SO3_Rotation(edge_rot_mat, self.lmax_list[i])
            )

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l=0,m=0 coefficients for each resolution
        for i in range(self.num_resolutions):
            x.embedding[:, offset_res, :] = self.sphere_embedding(
                atomic_numbers
            )[:, offset : offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # This can be expensive to compute (not implemented efficiently), so only do it once and pass it along to each layer
        mappingReduced = CoefficientMapping(
            self.lmax_list, self.mmax_list, self.device
        )

        # Node-wise force encoding during denoising positions
        if hasattr(data, 'denoising_pos_forward') and data.denoising_pos_forward and self.use_force_encoding:
            assert hasattr(data, 'forces')
            force_data = data.forces
            force_sh = o3.spherical_harmonics(
                l=self.irreps_sh,
                x=force_data, 
                normalize=True, 
                normalization='component')
            force_sh = force_sh.view(num_atoms, (max(self.lmax_list) + 1) ** 2, 1)
            if hasattr(data, 'noise_mask'):
                noise_mask_tensor = data.noise_mask.view(-1, 1, 1)
                force_sh = force_sh * noise_mask_tensor
            force_norm = force_data.norm(dim=-1, keepdim=True)
            force_norm = force_norm.view(-1, 1, 1)
            force_norm = force_norm / math.sqrt(3.0)    # since we use `component` normalization
            force_embedding = SO3_Embedding(num_atoms, self.lmax_list, 1, self.device, self.dtype)
            force_embedding.embedding = force_sh * force_norm
        else:
            # normal S2EF
            # we use zero tensors to remove force encoding 
            # we can create the zero tensors by just calling `SO3_Embedding()`
            #   https://github.com/Open-Catalyst-Project/ocp/blob/5a7738f9aa80b1a9a7e0ca15e33938b4d2557edd/ocpmodels/models/escn/so3.py#L130-L171
            force_embedding = SO3_Embedding(num_atoms, self.lmax_list, 1, self.device, self.dtype)
        force_embedding = self.force_embedding(force_embedding)
        x.embedding = x.embedding + force_embedding.embedding

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            if i > 0:
                x_message = self.layer_blocks[i](
                    x,
                    atomic_numbers,
                    edge_distance,
                    edge_index,
                    self.SO3_edge_rot,
                    mappingReduced,
                )

                # Residual layer for all layers past the first
                x.embedding = x.embedding + x_message.embedding

            else:
                # No residual for the first layer
                x = self.layer_blocks[i](
                    x,
                    atomic_numbers,
                    edge_distance,
                    edge_index,
                    self.SO3_edge_rot,
                    mappingReduced,
                )

        # Sample the spherical channels (node embeddings) at evenly distributed points on the sphere.
        # These values are fed into the output blocks.
        x_pt = torch.tensor([], device=self.device)
        offset = 0
        # Compute the embedding values at every sampled point on the sphere
        for i in range(self.num_resolutions):
            num_coefficients = int((x.lmax_list[i] + 1) ** 2)
            x_pt = torch.cat(
                [
                    x_pt,
                    torch.einsum(
                        "abc, pb->apc",
                        x.embedding[:, offset : offset + num_coefficients],
                        self.sphharm_weights[i],
                    ).contiguous(),
                ],
                dim=2,
            )
            offset = offset + num_coefficients

        x_pt = x_pt.view(-1, self.sphere_channels_all)

        ###############################################################
        # Energy estimation
        ###############################################################
        node_energy = self.energy_block(x_pt)
        energy = torch.zeros(len(data.natoms), device=pos.device)
        energy.index_add_(0, data.batch, node_energy.view(-1))
        # Scale energy to help balance numerical precision w.r.t. forces
        energy = energy * 0.001
        
        outputs = {'energy': energy}

        ###############################################################
        # Force estimation
        ###############################################################
        if self.regress_forces:
            forces = self.force_block(x_pt, self.sphere_points)
            denoising_pos_vec = self.denoising_pos_block(x_pt, self.sphere_points)

        self.counter = self.counter + 1

        if self.regress_forces:
            if hasattr(data, 'denoising_pos_forward') and data.denoising_pos_forward:
                if hasattr(data, 'noise_mask'):
                    noise_mask_tensor = data.noise_mask.view(-1, 1)
                    forces = denoising_pos_vec * noise_mask_tensor + forces * (~noise_mask_tensor)
                else:
                    forces = denoising_pos_vec + 0 * forces
            else:
                forces = 0 * denoising_pos_vec + forces
            
            outputs['forces'] = forces
        
        return outputs