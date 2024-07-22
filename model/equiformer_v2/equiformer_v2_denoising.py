import math
import torch

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad

try:
    from e3nn import o3
except ImportError:
    pass

from .so3 import (
    SO3_Embedding,
    SO3_LinearV2
)
from .transformer_block import (
    SO2EquivariantGraphAttention,
)
from .equiformer_v2 import EquiformerV2S_OC20


# Statistics of IS2RE 100K
_AVG_NUM_NODES = 77.81317
_AVG_DEGREE = 23.395238876342773    # IS2RE: 100k, max_radius = 5, max_neighbors = 100


@registry.register_model("equiformer_v2_dens")
class EquiformerV2S_OC20_DenoisingPos(EquiformerV2S_OC20):
    """
    Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon S2 activation

    Args:
        use_pbc (bool):         Use periodic boundary conditions
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_neighbors (int):    Maximum number of neighbors per atom
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_layers (int):             Number of layers in the GNN
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])

        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        grid_resolution (int):        Resolution of SO3_Grid

        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks

        edge_channels (int):                Number of channels for the edge invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel share radial function weights
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances

        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_tp_reparam (bool):      Whether to use tensor product re-parametrization for SO(2) convolution
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks

        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions

        avg_num_nodes (float):      Normalization factor for sum aggregation over nodes
        avg_degree (float):         Normalization factor for sum aggregation over edges

        enforce_max_neighbors_strictly (bool):      When edges are subselected based on the `max_neighbors` arg, arbitrarily select amongst equidistant / degenerate edges to have exactly the correct number.

        use_force_encoding (bool):                  For ablation study, whether to encode forces during denoising positions. Default: True.
        use_noise_schedule_sigma_encoding (bool):   For ablation study, whether to encode the sigma (sampled std of Gaussian noises) during
                                                    denoising positions when `fixed_noise_std` = False in config files. Default: False.
        use_denoising_energy (bool):                For ablation study, whether to predict the energy of the original structure given
                                                    a corrupted structure. If `False`, we zero out the energy prediction. Default: True. 
                                                    
        use_energy_lin_ref (bool):  Whether to add the per-atom energy references during prediction.
                                    During training and validation, this should be kept `False` since we use the `lin_ref` parameter in the OC22 dataloader to subtract the per-atom linear references from the energy targets.
                                    During prediction (where we don't have energy targets), this can be set to `True` to add the per-atom linear references to the predicted energies.
        load_energy_lin_ref (bool): Whether to add nn.Parameters for the per-element energy references.
                                    This additional flag is there to ensure compatibility when strict-loading checkpoints, since the `use_energy_lin_ref` flag can be either True or False even if the model is trained with linear references.
                                    You can't have use_energy_lin_ref = True and load_energy_lin_ref = False, since the model will not have the parameters for the linear references. All other combinations are fine.
    """
    def __init__(
        self,
        num_atoms,      # not used
        bond_feat_dim,  # not used
        num_targets,    # not used
        use_pbc=True,
        regress_forces=True,
        otf_graph=True,
        max_neighbors=500,
        max_radius=5.0,
        max_num_elements=90,

        num_layers=12,
        sphere_channels=128,
        attn_hidden_channels=128,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=512,

        norm_type='rms_norm_sh',

        lmax_list=[6],
        mmax_list=[2],
        grid_resolution=None,

        num_sphere_samples=128,

        edge_channels=128,
        use_atom_edge_embedding=True,
        share_atom_edge_embedding=False,
        use_m_share_rad=False,
        distance_function="gaussian",
        num_distance_basis=512,

        attn_activation='scaled_silu',
        use_tp_reparam=False,
        use_s2_act_attn=False,
        use_attn_renorm=True,
        ffn_activation='scaled_silu',
        use_gate_act=False,
        use_grid_mlp=False,
        use_sep_s2_act=True,

        alpha_drop=0.1,
        drop_path_rate=0.05,
        proj_drop=0.0,

        weight_init='normal',

        avg_num_nodes=_AVG_NUM_NODES,
        avg_degree=_AVG_DEGREE,

        enforce_max_neighbors_strictly=True,

        use_force_encoding=True,
        use_noise_schedule_sigma_encoding=False,
        use_denoising_energy=True,

        use_energy_lin_ref=False,
        load_energy_lin_ref=False,
    ):
        super().__init__(
            num_atoms,      # not used
            bond_feat_dim,  # not used
            num_targets,    # not used
            use_pbc,
            regress_forces,
            otf_graph,
            max_neighbors,
            max_radius,
            max_num_elements,

            num_layers,
            sphere_channels,
            attn_hidden_channels,
            num_heads,
            attn_alpha_channels,
            attn_value_channels,
            ffn_hidden_channels,

            norm_type,

            lmax_list,
            mmax_list,
            grid_resolution,

            num_sphere_samples,

            edge_channels,
            use_atom_edge_embedding,
            share_atom_edge_embedding,
            use_m_share_rad,
            distance_function,
            num_distance_basis,

            attn_activation,
            use_tp_reparam,
            use_s2_act_attn,
            use_attn_renorm,
            ffn_activation,
            use_gate_act,
            use_grid_mlp,
            use_sep_s2_act,

            alpha_drop,
            drop_path_rate,
            proj_drop,

            weight_init,

            avg_num_nodes,
            avg_degree,

            enforce_max_neighbors_strictly,

            use_energy_lin_ref,
            load_energy_lin_ref,
        )

        # for denoising position
        self.use_force_encoding = use_force_encoding
        self.use_noise_schedule_sigma_encoding = use_noise_schedule_sigma_encoding
        self.use_denoising_energy = use_denoising_energy
        
        # for denoising position, encode node-wise forces as node features
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=max(self.lmax_list), p=1)
        self.force_embedding = SO3_LinearV2(
            in_features=1,
            out_features=self.sphere_channels,
            lmax=max(self.lmax_list)
        )

        if self.use_noise_schedule_sigma_encoding:
            self.noise_schedule_sigma_embedding = torch.nn.Linear(
                in_features=1,
                out_features=self.sphere_channels
            )

        if self.regress_forces:
            self.denoising_pos_block = SO2EquivariantGraphAttention(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                1,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_tp_reparam,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.use_gate_act,
                self.use_sep_s2_act,
                alpha_drop=0.0
            )

        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)


    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device

        atomic_numbers = data.atomic_numbers.long()
        num_atoms = len(atomic_numbers)

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(
            data,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
        )

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, edge_index, edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

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
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(
                    atomic_numbers
                    )[:, offset : offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Node-wise force encoding during denoising positions
        force_embedding = SO3_Embedding(num_atoms, self.lmax_list, 1, self.device, self.dtype)
        if hasattr(data, "denoising_pos_forward") and data.denoising_pos_forward:
            assert hasattr(data, "forces")
            force_data = data.forces
            force_sh = o3.spherical_harmonics(
                l=self.irreps_sh,
                x=force_data,
                normalize=True,
                normalization='component'
            )
            force_sh = force_sh.view(num_atoms, (max(self.lmax_list) + 1) ** 2, 1)
            force_norm = force_data.norm(dim=-1, keepdim=True)
            if hasattr(data, "noise_mask"):
                noise_mask_tensor = data.noise_mask.view(-1, 1, 1)
                force_sh = force_sh * noise_mask_tensor
        else:
            force_sh = torch.zeros((num_atoms, (max(self.lmax_list) + 1) ** 2, 1), dtype=data.pos.dtype, device=data.pos.device)
            force_norm = torch.zeros((num_atoms, 1), dtype=data.pos.dtype, device=data.pos.device)

        if not self.use_force_encoding:
            # for ablation study, we enforce the force encoding to be zero.
            force_sh = torch.zeros((num_atoms, (max(self.lmax_list) + 1) ** 2, 1), dtype=data.pos.dtype, device=data.pos.device)
            force_norm = torch.zeros((num_atoms, 1), dtype=data.pos.dtype, device=data.pos.device)

        force_norm = force_norm.view(-1, 1, 1)
        force_norm = force_norm / math.sqrt(3.0)  # since we use `component` normalization
        force_embedding.embedding = force_sh * force_norm

        force_embedding = self.force_embedding(force_embedding)
        x.embedding = x.embedding + force_embedding.embedding

        # noise schedule sigma encoding
        if self.use_noise_schedule_sigma_encoding:
            if hasattr(data, 'denoising_pos_forward') and data.denoising_pos_forward:
                assert hasattr(data, 'sigmas')
                sigmas = data.sigmas
            else:
                sigmas = torch.zeros((num_atoms, 1), dtype=data.pos.dtype, device=data.pos.device)
            noise_schedule_sigma_enbedding = self.noise_schedule_sigma_embedding(sigmas)
            x.embedding[:, 0, :] = x.embedding[:, 0, :] + noise_schedule_sigma_enbedding

        # Edge encoding (distance and atom edge)
        edge_distance = self.distance_expansion(edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            atomic_numbers,
            edge_distance,
            edge_index
        )
        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            x = self.blocks[i](
                x,                  # SO3_Embedding
                atomic_numbers,
                edge_distance,
                edge_index,
                batch=data.batch    # for GraphDropPath
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        ###############################################################
        # Energy estimation
        ###############################################################
        node_energy = self.energy_block(x)
        node_energy = node_energy.embedding.narrow(1, 0, 1)
        energy = torch.zeros(len(data.natoms), device=node_energy.device, dtype=node_energy.dtype)
        energy.index_add_(0, data.batch, node_energy.view(-1))
        energy = energy / self.avg_num_nodes

        # Add the per-atom linear references to the energy.
        if self.use_energy_lin_ref and self.load_energy_lin_ref:
            # During training, target E = (E_DFT - E_ref - E_mean) / E_std, and
            # during inference, \hat{E_DFT} = \hat{E} * E_std + E_ref + E_mean
            # where
            #
            # E_DFT = raw DFT energy,
            # E_ref = reference energy,
            # E_mean = normalizer mean,
            # E_std = normalizer std,
            # \hat{E} = predicted energy,
            # \hat{E_DFT} = predicted DFT energy.
            #
            # We can also write this as
            # \hat{E_DFT} = E_std * (\hat{E} + E_ref / E_std) + E_mean,
            # which is why we save E_ref / E_std as the linear reference.
            with torch.cuda.amp.autocast(False):
                energy = energy.to(self.energy_lin_ref.dtype).index_add(
                    0,
                    data.batch,
                    self.energy_lin_ref[atomic_numbers],
                )

        # zero out denoising energy for ablation study
        if hasattr(data, "denoising_pos_forward") and data.denoising_pos_forward:
            if not self.use_denoising_energy:
                energy = energy * 0.0

        outputs = {"energy": energy}
        ###############################################################
        # Force estimation
        ###############################################################
        if self.regress_forces:
            forces = self.force_block(x,
                atomic_numbers,
                edge_distance,
                edge_index
            )
            forces = forces.embedding.narrow(1, 1, 3)
            forces = forces.view(-1, 3)

            # for denoising positions
            denoising_pos_vec = self.denoising_pos_block(
                x,
                atomic_numbers,
                edge_distance,
                edge_index
            )
            denoising_pos_vec = denoising_pos_vec.embedding.narrow(1, 1, 3)
            denoising_pos_vec = denoising_pos_vec.view(-1, 3)

        if self.regress_forces:
            if hasattr(data, "denoising_pos_forward") and data.denoising_pos_forward:
                if hasattr(data, "noise_mask"):
                    noise_mask_tensor = data.noise_mask.view(-1, 1)
                    forces = denoising_pos_vec * noise_mask_tensor + forces * (~noise_mask_tensor)
                else:
                    forces = denoising_pos_vec + 0 * forces
            else:
                forces = 0 * denoising_pos_vec + forces

            outputs["forces"] = forces

        return outputs
