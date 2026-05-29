import torch
import torch.nn as nn
from typing import Tuple

from .abstract_vae import VAE

from ...data.skeletal_data import MediaPipe50KeyPoints

from ..components import (
MLP,
build_adjacency_matrix, normalize_adjacency,
GenericSkelRegionMotionEncoder, GenericSkelRegionMotionDecoder,
add_positional_sinusoidal_encoding, SpatialTransformer1d,
)


def read_ids_file(ids_filepath: str) -> list[int]:
    with open(ids_filepath, "r") as f:
        list_ids = [int(line) for line in f]
    return list_ids


def read_edges_file(edges_filepath: str) -> list[Tuple[int, int]]:
    edges = []
    with open(edges_filepath, "r") as f:
        for line in f:
            i, j = map(int, line.split(","))
            edges.append((i, j))
    return edges


class SkelMotionMultiRegionVAE(VAE):
    """
    Variational Autoencoder for skeletal pose sequences.
    Allows to encode and decode a sequence of skeletal landmarks coordinates.

    It expects by default the format of SLRTP25 CVPR challenge,
    ie a 178 key-points skeleton, which structure is reminded
    in ROOT/data/skeleta_data.py.
    However, it can be easily adapted to other structures skeletal structures, given the correct IDs
    and edges.

    This VAE separates the torso+arms, right hand, left hand and face regions for encoding and decoding.

    Besides, it handles different design configurations:
        - different architectures for encoders or decoders
          (MLP, Graph Convolutional Networks, Residual Temporal Convolutions)
        - either to predict one mu and logvar per region (factorized latent distribution)
          or to concatenate each region encoders' outputs to
          predict a 'unified' mu and logvar (shared latent distribution)
    Please have a look at 'model' arguments in the YAML config files (in the ROOT/configs/ folder)
    to see how to define these options.

    For more details about the architecture details, have a look at the scheme
    in the repo's README file.

    Note that you can output reconstructed poses, mu, logvar and latent poses directly
    by applying your instantiated `your_vae = SkelMotionMultiRegionVAE(your_cfg)` to an batch
    of input poses x and their padding mask m as: `x, mu, logv, z = your_vae(x, pad_mask=m)`.
    """
    def __init__(self, cfg):
        super().__init__()

        # ===/ PARAMS
        T = cfg.get("T", 256)  # max num frames

        Npts_torsoarms = cfg.get("Npts_torsoarms", 10)  # 8 + 2 hands wrists
        Npts_rh = cfg.get("Npts_rh", 21)
        Npts_lh = cfg.get("Npts_lh", 21)
        Npts_face = cfg.get("Npts_face", 128)
        Npts_tot = cfg.get("Npts_tot", 178)
        assert Npts_tot == Npts_torsoarms - 2 + Npts_rh + Npts_lh + Npts_face, "`Npts_tot` not consistent w/ per-region"

        # ids for each region
        ids_torsoarms = cfg.get("ids_torsoarms", MediaPipe50KeyPoints.ids_torsoarms())
        ids_rh = cfg.get("ids_rh", MediaPipe50KeyPoints.ids_rh())
        ids_lh = cfg.get("ids_lh", MediaPipe50KeyPoints.ids_lh())
        ids_face = cfg.get("ids_face", list(range(50, 178)))
        # (if we provide a path to a .txt file with ids (as one ID per line))
        if isinstance(ids_torsoarms, str):
            ids_torsoarms = read_ids_file(ids_torsoarms)
        if isinstance(ids_rh, str):
            ids_rh = read_ids_file(ids_rh)
        if isinstance(ids_lh, str):
            ids_lh = read_ids_file(ids_lh)
        if isinstance(ids_face, str):
            ids_face = read_ids_file(ids_face)

        # wrists ids
        id_torsoarms_rw = cfg.get("id_torsoarms_rwrist", 8)  # Right wrist ID in `torsoarms`
        id_torsoarms_lw = cfg.get("id_torsoarms_lwrist", 9)  # Left wrist ID in `torsoarms`

        # shoulders ids
        id_torsoarms_rshould = cfg.get("id_torsoarms_rshoulder", 0)  # Right shoulder ID in `torsoarms`
        id_torsoarms_lshould = cfg.get("id_torsoarms_lshoulder", 3)  # Left shoulder ID in `torsoarms`

        # connections for graph structured regions
        edges_torsoarms = cfg.get("edges_torsoarms", MediaPipe50KeyPoints.connections_torsoarms())
        edges_rh = cfg.get("edges_rh", MediaPipe50KeyPoints.connections_rh())
        edges_lh = cfg.get("edges_lh", MediaPipe50KeyPoints.connections_lh())
        # (if we provide a path to a .txt file with edges (as NODEi, NODEj per line))
        if isinstance(edges_torsoarms, str):
            edges_torsoarms = read_edges_file(edges_torsoarms)
        if isinstance(edges_rh, str):
            edges_rh = read_edges_file(edges_rh)
        if isinstance(edges_lh, str):
            edges_lh = read_edges_file(edges_lh)

        # default latent dims choices as in "Disentangle and Regularize" paper https://arxiv.org/abs/2504.06610
        latent_dim_torsoarms = cfg.get("latent_dim_torsoarms", 8)
        latent_dim_rh = cfg.get("latent_dim_rh", 28)
        latent_dim_lh = cfg.get("latent_dim_lh", 28)
        latent_dim_face = cfg.get("latent_dim_face", 16)

        # whether to have one shared latent distribution N(mu, sigma2) for concatenated regions embeddings
        shared_latent_distribution = cfg.get("shared_latent_distribution", False)

        # ===/ MODULES
        encoder_torsoarms_default_cfg = {
            "dim_out": 256,
            "hidden_dims": [64, 128],
        }
        encoder_rh_default_cfg = {
            "dim_out": 256,
            "hidden_dims": [64, 128],
        }
        encoder_lh_default_cfg = {
            "dim_out": 256,
            "hidden_dims": [64, 128],
        }
        encoder_face_default_cfg = {
            "dim_out": 256,
            "hidden_dims": [64, 128],
        }

        decoder_torsoarms_default_cfg = {
            "hidden_dims": list(reversed(encoder_torsoarms_default_cfg["hidden_dims"])),
        }
        decoder_rh_default_cfg = {
            "hidden_dims": list(reversed(encoder_rh_default_cfg["hidden_dims"])),
        }
        decoder_lh_default_cfg = {
            "hidden_dims": list(reversed(encoder_lh_default_cfg["hidden_dims"])),
        }
        decoder_face_default_cfg = {
            "hidden_dims": list(reversed(encoder_face_default_cfg["hidden_dims"])),
        }

        encoder_torsoarms_cfg = cfg.get("encoder_torsoarms", encoder_torsoarms_default_cfg)
        encoder_rh_cfg = cfg.get("encoder_rh", encoder_rh_default_cfg)
        encoder_lh_cfg = cfg.get("encoder_lh", encoder_lh_default_cfg)
        encoder_face_cfg = cfg.get("encoder_face", encoder_face_default_cfg)

        decoder_torsoarms_cfg = cfg.get("decoder_torsoarms", decoder_torsoarms_default_cfg)
        decoder_rh_cfg = cfg.get("decoder_rh", decoder_rh_default_cfg)
        decoder_lh_cfg = cfg.get("decoder_lh", decoder_lh_default_cfg)
        decoder_face_cfg = cfg.get("decoder_face", decoder_face_default_cfg)

        # ===/ SET ATTRIBUTES
        self.T = T

        self.Npts_torsoarms = Npts_torsoarms
        self.Npts_rh = Npts_rh
        self.Npts_lh = Npts_lh
        self.Npts_face = Npts_face
        self.Npts_tot = Npts_tot

        self.ids_torsoarms = ids_torsoarms
        self.ids_rh = ids_rh
        self.ids_lh = ids_lh
        self.ids_face = ids_face

        self.id_torsoarms_rw = id_torsoarms_rw
        self.id_torsoarms_lw = id_torsoarms_lw

        self.id_torsoarms_rshould = id_torsoarms_rshould
        self.id_torsoarms_lshould = id_torsoarms_lshould

        self.A_torsoarms = None if edges_torsoarms is None else build_adjacency_matrix(Npts_torsoarms, edges_torsoarms)
        self.A_rh = None if edges_rh is None else build_adjacency_matrix(Npts_rh, edges_rh)
        self.A_lh = None if edges_lh is None else build_adjacency_matrix(Npts_lh, edges_lh)

        self.latent_dim_torsoarms = latent_dim_torsoarms
        self.latent_dim_rh = latent_dim_rh
        self.latent_dim_lh = latent_dim_lh
        self.latent_dim_face = latent_dim_face

        self.shared_latent_distribution = shared_latent_distribution

        # ------------------| REGIONS ENCODERS |-------------------
        self.encoder_torsoarms = GenericSkelRegionMotionEncoder(
            Npts=Npts_torsoarms,
            **encoder_torsoarms_cfg,
            A_norm=None if self.A_torsoarms is None else normalize_adjacency(self.A_torsoarms)
        )
        self.encoder_rh = GenericSkelRegionMotionEncoder(
            Npts=Npts_rh,
            **encoder_rh_cfg,
            A_norm=None if self.A_rh is None else normalize_adjacency(self.A_rh)
        )
        self.encoder_lh = GenericSkelRegionMotionEncoder(
            Npts=Npts_lh,
            **encoder_lh_cfg,
            A_norm=None if self.A_lh is None else normalize_adjacency(self.A_lh)
        )
        self.encoder_face = GenericSkelRegionMotionEncoder(
            Npts=Npts_face,
            **encoder_face_cfg,
        )
        encoder_modules_names = [
            "encoder_torsoarms",
            "encoder_rh",
            "encoder_lh",
            "encoder_face",
        ]
        self._encoder_modules_names = encoder_modules_names

        # ------------------| REGIONS HEADS |-------------------
        dim_enc_torsoarms = encoder_torsoarms_cfg["dim_out"]
        dim_enc_rh = encoder_rh_cfg["dim_out"]
        dim_enc_lh = encoder_lh_cfg["dim_out"]
        dim_enc_face = encoder_face_cfg["dim_out"]

        # == optional fully connected layers before last projections heads
        self.fc_torsoarms = None
        self.fc_rh = None
        self.fc_lh = None
        self.fc_face = None

        if cfg.get("fc_torsoarms", False):
            self.fc_torsoarms = MLP(sizes=3 * [dim_enc_torsoarms], act=nn.ReLU, final_act=nn.ReLU)
            self._encoder_modules_names += ["fc_torsoarms"]
        if cfg.get("fc_rh", False):
            self.fc_rh = MLP(sizes=3 * [dim_enc_rh], act=nn.ReLU, final_act=nn.ReLU)
            self._encoder_modules_names += ["fc_rh"]
        if cfg.get("fc_lh", False):
            self.fc_lh = MLP(sizes=3 * [dim_enc_lh], act=nn.ReLU, final_act=nn.ReLU)
            self._encoder_modules_names += ["fc_lh"]
        if cfg.get("fc_face", False):
            self.fc_face = MLP(sizes=3 * [dim_enc_face], act=nn.ReLU, final_act=nn.ReLU)
            self._encoder_modules_names += ["fc_face"]

        # == mu and logvar
        self.mu_proj = None

        self.mu_proj_torsoarms = None
        self.mu_proj_rh = None
        self.mu_proj_lh = None
        self.mu_proj_face = None

        # ---

        self.logvar_proj = None

        self.logvar_proj_torsoarms = None
        self.logvar_proj_rh = None
        self.logvar_proj_lh = None
        self.logvar_proj_face = None

        if shared_latent_distribution:
            # in case we want one shared learned distribution N(mu, sigma2) for concatenated regions
            dim_enc_concat = dim_enc_torsoarms + dim_enc_rh + dim_enc_lh + dim_enc_face
            latent_dim_concat = latent_dim_torsoarms + latent_dim_rh + latent_dim_lh + latent_dim_face

            self.mu_proj = nn.Linear(dim_enc_concat, latent_dim_concat)
            self.logvar_proj = nn.Linear(dim_enc_concat, latent_dim_concat)

            self._encoder_modules_names += ["mu_proj", "logvar_proj"]
        else:
            # --/ mu
            self.mu_proj_torsoarms = nn.Linear(dim_enc_torsoarms, latent_dim_torsoarms)
            self.mu_proj_rh = nn.Linear(dim_enc_rh, latent_dim_rh)
            self.mu_proj_lh = nn.Linear(dim_enc_lh, latent_dim_lh)
            self.mu_proj_face = nn.Linear(dim_enc_face, latent_dim_face)

            # --/ logvar
            self.logvar_proj_torsoarms = nn.Linear(dim_enc_torsoarms, latent_dim_torsoarms)
            self.logvar_proj_rh = nn.Linear(dim_enc_rh, latent_dim_rh)
            self.logvar_proj_lh = nn.Linear(dim_enc_lh, latent_dim_lh)
            self.logvar_proj_face = nn.Linear(dim_enc_face, latent_dim_face)

            self._encoder_modules_names += [
                "mu_proj_torsoarms", "logvar_proj_torsoarms",
                "mu_proj_rh", "logvar_proj_rh",
                "mu_proj_lh", "logvar_proj_lh",
                "mu_proj_face", "logvar_proj_face",
            ]

        # ------------------| REGIONS DECODERS |-------------------
        # == decoder projections (latent_dim -> dim_enc) for symmetry
        self.dec_proj_torsoarms = nn.Linear(latent_dim_torsoarms, dim_enc_torsoarms)
        self.dec_proj_rh = nn.Linear(latent_dim_rh, dim_enc_rh)
        self.dec_proj_lh = nn.Linear(latent_dim_lh, dim_enc_lh)
        self.dec_proj_face = nn.Linear(latent_dim_face, dim_enc_face)

        decoder_modules_names = [
            # -- projections
            "dec_proj_torsoarms",
            "dec_proj_rh",
            "dec_proj_lh",
            "dec_proj_face",
        ]
        self._decoder_modules_names = decoder_modules_names

        # == optional cross-region attention
        self.cross_region_attn_torsoarms = None
        self.attended_regions_cross_region_attn_torsoarms = None

        self.cross_region_attn_rh = None
        self.attended_regions_cross_region_attn_rh = None

        self.cross_region_attn_lh = None
        self.attended_regions_cross_region_attn_lh = None

        self.cross_region_attn_face = None
        self.attended_regions_cross_region_attn_face = None

        cross_region_attn_names = ["torsoarms", "rh", "lh", "face"]  # must match the suffix in attributes!!
        cross_region_attn_dims = {
            "torsoarms": dim_enc_torsoarms, "rh": dim_enc_rh, "lh": dim_enc_lh, "face": dim_enc_face
        }
        self.cross_region_attn_names = cross_region_attn_names  # just for checking later

        cross_region_attn = cfg.get("cross_region_attn", None)  # dictionary with `region`: `cfg`

        if cross_region_attn is not None:
            for region, cfg in cross_region_attn.items():
                assert region in cross_region_attn_names, f"Invalid region name '{region}'"

                attend_to_regions = cfg.get("attend_to_regions", None)  # list of regions to attend to
                assert attend_to_regions is not None
                assert set(attend_to_regions).issubset(cross_region_attn_names)

                d_q = cross_region_attn_dims[region]
                d_kv = sum([cross_region_attn_dims[reg] for reg in attend_to_regions])

                attn_dim_heads = cfg.get("attn_dim_heads", 32)
                assert d_q % attn_dim_heads == 0  # d_q must be divisible by attn_dim_heads
                attn_depth = cfg.get("attn_depth", 2)

                cross_attn = SpatialTransformer1d(
                    in_channels=d_q,
                    n_heads=d_q // attn_dim_heads,
                    d_head=attn_dim_heads,
                    depth=attn_depth,
                    context_dim=d_kv,
                )

                setattr(self, f"cross_region_attn_{region}", cross_attn)
                setattr(self, f"attended_regions_cross_region_attn_{region}", attend_to_regions)

                print(f"Using cross-region attention for region='{region}':"
                      f"\n- Query: {region}"
                      f"\n- Key/Value: {attend_to_regions}")

                self._decoder_modules_names += [f"cross_region_attn_{region}"]

        # == points embeddings
        # (to recover skeletal structure after expansion over joint dimension)
        self.point_emb_dim  = cfg.get("point_emb_dim", 64)

        self.point_emb_torsoarms = nn.Embedding(self.Npts_torsoarms, self.point_emb_dim)
        self.point_emb_rh = nn.Embedding(self.Npts_rh, self.point_emb_dim)
        self.point_emb_lh = nn.Embedding(self.Npts_lh, self.point_emb_dim)
        self.point_emb_face = nn.Embedding(self.Npts_face, self.point_emb_dim)

        self._decoder_modules_names += [
            "point_emb_torsoarms",
            "point_emb_rh",
            "point_emb_lh",
            "point_emb_face",
        ]

        # == decoder modules
        self.decoder_torsoarms = GenericSkelRegionMotionDecoder(
            dim_in=dim_enc_torsoarms + self.point_emb_dim,
            Npts=Npts_torsoarms,
            **decoder_torsoarms_cfg,
        )
        self.decoder_rh = GenericSkelRegionMotionDecoder(
            dim_in=dim_enc_rh + self.point_emb_dim,
            Npts=Npts_rh,
            **decoder_rh_cfg,
        )
        self.decoder_lh = GenericSkelRegionMotionDecoder(
            dim_in=dim_enc_lh + self.point_emb_dim,
            Npts=Npts_lh,
            **decoder_lh_cfg,
        )
        self.decoder_face = GenericSkelRegionMotionDecoder(
            dim_in=dim_enc_face + self.point_emb_dim,
            Npts=Npts_face,
            **decoder_face_cfg,
        )

        self._decoder_modules_names += [
            # -- motion decoders
            "decoder_torsoarms",
            "decoder_rh",
            "decoder_lh",
            "decoder_face",
        ]

        # == registering the neutral face
        # (to learn only variations relative to this reference later on)
        neutral_face = cfg.get("neutral_face", None)
        self.neutral_face_initialized = False
        if neutral_face is not None:
            self.register_buffer("neutral_face", neutral_face)
            self.neutral_face_initialized = True
        else:
            self.register_buffer("neutral_face", torch.zeros(self.Npts_face, 3))

        # == module to predict face center from shoulders
        # (to recenter face)
        # NB: takes as input N_shoulders * num_coordinates (i.e x, y, z) = 2 * 3
        self.face_center_predictor = MLP(sizes=[2 * 3, 32, 32, 3], act=nn.ReLU, final_act=None)
        self._decoder_modules_names += ["face_center_predictor"]

        # == optional final temporal convolution module to refine the face
        # (to reduce jitter / temporal noise)
        self.face_temporal_refiner = None
        if cfg.get("use_face_temp_refiner", False):
            self.face_temporal_refiner = nn.Conv1d(
                in_channels=3 * self.Npts_face,
                out_channels=3 * self.Npts_face,
                kernel_size=3,
                padding=1,
                groups=self.Npts_face
            )
            nn.init.zeros_(self.face_temporal_refiner.weight)
            nn.init.zeros_(self.face_temporal_refiner.bias)
            self._decoder_modules_names += ["face_temporal_refiner"]

    @property
    def encoder_modules_names(self):
        return self._encoder_modules_names

    def decoder_modules_names(self):
        return self._decoder_modules_names

    def center_pts(self, pts: torch.Tensor, id_center: int):
        """
        `pts` is a torch Tensor of shape (B, T, N, 3).
        Center `pts` so that pts[:, :, id_center] is at origin [0, 0, 0].
        """
        return pts - pts[:, :, [id_center]]

    def encode(self, X: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Split regions
        torsoarms = X[:, :, self.ids_torsoarms]  # (B, T, Npts_torsoarms, 3)
        rh = self.center_pts(X[:, :, self.ids_rh], 0)  # (B, T, Npts_rhs, 3)
        lh = self.center_pts(X[:, :, self.ids_lh], 0)  # (B, T, Npts_lh, 3)
        face = X[:, :, self.ids_face]  # (B, T, Npts_face, 3)
        # -- initializing neutral face on 1st batch if not provided
        if self.training and not self.neutral_face_initialized:
            with torch.no_grad():
                # compute mean across batch and time
                self.neutral_face.copy_(face.mean(dim=(0, 1)))  # (Npts_face, 3)
                self.neutral_face_initialized = True
        # -- subtracting neutral face from input face (to keep only relative displacements)
        face = face - self.neutral_face[None, None, :, :]  # (B, T, Npts_face, 3)

        # 2. Encode separately regions
        h_torsoarms = self.encoder_torsoarms(torsoarms)  # (B, T, dim_enc_torsoarms)
        h_rh = self.encoder_rh(rh)  # (B, T, dim_enc_rh)
        h_lh = self.encoder_lh(lh)  # (B, T, dim_enc_lh)
        h_face = self.encoder_face(face)  # (B, T, dim_enc_face)

        # 3. VAE heads (one per region) & sample z_region ~ N(mu_reg, exp(1/2 * logvar_reg))
        # --- apply FC (optional)
        if self.fc_torsoarms:
            h_torsoarms = self.fc_torsoarms(h_torsoarms)
        if self.fc_rh:
            h_rh = self.fc_rh(h_rh)
        if self.fc_lh:
            h_lh = self.fc_lh(h_lh)
        if self.fc_face:
            h_face = self.fc_face(h_face)

        # --- predict mu and logvar
        if self.shared_latent_distribution:
            h_concat = torch.cat(
                [h_torsoarms, h_rh, h_lh, h_face],
                dim=-1
            )  # (B, T, sum{dim_enc_region})
            mu = self.mu_proj(h_concat)
            logvar = self.logvar_proj(h_concat)
        else:
            mu_torsoarms = self.mu_proj_torsoarms(h_torsoarms)  # (B, T, latent_dim_torsoarms)
            mu_rh = self.mu_proj_rh(h_rh)  # (B, T, latent_dim_rh)
            mu_lh = self.mu_proj_lh(h_lh)  # (B, T, latent_dim_lh)
            mu_face = self.mu_proj_face(h_face)  # (B, T, latent_dim_face)

            logvar_torsoarms = self.logvar_proj_torsoarms(h_torsoarms)
            logvar_rh = self.logvar_proj_rh(h_rh)
            logvar_lh = self.logvar_proj_lh(h_lh)
            logvar_face = self.logvar_proj_face(h_face)

            # --- concatenate [TORSO ARMS | RH | LH | FACE]
            mu = torch.cat(
                [mu_torsoarms, mu_rh, mu_lh, mu_face],
                dim=-1
            )  # (B, T, sum{latent_dim_region})

            logvar = torch.cat(
                [logvar_torsoarms, logvar_rh, logvar_lh, logvar_face],
                dim=-1
            )  # (B, T, sum{latent_dim_region})
        z = self.reparameterize(mu, logvar)  # (B, T, sum{latent_dim_region})

        return mu, logvar, z

    def decode(self, Z: torch.Tensor, pad_mask=None) -> torch.Tensor:
        # 1. Split regions
        z_torsoarms, z_rh, z_lh, z_face = torch.split(
            Z,
            [
                self.latent_dim_torsoarms,
                self.latent_dim_rh,
                self.latent_dim_rh,
                self.latent_dim_face,
            ],
            dim=-1
        )
        B, T = z_torsoarms.shape[:2]

        # 2.a) Projections (d_lat -> d_enc)
        z_torsoarms = self.dec_proj_torsoarms(z_torsoarms)  # (B, T, dim_enc_torso_arms)
        z_rh = self.dec_proj_rh(z_rh)  # (B, T, dim_enc_rh)
        z_lh = self.dec_proj_lh(z_lh)  # (B, T, dim_enc_lh)
        z_face = self.dec_proj_face(z_face)  # (B, T, dim_enc_face)

        # 2.b) OPTIONAL cross-region attention
        named_zs = {
            "torsoarms": z_torsoarms,
            "rh": z_rh,
            "lh": z_lh,
            "face": z_face,
        }
        assert set(named_zs.keys()) == set(self.cross_region_attn_names), "Keys must match cross-region attn names"

        out_zs = {}

        for name in named_zs.keys():
            attn = getattr(self, f"cross_region_attn_{name}", None)
            if attn is None:
                out_zs[name] = named_zs[name]
                continue
            query = named_zs[name]
            context = torch.cat(
                [named_zs[r] for r in getattr(self, f"attended_regions_cross_region_attn_{name}")],
                dim=-1
            )
            query = add_positional_sinusoidal_encoding(query)  # (B, T, d_q)
            context = add_positional_sinusoidal_encoding(context)  # (B, T, d_c)
            out_zs[name] = attn(x=query, context=context, context_mask=pad_mask)

        z_torsoarms = out_zs["torsoarms"]
        z_rh = out_zs["rh"]
        z_lh = out_zs["lh"]
        z_face = out_zs["face"]

        # 3.a) Broadcast (retrieve N_reg) - expanding embeddings over regions joints
        z_torsoarms = z_torsoarms.unsqueeze(2).expand(-1, -1, self.Npts_torsoarms, -1)  # (B, T, Npts_torsoarms, dim_enc_torsoarms)
        z_rh = z_rh.unsqueeze(2).expand(-1, -1, self.Npts_rh, -1)  # (B, T, Npts_rh, dim_enc_rh)
        z_lh = z_lh.unsqueeze(2).expand(-1, -1, self.Npts_lh, -1)  # (B, T, Npts_lh, dim_enc_lh)
        z_face = z_face.unsqueeze(2).expand(-1, -1, self.Npts_face, -1)  # (B, T, Npts_face, dim_enc_face)

        # 3.b) Concatenate point embeddings
        p_torsoarms = self.point_emb_torsoarms.weight  # (Npts_torsoarms, point_emb_dim)
        p_rh = self.point_emb_rh.weight  # (Npts, d_point)
        p_lh = self.point_emb_lh.weight  # (Npts, d_point)
        p_face = self.point_emb_face.weight  # (Npts, d_point)

        p_torsoarms = p_torsoarms[None, None, :, :].expand(B, T, -1, -1)
        p_rh = p_rh[None, None, :, :].expand(B, T, -1, -1)
        p_lh = p_lh[None, None, :, :].expand(B, T, -1, -1)
        p_face = p_face[None, None, :, :].expand(B, T, -1, -1)

        dec_in_torsoarms = torch.cat([z_torsoarms, p_torsoarms], dim=-1)
        dec_in_rh = torch.cat([z_rh, p_rh], dim=-1)
        dec_in_lh = torch.cat([z_lh, p_lh], dim=-1)
        dec_in_face = torch.cat([z_face, p_face], dim=-1)

        # 3.c) Decode separately regions
        torsoarms_recon = self.decoder_torsoarms(dec_in_torsoarms)  # (B, T, Npts_torsoarms, 3)
        rh_recon = self.decoder_rh(dec_in_rh)  # (B, T, Npts_rh, 3)
        lh_recon = self.decoder_lh(dec_in_lh)  # (B, T, Npts_lh, 3)
        face_recon = self.decoder_face(dec_in_face)  # (B, T, Npts_face, 3)

        # 4.a) Connect-back centered hands wrists to `torsoarms` predicted wrists positions
        rh_recon = self.center_pts(rh_recon, 0) + torsoarms_recon[:, :, [self.id_torsoarms_rw]]
        lh_recon = self.center_pts(lh_recon, 0) + torsoarms_recon[:, :, [self.id_torsoarms_lw]]

        # 4.b) Predict face center & translate back the face
        shoulders_recon = torsoarms_recon[:, :, [self.id_torsoarms_rshould, self.id_torsoarms_lshould]]
        predicted_face_center = self.face_center_predictor(shoulders_recon.reshape(B, T, -1))  # (B, T, 3)
        face_recon = face_recon + self.neutral_face[None, None] + predicted_face_center[:, :, None]

        # 4.c) OPTIONAL temporal face refinement w/ convolution
        if self.face_temporal_refiner is not None:
            # (B, T, Npts_face, 3) -> (B, T, 3*Npts_face) -> (B, 3*Npts_face, T)
            face_flat = face_recon.reshape(B, T, 3 * self.Npts_face)
            face_flat = face_flat.transpose(1, 2)  # because Conv1d expects channels first
            # applying convolution
            face_flat = self.face_temporal_refiner(face_flat)
            # (B, 3*Npts_face, T) -> (B, T, 3*Npts_face) -> (B, T, Npts_face, 3)
            face_flat = face_flat.transpose(1, 2)
            face_recon = face_flat.reshape(B, T, self.Npts_face, 3)

        # 5. Re-assemble different regions as in the initial tensor
        B, T = torsoarms_recon.shape[:2]
        X_recon = torch.zeros(
            (B, T, self.Npts_tot, 3),
            device=torsoarms_recon.device,
            dtype=torsoarms_recon.dtype
        )
        X_recon[:, :, self.ids_torsoarms] = torsoarms_recon
        X_recon[:, :, self.ids_rh] = rh_recon
        X_recon[:, :, self.ids_lh] = lh_recon
        X_recon[:, :, self.ids_face] = face_recon

        return X_recon
