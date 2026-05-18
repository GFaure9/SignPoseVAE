import torch
import torch.nn as nn

from .abstract_vae import VAE

from ...data.skeletal_data import MediaPipe50KeyPoints

from ..components import (
build_adjacency_matrix, normalize_adjacency,
BodyEncSpatial, FaceEncSpatial,
BodyEncTemp, FaceEncTemp,
BodyFaceFusion,
BodyFaceLatentTempDec,
BodyFaceSplitter,
BodyDecSpatial, FaceDecSpatial,
)


class SkelMotionVAE(VAE):
    def __init__(self, cfg):
        super().__init__()

        # === PARAMS
        T = cfg.get("T", 300)  # max num. frames (10 sec at 30 FPS)
        Npts_body = cfg.get("Npts_body", 50)
        edges_body = cfg.get("edges_body", MediaPipe50KeyPoints.connections())
        if isinstance(edges_body, str):  # if we provide a path to a .txt file with edges (as NODEi, NODEj per line)
            edges = []
            with open(edges_body, "r") as f:
                for line in f:
                    i, j = map(int, line.split(","))
                    edges.append((i, j))
            edges_body = edges
        Npts_face = cfg.get("Npts_face", 128)
        latent_dim = cfg.get("latent_dim", 128)
        # latent_expanded_dim = cfg.get("latent_expanded_dim", 256)  # uncomment if vae head shape is (B, latent_dim)

        self.T = T
        self.Npts_body = Npts_body
        self.A = build_adjacency_matrix(Npts_body, edges_body)
        self.Npts_face = Npts_face
        self.latent_dim = latent_dim
        # self.latent_expanded_dim = latent_expanded_dim  # uncomment if vae head shape is (B, latent_dim)

        # --- BODY ENC
        body_enc_spatial_default_cfg = {
            "n_gcn_layers": 2,
            "dim_out": 64,
        }
        body_enc_temp_default_cfg = {
            "dim_out": 128,
            "n_restconv_blocks": 2,
            "convt_kernel_size": 3,
            "convt_dilation": 1,
            "use_attn": True,
            "attn_heads": 4,
        }

        # --- FACE ENC
        face_enc_spatial_default_cfg = {
            "dim_hidden": 64,
            "dim_out": 48,
        }
        face_enc_temp_default_cfg = {
            "dim_out": 128,
            "n_restconv_blocks": 2,
            "convt_kernel_size": 3,
            "convt_dilation": 1,
            "use_attn": True,
            "attn_heads": 4,
        }

        # --- FUSION
        bodyface_fusion_default_cfg = {
            "dim_out": 256,
            "use_attn": True,
            "attn_heads": 4,
        }

        # --- LATENT TEMP DEC
        bodyface_latent_temp_dec_default_cfg = {
            "dim_ft_body_dec": 32,
            "dim_ft_face_dec": 16,
            "n_tconv_blocks": 2,
            "use_attn": True,
            "attn_heads": 4,
        }

        # --- BODY DEC
        body_dec_spatial_default_cfg = {
            "dim_hidden": 64,
        }

        # --- FACE DEC
        face_dec_spatial_default_cfg = {
            "dim_hidden": 32,
        }

        body_enc_spatial_cfg = cfg.get("body_enc_spatial", body_enc_spatial_default_cfg)
        body_enc_temp_cfg = cfg.get("body_enc_temp", body_enc_temp_default_cfg)
        face_enc_spatial_cfg = cfg.get("face_enc_spatial", face_enc_spatial_default_cfg)
        face_enc_temp_cfg = cfg.get("face_enc_temp", face_enc_temp_default_cfg)
        bodyface_fusion_cfg = cfg.get("bodyface_fusion", bodyface_fusion_default_cfg)
        bodyface_latent_temp_dec_cfg = cfg.get("bodyface_latent_temp_dec", bodyface_latent_temp_dec_default_cfg)
        body_dec_spatial_cfg = cfg.get("body_dec_spatial", body_dec_spatial_default_cfg)
        face_dec_spatial_cfg = cfg.get("face_dec_spatial", face_dec_spatial_default_cfg)

        # === MODULES
        self.body_encoder = nn.Sequential(
            BodyEncSpatial(Npts_body=Npts_body, **body_enc_spatial_cfg, A_norm=normalize_adjacency(self.A)),
            BodyEncTemp(Npts_body=Npts_body, dim_in=body_enc_spatial_cfg["dim_out"], **body_enc_temp_cfg)
        )
        self.face_encoder = nn.Sequential(
            FaceEncSpatial(Npts_face=Npts_face, **face_enc_spatial_cfg),
            FaceEncTemp(Npts_face=Npts_face, dim_in=face_enc_spatial_cfg["dim_out"], **face_enc_temp_cfg)
        )
        self.bodyface_fusion = BodyFaceFusion(
            dim_in_body=body_enc_temp_cfg["dim_out"],
            dim_in_face=face_enc_temp_cfg["dim_out"],
            **bodyface_fusion_cfg,
        )
        fusion_dim = bodyface_fusion_cfg["dim_out"]
        fc = cfg.get("fc", False)
        self.fc = None
        if fc:  # optional?
            self.fc = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, fusion_dim),
                nn.ReLU()
            )
        self.mu_proj = nn.Linear(fusion_dim, latent_dim)
        self.logvar_proj = nn.Linear(fusion_dim, latent_dim)
        # ******** uncomment if vae head shape is (B, latent_dim) ********
        # self.latent_expander = nn.Sequential(  # to retrieve temporal dim
        #     nn.Linear(latent_dim, latent_expanded_dim),
        #     nn.ReLU(),
        #     nn.Linear(latent_expanded_dim, T * latent_expanded_dim)
        # )
        # ****************************************************************
        self.bodyface_latent_temp_decoder = BodyFaceLatentTempDec(
            T=T,
            # dim_in=latent_expanded_dim,  # uncomment if vae head shape is (B, latent_dim)
            dim_in=latent_dim,
            **bodyface_latent_temp_dec_cfg,
        )
        dim_ft_body_dec = bodyface_latent_temp_dec_cfg["dim_ft_body_dec"]
        dim_ft_face_dec = bodyface_latent_temp_dec_cfg["dim_ft_face_dec"]
        self.bodyface_splitter = BodyFaceSplitter(
            Npts_body=Npts_body,
            Npts_face=Npts_face,
            dim_ft_body=dim_ft_body_dec,
            dim_ft_face=dim_ft_face_dec,
        )
        self.body_decoder = BodyDecSpatial(
            dim_in=dim_ft_body_dec, **body_dec_spatial_cfg,
        )
        self.face_decoder = FaceDecSpatial(
            dim_in=dim_ft_face_dec, **face_dec_spatial_cfg,
        )

        encoder_modules_names = [
            "body_encoder",
            "face_encoder",
            "bodyface_fusion",
            "logvar_proj",
            "mu_proj",
        ]
        if fc:
            encoder_modules_names += ["fc"]
        self._encoder_modules_names = encoder_modules_names

        self._decoder_modules_names = [
            "bodyface_latent_temp_decoder",
            "bodyface_splitter",
            "body_decoder",
            "face_decoder",
        ]

    @property
    def encoder_modules_names(self):
        return self._encoder_modules_names

    @property
    def decoder_modules_names(self):
        return self._decoder_modules_names

    def encode(self, X: torch.Tensor, **kwargs):
        # split body & face
        body = X[:, :, :self.Npts_body]  # (B, T, Npts_body, 3)
        face = X[:, :, self.Npts_body:]  # (B, T, Npts_face, 3)

        # encode separately + fuse
        emb_body = self.body_encoder(body)  # (B, T, dim_enc_body)
        emb_face = self.face_encoder(face)  # (B, T, dim_enc_face)
        emb_bodyface = self.bodyface_fusion(emb_body=emb_body, emb_face=emb_face)  # (B, T, dim_fus) [or (B, dim_latent) if temp pooling]

        # VAE head (latent encoding, i.e. predict mu, logvar and sample z ~ N(mu, exp(1/2 * logvar))
        h = emb_bodyface
        if self.fc:
            h = self.fc(h)
        mu = self.mu_proj(h)  # (B, T, dim_latent) [or (B, dim_latent)]
        logvar = self.logvar_proj(h)  # (B, T, dim_latent) [or (B, dim_latent)]
        z = self.reparameterize(mu, logvar)  # (B, T, dim_latent) [or (B, dim_latent)]

        return mu, logvar, z

    def decode(self, Z: torch.Tensor, **kwargs):
        # B, D_lat = Z.shape  # uncomment if vae head shape is (B, latent_dim)
        B, T, D_lat = Z.shape  # comment if vae head shape is (B, latent_dim)
        assert D_lat == self.latent_dim
        # Z is of shape (B, T, dim_latent) [or (B, dim_latent)]

        # ******** uncomment if vae head shape is (B, latent_dim) ********
        # # expand
        # Z = self.latent_expander(Z).view(B, self.T, self.latent_expanded_dim)  # (B, T, latent_expanded_dim)
        # ****************************************************************

        # temp decoding + split
        z_temp_dec = self.bodyface_latent_temp_decoder(Z)  # (B, T, Npts_body*dim_ft_body + Npts_face*dim_ft_face)
        body_recon, face_recon = self.bodyface_splitter(z_temp_dec)  # (B, T, Npts_body, dim_ft_body) & (B, T, Npts_body, dim_ft_face)

        # spatial decoding
        body_recon = self.body_decoder(body_recon)  # (B, T, Npts_body, 3)
        face_recon = self.face_decoder(face_recon)  # (B, T, Npts_face, 3)

        X_recon = torch.cat([body_recon, face_recon], dim=2)  # (B, T, Npts, 3)

        return X_recon
