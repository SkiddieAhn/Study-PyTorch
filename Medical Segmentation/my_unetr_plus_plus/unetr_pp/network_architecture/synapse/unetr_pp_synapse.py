from torch import nn
from typing import Tuple, Union
from unetr_pp.network_architecture.neural_network import SegmentationNetwork
from unetr_pp.network_architecture.dynunet_block import UnetOutBlock, UnetResBlock, get_conv_layer
from unetr_pp.network_architecture.synapse.model_components import UnetrPPEncoder, UnetrUpBlock
from unetr_pp.network_architecture.my_module import My_Fusion, My_Fusion2

class UNETR_PP(SegmentationNetwork):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: [64, 128, 128],
            feature_size: int = 16,
            hidden_size: int = 256,
            num_heads: int = 4,
            pos_embed: str = "perceptron",  # TODO: Remove the argument
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.

        Examples::
                    network = UNETR_PP(in_channels=input_channels,
                                    out_channels=num_classes,
                                    img_size=crop_size,
                                    feature_size=16,
                                    num_heads=4,
                                    depths=[3, 3, 3, 3,3],
                                    dims=[16,32, 64, 128, 256],
                                    do_ds=True,)
        """

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.patch_size = (2, 4, 4)
        self.feat_size = (
            img_size[0] // self.patch_size[0] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[1] // self.patch_size[1] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[2] // self.patch_size[2] // 8,  # 8 is the downsampling happened through the four encoders stages
        )
        self.hidden_size = hidden_size

        self.unetr_pp_encoder = UnetrPPEncoder(dims=dims, depths=depths, num_heads=num_heads)

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=8 * 8 * 8,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=16 * 16 * 16,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=32 * 32 * 32,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(2, 4, 4),
            norm_name=norm_name,
            out_size=64 * 128 * 128,
            conv_decoder=True, # upsampling -> TrasposedConv
        )

        self.fusion3=My_Fusion2(
            HWD_e=8*8*8,
            proj_size=64, 
            dim_e=feature_size*8,
        )

        self.fusion2=My_Fusion2(
            HWD_e=16*16*16,
            proj_size=64, 
            dim_e=feature_size*4,
        )

        self.fusion1=My_Fusion2(
            HWD_e=32*32*32,
            proj_size=64, 
            dim_e=feature_size*2,
        )

        # self.fusion3=My_Fusion(
        #     HWD_e=8*8*8,
        #     HWD_r=4*4*4,
        #     proj_size=64, 
        #     dim_e=feature_size*8,
        #     dim_r=feature_size*16
        # )

        # self.fusion2=My_Fusion(
        #     HWD_e=16*16*16,
        #     HWD_r=8*8*8,
        #     proj_size=64, 
        #     dim_e=feature_size*4,
        #     dim_r=feature_size*8
        # )

        # self.fusion1=My_Fusion(
        #     HWD_e=32*32*32,
        #     HWD_r=16*16*16,
        #     proj_size=64, 
        #     dim_e=feature_size*2,
        #     dim_r=feature_size*4
        # )

        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        # x_in: H x W x D x C

        x_output, hidden_states = self.unetr_pp_encoder(x_in)

        convBlock = self.encoder1(x_in) # H x W x D x 16

        # Four encoders
        enc1 = hidden_states[0] # H/4 x W/4 x D/2 x 32
        enc2 = hidden_states[1] # H/8 x W/8 x D/4 x 64
        enc3 = hidden_states[2] # H/16 x W/16 x D/8 x 128
        enc4 = hidden_states[3] 
        enc4 = self.proj_feat(enc4, self.hidden_size, self.feat_size) # H/32 x W/32 x D/16 x 256

        # fusion
        fs3=self.fusion3(enc3,enc4)
        fs2=self.fusion2(enc2,enc3)
        fs1=self.fusion1(enc1,enc2)

        # Five decoders
        dec4 = self.decoder4(enc4, fs3)
        dec3 = self.decoder3(dec4, fs2)
        dec2 = self.decoder2(dec3, fs1)
        dec1 = self.decoder1(dec2, convBlock)

        if self.do_ds:
            logits = [self.out1(dec1), self.out2(dec2), self.out3(dec3)]
        else:
            logits = self.out1(dec1)

        return logits
